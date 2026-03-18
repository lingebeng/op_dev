"""
v3: 转置布局 (L, batch)

相比 v2 的变化:
- 数据布局从 (batch, L) 转为 (L, batch)
- L 在 dim0 (sublane, align 8), batch 在 dim1 (lane, align 128)
- matmul: tril @ chunk  (而非 chunk @ triu)
- carry 形状: (1, block_B) — sublane row, vreg 的最小自然单元

性能说明:
- 单独的 cumsum kernel, v3 不如 v2 (v2 的 block_L 对齐 128, MXU 利用率更高)
- v3 的 carry 操作在 vreg 层面更高效, 但 carry 占比 < 0.01%, 被 matmul 淹没

适用场景:
- 融合到更大的 kernel 中时 (如 GLA attention), 序列维度在 dim0 是 pipeline 约定
- L 很短时 (< 128), block_L 对齐 8 避免大量 pad 浪费
- 团队协定 (time, feature) 布局的代码库
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# =====================================================================
# TPU 对齐常量
# =====================================================================
_TPU_ROW_ALIGN = 8  # 第 0 维 (sublane) 对齐粒度
_TPU_COL_ALIGN = 128  # 第 1 维 (lane) 对齐粒度


def _align_up(x: int, align: int) -> int:
    return ((x + align - 1) // align) * align


# =====================================================================
# v3: 转置布局 (L, batch), carry = sublane row
# =====================================================================


def _pipeline_scan_kernel(x_ref, out_ref, acc_ref):
    """
    转置布局: tile 形状 (block_L, block_B)。

    tril_mat @ chunk 计算 local cumsum (沿 dim 0)。
    acc_ref: VMEM scratch (1, block_B) — 一个 sublane row。
    grid: (num_batch_blocks, num_L_blocks), ("parallel", "arbitrary")。
    """
    block_L = x_ref.shape[0]
    il = pl.program_id(1)  # L block index (arbitrary, 内层)
    tril_mat = jnp.tril(jnp.ones((block_L, block_L), dtype=jnp.float32))

    @pl.when(il == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    chunk = x_ref[...].astype(jnp.float32)
    # tril @ chunk: (block_L, block_L) @ (block_L, block_B) → (block_L, block_B)
    local_cs = jnp.dot(tril_mat, chunk, precision=jax.lax.Precision.HIGHEST)
    local_cs = local_cs + acc_ref[...]  # (1, block_B) 沿 sublane 广播
    out_ref[...] = local_cs.astype(out_ref.dtype)

    # carry: 最后一个 sublane row — 选寄存器, 无需 shift
    acc_ref[...] = local_cs[-1:, :]


def cumsum(
    x: jax.Array,
    axis: int = -1,
    dtype: jnp.dtype = jnp.float32,
    block_L: int = 512,
    block_B: int = 128,
) -> jax.Array:
    """
    v3 cumsum: 转置布局 (L, batch)。

    - block_L: L 方向分块大小, 对齐到 8 (sublane)
    - block_B: batch 方向分块大小, 对齐到 128 (lane)
    """
    ndim = x.ndim
    axis = axis % ndim

    x_moved = jnp.moveaxis(x, axis, 0)
    orig_shape = x_moved.shape
    L = orig_shape[0]

    batch_size = x_moved[0].size
    # axis 已在 dim0, 直接 reshape 成 (L, batch), 无需转置
    x_2d = x_moved.reshape(L, batch_size)

    # L 在 dim0 (sublane align 8), batch 在 dim1 (lane align 128)
    block_L = _align_up(block_L, _TPU_ROW_ALIGN)
    block_B = _align_up(block_B, _TPU_COL_ALIGN)
    L_padded = _align_up(L, block_L)
    batch_padded = _align_up(batch_size, block_B)

    pad_L = L_padded - L
    pad_B = batch_padded - batch_size
    if pad_L > 0 or pad_B > 0:
        x_2d = jnp.pad(x_2d, ((0, pad_L), (0, pad_B)))

    interpret = jax.default_backend() != "tpu"
    num_L_blocks = L_padded // block_L
    num_batch_blocks = batch_padded // block_B

    # BlockSpec: grid(batch, L) → data(L, batch)
    chunk_spec = pl.BlockSpec(
        index_map=lambda ib, il: (il, ib),
        block_shape=(block_L, block_B),
    )

    out_2d = pl.pallas_call(
        _pipeline_scan_kernel,
        out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(num_batch_blocks, num_L_blocks),
            in_specs=[chunk_spec],
            out_specs=chunk_spec,
            # scratch: (1, block_B) — 一个 sublane row
            scratch_shapes=[pltpu.VMEM((1, block_B), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            # batch parallel, L arbitrary (串行 + DMA pipeline)
            dimension_semantics=("parallel", "arbitrary"),
        ),
        interpret=interpret,
    )(x_2d)

    # 去掉 padding 并 reshape 回原始形状
    out_2d = out_2d[:L, :batch_size]
    out_moved = out_2d.reshape(orig_shape)
    return jnp.moveaxis(out_moved, 0, axis)


# =====================================================================
# 测试 + 性能对比
# =====================================================================

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    backend = jax.default_backend()
    print(f"Backend: {backend}")

    # ---------- 正确性测试 ----------
    print("=" * 60)
    print("正确性测试 (v3 转置布局)")
    print("=" * 60)

    test_cases = [
        ((8,), 0, 128, "1-D, axis=0"),
        ((4, 256), 1, 128, "2-D, axis=1"),
        ((3, 4, 256), 2, 128, "3-D, axis=2"),
        ((2, 3, 4, 256), 1, 128, "4-D, axis=1"),
        ((16, 16, 512), 2, 256, "3-D, 大 batch"),
        ((8, 2048), 1, 512, "2-D, L=2048"),
        ((4, 4, 4096), 2, 512, "3-D, L=4096"),
    ]

    all_pass = True
    for shape, axis, bl, desc in test_cases:
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        ref = jnp.cumsum(x, axis=axis)
        out = cumsum(x, axis=axis, block_L=bl)
        max_err = float(jnp.max(jnp.abs(out - ref)))
        ok = bool(jnp.allclose(out, ref, atol=1e-4))
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc:25s} max_err={max_err:.2e}")

    print()
    if all_pass:
        print("所有测试通过!")
    else:
        print("存在失败的测试!")
        raise SystemExit(1)

    # ---------- 性能对比 ----------
    import timeit
    from cumsum_pallas_v2 import cumsum as cumsum_v2

    print()
    print("=" * 60)
    print("性能对比: v3 (转置) vs v2 (原布局) vs jnp.cumsum")
    print("=" * 60)

    perf_shapes = [
        ((16, 16, 512), 2, "3-D (16,16,512)"),
        ((32, 32, 1024), 2, "3-D (32,32,1024)"),
        ((64, 64, 2048), 2, "3-D (64,64,2048)"),
        ((16, 16, 4096), 2, "3-D (16,16,4096)"),
    ]
    block_L_sizes = [128, 256, 512]

    warmup = 10
    repeat = 100

    for shape, axis, desc in perf_shapes:
        x = jax.random.normal(key, shape, dtype=jnp.float32)

        # jnp.cumsum baseline
        ref_jit = jax.jit(jnp.cumsum, static_argnums=(1,))
        for _ in range(warmup):
            ref_jit(x, axis).block_until_ready()
        t0 = timeit.default_timer()
        for _ in range(repeat):
            ref_jit(x, axis).block_until_ready()
        jnp_time = (timeit.default_timer() - t0) / repeat * 1000

        # v2: block_B = 8, 16, 64, 128, 256, 512
        v2_B_sizes = [8, 16, 64, 128, 256, 512]
        v2_times = {}
        for bb in v2_B_sizes:
            v2_times[bb] = {}
            for bl in block_L_sizes:
                fn = jax.jit(
                    lambda x, _bl=bl, _bb=bb: cumsum_v2(
                        x, axis=axis, block_L=_bl, block_B=_bb
                    )
                )
                for _ in range(warmup):
                    fn(x).block_until_ready()
                t0 = timeit.default_timer()
                for _ in range(repeat):
                    fn(x).block_until_ready()
                v2_times[bb][bl] = (timeit.default_timer() - t0) / repeat * 1000

        # v3: block_B = 128, 256, 512
        v3_B_sizes = [128, 256, 512]

        print(f"\n  {desc}  (jnp.cumsum: {jnp_time:.3f} ms)")
        print(f"    {'config':>15s}", end="")
        for bl in block_L_sizes:
            print(f"  {'L=' + str(bl):>10s}", end="")
        print()
        print(f"    {'-' * 15}", end="")
        for _ in block_L_sizes:
            print(f"  {'-' * 10}", end="")
        print()

        # v2 rows
        for bb in v2_B_sizes:
            print(f"    {f'v2 B={bb}':>15s}", end="")
            for bl in block_L_sizes:
                print(f"  {v2_times[bb][bl]:>8.3f} ms", end="")
            print()

        # v3 rows
        for bb in v3_B_sizes:
            print(f"    {f'v3 B={bb}':>15s}", end="")
            for bl in block_L_sizes:
                fn = jax.jit(
                    lambda x, _bl=bl, _bb=bb: cumsum(
                        x, axis=axis, block_L=_bl, block_B=_bb
                    )
                )
                for _ in range(warmup):
                    fn(x).block_until_ready()
                t0 = timeit.default_timer()
                for _ in range(repeat):
                    fn(x).block_until_ready()
                t = (timeit.default_timer() - t0) / repeat * 1000
                print(f"  {t:>8.3f} ms", end="")
            print()
