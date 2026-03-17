import functools

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
    """向上对齐到 align 的倍数。"""
    return ((x + align - 1) // align) * align


# =====================================================================
# Kernel 函数
# =====================================================================


def _serial_scan_kernel(x_ref, out_ref, *, block_L):
    L_padded = x_ref.shape[1]
    num_blocks = L_padded // block_L
    triu_mat = jnp.triu(jnp.ones((block_L, block_L), dtype=jnp.float32))

    def body(i, running_sum):
        chunk = x_ref[:, pl.dslice(i * block_L, block_L)].astype(jnp.float32)
        local_cs = jnp.dot(chunk, triu_mat, precision=jax.lax.Precision.HIGHEST)
        local_cs = local_cs + running_sum
        out_ref[:, pl.dslice(i * block_L, block_L)] = local_cs.astype(out_ref.dtype)
        return local_cs[:, -1:]

    init_sum = jnp.zeros((x_ref.shape[0], 1), dtype=jnp.float32)
    jax.lax.fori_loop(0, num_blocks, body, init_sum)


def _chunk_cumsum_kernel(x_ref, out_ref, *, block_L):
    triu_mat = jnp.triu(jnp.ones((block_L, block_L), dtype=jnp.float32))
    chunk = x_ref[...].astype(jnp.float32)
    local_cs = jnp.dot(chunk, triu_mat, precision=jax.lax.Precision.HIGHEST)
    out_ref[...] = local_cs.astype(out_ref.dtype)


def cumsum(
    x: jax.Array,
    axis: int = -1,
    dtype: jnp.dtype = jnp.float32,
    block_L: int = 512,
    block_B: int = 8,
    mode: str = "serial",  # "serial", "parallel"
) -> jax.Array:
    """
    支持任意 shape 和任意 axis 的 cumsum。

    Args:
        mode: "serial" → 单 kernel L 串行,
              "parallel" → 两 kernel L 并行。
    """
    ndim = x.ndim
    axis = axis % ndim

    x_moved = jnp.moveaxis(x, axis, -1)
    orig_shape = x_moved.shape
    L = orig_shape[-1]

    batch_size = x_moved[..., 0].size
    x_2d = x_moved.reshape(batch_size, L)

    block_L = _align_up(block_L, _TPU_COL_ALIGN)
    block_B = _align_up(block_B, _TPU_ROW_ALIGN)
    L_padded = _align_up(L, block_L)
    batch_padded = _align_up(batch_size, block_B)

    pad_L = L_padded - L
    pad_B = batch_padded - batch_size
    if pad_L > 0 or pad_B > 0:
        x_2d = jnp.pad(x_2d, ((0, pad_B), (0, pad_L)))

    interpret = jax.default_backend() != "tpu"
    num_batch_blocks = batch_padded // block_B
    num_L_blocks = L_padded // block_L

    if mode == "parallel" and num_L_blocks > 1:
        chunk_spec = pl.BlockSpec(
            index_map=lambda ib, il: (ib, il),
            block_shape=(block_B, block_L),
        )
        # Kernel 1: 每个 chunk 独立做 local cumsum（batch × L 两维全并行）
        intermediate = pl.pallas_call(
            functools.partial(_chunk_cumsum_kernel, block_L=block_L),
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_batch_blocks, num_L_blocks),
                in_specs=[chunk_spec],
                out_specs=chunk_spec,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel"),
            ),
            interpret=interpret,
        )(x_2d)

        # Pass 2 (host): 提取每个 chunk 的总和 → exclusive prefix sum
        chunk_sums = intermediate[
            :, block_L - 1 :: block_L
        ]  # (batch_padded, num_L_blocks)
        inclusive = jnp.cumsum(chunk_sums, axis=1)
        offsets = jnp.pad(
            inclusive[:, :-1], ((0, 0), (1, 0))
        )  # (batch_padded, num_L_blocks)

        # Pass 3: reshape + broadcast 加回 offset
        inter_3d = intermediate.reshape(batch_padded, num_L_blocks, block_L)
        out_2d = (inter_3d + offsets[:, :, None]).reshape(batch_padded, L_padded)

    else:
        # ---- 单 kernel, L 维串行 ----
        row_spec = pl.BlockSpec(
            index_map=lambda ib: (ib, 0),
            block_shape=(block_B, L_padded),
        )
        out_2d = pl.pallas_call(
            functools.partial(_serial_scan_kernel, block_L=block_L),
            out_shape=jax.ShapeDtypeStruct(x_2d.shape, x_2d.dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                grid=(num_batch_blocks,),
                in_specs=[row_spec],
                out_specs=row_spec,
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",),
            ),
            interpret=interpret,
        )(x_2d)

    out_2d = out_2d[:batch_size, :L]
    out_moved = out_2d.reshape(orig_shape)
    return jnp.moveaxis(out_moved, -1, axis)


# =====================================================================
# 测试 + 性能对比
# =====================================================================

if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    backend = jax.default_backend()
    print(f"Backend: {backend}")

    # ---------- 正确性测试 ----------
    print("=" * 60)
    print("正确性测试")
    print("=" * 60)

    test_cases = [
        ((8,), 0, 128, "1-D, axis=0"),
        ((4, 256), 1, 128, "2-D, axis=1"),
        ((3, 4, 256), 2, 128, "3-D, axis=2"),
        ((2, 3, 4, 256), 1, 128, "4-D, axis=1"),
        ((16, 16, 512), 2, 256, "3-D, 大 batch"),
    ]

    all_pass = True
    for shape, axis, bl, desc in test_cases:
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        ref = jnp.cumsum(x, axis=axis)
        for m in ["serial", "parallel"]:
            out = cumsum(x, axis=axis, block_L=bl, mode=m)
            max_err = float(jnp.max(jnp.abs(out - ref)))
            ok = bool(jnp.allclose(out, ref, atol=1e-4))
            if not ok:
                all_pass = False
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {m:8s} {desc:25s} max_err={max_err:.2e}")

    print()
    if all_pass:
        print("所有测试通过!")
    else:
        print("存在失败的测试!")
        raise SystemExit(1)

    # ---------- 性能对比 ----------
    import timeit

    print()
    print("=" * 60)
    print("性能对比: serial vs parallel vs jnp.cumsum")
    print("=" * 60)

    perf_shapes = [
        ((16, 16, 512), 2, "3-D (16,16,512)"),
        ((32, 32, 1024), 2, "3-D (32,32,1024)"),
        ((64, 64, 2048), 2, "3-D (64,64,2048)"),
    ]
    chunk_sizes = [128, 256, 512]

    warmup = 10
    repeat = 100

    for shape, axis, desc in perf_shapes:
        x = jax.random.normal(key, shape, dtype=jnp.float32)

        # jnp.cumsum baseline
        ref_jit = jax.jit(jnp.cumsum, static_argnums=(1,))

        def ref_fn(x):
            return ref_jit(x, axis)

        for _ in range(warmup):
            ref_fn(x).block_until_ready()
        t0 = timeit.default_timer()
        for _ in range(repeat):
            ref_fn(x).block_until_ready()
        jnp_time = (timeit.default_timer() - t0) / repeat * 1000

        print(f"\n  {desc}  (jnp: {jnp_time:.3f} ms)")
        print(f"    {'chunk_size':>10s}  {'serial':>10s}  {'parallel':>10s}")
        print(f"    {'-' * 10}  {'-' * 10}  {'-' * 10}")

        for bl in chunk_sizes:
            times = {}
            for mode in ["serial", "parallel"]:
                fn = jax.jit(
                    lambda x, _bl=bl, _m=mode: cumsum(
                        x, axis=axis, block_L=_bl, mode=_m
                    )
                )
                for _ in range(warmup):
                    fn(x).block_until_ready()
                t0 = timeit.default_timer()
                for _ in range(repeat):
                    fn(x).block_until_ready()
                times[mode] = (timeit.default_timer() - t0) / repeat * 1000
            print(
                f"    {bl:>10d}  {times['serial']:>8.3f} ms  {times['parallel']:>8.3f} ms"
            )
