"""
用原生 JAX 实现 cumsum: reshape + 上三角矩阵乘法

支持对序列维度和 batch 维度同时分块，精确控制 tile 大小。

例: [1,2,3,4,5,6,7,8,9]
  reshape → [[1,2,3],[4,5,6],[7,8,9]]
  每个 chunk 乘上三角矩阵 → local cumsum
  再加上前面 chunk 的总和 → 全局 cumsum
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

# 全局精度设置
_PRECISION = lax.Precision.HIGHEST


def _dot(a, b, contracting_a, contracting_b):
    """封装 jax.lax.dot_general，统一使用 HIGHEST 精度"""
    return lax.dot_general(
        a,
        b,
        dimension_numbers=((contracting_a, contracting_b), ((), ())),
        precision=_PRECISION,
    )


def cumsum_reshape_triu(x, axis=-1, chunk_size=128, batch_chunk_size=None):
    """
    reshape + 上三角矩阵乘法实现 cumsum。
    全部使用 jax.lax.dot_general + Precision.HIGHEST，确保 TPU 上 float32 精度。

    Args:
        x: 输入张量
        axis: cumsum 的轴
        chunk_size: 序列维度的 chunk 大小 (对应 MXU 的 K/N 维)
        batch_chunk_size: batch 维度的 chunk 大小 (对应 MXU 的 M 维)
            None = 不分块; 设为 128 可让每次 matmul 刚好填满 128x128 MXU

    布局 (batch_chunk_size=128, chunk_size=128):
        输入 (B, L) → (num_batch_chunks, 128, num_seq_chunks, 128)
        每次 matmul: (128, 128) @ (128, 128) → 刚好填满 MXU
    """
    ndim = x.ndim
    axis = axis % ndim
    L = x.shape[axis]

    # 1. Pad 序列维度
    seq_pad = (chunk_size - L % chunk_size) % chunk_size
    if seq_pad > 0:
        pad_widths = [(0, 0)] * ndim
        pad_widths[axis] = (0, seq_pad)
        x = jnp.pad(x, pad_widths)
    L_padded = L + seq_pad
    num_seq_chunks = L_padded // chunk_size

    # 2. 标准化为 (B, L_padded) —— 把 axis 移到最后，展平 batch 维
    x = jnp.moveaxis(x, axis, -1)
    batch_shape = x.shape[:-1]
    B = math.prod(batch_shape) if batch_shape else 1
    x = x.reshape(B, L_padded)

    # 3. Chunk 序列维度 → (B, num_seq_chunks, chunk_size)
    x = x.reshape(B, num_seq_chunks, chunk_size)

    # 4. 可选: chunk batch 维度 → (num_batch_chunks, batch_chunk_size, num_seq_chunks, chunk_size)
    if batch_chunk_size is not None:
        batch_pad = (batch_chunk_size - B % batch_chunk_size) % batch_chunk_size
        if batch_pad > 0:
            x = jnp.pad(x, [(0, batch_pad), (0, 0), (0, 0)])
        B_padded = B + batch_pad
        num_batch_chunks = B_padded // batch_chunk_size
        x = x.reshape(num_batch_chunks, batch_chunk_size, num_seq_chunks, chunk_size)
        cs_ax = 3  # chunk_size 所在维度
        nsc_ax = 2  # num_seq_chunks 所在维度
    else:
        B_padded = B
        cs_ax = 2
        nsc_ax = 1

    # 5. 上三角矩阵 + local cumsum
    triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=x.dtype))
    # dot_general 收缩 chunk_size，结果的 chunk_size 维在最后
    local_cs = _dot(x, triu, (cs_ax,), (0,))

    # 6. chunk sums + inter-chunk exclusive prefix sum (也用 triu matmul)
    chunk_sums = local_cs[..., -1]
    # strictly upper triangular (k=1) 直接算 exclusive prefix sum，无需 shift
    triu_chunks = jnp.triu(
        jnp.ones((num_seq_chunks, num_seq_chunks), dtype=x.dtype), k=1
    )
    offsets = _dot(chunk_sums, triu_chunks, (nsc_ax,), (0,))

    # 7. 加上 offsets
    result = local_cs + offsets[..., None]

    # 8. Reshape 回原始形状
    if batch_chunk_size is not None:
        result = result.reshape(B_padded, num_seq_chunks, chunk_size)
    result = result[:B, :, :]
    result = result.reshape(B, L_padded)[:, :L]
    result = result.reshape(*batch_shape, L)
    result = jnp.moveaxis(result, -1, axis)
    return result


if __name__ == "__main__":
    import timeit

    print("=" * 60)
    print("cumsum_reshape_triu (jax.lax.dot_general, HIGHEST precision)")
    print("=" * 60)

    # 简单示例
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)
    print(f"\n输入: {x}")
    print(f"jnp.cumsum:               {jnp.cumsum(x)}")
    print(f"triu(cs=3):               {cumsum_reshape_triu(x, chunk_size=3)}")
    print(f"triu(cs=3, bcs=3):        {cumsum_reshape_triu(x, chunk_size=3, batch_chunk_size=3)}")

    # 正确性测试
    print("\n正确性测试:")
    key = jax.random.PRNGKey(42)
    test_cases = [
        ((16,), 0, 4, None, "1-D"),
        ((4, 32), 1, 8, None, "2-D axis=1"),
        ((4, 32), 1, 8, 2, "2-D axis=1 bcs=2"),
        ((3, 4, 64), 2, 16, None, "3-D axis=2"),
        ((3, 4, 64), 2, 16, 4, "3-D axis=2 bcs=4"),
        ((3, 4, 64), 1, 4, None, "3-D axis=1"),
        ((3, 4, 64), 1, 4, 3, "3-D axis=1 bcs=3"),
        ((64, 64, 2048), 2, 128, 128, "3-D bcs=128 cs=128"),
    ]

    for *params, desc in test_cases:
        if len(params) == 3:
            shape, ax, cs = params
            bcs = None
        else:
            shape, ax, cs, bcs = params
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        ref = jnp.cumsum(x, axis=ax)
        out = cumsum_reshape_triu(x, axis=ax, chunk_size=cs, batch_chunk_size=bcs)
        max_err = float(jnp.max(jnp.abs(out - ref)))
        ok = bool(jnp.allclose(out, ref, atol=1e-4))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc:25s} max_err={max_err:.2e}")

    # 性能对比
    print()
    print("=" * 60)
    print("性能对比 vs jnp.cumsum")
    print("=" * 60)

    perf_shapes = [
        ((64, 64, 2048), 2, "3-D (64,64,2048)"),
        ((16, 16, 4096), 2, "3-D (16,16,4096)"),
    ]
    chunk_sizes = [128, 256, 512]
    batch_chunk_sizes = [None, 64, 128, 256]
    warmup = 10
    repeat = 100

    for shape, axis, desc in perf_shapes:
        x = jax.random.normal(key, shape, dtype=jnp.float32)

        # baseline: jnp.cumsum
        fn_ref = jax.jit(jnp.cumsum, static_argnums=(1,))
        for _ in range(warmup):
            fn_ref(x, axis).block_until_ready()
        t0 = timeit.default_timer()
        for _ in range(repeat):
            fn_ref(x, axis).block_until_ready()
        t_ref = (timeit.default_timer() - t0) / repeat * 1000

        print(f"\n  {desc}  (jnp.cumsum: {t_ref:.3f} ms)")
        print(
            f"    {'cs':>5s}  {'bcs':>5s}  {'time':>10s}"
        )
        print(f"    {'-----':>5s}  {'-----':>5s}  {'----------':>10s}")

        for cs in chunk_sizes:
            for bcs in batch_chunk_sizes:
                fn = jax.jit(
                    lambda x, _cs=cs, _bcs=bcs: cumsum_reshape_triu(
                        x, axis=axis, chunk_size=_cs, batch_chunk_size=_bcs
                    )
                )
                for _ in range(warmup):
                    fn(x).block_until_ready()
                t0 = timeit.default_timer()
                for _ in range(repeat):
                    fn(x).block_until_ready()
                t = (timeit.default_timer() - t0) / repeat * 1000
                bcs_str = str(bcs) if bcs else "-"
                print(f"    {cs:>5d}  {bcs_str:>5s}  {t:>8.3f} ms")
