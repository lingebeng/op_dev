"""
想法: 用原生 JAX 实现 cumsum 的两种方式对比

方式 1: 直接 jnp.cumsum
方式 2: reshape 成 chunks + 上三角矩阵乘法 + chunk 间 prefix sum

例: [1,2,3,4,5,6,7,8,9]
  reshape → [[1,2,3],[4,5,6],[7,8,9]]
  每个 chunk 乘上三角矩阵 → local cumsum
  再加上前面 chunk 的总和 → 全局 cumsum
"""

import jax
import jax.numpy as jnp


def cumsum_native(x, axis=-1):
    """方式 1: 直接 jnp.cumsum"""
    return jnp.cumsum(x, axis=axis)


def cumsum_reshape_triu(x, axis=-1, chunk_size=3):
    """
    方式 2: reshape + 上三角矩阵乘法
    把 axis 维度切成 chunk，每个 chunk 用 triu matmul 做 local cumsum，
    再用 jnp.cumsum 算 chunk 间的 prefix sum 加回去。
    """
    ndim = x.ndim
    axis = axis % ndim
    L = x.shape[axis]

    # padding
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len > 0:
        pad_widths = [(0, 0)] * ndim
        pad_widths[axis] = (0, pad_len)
        x = jnp.pad(x, pad_widths)

    L_padded = L + pad_len
    num_chunks = L_padded // chunk_size

    # 把 axis 维度 reshape 成 (num_chunks, chunk_size)
    shape = list(x.shape)
    shape[axis : axis + 1] = [num_chunks, chunk_size]
    x_chunked = x.reshape(shape)  # (..., num_chunks, chunk_size, ...)

    # 上三角矩阵
    triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=x.dtype))

    # 每个 chunk 做 local cumsum: x @ triu (沿最后一维)
    # x_chunked 的 chunk_size 在 axis+1 维
    local_cs = jnp.tensordot(x_chunked, triu, axes=[[axis + 1], [0]])
    # tensordot 会把结果维度放到最后，需要 moveaxis 回去
    dest = axis + 1
    src = local_cs.ndim - 1
    if src != dest:
        local_cs = jnp.moveaxis(local_cs, src, dest)

    # 每个 chunk 的总和 = local_cs 在 chunk_size 维的最后一个元素
    chunk_sums = jnp.take(local_cs, -1, axis=axis + 1)  # (..., num_chunks, ...)

    # chunk 间的 inclusive prefix sum（也用上三角矩阵乘法，避免 jnp.cumsum）
    triu_chunks = jnp.triu(jnp.ones((num_chunks, num_chunks), dtype=x.dtype))
    inclusive = jnp.tensordot(chunk_sums, triu_chunks, axes=[[axis], [0]])
    # tensordot 把结果放到最后一维，moveaxis 回去
    if inclusive.ndim > 1:
        inclusive = jnp.moveaxis(inclusive, -1, axis)

    # exclusive prefix sum (shift right, pad 0)
    pad_widths = [(0, 0)] * inclusive.ndim
    pad_widths[axis] = (1, 0)
    slices = [slice(None)] * inclusive.ndim
    slices[axis] = slice(None, -1)
    offsets = jnp.pad(inclusive[tuple(slices)], pad_widths)

    # broadcast 加回去
    offsets = jnp.expand_dims(offsets, axis=axis + 1)
    result = local_cs + offsets

    # reshape 回去并去掉 padding
    out_shape = list(x.shape)  # padded shape
    result = result.reshape(out_shape)
    slices = [slice(None)] * ndim
    slices[axis] = slice(None, L)
    return result[tuple(slices)]


# =====================================================================
# 测试
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("对比: jnp.cumsum vs reshape+triu matmul")
    print("=" * 60)

    # 简单示例
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)
    print(f"\n输入: {x}")
    print(f"jnp.cumsum:       {cumsum_native(x)}")
    print(f"reshape+triu(cs=3): {cumsum_reshape_triu(x, chunk_size=3)}")

    # 正确性测试
    print("\n正确性测试:")
    key = jax.random.PRNGKey(42)
    test_cases = [
        ((16,), 0, 4, "1-D"),
        ((4, 32), 1, 8, "2-D axis=1"),
        ((3, 4, 64), 2, 16, "3-D axis=2"),
        ((3, 4, 64), 1, 4, "3-D axis=1"),
    ]

    for shape, axis, cs, desc in test_cases:
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        ref = jnp.cumsum(x, axis=axis)
        out = cumsum_reshape_triu(x, axis=axis, chunk_size=cs)
        max_err = float(jnp.max(jnp.abs(out - ref)))
        ok = bool(jnp.allclose(out, ref, atol=1e-4))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc:20s} chunk_size={cs:3d}  max_err={max_err:.2e}")

    # 性能对比
    import timeit

    print()
    print("=" * 60)
    print("性能对比")
    print("=" * 60)

    perf_shapes = [
        ((64, 64, 2048), 2, "3-D (64,64,2048)"),
        ((16, 16, 4096), 2, "3-D (16,16,4096)"),
    ]
    chunk_sizes = [128, 256, 512]
    warmup = 10
    repeat = 100

    for shape, axis, desc in perf_shapes:
        x = jax.random.normal(key, shape, dtype=jnp.float32)

        # baseline
        fn_ref = jax.jit(jnp.cumsum, static_argnums=(1,))
        for _ in range(warmup):
            fn_ref(x, axis).block_until_ready()
        t0 = timeit.default_timer()
        for _ in range(repeat):
            fn_ref(x, axis).block_until_ready()
        t_ref = (timeit.default_timer() - t0) / repeat * 1000

        print(f"\n  {desc}  (jnp.cumsum: {t_ref:.3f} ms)")
        print(f"    {'chunk_size':>10s}  {'reshape+triu':>12s}")
        print(f"    {'-' * 10}  {'-' * 12}")

        for cs in chunk_sizes:
            fn = jax.jit(
                lambda x, _cs=cs: cumsum_reshape_triu(x, axis=axis, chunk_size=_cs)
            )
            for _ in range(warmup):
                fn(x).block_until_ready()
            t0 = timeit.default_timer()
            for _ in range(repeat):
                fn(x).block_until_ready()
            t = (timeit.default_timer() - t0) / repeat * 1000
            print(f"    {cs:>10d}  {t:>10.3f} ms")
