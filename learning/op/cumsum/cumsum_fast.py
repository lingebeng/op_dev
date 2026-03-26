"""
递归层级 triu matmul cumsum — 纯 JAX 实现

算法:
  将序列分成 chunk_size 大小的块，每块用上三角矩阵乘法做 local cumsum，
  再对 chunk 总和递归地做 prefix sum（也用 triu matmul），最后加上 inter-chunk offsets。

  递归避免了构造 O(num_chunks²) 的大 triu 矩阵，实际递归深度 ≤ 3 层。
  全部使用 jax.lax.dot_general + Precision.HIGHEST，确保 TPU 上 float32 精度。

用法:
  from cumsum_fast import cumsum
  out = cumsum(x, axis=-1)
  out = cumsum(x, axis=1, chunk_size=256)
"""

import math
import timeit

import jax
import jax.numpy as jnp
from jax import lax

_PRECISION = lax.Precision.HIGHEST


# ============================================================
# 内部工具
# ============================================================

def _dot(a, b, contracting_a, contracting_b):
    return lax.dot_general(
        a, b,
        dimension_numbers=((contracting_a, contracting_b), ((), ())),
        precision=_PRECISION,
    )


def _normalize(x, axis):
    """将任意 ndim 张量标准化为 (B, L) 用于 cumsum。
    Returns: (x_2d, batch_shape, L_orig, axis_normalized)
    """
    ndim = x.ndim
    axis = axis % ndim
    L = x.shape[axis]
    x = jnp.moveaxis(x, axis, -1)
    batch_shape = x.shape[:-1]
    B = math.prod(batch_shape) if batch_shape else 1
    return x.reshape(B, L), batch_shape, L, axis


def _denormalize(result, batch_shape, L, axis):
    """从 (B, L) 恢复原始形状。"""
    result = result.reshape(*batch_shape, L) if batch_shape else result.reshape(L)
    return jnp.moveaxis(result, -1, axis)


# ============================================================
# 核心算法: 递归层级 triu matmul
# ============================================================

def _recursive_cumsum_2d(x_2d, chunk_size):
    """递归 triu matmul cumsum on (B, L) along axis=1。

    Base case: L <= chunk_size → 直接 triu matmul
    Recursive: chunk → local triu → 递归 cumsum chunk_totals → exclusive offsets

    递归深度: ⌈log_{chunk_size}(L)⌉，实际 ≤ 3 层。
    """
    B, L = x_2d.shape

    if L <= chunk_size:
        triu = jnp.triu(jnp.ones((L, L), dtype=x_2d.dtype))
        return _dot(x_2d, triu, (1,), (0,))

    # Pad
    seq_pad = (chunk_size - L % chunk_size) % chunk_size
    if seq_pad > 0:
        x_2d = jnp.pad(x_2d, [(0, 0), (0, seq_pad)])
    L_padded = L + seq_pad
    num_chunks = L_padded // chunk_size

    # (B, num_chunks, chunk_size)
    x_3d = x_2d.reshape(B, num_chunks, chunk_size)

    # Local cumsum via triu
    triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=x_3d.dtype))
    local_cs = _dot(x_3d, triu, (2,), (0,))  # (B, num_chunks, chunk_size)

    # Chunk totals
    chunk_totals = local_cs[:, :, -1]  # (B, num_chunks)

    # 递归: inclusive cumsum of chunk_totals
    totals_cumsum = _recursive_cumsum_2d(chunk_totals, chunk_size)

    # Exclusive offsets: cumsum[i] - totals[i] = sum(t[0..i-1])
    offsets = totals_cumsum - chunk_totals  # (B, num_chunks)

    result = local_cs + offsets[:, :, None]
    return result.reshape(B, L_padded)[:, :L]


def cumsum(x, axis=-1, chunk_size=128):
    """递归层级 triu matmul cumsum。

    Args:
        x: 输入张量
        axis: cumsum 的轴
        chunk_size: 每层分块大小 (默认 128，对齐 TPU MXU)
    """
    x_2d, batch_shape, L, axis = _normalize(x, axis)
    result = _recursive_cumsum_2d(x_2d, chunk_size)
    return _denormalize(result, batch_shape, L, axis)


# ============================================================
# main: 正确性 + 性能对比
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("cumsum_fast: 递归层级 triu matmul cumsum")
    print("=" * 70)

    key = jax.random.PRNGKey(42)

    # ── 正确性测试 ──
    print("\n正确性测试:")
    test_cases = [
        ((16,), 0, 4, "1-D 小序列"),
        ((4, 32), 1, 8, "2-D axis=1"),
        ((4, 32), 0, 4, "2-D axis=0"),
        ((8, 8, 256), 2, 64, "3-D axis=2"),
        ((8, 8, 256), 1, 8, "3-D axis=1"),
        ((3, 4, 64), 2, 16, "3-D axis=2 不对齐 batch"),
        ((7, 13, 100), 2, 32, "3-D 全不对齐"),
        ((7, 13, 100), 0, 4, "3-D axis=0 不对齐"),
        ((64, 64, 2048), 2, 128, "3-D 大张量"),
        ((4, 65536), 1, 128, "2-D 极长序列"),
        ((1, 262144), 1, 256, "2-D 超长单 batch"),
    ]

    all_pass = True
    for shape, ax, cs, desc in test_cases:
        x = jax.random.normal(key, shape, dtype=jnp.float32)
        ref = jnp.cumsum(x, axis=ax)
        L = shape[ax]
        tol = 1e-4 * math.sqrt(L)

        out = cumsum(x, axis=ax, chunk_size=cs)
        max_err = float(jnp.max(jnp.abs(out - ref)))
        ok = bool(jnp.allclose(out, ref, atol=tol))
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}] {desc:28s} max_err={max_err:.2e}  (tol={tol:.2e})")

    if not all_pass:
        print("\n!! 有测试未通过 !!")
        exit(1)

    # ── 性能对比 vs jnp.cumsum ──
    print()
    print("=" * 70)
    print("性能对比 vs jnp.cumsum")
    print("=" * 70)

    warmup, repeat = 10, 100
    perf_cases = [
        ("小序列", (16,), 0, 4),
        ("中序列 2-D", (64, 512), 1, 128),
        ("大 3-D batch", (64, 64, 2048), 2, 128),
        ("大 3-D seq", (16, 16, 4096), 2, 128),
        ("极长序列 B=4", (4, 65536), 1, 128),
        ("超长单 batch", (1, 262144), 1, 128),
        ("长序列 B=2", (2, 131072), 1, 128),
        ("长序列 B=8", (8, 32768), 1, 128),
        ("不对齐", (7, 13, 100), 2, 32),
    ]

    print(f"\n    {'场景':16s}  {'shape':20s}  {'jnp.cumsum':>10s}  {'triu_rec':>10s}  {'加速比':>8s}")
    print(f"    {'─' * 16}  {'─' * 20}  {'─' * 10}  {'─' * 10}  {'─' * 8}")

    for desc, shape, ax, cs in perf_cases:
        x = jax.random.normal(key, shape, dtype=jnp.float32)

        # jnp.cumsum
        fn_ref = jax.jit(jnp.cumsum, static_argnums=(1,))
        for _ in range(warmup):
            fn_ref(x, ax).block_until_ready()
        t0 = timeit.default_timer()
        for _ in range(repeat):
            fn_ref(x, ax).block_until_ready()
        t_ref = (timeit.default_timer() - t0) / repeat * 1000

        # triu_rec
        fn_rec = jax.jit(lambda x, _ax=ax, _cs=cs: cumsum(x, axis=_ax, chunk_size=_cs))
        for _ in range(warmup):
            fn_rec(x).block_until_ready()
        t0 = timeit.default_timer()
        for _ in range(repeat):
            fn_rec(x).block_until_ready()
        t_rec = (timeit.default_timer() - t0) / repeat * 1000

        speedup = t_ref / t_rec
        mark = " *" if speedup > 1.05 else ""
        print(f"    {desc:16s}  {str(shape):20s}  {t_ref:>8.3f}ms  {t_rec:>8.3f}ms  {speedup:>6.2f}x{mark}")

    # ── 简单示例 ──
    print()
    print("=" * 70)
    print("示例")
    print("=" * 70)
    x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=jnp.float32)
    print(f"  输入:       {x}")
    print(f"  jnp.cumsum: {jnp.cumsum(x)}")
    print(f"  cumsum:     {cumsum(x, chunk_size=3)}")
