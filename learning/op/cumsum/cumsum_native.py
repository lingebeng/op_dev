import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# =====================================================================
# Pass 1: 局部扫描与归约 (Local Scan & Reduce)
# 目标：算出每个 Block 内部的 cumsum，并提取出每个 Block 的总和。
# =====================================================================
def local_scan_kernel(x_ref, local_cumsum_ref, block_sum_ref):
    x = x_ref[...]
    local_cumsum = jnp.cumsum(x, axis=0)

    local_cumsum_ref[...] = local_cumsum
    block_sum_ref[0, 0] = local_cumsum[-1]


# =====================================================================
# Pass 3: 基数偏移更新 (Add Base Offset)
# 目标：把前面所有 Block 的累加总和，加到当前 Block 的每一个元素上。
# =====================================================================
def add_offset_kernel(local_cumsum_ref, block_offset_ref, out_ref):
    local_cumsum = local_cumsum_ref[...]
    block_offset = block_offset_ref[0, 0]

    out_ref[...] = local_cumsum + block_offset


# =====================================================================
# Host 端调度逻辑：串联 3-Pass
# =====================================================================
def pallas_cumsum(x: jax.Array, block_size: int) -> jax.Array:
    seq_len = x.shape[0]
    num_blocks = seq_len // block_size
    grid = (num_blocks,)

    block_spec = pl.BlockSpec(index_map=lambda i: (i,), block_shape=(block_size,))
    scalar_spec = pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(1, 1))

    # --- Pass 1: 计算局部 cumsum 和各个 Block 的总和 ---
    local_cumsum, block_sums = pl.pallas_call(
        local_scan_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct((num_blocks, 1), x.dtype),
        ],
        grid=grid,
        in_specs=[block_spec],
        out_specs=[block_spec, scalar_spec],
        interpret=True,
    )(x)
    block_sums = block_sums.squeeze(-1)  # (num_blocks, 1) -> (num_blocks,)
    print(f"Block sums: {block_sums}")

    # --- Pass 2: 计算 Block 级别的 exclusive prefix sum (基准偏移量) ---
    # num_blocks 通常很小，直接在 JAX 层面用标准 cumsum 即可
    inclusive_sums = jnp.cumsum(block_sums, axis=0)
    block_offsets = jnp.concatenate(
        [jnp.array([0.0], dtype=x.dtype), inclusive_sums[:-1]]
    )

    # --- Pass 3: 将基准偏移量加回局部 cumsum 中 ---
    block_offsets_2d = block_offsets[:, None]  # (num_blocks,) -> (num_blocks, 1)
    out = pl.pallas_call(
        add_offset_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=grid,
        in_specs=[block_spec, scalar_spec],
        out_specs=block_spec,
        interpret=True,
    )(local_cumsum, block_offsets_2d)

    return out


# 测试验证
if __name__ == "__main__":
    x = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    block_size = 4

    out = pallas_cumsum(x, block_size)
    ref = jnp.cumsum(x, axis=0)
    print(f"Pallas 输出: {out}")
    print(f"参考输出:     {ref}")
    assert jnp.allclose(out, ref), f"Mismatch! diff={jnp.abs(out - ref)}"
    print("[PASS] 结果匹配!")
