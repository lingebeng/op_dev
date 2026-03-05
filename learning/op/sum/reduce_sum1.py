import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def reduce_sum_kernel(x_ref, o_ref):
    # x_ref 被切出来后的形状是 (8, block_M, block_N)
    # 直接在 SRAM 内沿第 0 个维度“一口吞”求和
    o_ref[...] = jnp.sum(x_ref[...], axis=0)


@jax.jit
def reduce_sum(x):
    B, M, N = x.shape  # 8, 1024, 1024

    # 针对 1024x1024 进行二维分块
    block_M, block_N = 128, 128

    # Grid 只有两维！网格完全不管第 0 维
    grid = (M // block_M, N // block_N)
    interpret = jax.default_backend() == "cpu"
    out = pl.pallas_call(
        reduce_sum_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        in_specs=[
            pl.BlockSpec(
                # 【核心映射逻辑】：
                # 无论网格坐标 (i, j) 怎么跑，输入张量的第 0 维永远取第 0 块
                index_map=lambda i, j: (0, i, j),
                # 因为第 0 块的大小直接等于完整的 B (即 8)
                block_shape=(B, block_M, block_N),
            )
        ],
        out_specs=pl.BlockSpec(
            # 输出映射：网格 (i, j) 写回输出的 (i, j) 块
            index_map=lambda i, j: (i, j),
            block_shape=(block_M, block_N),
        ),
        grid=grid,
        interpret=interpret,
    )(x)
    return out


# 测试代码
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jnp.ones((8, 1024, 1024))

    out_pallas = reduce_sum(x)
    out_jax = jnp.sum(x, axis=0)

    print("Pallas 输出形状:", out_pallas.shape)
    print("最大误差:", jnp.max(jnp.abs(out_pallas - out_jax)))
    print("结果是否一致?", jnp.allclose(out_pallas, out_jax, atol=1e-6))
    print(out_pallas)
