import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def softmax_kernel(x_ref, o_ref):
    """
    Pallas 底层 Kernel 函数
    x_ref 和 o_ref 是指向片上 SRAM (如 GPU 的 Shared Memory 或 TPU 的 VMEM) 的引用。
    """
    # 1. 从 SRAM 引用中加载数据块到寄存器
    x = x_ref[...]

    # 2. 寻找当前行的最大值 (Keepdims 保证可以广播)
    # 在 TPU 上这会映射为 VMAX 指令，在 GPU 上会映射为对应的归约指令
    row_max = jnp.max(x, axis=-1, keepdims=True)

    # 3. 减去最大值并求指数 (保证数值稳定)
    numerator = jnp.exp(x - row_max)

    # 4. 对指数求和
    denominator = jnp.sum(numerator, axis=-1, keepdims=True)

    # 5. 归一化并将结果写回 SRAM 引用
    o_ref[...] = numerator / denominator


@jax.jit
def pallas_softmax(x):
    """
    使用 pallas_call 包装 kernel，定义分块（BlockSpec）和网格（Grid）
    """
    B, L = x.shape
    block_size = 64
    # 设定 Grid: 针对第一维 (Batch) 并行，每个 Block 负责一行
    grid = (B // block_size,)

    return pl.pallas_call(
        softmax_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        # BlockSpec 定义了如何将 HBM 中的数据映射到 SRAM 中
        # lambda i: (i, 0) 表示第 i 个 grid 处理输入张量的第 i 行，列方向从 0 开始
        # (1, L) 表示每次加载 1 行，长度为 L
        in_specs=[
            pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(block_size, L))
        ],
        out_specs=pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(block_size, L)),
        grid=grid,
        interpret=True,
    )(x)


# 测试代码
if __name__ == "__main__":
    # 生成随机测试数据
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1024, 2048))

    # Pallas 版本
    out_pallas = pallas_softmax(x)

    # JAX 原生版本 (参考基准)
    out_jax = jax.nn.softmax(x, axis=-1)
    print("Pallas 输出形状:", out_pallas.shape)

    # 验证正确性
    print("最大误差:", jnp.max(jnp.abs(out_pallas - out_jax)))
    print("结果是否一致?", jnp.allclose(out_pallas, out_jax, atol=1e-5))
