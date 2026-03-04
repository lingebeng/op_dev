import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def online_softmax_kernel(x_ref, o_ref, *, block_L):
    """
    Pallas Online Softmax 底层 Kernel
    注意：这里的 x_ref 和 o_ref 接收到的是一整行数据 (1, L) 的引用视图。
    """
    padded_L = x_ref.shape[1]
    num_blocks = padded_L // block_L

    # ==========================================
    # Pass 1: 分块扫描，增量计算 m_final 和 d_final
    # ==========================================
    def pass1_body(i, state):
        m_prev, d_prev = state

        # 动态加载当前的数据块到 Vmem (TPU 向量寄存器)
        # pl.dslice(start, size) 用于动态切片
        x_chunk = x_ref[0, pl.dslice(i * block_L, block_L)].astype(jnp.float32)

        # 计算局部统计量
        m_local = jnp.max(x_chunk)
        m_new = jnp.maximum(m_prev, m_local)

        # 核心步骤：修正之前的指数和，并累加当前块
        d_prev_scaled = d_prev * jnp.exp(m_prev - m_new)
        d_local = jnp.sum(jnp.exp(x_chunk - m_new))
        d_new = d_prev_scaled + d_local

        return m_new, d_new

    # 初始化 m 为极小值，d 为 0
    init_state = (
        jnp.array(-jnp.inf, dtype=jnp.float32),
        jnp.array(0.0, dtype=jnp.float32),
    )
    m_final, d_final = jax.lax.fori_loop(0, num_blocks, pass1_body, init_state)

    # ==========================================
    # Pass 2: 再次扫描原始数据，计算概率并写入输出
    # ==========================================
    def pass2_body(i, _):
        # 重新将块加载进 SRAM
        x_chunk = x_ref[0, pl.dslice(i * block_L, block_L)].astype(jnp.float32)

        # 使用 Pass 1 算出的全局 m_final 和 d_final 进行归一化
        o_chunk = (jnp.exp(x_chunk - m_final) / d_final).astype(o_ref.dtype)

        # 将结果写回输出引用
        o_ref[0, pl.dslice(i * block_L, block_L)] = o_chunk

    jax.lax.fori_loop(0, num_blocks, pass2_body, None)


@jax.jit
def pallas_online_softmax(x):
    B, L = x.shape
    block_L = 256
    padded_L = ((L + block_L - 1) // block_L) * block_L

    # 对齐到 block_L，padding 为 -inf，不影响 softmax 结果
    if padded_L != L:
        x = jnp.pad(x, ((0, 0), (0, padded_L - L)), constant_values=-jnp.inf)

    # Grid 仅在 Batch 维度并行，启动 B 个底层线程
    grid = (B,)
    interpret = jax.default_backend() == "cpu"
    kernel = functools.partial(online_softmax_kernel, block_L=block_L)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        # BlockSpec 映射整行数据，将底层的显存加载控制权完全交给 Kernel 内部的循环
        in_specs=[pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(1, padded_L))],
        out_specs=pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(1, padded_L)),
        grid=grid,
        interpret=interpret,
    )(x)
    return out[:, :L]


# 测试代码
if __name__ == "__main__":
    # 生成随机测试数据（故意使用非 block 对齐长度）
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (1024, 2050))

    # Pallas 版本
    out_pallas = pallas_online_softmax(x)

    # JAX 原生版本 (参考基准)
    out_jax = jax.nn.softmax(x, axis=-1)
    print("Pallas 输出形状:", out_pallas.shape)

    # 验证正确性
    print("最大误差:", jnp.max(jnp.abs(out_pallas - out_jax)))
    print("结果是否一致?", jnp.allclose(out_pallas, out_jax, atol=1e-5))
