import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def online_softmax_kernel(x_ref, o_ref, *, block_L, block_B):
    """
    Pallas Online Softmax (TPU 8x128 对齐版)
    x_ref 和 o_ref 接收的是 (block_B, padded_L) 的二维块
    """
    padded_L = x_ref.shape[1]
    num_blocks = padded_L // block_L

    # ==========================================
    # Pass 1: 分块扫描，增量计算 m_final 和 d_final
    # ==========================================
    def pass1_body(i, state):
        m_prev, d_prev = state

        # 加载当前的数据块，此时提取的是 block_B 行
        x_chunk = x_ref[:, pl.dslice(i * block_L, block_L)].astype(jnp.float32)

        # 按行计算局部统计量，保持 (block_B, 1) 的形状
        m_local = jnp.max(x_chunk, axis=-1, keepdims=True)
        m_new = jnp.maximum(m_prev, m_local)

        # 修正指数和，并累加当前块
        d_prev_scaled = d_prev * jnp.exp(m_prev - m_new)
        d_local = jnp.sum(jnp.exp(x_chunk - m_new), axis=-1, keepdims=True)
        d_new = d_prev_scaled + d_local

        return m_new, d_new

    # 初始化状态为 (block_B, 1) 的列向量
    init_state = (
        jnp.full((block_B, 1), -jnp.inf, dtype=jnp.float32),
        jnp.full((block_B, 1), 0.0, dtype=jnp.float32),
    )
    m_final, d_final = jax.lax.fori_loop(0, num_blocks, pass1_body, init_state)

    # ==========================================
    # Pass 2: 再次扫描原始数据，计算概率并写入输出
    # ==========================================
    def pass2_body(i, _):
        x_chunk = x_ref[:, pl.dslice(i * block_L, block_L)].astype(jnp.float32)

        # 向量广播机制：(block_B, block_L) 减去 (block_B, 1)
        o_chunk = (jnp.exp(x_chunk - m_final) / d_final).astype(o_ref.dtype)

        o_ref[:, pl.dslice(i * block_L, block_L)] = o_chunk

    jax.lax.fori_loop(0, num_blocks, pass2_body, None)


@jax.jit
def pallas_online_softmax(x):
    B, L = x.shape

    # --- 硬件硬性对齐参数 ---
    block_L = 256  # 128 的倍数
    block_B = 8  # 8 的倍数

    padded_L = ((L + block_L - 1) // block_L) * block_L
    padded_B = ((B + block_B - 1) // block_B) * block_B

    # 沿两个维度进行 padding
    pad_B_len = padded_B - B
    pad_L_len = padded_L - L
    if pad_B_len > 0 or pad_L_len > 0:
        x = jnp.pad(x, ((0, pad_B_len), (0, pad_L_len)), constant_values=-jnp.inf)

    # Grid：每个线程块处理 block_B 行，总网格数除以 block_B
    grid = (padded_B // block_B,)
    interpret = jax.default_backend() == "cpu"

    kernel = functools.partial(online_softmax_kernel, block_L=block_L, block_B=block_B)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        # in_specs 的 block_shape 更新为 (block_B, padded_L)
        in_specs=[
            pl.BlockSpec(index_map=lambda i: (i, 0), block_shape=(block_B, padded_L))
        ],
        out_specs=pl.BlockSpec(
            index_map=lambda i: (i, 0), block_shape=(block_B, padded_L)
        ),
        grid=grid,
        interpret=interpret,
    )(x)

    # 裁剪掉 Padding 的部分，返回真实大小
    return out[:B, :L]


# 测试代码
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    # 使用 bfloat16 进一步榨干 TPU 带宽
    x = jax.random.normal(key, (1024, 2050), dtype=jnp.bfloat16)

    out_pallas = pallas_online_softmax(x)
    out_jax = jax.nn.softmax(x, axis=-1)

    print("Pallas 输出形状:", out_pallas.shape)
    print("最大误差:", jnp.max(jnp.abs(out_pallas - out_jax)))
    print("结果是否一致?", jnp.allclose(out_pallas, out_jax, atol=1e-3))
