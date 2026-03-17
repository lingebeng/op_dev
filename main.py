import numpy as np

np.random.seed(42)

# === 数据准备 ===
# 对齐 kernel 的形状: Q[M, D], K[N, D], V[N, D]
M, N, D = 4, 8, 4  # M=query数, N=key数, D=head_dim
BLOCK_N = 2  # 每次处理 BLOCK_N 个 key（模拟 kernel 的内循环）
scale = 1.0 / np.sqrt(D)

Q = np.random.randn(M, D)
K = np.random.randn(N, D)
V = np.random.randn(N, D)


# === 1. Ground Truth: 标准 Softmax Attention ===
S = Q @ K.T * scale  # [M, N]
S_max = S.max(axis=1, keepdims=True)  # [M, 1]
P = np.exp(S - S_max)
P = P / P.sum(axis=1, keepdims=True)
out_true = P @ V  # [M, D]


# === 2. Online Softmax (对齐 kernel 的写法) ===
# kernel 里每个 program 处理一行 query，这里用 for 模拟
# 每行维护: m_i(当前最大值), lse_i(log-sum-exp), acc_o(加权累加)
m_i = np.full(M, -np.inf)
lse_i = np.full(M, -np.inf)
acc_o = np.zeros((M, D))
print(acc_o)
for start_n in range(0, N, BLOCK_N):
    print("-" * 40)
    k_block = K[start_n : start_n + BLOCK_N]  # [BLOCK_N, D]
    v_block = V[start_n : start_n + BLOCK_N]  # [BLOCK_N, D]

    # qk = Q @ K_block^T，对齐 kernel: qk += tl.dot(q, tl.trans(k))
    qk = Q @ k_block.T * scale  # [M, BLOCK_N]

    # -- online softmax (完全对齐 kernel) --
    m_ij = np.maximum(qk.max(axis=1), lse_i)  # [M,], 每行的全局最大值
    p = np.exp(qk - m_ij[:, None])  # 局部 softmax 分子
    l_ij = p.sum(axis=1)  # 局部 sum

    # 缩放历史累加器: acc_o *= exp(m_i - m_ij)
    acc_o = acc_o * np.exp(m_i - m_ij)[:, None]

    # 累加: acc_o += p @ v_block
    acc_o += p @ v_block

    # 更新统计量
    m_i = m_ij
    l_i_new = np.exp(lse_i - m_ij) + l_ij
    lse_i = m_i + np.log(l_i_new)
    print(acc_o)

# 最终 rescale: o *= exp(m_i - lse_i)
out_online = acc_o * np.exp(m_i - lse_i)[:, None]


# === 3. 校验 ===
print("Ground Truth:\n", np.round(out_true, 4))
print("Online Softmax:\n", np.round(out_online, 4))
print(f"Match: {np.allclose(out_true, out_online)}")
