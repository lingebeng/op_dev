import torch


def gated_linear_attention_chunked(q, k, v, g, scale=1.0, chunk_size=4):
    """
    最朴素的分块 Gated Linear Attention。
    去掉 batch 和 head，只关注核心逻辑。

    Args:
        q: [T, K]
        k: [T, K]
        v: [T, V]
        g: [T, K]  — 每个位置的门控值（通常为负数）
        scale: 缩放因子
        chunk_size: 分块大小

    Returns:
        o:  [T, V]  — 输出
        ht: [K, V]  — 最终隐状态
    """
    T, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    # =============================================
    # 第 1 步：chunk local cumsum
    # 每个 chunk 内部对 g 做累积求和，chunk 之间互不影响
    #
    # 例: g = [g0, g1, g2, g3 | g4, g5, g6, g7]
    # cumsum = [g0, g0+g1, g0+g1+g2, g0+g1+g2+g3 | g4, g4+g5, ...]
    # =============================================
    g_cumsum = torch.zeros_like(g)  # [T, K]
    for t_chunk in range(NT):
        s = t_chunk * C
        cumsum = torch.zeros(K)
        for i in range(C):
            cumsum = cumsum + g[s + i]
            g_cumsum[s + i] = cumsum

    # =============================================
    # 第 2 步：chunk_fwd_h
    # 串行扫描，算每个 chunk 开头的记忆矩阵 h
    #
    # h 就是一个 [K, V] 的矩阵，累积了之前所有 k^T@v 的信息（带衰减）
    # =============================================
    h_all = torch.zeros(NT, K, V)  # 每个 chunk 开头的记忆快照
    h = torch.zeros(K, V)  # 当前记忆

    for t_chunk in range(NT):
        # 先拍快照（更新前的状态）
        h_all[t_chunk] = h

        s = t_chunk * C
        gk_last = g_cumsum[s + C - 1]  # [K]，整个 chunk 的总衰减量

        # 旧记忆衰减
        h = h * torch.exp(gk_last[:, None])  # [K, V]

        # 调整 k 并累加 k^T @ v
        for i in range(C):
            k_adjusted = k[s + i] * torch.exp(gk_last - g_cumsum[s + i])  # [K]
            h = h + k_adjusted[:, None] * v[s + i][None, :]  # [K,1] * [1,V] → [K,V]

    ht = h  # 最终隐状态

    # =============================================
    # 第 3 步：chunk_gla_fwd_intra
    # 每个 chunk 内部算 q 和 k 的注意力矩阵 A (C×C)
    # 各 chunk 完全独立，可并行
    # =============================================
    A = torch.zeros(NT, C, C)

    for t_chunk in range(NT):
        s = t_chunk * C
        for i in range(C):
            q_gated = q[s + i] * torch.exp(g_cumsum[s + i])  # [K]
            for j in range(C):
                k_gated = k[s + j] * torch.exp(-g_cumsum[s + j])  # [K]
                A[t_chunk, i, j] = torch.dot(q_gated, k_gated)

    A = A * scale

    # =============================================
    # 第 4 步：chunk_fwd_o
    # 合并两部分：
    #   跨 chunk：q 读取历史记忆 h
    #   chunk 内：用注意力矩阵 A 加权 v
    # =============================================
    o = torch.zeros(T, V)

    for t_chunk in range(NT):
        s = t_chunk * C

        for i in range(C):
            q_gated = q[s + i] * torch.exp(g_cumsum[s + i])  # [K]

            # 跨 chunk：q @ h（读取之前所有 chunk 的累积记忆）
            o_inter = q_gated @ h_all[t_chunk] * scale  # [K] @ [K,V] → [V]

            # chunk 内：sum_{j<=i} A[i,j] * v[j]（causal，只看 j<=i）
            o_intra = torch.zeros(V)
            for j in range(i + 1):
                o_intra = o_intra + A[t_chunk, i, j] * v[s + j]

            o[s + i] = o_inter + o_intra

    return o, ht


# =============================================
# 验证：和逐步递推的朴素版本对比
# =============================================
def gated_linear_attention_naive(q, k, v, g, scale=1.0):
    """逐步递推，最朴素的 GLA，作为正确性参考。"""
    T, K = q.shape
    V = v.shape[-1]
    h = torch.zeros(K, V)
    o = torch.zeros(T, V)

    for t in range(T):
        h = h * torch.exp(g[t])[:, None]  # 衰减
        h = h + k[t][:, None] * v[t][None, :]  # 写入
        o[t] = (q[t] @ h) * scale  # 读取

    return o, h


if __name__ == "__main__":
    torch.manual_seed(42)
    T, K, V = 8, 4, 4
    C = 4

    q = torch.randn(T, K)
    k = torch.randn(T, K)
    v = torch.randn(T, V)
    g = -torch.rand(T, K).abs()  # 负数，表示遗忘

    o_naive, h_naive = gated_linear_attention_naive(q, k, v, g, scale=1.0)
    o_chunk, h_chunk = gated_linear_attention_chunked(
        q, k, v, g, scale=1.0, chunk_size=C
    )

    print("output diff:", (o_naive - o_chunk).abs().max().item())
    print("hidden diff:", (h_naive - h_chunk).abs().max().item())
