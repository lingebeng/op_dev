import torch


# ============================================================================
# Gated Linear Attention (GLA) 的三种等价实现
# 核心区别（相比普通 Linear Attention）：S_t = exp(g_t) * S_{t-1} + k_t ⊗ v_t
# g 是每个时间步的标量门控，控制旧信息的衰减速度
# ============================================================================


def gated_linear_attention_parallel(Q, K, V, g):
    """
    带门控的并行版本
    输入维度: Q, K, V: (B, L, d), g: (B, L) 每个时间步一个标量门控
    """
    B, L, d = Q.shape

    # 构造门控的因果衰减矩阵 G (B, L, L)
    # G[i][j] = exp(g_{j+1} + g_{j+2} + ... + g_i)，表示从时间步 j 到 i 的累积衰减
    # 先算前缀和: cum_g[t] = g_0 + g_1 + ... + g_t
    cum_g = torch.cumsum(g, dim=1)  # (B, L)

    # G[i][j] = exp(cum_g[i] - cum_g[j])
    # 用广播: (B, L, 1) - (B, 1, L) -> (B, L, L)
    G = torch.exp(cum_g.unsqueeze(2) - cum_g.unsqueeze(1))  # (B, L, L)

    # 乘上因果掩码，只保留 i >= j 的部分（下三角）
    causal_mask = torch.tril(torch.ones(L, L, device=Q.device)).unsqueeze(0)
    G = G * causal_mask  # (B, L, L)

    # 计算带衰减的注意力分数
    # scores[i][j] = (q_i · k_j) * G[i][j]
    scores = torch.bmm(Q, K.transpose(1, 2))  # (B, L, L)
    gated_scores = scores * G

    # 乘以 Value 矩阵
    O = torch.bmm(gated_scores, V)  # (B, L, d)

    return O


def gated_linear_attention_recurrent(Q, K, V, g):
    """
    带门控的递归版本
    输入维度: Q, K, V: (B, L, d), g: (B, L) 每个时间步一个标量门控
    """
    B, L, d = Q.shape
    O = torch.zeros_like(Q)
    S = torch.zeros(B, d, d, device=Q.device)

    for t in range(L):
        q_t = Q[:, t, :]  # (B, d)
        k_t = K[:, t, :]
        v_t = V[:, t, :]
        g_t = g[:, t]  # (B,)

        # 关键区别：先让旧状态衰减，再累加新信息
        # exp(g_t) 是一个标量，对 S 的每个元素统一缩放
        decay = torch.exp(g_t).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
        S = S * decay + torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))

        o_t = torch.bmm(q_t.unsqueeze(1), S).squeeze(1)
        O[:, t, :] = o_t

    return O


def gated_chunkwise_linear_attention(Q, K, V, g, chunk_size=64):
    """
    带门控的分块版本
    输入维度: Q, K, V: (B, L, d), g: (B, L) 每个时间步一个标量门控
    """
    B, L, d = Q.shape
    C = chunk_size
    num_chunks = L // C

    O = torch.zeros_like(Q)
    S = torch.zeros(B, d, d, device=Q.device)

    for i in range(num_chunks):
        Q_i = Q[:, i * C : (i + 1) * C, :]  # (B, C, d)
        K_i = K[:, i * C : (i + 1) * C, :]
        V_i = V[:, i * C : (i + 1) * C, :]
        g_i = g[:, i * C : (i + 1) * C]  # (B, C)

        # --------------------------------------------------------
        # 第一部分：块内计算 (Intra-chunk) - 带门控的局部注意力
        # --------------------------------------------------------
        # 构造块内的门控因果矩阵 G_local (B, C, C)
        cum_g_local = torch.cumsum(g_i, dim=1)  # (B, C)
        G_local = torch.exp(
            cum_g_local.unsqueeze(2) - cum_g_local.unsqueeze(1)
        )  # (B, C, C)
        causal_mask = torch.tril(torch.ones(C, C, device=Q.device)).unsqueeze(0)
        G_local = G_local * causal_mask

        local_scores = torch.bmm(Q_i, K_i.transpose(1, 2))  # (B, C, C)
        O_intra = torch.bmm(local_scores * G_local, V_i)  # (B, C, d)

        # --------------------------------------------------------
        # 第二部分：块间计算 (Inter-chunk) - 查询全局历史，但要乘上衰减
        # --------------------------------------------------------
        # 块内第 j 个 token 查询全局状态时，需要考虑从块起始到第 j 个位置的累积衰减
        # decay_for_query[j] = exp(g_{i*C} + g_{i*C+1} + ... + g_{i*C+j})
        decay_for_query = torch.exp(cum_g_local).unsqueeze(2)  # (B, C, 1)
        O_inter = torch.bmm(Q_i, S) * decay_for_query  # (B, C, d)

        # --------------------------------------------------------
        # 第三部分：合并输出 & 更新全局状态
        # --------------------------------------------------------
        O[:, i * C : (i + 1) * C, :] = O_inter + O_intra

        # 更新全局状态 S：先衰减，再加上当前块的贡献
        # 整个块对 S 的总衰减 = exp(g_{i*C} + ... + g_{(i+1)*C-1})
        chunk_total_decay = (
            torch.exp(cum_g_local[:, -1]).unsqueeze(1).unsqueeze(2)
        )  # (B, 1, 1)
        S = S * chunk_total_decay

        # 当前块中第 j 个 token 的 kv 外积，到块末尾还要经历 (C-1-j) 步衰减
        # decay_for_kv[j] = exp(g_{j+1} + ... + g_{C-1}) = exp(cum_g[C-1] - cum_g[j])
        decay_for_kv = torch.exp(cum_g_local[:, -1:] - cum_g_local).unsqueeze(
            2
        )  # (B, C, 1)
        # K_i^T @ (decay_for_kv * V_i): 让每个 token 的 v 带上对应的衰减权重
        S = S + torch.bmm(
            K_i.transpose(1, 2),
            V_i * decay_for_kv,
        )

    return O


if __name__ == "__main__":
    B, L, d = 2, 8, 3
    Q = torch.randn(B, L, d)
    K = torch.randn(B, L, d)
    V = torch.randn(B, L, d)
    g = torch.randn(B, L) * 0.1  # 门控值，乘 0.1 让衰减不要太剧烈

    O_par = gated_linear_attention_parallel(Q, K, V, g)
    O_rec = gated_linear_attention_recurrent(Q, K, V, g)
    O_chunk = gated_chunkwise_linear_attention(Q, K, V, g, chunk_size=2)

    assert torch.allclose(O_par, O_rec, atol=1e-5), (
        f"Recurrent != Parallel! max diff={torch.max(torch.abs(O_par - O_rec))}"
    )
    assert torch.allclose(O_par, O_chunk, atol=1e-5), (
        f"Chunkwise != Parallel! max diff={torch.max(torch.abs(O_par - O_chunk))}"
    )
    print("[PASS] Gated Linear Attention: all 3 versions match.")
