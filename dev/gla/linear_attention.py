import torch


def naive_linear_attention(Q, K, V):
    """
    最朴素的并行版本，直接计算 L x L 矩阵
    输入维度: (Batch, Seq_Len, Head_Dim) 简写为 (B, L, d)
    """
    B, L, d = Q.shape

    # 1. 强行计算全量的 QK^T，得到 (B, L, L) 的巨大打分矩阵
    # 这里的计算复杂度是 O(L^2 * d)
    scores = torch.bmm(Q, K.transpose(1, 2))

    # 2. 生成一个下三角的因果掩码矩阵 M (L, L)
    # 右上角（未来词）全为 0，左下角（包含对角线）全为 1
    mask = torch.tril(torch.ones(L, L, device=Q.device)).unsqueeze(0)

    # 3. 致命的逐元素相乘操作 (Hadamard Product)
    # 就是这一步破坏了结合律，且占据了极大的显存
    masked_scores = scores * mask

    # 4. 乘以 Value 矩阵，得到最终输出 (B, L, d)
    O = torch.bmm(masked_scores, V)

    return O


def naive_linear_attention_recurrent(Q, K, V):
    """
    等价的 RNN 形式版本，内存复杂度只有 O(1) (对于序列长度而言)
    输入维度: (Batch, Seq_Len, Head_Dim) -> (B, L, d)
    """
    B, L, d = Q.shape

    # 准备一个空张量来装最终的输出
    O = torch.zeros_like(Q)

    # 这就是那个神奇的 2D 隐藏状态矩阵 S！
    # 它的形状永远是 (B, d, d)，跟序列长度 L 毫无关系，绝不爆显存。
    S = torch.zeros(B, d, d, device=Q.device)

    # 顺着时间步 t，一个词一个词地往前扫
    for t in range(L):
        # 抽出当前第 t 个词的向量，形状 (B, d)
        q_t = Q[:, t, :]
        k_t = K[:, t, :]
        v_t = V[:, t, :]

        # 1. 吸收新知识，更新隐藏状态 S_t
        # k_t^T * v_t 是外积: (B, d, 1) @ (B, 1, d) -> (B, d, d)
        # 把当前词的信息浓缩进这个 d x d 的小方阵里
        S = S + torch.bmm(k_t.unsqueeze(2), v_t.unsqueeze(1))

        # 2. 计算当前时间步的输出 o_t
        # q_t * S_t: (B, 1, d) @ (B, d, d) -> (B, 1, d)
        o_t = torch.bmm(q_t.unsqueeze(1), S).squeeze(1)  # 最后把多余的 1 维挤掉

        # 把结果存进输出张量
        O[:, t, :] = o_t

    return O


def naive_chunkwise_linear_attention(Q, K, V, chunk_size=64):
    """
    朴素的分块线性注意力 (Chunkwise Linear Attention)
    输入维度: (Batch, Seq_Len, Head_Dim) -> (B, L, d)
    """
    B, L, d = Q.shape
    C = chunk_size
    num_chunks = L // C

    # 准备最终的输出张量
    O = torch.zeros_like(Q)

    # 初始化全局隐藏状态矩阵 S，形状为 (B, d, d)
    # 在写 Triton 时，这个 S 就是一直驻留在 SRAM (Shared Memory) 里的宝贝
    S = torch.zeros(B, d, d, device=Q.device)

    # 提前准备好一个 C x C 的局部因果掩码 M
    # 因为每个 Chunk 内部的词是有先后顺序的，不能看未来的词
    mask = torch.tril(torch.ones(C, C, device=Q.device)).unsqueeze(0)

    # 沿着时间维度，一块一块地处理 (这就相当于 Triton 里的主循环)
    for i in range(num_chunks):
        # 1. 切片：把当前第 i 个 Chunk 的 Q, K, V 取出来
        # 形状都是 (B, C, d)。在底层算子里，这步相当于从 HBM 加载数据到 SRAM
        Q_i = Q[:, i * C : (i + 1) * C, :]
        K_i = K[:, i * C : (i + 1) * C, :]
        V_i = V[:, i * C : (i + 1) * C, :]

        # --------------------------------------------------------
        # 第一部分：块内计算 (Intra-chunk) - 处理这 C 个词内部的局部注意力
        # --------------------------------------------------------
        # Q_i @ K_i^T 得到 (B, C, C) 的局部打分矩阵
        local_scores = torch.bmm(Q_i, K_i.transpose(1, 2))

        # 乘上局部因果掩码 M，挡住未来的词
        masked_local_scores = local_scores * mask

        # 再乘上 V_i，得到局部的输出 (B, C, d) [cite: 74, 76]
        O_intra = torch.bmm(masked_local_scores, V_i)

        # --------------------------------------------------------
        # 第二部分：块间计算 (Inter-chunk) - 吸收之前所有块的全局历史信息
        # --------------------------------------------------------
        # Q_i 直接乘以全局状态 S：(B, C, d) @ (B, d, d) -> (B, C, d) [cite: 74, 76]
        O_inter = torch.bmm(Q_i, S)

        # --------------------------------------------------------
        # 第三部分：合并输出 & 更新全局状态
        # --------------------------------------------------------
        # 1. 当前块的最终输出 = 全局历史输出 + 局部当前输出 [cite: 74, 76]
        O[:, i * C : (i + 1) * C, :] = O_inter + O_intra

        # 2. 当前块处理完了，把当前块的 K 和 V 的外积更新到全局状态 S 中，留给下一个块用
        # K_i^T @ V_i: (B, d, C) @ (B, C, d) -> (B, d, d) [cite: 67]
        S = S + torch.bmm(K_i.transpose(1, 2), V_i)

    return O


if __name__ == "__main__":
    # 测试一下这两个版本的输出是否一致
    B, L, d = 2, 4, 3  # 小规模测试
    Q = torch.randn(B, L, d)
    K = torch.randn(B, L, d)
    V = torch.randn(B, L, d)

    O_parallel = naive_linear_attention(Q, K, V)
    O_recurrent = naive_linear_attention_recurrent(Q, K, V)
    O_chunkwise = naive_chunkwise_linear_attention(Q, K, V, chunk_size=2)

    print("Parallel Output:\n", O_parallel)
    print("Recurrent Output:\n", O_recurrent)

    # 验证两者是否近似相等
    assert torch.allclose(O_parallel, O_recurrent, atol=1e-6), "Outputs do not match!"
    print("Test passed! Both outputs are approximately equal.")
    print("Chunkwise Output:\n", O_chunkwise)
    assert torch.allclose(O_parallel, O_chunkwise, atol=1e-6), (
        "Chunkwise output does not match parallel output!"
    )
    print("Chunkwise test passed! Output is approximately equal to parallel output.")
