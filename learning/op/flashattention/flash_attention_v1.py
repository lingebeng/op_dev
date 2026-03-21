"""
FlashAttention V1 — Triton 实现
================================

严格按照 FlashAttention V1 论文 (Dao et al., 2022) Algorithm 1 实现。

V1 核心特征 (与 V2 的关键区别):
  每一步都对输出 O_i 做归一化 (除以 l_i^new)，而 V2 只在最后归一化一次。
  这意味着 O_i 在每次迭代后都是"当前最佳估计"的归一化输出。

算法 (对每个 Q 块并行，内部串行遍历 K/V 块):
  初始化: m_i = -∞,  l_i = 0,  O_i = 0
  for j in K/V blocks:
      S_ij   = Q_i @ K_j^T / √d
      m̃_ij  = rowmax(S_ij)
      P̃_ij  = exp(S_ij − m̃_ij)
      l̃_ij  = rowsum(P̃_ij)
      m_i^new = max(m_i, m̃_ij)
      α = exp(m_i − m_i^new),  β = exp(m̃_ij − m_i^new)
      l_i^new = α·l_i + β·l̃_ij
      O_i     = (l_i·α·O_i + β·P̃_ij@V_j) / l_i^new     ← V1: 每步归一化
      m_i, l_i ← m_i^new, l_i^new
"""

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton Kernel
# ---------------------------------------------------------------------------
@triton.jit
def flash_attention_v1_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qm, stride_qk,   # Q strides: (batch*heads, seq, dim)
    stride_kb, stride_kn, stride_kk,   # K strides
    stride_vb, stride_vn, stride_vk,   # V strides
    stride_ob, stride_om, stride_ok,   # O strides
    N_CTX,                              # 序列长度
    sm_scale,                           # 1 / √d_k
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,             # Q 块行数
    BLOCK_N: tl.constexpr,             # K/V 块行数
    BLOCK_D: tl.constexpr,             # head dimension (需 == 实际 head_dim)
):
    """
    Grid: (cdiv(N_CTX, BLOCK_M),  batch * n_heads)
      - program_id(0) → 第几个 Q 块
      - program_id(1) → 第几个 batch×head
    """
    # ── 1. 定位当前 program 的 Q 块 ──────────────────────────
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    Q_ptr += pid_b * stride_qb
    K_ptr += pid_b * stride_kb
    V_ptr += pid_b * stride_vb
    O_ptr += pid_b * stride_ob

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_d = tl.arange(0, BLOCK_D)                       # [BLOCK_D]
    offs_n = tl.arange(0, BLOCK_N)                       # [BLOCK_N]

    # ── 2. 加载 Q 块 (驻留 SRAM，整个循环复用) ──────────────
    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)          # [BLOCK_M, BLOCK_D]

    # ── 3. 初始化在线统计量 ──────────────────────────────────
    m_i = tl.full([BLOCK_M], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # 因果注意力: 只遍历到当前 Q 块末尾 (key 不能晚于 query)
    if IS_CAUSAL:
        loop_end = min((pid_m + 1) * BLOCK_M, N_CTX)
    else:
        loop_end = N_CTX

    # ── 4. 遍历 K/V 块 ─────────────────────────────────────
    for j in range(0, loop_end, BLOCK_N):
        k_offs = j + offs_n

        # 加载 K 块: [BLOCK_N, BLOCK_D]
        k_ptrs = K_ptr + k_offs[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = k_offs[:, None] < N_CTX
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # 加载 V 块: [BLOCK_N, BLOCK_D]
        v_ptrs = V_ptr + k_offs[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # ---- 4a. 注意力分数 S_ij = Q_i K_j^T · scale ----
        s_ij = tl.dot(q, tl.trans(k)) * sm_scale         # [BLOCK_M, BLOCK_N]

        # 边界掩码
        s_ij = tl.where(k_offs[None, :] < N_CTX, s_ij, float("-inf"))

        # 因果掩码: S[m, n] = -∞  when  query_pos < key_pos
        if IS_CAUSAL:
            s_ij = tl.where(offs_m[:, None] >= k_offs[None, :], s_ij, float("-inf"))

        # ---- 4b. 在线 softmax (V1) ----
        m_ij_tilde = tl.max(s_ij, axis=1)                # [BLOCK_M]
        p_ij_tilde = tl.exp(s_ij - m_ij_tilde[:, None])  # [BLOCK_M, BLOCK_N]
        l_ij_tilde = tl.sum(p_ij_tilde, axis=1)          # [BLOCK_M]

        m_i_new = tl.maximum(m_i, m_ij_tilde)
        alpha = tl.exp(m_i - m_i_new)                    # 旧 max 修正因子
        beta  = tl.exp(m_ij_tilde - m_i_new)             # 新块 max 修正因子
        l_i_new = alpha * l_i + beta * l_ij_tilde

        # ---- 4c. P̃_ij @ V_j ----
        pv = tl.dot(p_ij_tilde.to(q.dtype), v)           # [BLOCK_M, BLOCK_D]

        # ---- 4d. V1 特有: 每步归一化 ----
        # O_i = diag(l_i^new)^{-1} · (l_i · α · O_i  +  β · P̃V)
        o_i = (l_i[:, None] * alpha[:, None] * o_i + beta[:, None] * pv) / l_i_new[:, None]

        m_i = m_i_new
        l_i = l_i_new

    # ── 5. 写回 HBM ────────────────────────────────────────
    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < N_CTX
    tl.store(o_ptrs, o_i.to(O_ptr.dtype.element_ty), mask=o_mask)


# ---------------------------------------------------------------------------
# Python Wrapper
# ---------------------------------------------------------------------------
def flash_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    FlashAttention V1 前向传播

    Args:
        q, k, v: shape (batch, n_heads, seq_len, head_dim)，head_dim 须为 2 的幂
        causal:  是否施加因果掩码

    Returns:
        shape (batch, n_heads, seq_len, head_dim) 的输出张量
    """
    B, H, N, D = q.shape
    assert k.shape == v.shape == (B, H, N, D), "Q, K, V shape 必须一致"
    assert D == triton.next_power_of_2(D), f"head_dim ({D}) 必须是 2 的幂"

    sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)

    # reshape 为 (B*H, N, D) 方便 kernel 索引
    q_flat = q.reshape(B * H, N, D)
    k_flat = k.reshape(B * H, N, D)
    v_flat = v.reshape(B * H, N, D)
    o_flat = o.reshape(B * H, N, D)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D

    grid = (triton.cdiv(N, BLOCK_M), B * H)

    flash_attention_v1_fwd_kernel[grid](
        q_flat, k_flat, v_flat, o_flat,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        o_flat.stride(0), o_flat.stride(1), o_flat.stride(2),
        N_CTX=N,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return o


# ---------------------------------------------------------------------------
# 参考实现 (PyTorch，用于验证)
# ---------------------------------------------------------------------------
def reference_attention(q, k, v, causal=False):
    """标准 Attention (会实例化完整 N×N 矩阵，仅用于正确性对比)"""
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if causal:
        mask = torch.triu(torch.ones(s.shape[-2], s.shape[-1],
                                     device=q.device, dtype=torch.bool), diagonal=1)
        s.masked_fill_(mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("FlashAttention V1 需要 CUDA 环境。")
        raise SystemExit(1)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    print("FlashAttention V1  —  Triton 正确性验证")
    print("=" * 60)

    configs = [
        ("非因果, B=2 H=4 N=512  D=64",  2, 4,  512,  64, False),
        ("因果,   B=2 H=4 N=512  D=64",  2, 4,  512,  64, True),
        ("因果,   B=1 H=2 N=1024 D=64",  1, 2, 1024,  64, True),
        ("因果,   B=1 H=8 N=2048 D=128", 1, 8, 2048, 128, True),
    ]

    all_pass = True
    for tag, B, H, N, D, causal in configs:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        out_flash = flash_attention_v1(q, k, v, causal=causal)
        out_ref   = reference_attention(q, k, v, causal=causal)

        max_err = torch.max(torch.abs(out_flash - out_ref)).item()
        ok = torch.allclose(out_flash, out_ref, atol=1e-2, rtol=1e-2)
        all_pass = all_pass and ok

        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {tag}  |  max_err={max_err:.4e}")

    print("=" * 60)
    print("全部通过!" if all_pass else "存在失败的测试!")
