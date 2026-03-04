import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    o_ptr,  # 指向输入和输出张量的指针
    stride_x_row,
    stride_o_row,  # 每一行在内存中的步长 (Stride)
    N_COLS,  # 实际的序列长度 (列数)
    BLOCK_SIZE: tl.constexpr,  # 编译期常量，SRAM 中分配的块大小 (必须是 2 的幂)
):
    """
    Triton 底层 Kernel 函数
    类似于 Pallas，每个 Program Instance (Grid) 负责处理张量的一行。
    """
    # 1. 获取当前 Grid 的行索引 (对应 Pallas 的 lambda i: (i, 0) 中的 i)
    row_idx = tl.program_id(0)

    # 2. 指针偏移计算 (Pallas 自动处理，Triton 需手动完成)
    # 找到当前行的起始内存地址
    x_row_start_ptr = x_ptr + row_idx * stride_x_row
    o_row_start_ptr = o_ptr + row_idx * stride_o_row

    # 生成当前块内元素的偏移量序列: [0, 1, 2, ..., BLOCK_SIZE-1]
    col_offsets = tl.arange(0, BLOCK_SIZE)

    # 构建指向当前行所有元素的指针数组
    x_ptrs = x_row_start_ptr + col_offsets
    o_ptrs = o_row_start_ptr + col_offsets

    # 3. 从 HBM 加载数据块到 SRAM/寄存器
    # 掩码 mask 极其重要：当实际列数 N_COLS 无法填满 2的幂次方的 BLOCK_SIZE 时，防止内存越界
    # 越界部分填充 -inf，这样计算 max 时不会受影响
    mask = col_offsets < N_COLS
    row = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    # 4. 数值稳定 Softmax 计算 (全部在底层 SRAM / 寄存器中发生)
    # Triton 会自动将这些操作映射为 Warp 级别的归约指令 (如 __shfl_down_sync) 或 Shared Memory 操作
    row_max = tl.max(row, axis=0)

    # 减去最大值并求指数。对于 mask 外填充的 -inf，exp(-inf) = 0，因此后续求和完全正确
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)

    # 归一化
    output = numerator / denominator

    # 5. 结果写回 HBM/VRAM
    tl.store(o_ptrs, output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    使用 Triton 启动器包装 kernel
    """
    B, L = x.shape

    # 分配输出张量
    out = torch.empty_like(x)

    # 设定 Grid: 一维，针对 Batch 并行
    grid = (B,)

    # 计算 BLOCK_SIZE: 必须是大于等于 L 的最小的 2 的幂
    BLOCK_SIZE = triton.next_power_of_2(L)

    # 启动 Kernel
    softmax_kernel[grid](
        x,
        out,  # 张量会自动转换为首地址指针
        x.stride(0),
        out.stride(0),  # 传入第 0 维 (行) 的步长
        N_COLS=L,
        BLOCK_SIZE=BLOCK_SIZE,  # 编译期常量，影响内核的寄存器和 Shared Memory 分配
    )

    return out


# 测试代码
if __name__ == "__main__":
    # 确保在 CUDA 环境下运行 (Triton 强依赖 GPU/特定 AI 加速器)
    if not torch.cuda.is_available():
        print("Triton 需要 CUDA 支持的环境。")
    else:
        torch.manual_seed(0)
        # 生成在 GPU 上的随机测试数据，模拟 Pallas 的输入维度
        x = torch.randn(1024, 2048, device="cuda")

        # Triton 版本
        out_triton = triton_softmax(x)

        # PyTorch 原生版本 (参考基准)
        out_torch = torch.nn.functional.softmax(x, dim=-1)

        print("Triton 输出形状:", out_triton.shape)

        # 验证正确性
        max_error = torch.max(torch.abs(out_triton - out_torch))
        print("最大误差:", max_error.item())
        print("结果是否一致?", torch.allclose(out_triton, out_torch, atol=1e-5))
