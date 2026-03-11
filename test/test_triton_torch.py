import argparse
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass
class ErrorStats:
    max_abs: float
    mean_abs: float
    rmse: float
    max_rel: float


def summarize_error(out: torch.Tensor, ref: torch.Tensor) -> ErrorStats:
    out64 = out.detach().cpu().to(torch.float64)
    ref64 = ref.detach().cpu().to(torch.float64)
    abs_err = (out64 - ref64).abs()
    rel_err = abs_err / ref64.abs().clamp_min(1e-12)
    return ErrorStats(
        max_abs=abs_err.max().item(),
        mean_abs=abs_err.mean().item(),
        rmse=abs_err.square().mean().sqrt().item(),
        max_rel=rel_err.max().item(),
    )


def print_stats(title: str, stats: ErrorStats) -> None:
    print(
        f"{title:<20} "
        f"max_abs={stats.max_abs:.6e} "
        f"mean_abs={stats.mean_abs:.6e} "
        f"rmse={stats.rmse:.6e} "
        f"max_rel={stats.max_rel:.6e}"
    )


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[name]


@triton.jit
def exp_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.exp(x.to(tl.float32))
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_exp(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("triton_exp requires a CUDA tensor")
    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    exp_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024)
    return y


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton_matmul requires CUDA tensors")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("triton_matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible shapes for matmul")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("triton_matmul expects contiguous inputs")

    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
    )
    return c


def compare_exp(dtype: torch.dtype, size: int, seed: int) -> None:
    print(f"\n[exp] dtype={dtype} size={size}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x_cpu = torch.empty(size, dtype=torch.float32).uniform_(-8.0, 8.0, generator=gen)
    ref = torch.exp(x_cpu.to(torch.float64))

    torch_out_cpu = torch.exp(x_cpu.to(dtype))
    print_stats("torch.exp(cpu) vs ref", summarize_error(torch_out_cpu, ref))

    if not torch.cuda.is_available():
        print("triton.exp skipped: CUDA is not available")
        return

    x_cuda = x_cpu.to(device="cuda", dtype=dtype)
    torch_out = torch.exp(x_cuda)
    triton_out = triton_exp(x_cuda)
    print_stats("torch.exp(cuda) vs ref", summarize_error(torch_out, ref))
    print_stats("triton.exp vs ref", summarize_error(triton_out, ref))
    print_stats("torch vs triton", summarize_error(torch_out, triton_out))


def compare_matmul(dtype: torch.dtype, m: int, n: int, k: int, seed: int) -> None:
    print(f"\n[matmul] dtype={dtype} shape=({m}, {k}) x ({k}, {n})")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a_cpu = torch.empty((m, k), dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
    b_cpu = torch.empty((k, n), dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
    ref = a_cpu.to(torch.float64) @ b_cpu.to(torch.float64)

    torch_out_cpu = a_cpu.to(dtype) @ b_cpu.to(dtype)
    print_stats("torch.mm(cpu) vs ref", summarize_error(torch_out_cpu, ref))

    if not torch.cuda.is_available():
        print("triton matmul skipped: CUDA is not available")
        return

    a_cuda = a_cpu.to(device="cuda", dtype=dtype).contiguous()
    b_cuda = b_cpu.to(device="cuda", dtype=dtype).contiguous()
    torch_out = a_cuda @ b_cuda
    triton_out = triton_matmul(a_cuda, b_cuda)
    print_stats("torch.mm(cuda) vs ref", summarize_error(torch_out, ref))
    print_stats("triton mm vs ref", summarize_error(triton_out, ref))
    print_stats("torch vs triton", summarize_error(torch_out, triton_out))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare precision of torch/triton exp and matmul."
    )
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--exp-size", type=int, default=1 << 20)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_name(args.dtype)
    compare_exp(dtype=dtype, size=args.exp_size, seed=args.seed)
    compare_matmul(dtype=dtype, m=args.m, n=args.n, k=args.k, seed=args.seed + 1)


if __name__ == "__main__":
    main()
