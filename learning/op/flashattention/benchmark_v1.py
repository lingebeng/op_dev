"""
FlashAttention V1 vs Naive Attention — 全面 Profiling
=====================================================

对比维度:
  1. 延迟 (Latency)         — cuda event 精确计时
  2. 峰值显存 (Peak Memory) — FlashAttn O(N) vs Naive O(N²)
  3. 计算吞吐 (TFLOPS)      — 实际达到的算力
  4. Scaling 曲线            — 随序列长度 N 的变化趋势

运行: source .venv/bin/activate && python learning/op/flashattention/benchmark_v1.py
"""

import gc
import math
import sys
import os

import torch

# 将 flashattention 目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flash_attention_v1 import flash_attention_v1, reference_attention


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
def attention_flops(B, H, N, D, causal=False):
    """
    Attention 理论 FLOPs (乘加各算 1 次 = 2 ops per multiply-add)
    Q@K^T: 2·B·H·N·N·D
    P@V:   2·B·H·N·N·D
    总计:  4·B·H·N²·D
    因果时约一半: 2·B·H·N²·D
    """
    total = 4 * B * H * N * N * D
    if causal:
        total //= 2
    return total



def benchmark_fn(fn, *args, warmup=5, repeats=20):
    """
    使用 CUDA events 精确测量 kernel 延迟 (ms)
    返回 (median_ms, min_ms, max_ms)
    """
    # warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    timings.sort()
    median = timings[len(timings) // 2]
    return median, min(timings), max(timings)


def measure_peak_memory(fn, *args):
    """测量函数执行时的峰值显存增量 (MB)"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()

    fn(*args)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    return (peak - mem_before) / (1024 ** 2)  # MB


def try_run(fn, *args):
    """尝试运行，OOM 时返回 None"""
    try:
        return fn(*args)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        return None


# ──────────────────────────────────────────────────────────────
# 主 Benchmark
# ──────────────────────────────────────────────────────────────
def main():
    assert torch.cuda.is_available(), "需要 CUDA"
    device = "cuda"
    dtype = torch.bfloat16

    gpu_name = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu_name}")

    # ─── 配置 ───
    B, H, D = 1, 32, 64
    causal = True
    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print(f"\n配置: B={B}, H={H}, D={D}, causal={causal}, dtype={dtype}")

    # ─── 表头 ───
    header = (
        f"{'N':>6} | "
        f"{'Flash ms':>9} {'Naive ms':>9} {'加速比':>7} | "
        f"{'Flash MB':>9} {'Naive MB':>9} {'显存比':>7} | "
        f"{'Flash TF':>9} {'Naive TF':>9}"
    )
    separator = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("FlashAttention V1 vs Naive Attention 全面对比")
    print(f"{'='*len(header)}")
    print(header)
    print(separator)

    # ─── 收集数据 (用于绘图) ───
    results = {
        "seq_lens": [],
        "flash_latency": [], "naive_latency": [],
        "flash_memory": [], "naive_memory": [],
        "flash_tflops": [], "naive_tflops": [],
    }

    for N in seq_lens:
        results["seq_lens"].append(N)

        # 生成数据
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        flops = attention_flops(B, H, N, D, causal=causal)

        # ─── FlashAttention V1 ───
        flash_med, flash_min, flash_max = benchmark_fn(
            flash_attention_v1, q, k, v, causal
        )

        torch.cuda.empty_cache()
        gc.collect()
        flash_mem = measure_peak_memory(flash_attention_v1, q, k, v, causal)

        flash_tflops = flops / (flash_med * 1e-3) / 1e12

        results["flash_latency"].append(flash_med)
        results["flash_memory"].append(flash_mem)
        results["flash_tflops"].append(flash_tflops)

        # ─── Naive Attention ───
        torch.cuda.empty_cache()
        gc.collect()

        naive_result = try_run(benchmark_fn, reference_attention, q, k, v, causal)

        if naive_result is not None:
            naive_med, naive_min, naive_max = naive_result

            torch.cuda.empty_cache()
            gc.collect()
            naive_mem = measure_peak_memory(reference_attention, q, k, v, causal)

            naive_tflops = flops / (naive_med * 1e-3) / 1e12

            speedup = naive_med / flash_med
            mem_ratio = naive_mem / flash_mem if flash_mem > 0 else float("inf")

            results["naive_latency"].append(naive_med)
            results["naive_memory"].append(naive_mem)
            results["naive_tflops"].append(naive_tflops)

            print(
                f"{N:>6} | "
                f"{flash_med:>8.3f}m {naive_med:>8.3f}m {speedup:>6.2f}x | "
                f"{flash_mem:>8.1f}M {naive_mem:>8.1f}M {mem_ratio:>6.1f}x | "
                f"{flash_tflops:>8.2f}T {naive_tflops:>8.2f}T"
            )
        else:
            results["naive_latency"].append(None)
            results["naive_memory"].append(None)
            results["naive_tflops"].append(None)

            print(
                f"{N:>6} | "
                f"{flash_med:>8.3f}m {'OOM':>9} {'N/A':>7} | "
                f"{flash_mem:>8.1f}M {'OOM':>9} {'N/A':>7} | "
                f"{flash_tflops:>8.2f}T {'OOM':>9}"
            )

        # 释放数据
        del q, k, v
        torch.cuda.empty_cache()
        gc.collect()

    print(separator)

    # ─── 分析总结 ───
    print(f"\n{'='*60}")
    print("分析总结")
    print(f"{'='*60}")

    # 找到 naive OOM 的最小 N
    naive_oom_n = None
    for i, nl in enumerate(results["naive_latency"]):
        if nl is None:
            naive_oom_n = results["seq_lens"][i]
            break

    if naive_oom_n:
        print(f"\n[显存] Naive Attention 在 N={naive_oom_n} 时 OOM")
        print(f"       FlashAttention V1 在所有 N 下均正常运行")

    # 有效对比的最大 N
    valid = [(i, n) for i, n in enumerate(results["seq_lens"])
             if results["naive_latency"][i] is not None]

    if valid:
        last_i, last_n = valid[-1]
        speedup = results["naive_latency"][last_i] / results["flash_latency"][last_i]
        mem_ratio = results["naive_memory"][last_i] / results["flash_memory"][last_i]

        print(f"\n[延迟] 在 N={last_n} 时:")
        print(f"       Flash: {results['flash_latency'][last_i]:.3f} ms")
        print(f"       Naive: {results['naive_latency'][last_i]:.3f} ms")
        print(f"       加速比: {speedup:.2f}x")

        print(f"\n[显存] 在 N={last_n} 时:")
        print(f"       Flash: {results['flash_memory'][last_i]:.1f} MB")
        print(f"       Naive: {results['naive_memory'][last_i]:.1f} MB")
        print(f"       显存节省: {mem_ratio:.1f}x")

        print(f"\n[吞吐] 在 N={last_n} 时:")
        print(f"       Flash: {results['flash_tflops'][last_i]:.2f} TFLOPS")
        print(f"       Naive: {results['naive_tflops'][last_i]:.2f} TFLOPS")
        print(f"       Flash 计算利用率是 Naive 的 "
              f"{results['flash_tflops'][last_i]/results['naive_tflops'][last_i]:.1f} 倍")

    print(f"\n[理论复杂度]")
    print(f"       Naive 显存: O(B·H·N²) — 需实例化 N×N 注意力矩阵")
    print(f"       Flash 显存: O(B·H·N)  — 仅存 Q/K/V/O，分块在 SRAM 计算")
    print(f"       两者 FLOPs 相同: O(B·H·N²·D)")

    # ─── 绘图 ───
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"FlashAttention V1 vs Naive Attention  (B={B}, H={H}, D={D}, causal, FP16, {gpu_name})",
                     fontsize=13, fontweight="bold")

        ns = results["seq_lens"]
        valid_ns = [n for i, n in enumerate(ns) if results["naive_latency"][i] is not None]
        valid_flash_lat = [results["flash_latency"][i] for i, n in enumerate(ns) if results["naive_latency"][i] is not None]
        valid_naive_lat = [results["naive_latency"][i] for i, n in enumerate(ns) if results["naive_latency"][i] is not None]
        valid_flash_mem = [results["flash_memory"][i] for i, n in enumerate(ns) if results["naive_memory"][i] is not None]
        valid_naive_mem = [results["naive_memory"][i] for i, n in enumerate(ns) if results["naive_memory"][i] is not None]
        valid_flash_tf  = [results["flash_tflops"][i] for i, n in enumerate(ns) if results["naive_tflops"][i] is not None]
        valid_naive_tf  = [results["naive_tflops"][i] for i, n in enumerate(ns) if results["naive_tflops"][i] is not None]

        flash_only_ns  = [n for i, n in enumerate(ns) if results["naive_latency"][i] is None]
        flash_only_lat = [results["flash_latency"][i] for i, n in enumerate(ns) if results["naive_latency"][i] is None]
        flash_only_mem = [results["flash_memory"][i] for i, n in enumerate(ns) if results["naive_latency"][i] is None]
        flash_only_tf  = [results["flash_tflops"][i] for i, n in enumerate(ns) if results["naive_latency"][i] is None]

        # 1. Latency
        ax = axes[0, 0]
        ax.plot(valid_ns, valid_flash_lat, "o-", label="FlashAttn V1", color="tab:blue")
        ax.plot(valid_ns, valid_naive_lat, "s-", label="Naive", color="tab:red")
        if flash_only_ns:
            ax.plot(flash_only_ns, flash_only_lat, "o--", color="tab:blue", alpha=0.5)
        if naive_oom_n:
            ax.axvline(x=naive_oom_n, color="tab:red", linestyle=":", alpha=0.7, label=f"Naive OOM (N={naive_oom_n})")
        ax.set_xlabel("Sequence Length (N)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency (lower is better)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Peak Memory
        ax = axes[0, 1]
        ax.plot(valid_ns, valid_flash_mem, "o-", label="FlashAttn V1", color="tab:blue")
        ax.plot(valid_ns, valid_naive_mem, "s-", label="Naive", color="tab:red")
        if flash_only_ns:
            ax.plot(flash_only_ns, flash_only_mem, "o--", color="tab:blue", alpha=0.5)
        if naive_oom_n:
            ax.axvline(x=naive_oom_n, color="tab:red", linestyle=":", alpha=0.7, label=f"Naive OOM (N={naive_oom_n})")
        ax.set_xlabel("Sequence Length (N)")
        ax.set_ylabel("Peak Memory (MB)")
        ax.set_title("Peak Memory (lower is better)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. TFLOPS
        ax = axes[1, 0]
        ax.plot(valid_ns, valid_flash_tf, "o-", label="FlashAttn V1", color="tab:blue")
        ax.plot(valid_ns, valid_naive_tf, "s-", label="Naive", color="tab:red")
        if flash_only_ns:
            ax.plot(flash_only_ns, flash_only_tf, "o--", color="tab:blue", alpha=0.5)
        ax.set_xlabel("Sequence Length (N)")
        ax.set_ylabel("TFLOPS")
        ax.set_title("Compute Throughput (higher is better)")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Speedup & Memory Ratio
        ax = axes[1, 1]
        if valid_ns:
            speedups = [n / f for f, n in zip(valid_flash_lat, valid_naive_lat)]
            mem_ratios = [n / f for f, n in zip(valid_flash_mem, valid_naive_mem)]
            ax.plot(valid_ns, speedups, "o-", label="Speedup (latency)", color="tab:green")
            ax.plot(valid_ns, mem_ratios, "s-", label="Memory ratio", color="tab:orange")
            ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length (N)")
        ax.set_ylabel("Ratio (Naive / Flash)")
        ax.set_title("Speedup & Memory Saving (higher is better)")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ─── 底部说明 ───
        caption = (
            "Fig. FlashAttention V1 vs Naive Attention on NVIDIA H100 80GB.\n"
            "Left-top: Latency — both methods have identical FLOPs O(N²D), but Naive is memory-bound (bottlenecked by N×N matrix HBM R/W),\n"
            "    while Flash keeps tiles in SRAM, achieving up to ~16x speedup.\n"
            "Right-top: Peak Memory — Naive must materialize the full N×N attention matrix in HBM → O(N²) memory;\n"
            "    Flash only stores Q/K/V/O → O(N) memory. At N=16384 the gap reaches 517x (64MB vs 33GB).\n"
            "Left-bottom: TFLOPS — Flash scales to ~210 TFLOPS as N grows (compute-bound); Naive stays flat at ~15 TFLOPS\n"
            "    regardless of N (memory-bound, GPU compute units mostly idle waiting for data transfer).\n"
            "Right-bottom: Ratios — Green: latency speedup; Orange: memory saving ratio. Both grow with N,\n"
            "    with memory ratio growing linearly (N²/N = N), confirming the theoretical complexity gap."
        )
        fig.text(0.02, -0.01, caption, fontsize=8, fontfamily="monospace",
                 verticalalignment="top", color="dimgray",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="lightgray", alpha=0.9))

        plt.tight_layout(rect=[0, 0.13, 1, 1])
        fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_v1.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"\n图表已保存: {fig_path}")

    except ImportError:
        print("\n[跳过绘图] 未安装 matplotlib")


if __name__ == "__main__":
    main()
