"""
Reusable JAX/Pallas operator profiler.

Usage (as library):
    from profiler import profile
    profile(fn, *args, trace_dir="/tmp/jax_profile")

Usage (standalone, called by Claude skill):
    python profiler.py --driver /tmp/profile_driver.py
"""

import gzip
import json
import os
import timeit
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Trace Analysis
# ---------------------------------------------------------------------------

def _find_latest_xplane(trace_dir: str) -> str | None:
    """Find the most recently modified .xplane.pb file under trace_dir."""
    candidates = sorted(Path(trace_dir).rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def _find_latest_trace_json(trace_dir: str) -> str | None:
    """Find the most recently modified .trace.json.gz file under trace_dir."""
    candidates = sorted(Path(trace_dir).rglob("*.trace.json.gz"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def analyze_xplane(trace_dir: str, flops: float = 0, bytes_accessed: float = 0) -> bool:
    """Parse .xplane.pb via jax._src.lib._profile_data and print analysis.

    Args:
        trace_dir: Directory containing xplane trace files.
        flops: Total FLOPs per invocation (from cost_analysis).
        bytes_accessed: Total bytes accessed per invocation (from cost_analysis).

    Returns True if analysis succeeded, False otherwise.
    """
    xplane_path = _find_latest_xplane(trace_dir)
    if not xplane_path:
        return False

    try:
        from jax._src.lib import _profile_data
    except ImportError:
        return False

    try:
        pd = _profile_data.ProfileData.from_file(xplane_path)
    except Exception:
        return False

    plane = pd.find_plane_with_name("/device:TPU:0")
    if plane is None:
        # Try other device planes
        for p in pd.planes:
            if p.name.startswith("/device:"):
                plane = p
                break
    if plane is None:
        return False

    # --- Hardware specs ---
    hw = {}
    for key, val in plane.stats:
        hw[key] = val

    device_type = hw.get("device_type_string", "Unknown")
    peak_tflops = hw.get("peak_teraflops_per_second", 0)
    peak_hbm_bw = hw.get("peak_hbm_bw_gigabytes_per_second", 0)

    print()
    print("=" * 60)
    print("Trace Analysis")
    print("=" * 60)
    print(f"  Device: {device_type}")
    specs = []
    if peak_tflops:
        specs.append(f"{peak_tflops:.1f} TFLOPS")
    if peak_hbm_bw:
        specs.append(f"HBM BW: {peak_hbm_bw:.1f} GB/s")
    if specs:
        print(f"  Peak:   {' | '.join(specs)}")

    # --- XLA Modules (per-invocation kernel time) ---
    modules_line = None
    ops_line = None
    for line in plane.lines:
        if line.name == "XLA Modules":
            modules_line = line
        elif line.name == "XLA Ops":
            ops_line = line

    if modules_line is not None:
        events = list(modules_line.events)
        if events:
            durations_us = [ev.duration_ns / 1000.0 for ev in events]
            n = len(durations_us)
            mean_us = sum(durations_us) / n
            min_us = min(durations_us)
            max_us = max(durations_us)
            print()
            print(f"  Module invocations: {n}")
            print(f"  Mean kernel time:   {mean_us:.2f} us")
            if n > 1:
                print(f"  Min/Max:            {min_us:.2f} / {max_us:.2f} us")

    # --- XLA Ops (per-op timing breakdown) ---
    if ops_line is not None:
        events = list(ops_line.events)
        if events:
            # Aggregate by op name
            op_times: dict[str, float] = defaultdict(float)
            for ev in events:
                op_times[ev.name] += ev.duration_ns / 1000.0  # ns -> us

            total_us = sum(op_times.values())
            sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)

            # Extract short op names for display
            def short_name(full: str) -> str:
                # Format: "%name = f32[...]{...} copy(...)" or "custom-call(...)"
                parts = full.split("=", 1)
                name = parts[0].strip()
                if len(parts) > 1:
                    rest = parts[1].strip()
                    # Skip type tokens (e.g. "f32[1024]{1,0:T(8,128)S(1)}")
                    # and find the actual HLO op keyword
                    tokens = rest.split()
                    for t in tokens:
                        # Skip type descriptors like f32[...]{...}
                        if t.startswith(("f16", "f32", "f64", "bf16", "s8",
                                         "s16", "s32", "s64", "u8", "u16",
                                         "u32", "u64", "pred", "c64", "c128")):
                            continue
                        # This should be the op keyword, possibly with args
                        op_type = t.split("(")[0]
                        return f"{name} ({op_type})"
                return name

            print()
            print("  Top ops by device time:")
            print(f"    {'Op':<50s} {'Time (us)':>10s}  {'%':>6s}")
            print(f"    {'-'*50} {'-'*10}  {'-'*6}")
            for full_name, time_us in sorted_ops[:15]:
                pct = time_us / total_us * 100 if total_us > 0 else 0
                sname = short_name(full_name)
                if len(sname) > 50:
                    sname = sname[:47] + "..."
                print(f"    {sname:<50s} {time_us:>10.2f}  {pct:>5.1f}%")

            print(f"    {'-'*50} {'-'*10}  {'-'*6}")
            print(f"    {'Total':<50s} {total_us:>10.2f}")

    # --- Device utilization ---
    mean_kernel_us = None
    if modules_line is not None:
        events = list(modules_line.events)
        if events:
            durations_us = [ev.duration_ns / 1000.0 for ev in events]
            mean_kernel_us = sum(durations_us) / len(durations_us)
            # Profile window = first event start to last event end
            first_start = min(ev.start_ns for ev in events)
            last_end = max(ev.end_ns for ev in events)
            window_ns = last_end - first_start
            busy_ns = sum(ev.duration_ns for ev in events)
            if window_ns > 0:
                util_pct = busy_ns / window_ns * 100
                print()
                print(f"  Device utilization: {util_pct:.1f}% (over {window_ns / 1000:.1f} us window)")

    # --- Roofline analysis ---
    if mean_kernel_us and mean_kernel_us > 0 and (bytes_accessed > 0 or flops > 0):
        mean_kernel_s = mean_kernel_us / 1e6
        print()
        print("  Roofline:")

        # Ridge point
        ridge_point = 0.0
        if peak_tflops and peak_hbm_bw:
            ridge_point = peak_tflops * 1e3 / peak_hbm_bw  # GFLOP/s / GB/s = FLOPs/byte
            print(f"    Ridge point:       {ridge_point:.1f} FLOPs/byte")

        # Arithmetic intensity
        ai = 0.0
        if flops > 0 and bytes_accessed > 0:
            ai = flops / bytes_accessed
            print(f"    Arith intensity:   {ai:.2f} FLOPs/byte")

        # Determine bound
        if ridge_point > 0 and ai > 0:
            if ai < ridge_point:
                print(f"    Bound:             MEMORY  (AI {ai:.1f} < ridge {ridge_point:.1f})")
            else:
                print(f"    Bound:             COMPUTE (AI {ai:.1f} >= ridge {ridge_point:.1f})")

        # Achieved bandwidth
        if bytes_accessed > 0 and peak_hbm_bw:
            achieved_bw = bytes_accessed / mean_kernel_s / 1e9  # GB/s
            bw_util = achieved_bw / peak_hbm_bw * 100
            print(f"    Achieved BW:       {achieved_bw:.1f} GB/s ({bw_util:.1f}% of {peak_hbm_bw:.0f} GB/s peak)")

        # Achieved compute
        if flops > 0 and peak_tflops:
            achieved_tflops = flops / mean_kernel_s / 1e12
            compute_util = achieved_tflops / peak_tflops * 100
            print(f"    Achieved compute:  {achieved_tflops:.3f} TFLOPS ({compute_util:.1f}% of {peak_tflops:.0f} TFLOPS peak)")

    print(f"  Source: {xplane_path}")
    return True


def analyze_trace_json(trace_dir: str) -> bool:
    """Fallback: parse .trace.json.gz Chrome Trace Event format.

    Returns True if analysis succeeded, False otherwise.
    """
    json_path = _find_latest_trace_json(trace_dir)
    if not json_path:
        return False

    try:
        with gzip.open(json_path, "rt") as f:
            data = json.load(f)
    except Exception:
        return False

    # Chrome Trace format: {"traceEvents": [...]}
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    if not events:
        return False

    # Find device events (look for TPU device process)
    device_events = []
    for ev in events:
        if ev.get("ph") != "X":  # complete events only
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if "Op" in cat or "xla" in cat.lower() or "custom" in name.lower():
            device_events.append(ev)

    if not device_events:
        return False

    # Aggregate by name
    op_times: dict[str, float] = defaultdict(float)
    for ev in device_events:
        op_times[ev["name"]] += ev.get("dur", 0)  # dur in microseconds

    total_us = sum(op_times.values())
    sorted_ops = sorted(op_times.items(), key=lambda x: x[1], reverse=True)

    print()
    print("=" * 60)
    print("Trace Analysis (from trace.json.gz)")
    print("=" * 60)
    print(f"  {'Op':<50s} {'Time (us)':>10s}  {'%':>6s}")
    print(f"  {'-'*50} {'-'*10}  {'-'*6}")
    for name, time_us in sorted_ops[:15]:
        pct = time_us / total_us * 100 if total_us > 0 else 0
        sname = name[:50] if len(name) <= 50 else name[:47] + "..."
        print(f"  {sname:<50s} {time_us:>10.2f}  {pct:>5.1f}%")
    print(f"  {'-'*50} {'-'*10}  {'-'*6}")
    print(f"  {'Total':<50s} {total_us:>10.2f}")
    print(f"  Source: {json_path}")
    return True


def profile(
    fn,
    *args,
    trace_dir: str = "/tmp/jax_profile",
    hlo_path: str = "/tmp/profile_hlo.txt",
    warmup: int = 10,
    repeat: int = 100,
    fn_name: str = "",
    **kwargs,
):
    """
    Profile a JAX function end-to-end.

    Produces:
      1. Cost analysis (FLOPs)
      2. Memory analysis
      3. HLO text dump
      4. jax.profiler.trace capture
      5. Wall-clock timing (mean/std/median/min/max)
      6. Arithmetic intensity estimate

    Returns:
      dict with timing results and file paths.
    """
    backend = jax.default_backend()
    label = fn_name or getattr(fn, "__name__", str(fn))
    print(f"Profiling: {label}")
    print(f"Backend:   {backend}")
    print()

    # ---- Input/Output shapes ----
    print("=" * 60)
    print("Input / Output Shapes")
    print("=" * 60)
    for i, a in enumerate(args):
        if hasattr(a, "shape"):
            print(f"  arg[{i}]: shape={a.shape}, dtype={a.dtype}")
        else:
            print(f"  arg[{i}]: {type(a).__name__} = {a}")
    for k, v in kwargs.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")

    # JIT compile
    jitted = jax.jit(fn)
    lowered = jitted.lower(*args, **kwargs)
    compiled = lowered.compile()

    # Probe output shape
    out = jitted(*args, **kwargs)
    if hasattr(out, "shape"):
        print(f"  output: shape={out.shape}, dtype={out.dtype}")
    elif isinstance(out, (tuple, list)):
        for i, o in enumerate(out):
            if hasattr(o, "shape"):
                print(f"  output[{i}]: shape={o.shape}, dtype={o.dtype}")

    # ---- Cost analysis ----
    print()
    print("=" * 60)
    print("Cost Analysis")
    print("=" * 60)
    flops = 0
    bytes_accessed = 0
    try:
        costs = compiled.cost_analysis()
        if costs:
            # costs may be a list of dicts, a list of strings, or a single dict
            if isinstance(costs, dict):
                costs = [costs]
            for i, cost in enumerate(costs):
                tag = f"  [device {i}] " if len(costs) > 1 else "  "
                if isinstance(cost, dict):
                    for key, value in cost.items():
                        print(f"{tag}{key}: {value:,}" if isinstance(value, (int, float)) else f"{tag}{key}: {value}")
                        if key == "flops":
                            flops = value
                        if key == "bytes accessed":
                            bytes_accessed = value
                elif cost is not None:
                    print(f"{tag}{cost}")
            # Fallback: if no total "bytes accessed", sum per-operand keys
            if bytes_accessed == 0:
                for cost in costs:
                    if isinstance(cost, dict):
                        for key, value in cost.items():
                            if key.startswith("bytes accessed") and key != "bytes accessed":
                                bytes_accessed += value
        else:
            print("  (not available)")
    except Exception as e:
        print(f"  Error: {e}")

    # ---- Memory analysis ----
    print()
    print("=" * 60)
    print("Memory Analysis")
    print("=" * 60)
    try:
        mem = compiled.memory_analysis()
        if mem is not None:
            attrs = [
                "argument_size_in_bytes",
                "output_size_in_bytes",
                "alias_size_in_bytes",
                "temp_size_in_bytes",
                "generated_code_size_in_bytes",
                "host_temp_size_in_bytes",
                "host_argument_size_in_bytes",
                "host_generated_code_size_in_bytes",
            ]
            for attr in attrs:
                val = getattr(mem, attr, None)
                if val is not None and val > 0:
                    print(f"  {attr}: {val:,} bytes ({val / 1024 / 1024:.2f} MB)")
        else:
            print("  (not available)")
    except Exception as e:
        print(f"  Error: {e}")

    # ---- HLO dump ----
    print()
    print("=" * 60)
    print(f"HLO (saved to {hlo_path})")
    print("=" * 60)
    hlo_lines = 0
    try:
        hlo_text = compiled.as_text()
        Path(hlo_path).write_text(hlo_text)
        lines = hlo_text.split("\n")
        hlo_lines = len(lines)
        for line in lines[:80]:
            print(f"  {line}")
        if hlo_lines > 80:
            print(f"  ... ({hlo_lines - 80} more lines, see {hlo_path})")
    except Exception as e:
        print(f"  Error: {e}")

    # ---- Warmup ----
    for _ in range(warmup):
        jitted(*args, **kwargs).block_until_ready()

    # ---- jax.profiler.trace ----
    print()
    print("=" * 60)
    print("JAX Profiler Trace")
    print("=" * 60)
    os.makedirs(trace_dir, exist_ok=True)
    try:
        with jax.profiler.trace(trace_dir):
            for _ in range(5):
                jitted(*args, **kwargs).block_until_ready()

        # Auto-analyze the trace
        analyzed = analyze_xplane(trace_dir, flops=flops, bytes_accessed=bytes_accessed)
        if not analyzed:
            analyzed = analyze_trace_json(trace_dir)
        if not analyzed:
            # Fallback: just list files
            trace_files = []
            for root, _, files in os.walk(trace_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    size = os.path.getsize(fp)
                    trace_files.append((fp, size))
            if trace_files:
                print("  Generated files:")
                for fp, size in trace_files:
                    print(f"    {fp}  ({size:,} bytes)")
            else:
                print("  (no trace files generated)")

        print(f"\n  TensorBoard: tensorboard --logdir={trace_dir}")
    except Exception as e:
        print(f"  Trace error: {e}")

    # ---- Timing ----
    print()
    print("=" * 60)
    print("Timing Benchmark")
    print("=" * 60)
    times = []
    for _ in range(repeat):
        t0 = timeit.default_timer()
        jitted(*args, **kwargs).block_until_ready()
        times.append(timeit.default_timer() - t0)

    times_ms = np.array([t * 1000 for t in times])
    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    median_ms = float(np.median(times_ms))
    min_ms = float(np.min(times_ms))
    max_ms = float(np.max(times_ms))

    print(f"  Iterations: {repeat}")
    print(f"  Mean:   {mean_ms:.4f} ms")
    print(f"  Std:    {std_ms:.4f} ms")
    print(f"  Median: {median_ms:.4f} ms")
    print(f"  Min:    {min_ms:.4f} ms")
    print(f"  Max:    {max_ms:.4f} ms")

    # ---- Summary ----
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Operator:  {label}")
    print(f"  Time:      {mean_ms:.4f} ± {std_ms:.4f} ms")

    if flops > 0:
        total_bytes = sum(
            a.size * a.dtype.itemsize for a in args if hasattr(a, "size")
        )
        if hasattr(out, "size"):
            total_bytes += out.size * out.dtype.itemsize

        gflops_s = flops / (mean_ms / 1000) / 1e9
        print(f"  FLOPs:     {flops:,}")
        print(f"  GFLOP/s:   {gflops_s:.2f}")
        if total_bytes > 0:
            ai = flops / total_bytes
            bw_gb = total_bytes / (mean_ms / 1000) / 1e9
            print(f"  Data:      {total_bytes:,} bytes ({total_bytes / 1024 / 1024:.2f} MB)")
            print(f"  Arith intensity: {ai:.2f} FLOPs/byte")
            print(f"  Bandwidth: {bw_gb:.2f} GB/s")

    print(f"  HLO:       {hlo_lines} lines -> {hlo_path}")
    print(f"  Trace:     {trace_dir}")
    print()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "flops": flops,
        "trace_dir": trace_dir,
        "hlo_path": hlo_path,
    }
