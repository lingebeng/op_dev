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
# Trace Analysis helpers
# ---------------------------------------------------------------------------

def _find_latest_xplane(trace_dir: str) -> str | None:
    """Find the most recently modified .xplane.pb file under trace_dir."""
    candidates = sorted(Path(trace_dir).rglob("*.xplane.pb"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def _find_latest_trace_json(trace_dir: str) -> str | None:
    """Find the most recently modified .trace.json.gz file under trace_dir."""
    candidates = sorted(Path(trace_dir).rglob("*.trace.json.gz"), key=lambda p: p.stat().st_mtime)
    return str(candidates[-1]) if candidates else None


def _parse_op_type(full_name: str) -> tuple[str, str]:
    """Extract (short_name, op_type) from a full HLO op name.

    Returns e.g. ("%cumsum.1", "custom-call") or ("%copy.2", "copy").
    """
    parts = full_name.split("=", 1)
    name = parts[0].strip()
    op_type = ""
    if len(parts) > 1:
        rest = parts[1].strip()
        for t in rest.split():
            # Skip type descriptors like f32[...]{...}
            if t.startswith(("f16", "f32", "f64", "bf16", "s8", "s16", "s32",
                             "s64", "u8", "u16", "u32", "u64", "pred",
                             "c64", "c128")):
                continue
            op_type = t.split("(")[0]
            break
    return name, op_type


# XLA overhead op types (not user kernel logic)
_OVERHEAD_OPS = frozenset({"copy", "pad", "fusion", "bitcast", "reshape",
                           "slice", "concatenate", "broadcast", "tuple",
                           "get-tuple-element", "dynamic-slice"})


def _parse_xplane(trace_dir: str) -> dict | None:
    """Parse .xplane.pb and return structured analysis data.

    Returns dict with hardware specs, kernel timing, op breakdown,
    and device busy ratio. Returns None if parsing fails.
    """
    xplane_path = _find_latest_xplane(trace_dir)
    if not xplane_path:
        return None

    try:
        from jax._src.lib import _profile_data
    except ImportError:
        return None

    try:
        pd = _profile_data.ProfileData.from_file(xplane_path)
    except Exception:
        return None

    plane = pd.find_plane_with_name("/device:TPU:0")
    if plane is None:
        for p in pd.planes:
            if p.name.startswith("/device:"):
                plane = p
                break
    if plane is None:
        return None

    # --- Hardware specs ---
    hw = {}
    for key, val in plane.stats:
        hw[key] = val

    result = {
        "xplane_path": xplane_path,
        "device_type": hw.get("device_type_string", "Unknown"),
        "peak_tflops": hw.get("peak_teraflops_per_second", 0),
        "peak_hbm_bw": hw.get("peak_hbm_bw_gigabytes_per_second", 0),
    }

    # --- Collect lines ---
    modules_line = None
    ops_line = None
    for line in plane.lines:
        if line.name == "XLA Modules":
            modules_line = line
        elif line.name == "XLA Ops":
            ops_line = line

    # Module-level timing (ground truth for per-invocation device time)
    if modules_line is not None:
        events = list(modules_line.events)
        if events:
            durations_us = [ev.duration_ns / 1000.0 for ev in events]
            result["n_invocations"] = len(durations_us)
            result["mean_kernel_us"] = sum(durations_us) / len(durations_us)
            result["min_kernel_us"] = min(durations_us)
            result["max_kernel_us"] = max(durations_us)

            # Extract module name: "jit_cumsum(hash)" -> "cumsum"
            raw = events[0].name
            if "(" in raw:
                raw = raw[:raw.index("(")]
            raw = raw.removeprefix("jit_").strip("_")
            if raw and raw != "lambda":
                result["module_name"] = raw


    # Op-level breakdown: separate user kernel from XLA overhead
    kernel_ops = []
    overhead_ops = []
    if ops_line is not None:
        op_times: dict[str, tuple[str, float]] = {}
        for ev in list(ops_line.events):
            dur = ev.duration_ns / 1000.0
            if ev.name in op_times:
                _, prev = op_times[ev.name]
                op_times[ev.name] = (op_times[ev.name][0], prev + dur)
            else:
                _, ot = _parse_op_type(ev.name)
                op_times[ev.name] = (ot, dur)

        for full_name, (op_type, time_us) in op_times.items():
            short, _ = _parse_op_type(full_name)
            if op_type in _OVERHEAD_OPS:
                overhead_ops.append((short, op_type, time_us))
            else:
                kernel_ops.append((short, op_type, time_us))

    result["kernel_ops"] = kernel_ops
    result["overhead_ops"] = overhead_ops
    result["kernel_us"] = sum(t for _, _, t in kernel_ops)
    result["overhead_us"] = sum(t for _, _, t in overhead_ops)

    return result


def _parse_trace_json(trace_dir: str) -> dict | None:
    """Fallback: parse .trace.json.gz Chrome Trace Event format.

    Returns dict with op_times and total_us, or None if parsing fails.
    """
    json_path = _find_latest_trace_json(trace_dir)
    if not json_path:
        return None

    try:
        with gzip.open(json_path, "rt") as f:
            data = json.load(f)
    except Exception:
        return None

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    if not events:
        return None

    device_events = []
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if "Op" in cat or "xla" in cat.lower() or "custom" in name.lower():
            device_events.append(ev)

    if not device_events:
        return None

    op_times: dict[str, float] = defaultdict(float)
    for ev in device_events:
        op_times[ev["name"]] += ev.get("dur", 0)

    return {
        "json_path": json_path,
        "op_times": dict(op_times),
        "total_us": sum(op_times.values()),
    }


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

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
    Profile a JAX function and output a three-layer analysis:
      Layer 1: Timing   — wall-clock + device kernel breakdown
      Layer 2: Utilization — roofline analysis (BW, FLOPS vs hardware peak)
      Layer 3: Optimization Suggestions — actionable observations

    Returns:
      dict with timing results and file paths.
    """
    backend = jax.default_backend()
    label = fn_name or getattr(fn, "__name__", str(fn))

    # ==================================================================
    # Data collection (silent)
    # ==================================================================

    # ---- Compile ----
    jitted = jax.jit(fn)
    lowered = jitted.lower(*args, **kwargs)
    compiled = lowered.compile()
    out = jitted(*args, **kwargs)

    # ---- Cost analysis ----
    flops = 0
    bytes_accessed = 0
    cost_details = []
    try:
        costs = compiled.cost_analysis()
        if costs:
            if isinstance(costs, dict):
                costs = [costs]
            for cost in costs:
                if isinstance(cost, dict):
                    for key, value in cost.items():
                        cost_details.append((key, value))
                        if key == "flops":
                            flops = value
                        if key == "bytes accessed":
                            bytes_accessed = value
            # Fallback: sum per-operand "bytes accessed*" keys
            if bytes_accessed == 0:
                for cost in costs:
                    if isinstance(cost, dict):
                        for key, value in cost.items():
                            if key.startswith("bytes accessed") and key != "bytes accessed":
                                bytes_accessed += value
    except Exception:
        pass

    # ---- Memory analysis ----
    mem_info = {}
    try:
        mem = compiled.memory_analysis()
        if mem is not None:
            for attr in ["argument_size_in_bytes", "output_size_in_bytes",
                         "temp_size_in_bytes"]:
                val = getattr(mem, attr, None)
                if val is not None and val > 0:
                    mem_info[attr] = val
    except Exception:
        pass

    # ---- HLO (save to file) ----
    hlo_lines = 0
    try:
        hlo_text = compiled.as_text()
        Path(hlo_path).write_text(hlo_text)
        hlo_lines = len(hlo_text.split("\n"))
    except Exception:
        pass

    # ---- Warmup ----
    for _ in range(warmup):
        jitted(*args, **kwargs).block_until_ready()

    # ---- Trace capture ----
    os.makedirs(trace_dir, exist_ok=True)
    trace_data = None
    try:
        with jax.profiler.trace(trace_dir):
            for _ in range(5):
                jitted(*args, **kwargs).block_until_ready()
        trace_data = _parse_xplane(trace_dir)
    except Exception:
        pass

    # ---- Wall-clock timing ----
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

    # ==================================================================
    # OUTPUT: Three-layer analysis
    # ==================================================================

    # Header
    print(f"Profiling: {label}")
    print(f"Backend:   {backend}")
    for i, a in enumerate(args):
        if hasattr(a, "shape"):
            print(f"  in[{i}]:  {a.shape} {a.dtype}")
    if hasattr(out, "shape"):
        print(f"  out:    {out.shape} {out.dtype}")
    elif isinstance(out, (tuple, list)):
        for i, o in enumerate(out):
            if hasattr(o, "shape"):
                print(f"  out[{i}]: {o.shape} {o.dtype}")

    # ================================================================
    # Layer 1: Timing
    # ================================================================
    print()
    print("=" * 60)
    print("Layer 1: Timing")
    print("=" * 60)

    print(f"  Wall-clock: {mean_ms:.4f} ± {std_ms:.4f} ms"
          f"  (median {median_ms:.4f}, min {min_ms:.4f}, max {max_ms:.4f},"
          f" n={repeat})")

    # Resolve display name: fn_name > module_name from trace > label
    display_name = label
    device_kernel_us = None

    if trace_data:
        if not fn_name and trace_data.get("module_name"):
            display_name = trace_data["module_name"]

        mean_k = trace_data.get("mean_kernel_us")
        n_inv = trace_data.get("n_invocations", 0)
        if mean_k is not None:
            device_kernel_us = mean_k
            extra = ""
            if n_inv > 1:
                extra = (f", min {trace_data['min_kernel_us']:.2f}"
                         f" / max {trace_data['max_kernel_us']:.2f}")
            print(f"  Device kernel: {mean_k:.2f} us"
                  f"  (mean of {n_inv} invocations{extra})")

        # Normalize op times to per-invocation averages
        divisor = n_inv if n_inv > 1 else 1
        total_ops_us = (trace_data["kernel_us"] + trace_data["overhead_us"]) / divisor
        if total_ops_us > 0:
            # User kernel ops (the ones the user cares about)
            kernel_ops = sorted(trace_data["kernel_ops"],
                                key=lambda x: x[2], reverse=True)
            for short, op_type, t_total in kernel_ops:
                t = t_total / divisor
                pct = t / total_ops_us * 100
                if op_type == "custom-call":
                    op_label = f"{display_name} (pallas_call)"
                elif op_type:
                    op_label = f"{short} ({op_type})"
                else:
                    op_label = short
                if len(op_label) > 50:
                    op_label = op_label[:47] + "..."
                print(f"    {op_label:<50s} {t:>8.2f} us  {pct:>5.1f}%")

            # Summarize XLA overhead as one line
            overhead_us = trace_data["overhead_us"] / divisor
            if overhead_us > 0:
                overhead_pct = overhead_us / total_ops_us * 100
                overhead_detail = ", ".join(
                    f"{op}={t / divisor:.1f}us" for _, op, t in
                    sorted(trace_data["overhead_ops"],
                           key=lambda x: x[2], reverse=True)
                )
                print(f"    {'[XLA overhead]':<50s}"
                      f" {overhead_us:>8.2f} us  {overhead_pct:>5.1f}%"
                      f"  ({overhead_detail})")
    else:
        # Fallback: try trace.json.gz
        json_data = _parse_trace_json(trace_dir)
        if json_data:
            print(f"\n  (from trace.json.gz — xplane unavailable)")
            sorted_ops = sorted(json_data["op_times"].items(),
                                key=lambda x: x[1], reverse=True)
            total_us = json_data["total_us"]
            for name, t in sorted_ops[:10]:
                pct = t / total_us * 100 if total_us > 0 else 0
                sname = name[:50] if len(name) <= 50 else name[:47] + "..."
                print(f"    {sname:<50s} {t:>8.2f} us  {pct:>5.1f}%")

    # ================================================================
    # Layer 2: Utilization / BW / Resources
    # ================================================================
    print()
    print("=" * 60)
    print("Layer 2: Utilization")
    print("=" * 60)

    def _fmt_bytes(n: int | float) -> str:
        """Format byte count with adaptive units."""
        n = float(n)
        if n < 1024:
            return f"{n:.0f} B"
        elif n < 1024 ** 2:
            return f"{n / 1024:.1f} KB"
        elif n < 1024 ** 3:
            return f"{n / 1024 ** 2:.1f} MB"
        else:
            return f"{n / 1024 ** 3:.2f} GB"

    # -- Hardware specs --
    peak_tflops = 0
    peak_hbm_bw = 0
    if trace_data:
        device_type = trace_data["device_type"]
        peak_tflops = trace_data["peak_tflops"]
        peak_hbm_bw = trace_data["peak_hbm_bw"]

        print(f"  Device: {device_type}")
        specs = []
        if peak_tflops:
            specs.append(f"{peak_tflops:.1f} TFLOPS")
        if peak_hbm_bw:
            specs.append(f"HBM BW: {peak_hbm_bw:.1f} GB/s")
        if specs:
            print(f"  Peak:   {' | '.join(specs)}")

    else:
        print("  (trace analysis unavailable, using wall-clock — less accurate)")

    # -- I/O tensor sizes --
    in_bytes = sum(
        np.prod(a.shape) * a.dtype.itemsize
        for a in args if hasattr(a, "shape") and hasattr(a, "dtype")
    )
    out_bytes = 0
    if hasattr(out, "shape") and hasattr(out, "dtype"):
        out_bytes = int(np.prod(out.shape)) * out.dtype.itemsize
    elif isinstance(out, (tuple, list)):
        for o in out:
            if hasattr(o, "shape") and hasattr(o, "dtype"):
                out_bytes += int(np.prod(o.shape)) * o.dtype.itemsize
    if in_bytes or out_bytes:
        print(f"  I/O size:        in {_fmt_bytes(in_bytes)}, out {_fmt_bytes(out_bytes)}"
              f"  (total {_fmt_bytes(in_bytes + out_bytes)})")

    # -- cost_analysis details --
    if cost_details:
        print(f"  Cost analysis:")
        for key, value in cost_details:
            if isinstance(value, float) and value == int(value):
                value = int(value)
            if isinstance(value, (int, float)) and value >= 1024 and "byte" in key.lower():
                print(f"    {key}: {value:,} ({_fmt_bytes(value)})")
            elif isinstance(value, (int, float)):
                print(f"    {key}: {value:,}")
            else:
                print(f"    {key}: {value}")

    # -- Roofline analysis --
    # Only use device kernel time; if unavailable, show None
    if device_kernel_us and device_kernel_us > 0 and (bytes_accessed > 0 or flops > 0):
        kernel_s = device_kernel_us / 1e6

        # Ridge point
        ridge_point = 0.0
        if peak_tflops and peak_hbm_bw:
            ridge_point = peak_tflops * 1e3 / peak_hbm_bw

        # Arithmetic intensity & bound classification
        if flops > 0 and bytes_accessed > 0:
            ai = flops / bytes_accessed
            bound_str = ""
            if ridge_point > 0:
                if ai < ridge_point:
                    bound_str = f"  -> MEMORY-bound (ridge = {ridge_point:.0f})"
                else:
                    bound_str = f"  -> COMPUTE-bound (ridge = {ridge_point:.0f})"
            print(f"  Arith intensity: {ai:.2f} FLOPs/byte{bound_str}")
        elif bytes_accessed > 0:
            print(f"  No FLOPs reported  -> likely MEMORY-bound")

        # Achieved bandwidth
        if bytes_accessed > 0:
            achieved_bw = bytes_accessed / kernel_s / 1e9
            if peak_hbm_bw:
                bw_util = achieved_bw / peak_hbm_bw * 100
                print(f"  Achieved BW:     {achieved_bw:.1f} GB/s"
                      f"  ({bw_util:.1f}% of {peak_hbm_bw:.0f} peak)")
            else:
                print(f"  Achieved BW:     {achieved_bw:.1f} GB/s")

        # Achieved compute
        if flops > 0:
            achieved_tflops = flops / kernel_s / 1e12
            if peak_tflops:
                compute_util = achieved_tflops / peak_tflops * 100
                print(f"  Achieved FLOPS:  {achieved_tflops:.3f} TFLOPS"
                      f"  ({compute_util:.1f}% of {peak_tflops:.0f} peak)")
            else:
                print(f"  Achieved FLOPS:  {achieved_tflops:.3f} TFLOPS")
    else:
        if not (bytes_accessed > 0 or flops > 0):
            print("  Achieved BW:     None (cost_analysis returned no bytes)")
            print("  Achieved FLOPS:  None (cost_analysis returned no FLOPs)")
        elif not device_kernel_us:
            print("  Achieved BW:     None (device kernel time unavailable)")
            print("  Achieved FLOPS:  None (device kernel time unavailable)")

    # -- Memory usage from compiled.memory_analysis() --
    if mem_info:
        parts = []
        for attr, val in mem_info.items():
            name = attr.replace("_in_bytes", "").replace("_size", "")
            parts.append(f"{name}: {_fmt_bytes(val)}")
        print(f"  Memory alloc:    {', '.join(parts)}")

    # ================================================================
    # Layer 3: Optimization Suggestions
    # ================================================================
    observations = []

    if trace_data:
        total_ops_us = trace_data["kernel_us"] + trace_data["overhead_us"]
        overhead_us = trace_data["overhead_us"]

        # 1. XLA overhead check
        if total_ops_us > 0 and overhead_us > 0:
            overhead_pct = overhead_us / total_ops_us * 100
            if overhead_pct > 15:
                overhead_types = set(op for _, op, _ in trace_data["overhead_ops"])
                observations.append(
                    f"XLA overhead is {overhead_pct:.0f}% of device time "
                    f"({', '.join(sorted(overhead_types))}). "
                    f"Consider fusing into a larger kernel or fixing input layout.")

        # 2. BW utilization check
        if (device_kernel_us and bytes_accessed > 0
                and trace_data["peak_hbm_bw"]):
            achieved_bw = bytes_accessed / (device_kernel_us / 1e6) / 1e9
            bw_util = achieved_bw / trace_data["peak_hbm_bw"] * 100
            if bw_util < 20:
                observations.append(
                    f"Low BW utilization ({bw_util:.0f}%). "
                    f"Kernel may be too small for the DMA pipeline to saturate, "
                    f"or tile dimensions underutilize MXU lanes.")


    if observations:
        print()
        print("=" * 60)
        print("Layer 3: Optimization Suggestions")
        print("=" * 60)
        for i, obs in enumerate(observations, 1):
            print(f"  {i}. {obs}")

    # ================================================================
    # File references
    # ================================================================
    print()
    print("-" * 60)
    if hlo_lines:
        print(f"  HLO:         {hlo_path} ({hlo_lines} lines)")
    if trace_data:
        print(f"  Trace:       {trace_data['xplane_path']}")
    else:
        print(f"  Trace:       {trace_dir}")
    print(f"  Xprof:       xprof {trace_dir}")
    print()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "flops": flops,
        "bytes_accessed": bytes_accessed,
        "trace_dir": trace_dir,
        "hlo_path": hlo_path,
    }
