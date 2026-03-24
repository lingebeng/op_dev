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


def _parse_tensor_core(tc_line, module_events) -> dict | None:
    """Parse Tensor Core line for LLO bundle-level timing.

    Bundle events are zero-duration markers. Timing is computed from
    inter-event gaps (next event's device_offset_ps - current event's).
    """
    tc_events_raw = list(tc_line.events)
    if not tc_events_raw or not module_events:
        return None

    # Build invocation time ranges from XLA Modules events
    inv_ranges = []
    for ev in module_events:
        ev_stats = {}
        try:
            for sk, sv in ev.stats:
                ev_stats[sk] = sv
        except Exception:
            pass
        start_ps = ev_stats.get("device_offset_ps", 0)
        dur_ps = ev_stats.get("device_duration_ps", 0)
        if start_ps and dur_ps:
            inv_ranges.append((float(start_ps), float(start_ps) + float(dur_ps)))

    if not inv_ranges:
        return None

    # Collect TC events with offset and name
    tc_events = []
    for ev in tc_events_raw:
        ev_stats = {}
        try:
            for sk, sv in ev.stats:
                ev_stats[sk] = sv
        except Exception:
            pass
        offset_ps = float(ev_stats.get("device_offset_ps", 0))
        long_name = str(ev_stats.get("long_name", ev.name))
        tc_events.append((offset_ps, ev.name, long_name))

    tc_events.sort(key=lambda x: x[0])

    # Segment TC events into invocations and compute inter-event timing
    n_inv = len(inv_ranges)
    bundle_times: dict[str, list[float]] = defaultdict(list)  # name -> [duration_ps]
    bundle_long_names: dict[str, str] = {}

    inv_idx = 0
    inv_events = []

    def _flush_inv_events(events, inv_end_ps):
        for i in range(len(events)):
            offset, name, long_name = events[i]
            if i + 1 < len(events):
                dur_ps = events[i + 1][0] - offset
            else:
                dur_ps = inv_end_ps - offset
            if dur_ps > 0:
                bundle_times[name].append(dur_ps)
            bundle_long_names[name] = long_name

    for offset_ps, name, long_name in tc_events:
        # Find which invocation this event belongs to
        while inv_idx < n_inv and offset_ps > inv_ranges[inv_idx][1]:
            if inv_events:
                _flush_inv_events(inv_events, inv_ranges[inv_idx][1])
                inv_events = []
            inv_idx += 1
        if inv_idx >= n_inv:
            break
        if offset_ps >= inv_ranges[inv_idx][0]:
            inv_events.append((offset_ps, name, long_name))

    # Flush last invocation
    if inv_events and inv_idx < n_inv:
        _flush_inv_events(inv_events, inv_ranges[inv_idx][1])

    if not bundle_times:
        return None

    # Aggregate per-bundle stats
    bundles = []
    total_us = 0
    for name, times_ps in bundle_times.items():
        total_ps = sum(times_ps)
        total_us += total_ps / 1e6
        mean_per_inv_us = total_ps / n_inv / 1e6
        count_per_inv = len(times_ps) / n_inv

        # Parse hex address from long_name: "[ID]  0xABC: bundle.ID"
        hex_addr = ""
        ln = bundle_long_names.get(name, "")
        if "0x" in ln:
            part = ln.split("0x", 1)[1]
            hex_addr = "0x" + part.split(":")[0].split()[0]

        bundles.append({
            "name": name,
            "hex_addr": hex_addr,
            "total_us": total_ps / 1e6,
            "mean_us_per_inv": mean_per_inv_us,
            "count_per_inv": count_per_inv,
        })

    # Sort by total time descending, compute percentages
    bundles.sort(key=lambda b: b["total_us"], reverse=True)
    total_us_per_inv = total_us / n_inv if n_inv > 0 else total_us
    for b in bundles:
        b["pct"] = b["mean_us_per_inv"] / total_us_per_inv * 100 if total_us_per_inv > 0 else 0

    events_per_inv = len(tc_events_raw) / n_inv if n_inv > 0 else len(tc_events_raw)

    return {
        "bundles": bundles,
        "n_bundles": len(bundles),
        "events_per_inv": int(events_per_inv),
        "total_bundle_us_per_inv": total_us_per_inv,
    }


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
    tc_line = None
    for line in plane.lines:
        if line.name == "XLA Modules":
            modules_line = line
        elif line.name == "XLA Ops":
            ops_line = line
        elif line.name == "Tensor Core":
            tc_line = line

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
    # Tuples: (short_name, op_type, time_us, flops, bytes_accessed)
    kernel_ops = []
    overhead_ops = []
    xla_ops_flops = 0
    xla_ops_bytes = 0
    if ops_line is not None:
        op_times: dict[str, tuple[str, float, float, float]] = {}
        for ev in list(ops_line.events):
            dur = ev.duration_ns / 1000.0
            # Extract per-event stats (available with LLO flags)
            ev_flops = 0
            ev_bytes = 0
            try:
                for sk, sv in ev.stats:
                    if sk == "flops":
                        ev_flops = float(sv)
                    elif sk == "bytes_accessed":
                        ev_bytes = float(sv)
            except Exception:
                pass
            if ev.name in op_times:
                ot, prev_dur, prev_f, prev_b = op_times[ev.name]
                op_times[ev.name] = (ot, prev_dur + dur,
                                     prev_f + ev_flops, prev_b + ev_bytes)
            else:
                _, ot = _parse_op_type(ev.name)
                op_times[ev.name] = (ot, dur, ev_flops, ev_bytes)

        for full_name, (op_type, time_us, op_f, op_b) in op_times.items():
            short, _ = _parse_op_type(full_name)
            if op_type in _OVERHEAD_OPS:
                overhead_ops.append((short, op_type, time_us, op_f, op_b))
            else:
                kernel_ops.append((short, op_type, time_us, op_f, op_b))
            xla_ops_flops += op_f
            xla_ops_bytes += op_b

    result["kernel_ops"] = kernel_ops
    result["overhead_ops"] = overhead_ops
    result["kernel_us"] = sum(t for _, _, t, _, _ in kernel_ops)
    result["overhead_us"] = sum(t for _, _, t, _, _ in overhead_ops)
    result["xla_ops_flops"] = xla_ops_flops
    result["xla_ops_bytes"] = xla_ops_bytes

    # --- Tensor Core bundle analysis (with LLO flags) ---
    if tc_line is not None and modules_line is not None:
        tc_data = _parse_tensor_core(tc_line, list(modules_line.events))
        if tc_data:
            result["tensor_core"] = tc_data

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

    # Override cost_analysis flops/bytes with xplane values when available
    # (xplane per-op stats are more accurate for custom-call ops)
    if trace_data:
        n_inv = trace_data.get("n_invocations", 1)
        xplane_flops = trace_data.get("xla_ops_flops", 0)
        xplane_bytes = trace_data.get("xla_ops_bytes", 0)
        if xplane_flops > 0 and flops == 0:
            flops = int(xplane_flops / n_inv)
        if xplane_bytes > 0 and bytes_accessed == 0:
            bytes_accessed = int(xplane_bytes / n_inv)

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
            for short, op_type, t_total, op_f, op_b in kernel_ops:
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
                extra_info = ""
                if op_f > 0:
                    gf = op_f / divisor / 1e9
                    extra_info += f" {gf:.1f}GF"
                if op_b > 0:
                    mb = op_b / divisor / 1e6
                    extra_info += f" {mb:.1f}MB"
                print(f"    {op_label:<50s} {t:>8.2f} us  {pct:>5.1f}%{extra_info}")

            # Summarize XLA overhead as one line
            overhead_us = trace_data["overhead_us"] / divisor
            if overhead_us > 0:
                overhead_pct = overhead_us / total_ops_us * 100
                overhead_detail = ", ".join(
                    f"{op}={t / divisor:.1f}us" for _, op, t, _, _ in
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
                overhead_types = set(op for _, op, _, _, _ in trace_data["overhead_ops"])
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
    # Layer 4: LLO Bundle Analysis (when LLO flags are enabled)
    # ================================================================
    if trace_data and trace_data.get("tensor_core"):
        tc = trace_data["tensor_core"]
        print()
        print("=" * 60)
        print("Layer 4: LLO Bundle Analysis")
        print("=" * 60)
        print(f"  Bundles: {tc['n_bundles']}  |  "
              f"Events/invocation: {tc['events_per_inv']}  |  "
              f"Total: {tc['total_bundle_us_per_inv']:.1f} us/invocation")
        print()
        print(f"    {'Bundle':<28s} {'Addr':<8s} "
              f"{'Time (us)':>10s} {'Count':>6s} {'%':>6s}")
        print(f"    {'-'*28} {'-'*8} {'-'*10} {'-'*6} {'-'*6}")
        for b in tc["bundles"]:
            print(f"    {b['name']:<28s} {b['hex_addr']:<8s} "
                  f"{b['mean_us_per_inv']:>10.2f} "
                  f"{b['count_per_inv']:>6.0f} "
                  f"{b['pct']:>5.1f}%")

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
