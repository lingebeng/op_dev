import os
import sys
import timeit
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu

# Add project root for profiler import
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_this_dir, "../../.."))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, ".claude/skills/profile/scripts"))
from profiler import _parse_xplane


# ============================================================
# Kernel & wrapper
# ============================================================
def dslice_only_kernel_batched(
    s_ref, o_ref,
    *, BT: int, BS: int, BB: int,
):
    i_s, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)
    start_t = i_t * BT
    start_s = i_s * BS
    start_bh = i_bh * BB

    s = s_ref[dslice(start_bh, BB), dslice(start_t, BT), dslice(start_s, BS)]
    o_ref[dslice(start_bh, BB), dslice(start_t, BT), dslice(start_s, BS)] = s


def dslice_only_batched(g, *, BT, BS, BB):
    B, H, T, S = g.shape
    BH = B * H
    g_flat = g.reshape(BH, T, S)

    NT = T // BT
    NS = S // BS
    NBH = BH // BB

    grid = (NS, NT, NBH)
    kernel = functools.partial(dslice_only_kernel_batched, BT=BT, BS=BS, BB=BB)

    o_flat = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=pl.no_block_spec,
            out_specs=pl.no_block_spec,
        ),
        out_shape=jax.ShapeDtypeStruct(g_flat.shape, jnp.float32),
        interpret=jax.default_backend() != "tpu",
    )(g_flat)

    return o_flat.reshape(B, H, T, S)


# ============================================================
# Benchmark
# ============================================================
if __name__ == "__main__":
    B, H, T, S = 2, 8, 1024, 128
    BT = 64
    BS = 128
    BH = B * H  # 16

    key = jax.random.PRNGKey(42)
    g = jax.random.normal(key, (B, H, T, S), dtype=jnp.float32)

    bb_candidates = [bb for bb in [1, 2, 4, 8, 16] if BH % bb == 0]

    warmup = 20
    repeat = 200

    results = []

    for bb in bb_candidates:
        fn = jax.jit(functools.partial(dslice_only_batched, BT=BT, BS=BS, BB=bb))

        # Correctness
        out = fn(g)
        diff = jnp.abs(out - g).max()

        # Warmup
        for _ in range(warmup):
            fn(g).block_until_ready()

        # Trace: get device kernel time via xplane
        trace_dir = f"/tmp/jax_profile_bb{bb}"
        os.makedirs(trace_dir, exist_ok=True)
        device_us = None
        try:
            with jax.profiler.trace(trace_dir):
                for _ in range(5):
                    fn(g).block_until_ready()
            trace_data = _parse_xplane(trace_dir)
            if trace_data:
                device_us = trace_data.get("mean_kernel_us")
        except Exception:
            pass

        # Wall-clock
        times = []
        for _ in range(repeat):
            t0 = timeit.default_timer()
            fn(g).block_until_ready()
            times.append((timeit.default_timer() - t0) * 1000)
        times_ms = np.array(times)

        grid_size = (S // BS) * (T // BT) * (BH // bb)
        block_bytes = bb * BT * BS * 4  # float32

        results.append({
            "BB": bb,
            "grid": f"({S // BS},{T // BT},{BH // bb})",
            "programs": grid_size,
            "block_KB": block_bytes / 1024,
            "device_us": device_us,
            "wall_mean_ms": float(np.mean(times_ms)),
            "wall_std_ms": float(np.std(times_ms)),
            "diff": float(diff),
        })

    # Print results
    print(f"\nConfig: (B,H,T,S)=({B},{H},{T},{S}), BT={BT}, BS={BS}, B*H={BH}")
    print(f"Warmup={warmup}, Repeat={repeat}\n")

    header = f"{'BB':>4s} | {'grid':>12s} | {'programs':>8s} | {'block_KB':>9s} | {'device(us)':>10s} | {'wall(ms)':>16s} | {'diff':>8s}"
    print(header)
    print("-" * len(header))

    best_bb = None
    best_us = float("inf")

    for r in results:
        dev_str = f"{r['device_us']:.2f}" if r["device_us"] else "N/A"
        wall_str = f"{r['wall_mean_ms']:.4f} ± {r['wall_std_ms']:.4f}"
        print(
            f"{r['BB']:>4d} | {r['grid']:>12s} | {r['programs']:>8d} | "
            f"{r['block_KB']:>8.1f}K | {dev_str:>10s} | {wall_str:>16s} | {r['diff']:.1e}"
        )

        if r["device_us"] and r["device_us"] < best_us:
            best_us = r["device_us"]
            best_bb = r["BB"]

    print()
    if best_bb:
        print(f">>> Best BB = {best_bb}  (device kernel = {best_us:.2f} us)")
