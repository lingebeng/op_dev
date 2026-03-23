---
name: profile
description: Profile a Pallas kernel or JAX operator — captures timing, FLOPs, HLO, and jax.profiler trace
argument-hint: <file_path> [function_name] [B,H,T,S]
allowed-tools: Read, Grep, Glob, Bash, Write, Edit
---

Profile a JAX/Pallas operator. Target: $ARGUMENTS

## Workflow

### 1. Locate the target

- Read the file specified in the first argument.
- If a function name is given (second argument), profile that function.
- If no function name is given, look for the primary Pallas `pallas_call` wrapper or the main callable in the file.

### 2. Understand the operator signature

- Identify the function's parameters: input shapes, dtypes, and any static/keyword args (like `chunk_size`, `head_first`, `method`, `reverse`, `axis`, etc.).
- If a shape hint is given (third argument, e.g. `2,4,1024,128`), use it. Otherwise infer typical shapes from the file's `__main__` block or test code.

### 3. Write a driver script

Create `/tmp/profile_driver.py` that:

```python
import sys
sys.path.insert(0, "<project_root>")
sys.path.insert(0, "${CLAUDE_SKILL_DIR}/scripts")

from profiler import profile
# Import the target function
from <module> import <function>

import jax
import jax.numpy as jnp

# Create test inputs (adapt to the actual signature)
key = jax.random.PRNGKey(42)
x = jax.random.normal(key, (<shapes>), dtype=jnp.float32)

# Wrap with any required static args using lambda/functools.partial
fn = lambda x: <function>(x, <static_args>)

profile(fn, x, fn_name="<function_name>")
```

Key points:
- Use `sys.path.insert` to make both the project root and the profiler script importable.
- Use `functools.partial` or a lambda to freeze static arguments so the profiled function only takes array inputs.
- Match the exact calling convention of the target function.

### 4. Run the driver

Always use `uv run` to execute the driver script, so that the project's virtual environment (with TPU-enabled JAX) is used:

```bash
uv run python /tmp/profile_driver.py
```

**Important**: Do NOT use bare `python` — it may resolve to the system Python which lacks JAX or has a CPU-only version.

### 5. Analyze the output

Read the stdout and present a concise summary to the user:
- **Timing**: mean +/- std in ms
- **FLOPs / GFLOP/s** (if available from cost_analysis; if not reported, just show "None" — do NOT speculate about the reason)
- **Arithmetic intensity** (FLOPs/byte)
- **Memory** usage breakdown
- **Key HLO observations**: dominant ops (dot_general, reduce, etc.), fusion decisions
- **Trace analysis** is automatic — the profiler parses `.xplane.pb` and prints per-op device timing, hardware specs, and device utilization directly. No need for TensorBoard for basic analysis.
- If the user wants interactive visualization, suggest: `xprof /tmp/jax_profile`

### 6. Optional: compare multiple configs

If the user specifies multiple shapes or parameter variants, run the profiler for each and present a comparison table.

### 7. Language

Always present the analysis summary to the user in **Chinese (中文)**.
