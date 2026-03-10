import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import dslice
from jax.experimental.pallas import tpu as pltpu


def prepare_lens(cu_seqlens: jax.Array) -> jax.Array:
    """
    Compute the actual length of each sequence.
    [0, 48, 64] -> [48, 16]
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
) -> jax.Array:
    """Build a mapping from physical chunks to logical sequences for varlen inputs."""
    lens = prepare_lens(cu_seqlens)
    lens_list = [int(x) for x in lens]

    seq_ids = []
    block_ids = []
    for i, length in enumerate(lens_list):
        nt = math.ceil(length / chunk_size)
        seq_ids.extend([i] * nt)
        block_ids.extend(range(nt))

    return jnp.array(
        list(zip(seq_ids, block_ids)),
        dtype=jnp.int32,
    )


def chunk_cumsum_kernel(
    cu_seqlens_ref,
    chunk_indices_ref,
    s_ref,
    o_ref,
    *,
    BT: int,
    BS: int,
    REVERSE: bool,
    HAS_SCALE: bool,
    scale: float,
):
    i_s, i_t, i_bh = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    i_n, local_i_t = chunk_indices_ref[i_t, 0], chunk_indices_ref[i_t, 1]

    bos, eos = cu_seqlens_ref[i_n], cu_seqlens_ref[i_n + 1]
    T_seq = eos - bos

    start_t, start_s = bos + local_i_t * BT, i_s * BS

    # Each program handles one (BT, BS) tile.
    s = s_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)]

    # Mask out-of-bounds elements.
    valid_len = T_seq - local_i_t * BT
    valid_mask = (jnp.arange(BT) < valid_len).astype(jnp.float32)[:, None]
    s = s.astype(jnp.float32) * valid_mask

    if REVERSE:
        o = jnp.cumsum(s[::-1], axis=0)[::-1]
    else:
        # Forward mode: directly use the underlying scan.
        o = jnp.cumsum(s, axis=0)

    if HAS_SCALE:
        o = o * scale

    o = (o * valid_mask).astype(o_ref.dtype)
    o_ref[i_bh, dslice(start_t, BT), dslice(start_s, BS)] = o


def chunk_local_cumsum_vector(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:

    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), (
        "chunk_size must be power of 2"
    )

    if head_first:
        B, H, T, S = g.shape
        # Normalize to (B*H, T, S) to greatly simplify pointer offsets in the kernel.
        g_flat = g.reshape(B * H, T, S)
    else:
        B, T, H, S = g.shape
        g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

    BT = chunk_size
    BS = min(128, S)
    out_dtype = output_dtype or g.dtype
    HAS_SCALE = scale is not None
    scale_val = scale if scale is not None else 1.0

    interpret = jax.local_devices()[0].platform != "tpu"

    # Pad the S dimension to satisfy TPU shape constraints.
    pad_S = (BS - (S % BS)) % BS
    if pad_S > 0:
        g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))

    S_padded = S + pad_S
    NS = S_padded // BS

    # For fixed-length inputs, synthesize cu_seqlens/chunk_indices to simplify kernel control flow.
    if chunk_indices is not None or cu_seqlens is None:
        cu_seqlens = jnp.arange(0, B * T + 1, BT, dtype=jnp.int32)

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)
    grid = (NS, NT, B * H)

    # In varlen mode, append BT padding at the end to prevent dslice overflow.
    g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))

    kernel = functools.partial(
        chunk_cumsum_kernel,
        BT=BT,
        BS=BS,
        REVERSE=reverse,
        HAS_SCALE=HAS_SCALE,
        scale=scale_val,
    )

    o_flat = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=pl.no_block_spec,
            out_specs=pl.no_block_spec,
        ),
        out_shape=jax.ShapeDtypeStruct(g_flat.shape, out_dtype),
        interpret=interpret,
    )(cu_seqlens, chunk_indices, g_flat)

    # Remove the padding added earlier.
    o_flat = o_flat[:, :T, :S]

    # Convert the normalized layout back to the user-facing layout.
    if head_first:
        return o_flat.reshape(B, H, T, S)
    else:
        return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# ============================================================
# Reference implementation (pure JAX, used for numerical alignment tests)
# ============================================================
def chunk_local_cumsum_ref(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    head_first: bool = False,
) -> jax.Array:
    if head_first:
        B, H, T, S = g.shape
        g_chunked = g.reshape(B, H, T // chunk_size, chunk_size, S)
        if reverse:
            o_chunked = jnp.cumsum(g_chunked[:, :, :, ::-1], axis=3)[:, :, :, ::-1]
        else:
            o_chunked = jnp.cumsum(g_chunked, axis=3)
        return o_chunked.reshape(B, H, T, S)
    else:
        B, T, H, S = g.shape
        g_chunked = g.reshape(B, T // chunk_size, chunk_size, H, S)
        if reverse:
            o_chunked = jnp.cumsum(g_chunked[:, :, ::-1], axis=2)[:, :, ::-1]
        else:
            o_chunked = jnp.cumsum(g_chunked, axis=2)
        return o_chunked.reshape(B, T, H, S)


# ============================================================
# Test module
# ============================================================
if __name__ == "__main__":
    # --- Fixed-length mode test (head_first=False) ---
    B, T, H, S = 2, 64, 4, 128
    BT = 16

    key = jax.random.PRNGKey(42)
    g = jax.random.normal(key, (B, T, H, S), dtype=jnp.float32) * 0.1

    # --- Variable-length mode test ---

    T_total = 64
    cu_seqlens = jnp.array([0, 48, 64], dtype=jnp.int32)
    g = jax.random.normal(key, (1, T_total, H, S), dtype=jnp.float32) * 0.1

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    print(f"chunk_indices:\n{chunk_indices}")

    o_varlen = chunk_local_cumsum_vector(
        g,
        BT,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        head_first=False,
    )

    # Validate the first sequence (0:48).
    g_seq0 = g[0, :48]
    g_seq0_padded = jnp.pad(g_seq0, ((0, T_total - 48), (0, 0), (0, 0)))[None]
    o_seq0_ref = chunk_local_cumsum_ref(g_seq0_padded, BT, head_first=False)
    diff_v0 = jnp.abs(o_varlen[0, :48] - o_seq0_ref[0, :48]).max()
    print(f"Varlen seq0 max diff: {diff_v0:.2e}")

    # Validate the second sequence (48:64).
    g_seq1 = g[0, 48:64]
    g_seq1_padded = jnp.pad(g_seq1, ((0, T_total - 16), (0, 0), (0, 0)))[None]
    o_seq1_ref = chunk_local_cumsum_ref(g_seq1_padded, BT, head_first=False)
    diff_v1 = jnp.abs(o_varlen[0, 48:64] - o_seq1_ref[0, :16]).max()
    print(f"Varlen seq1 max diff: {diff_v1:.2e}")

    print("\nAll tests completed.")
