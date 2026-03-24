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
    n_chunks = -(-lens // chunk_size)  # ceil division
    total_nt = int(jnp.sum(n_chunks))
    num_seqs = len(lens)

    seq_ids = jnp.repeat(
        jnp.arange(num_seqs, dtype=jnp.int32), n_chunks, total_repeat_length=total_nt
    )
    prefix_chunks = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(n_chunks)]
    )
    seq_offsets = jnp.repeat(prefix_chunks[:-1], n_chunks, total_repeat_length=total_nt)
    block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets

    return jnp.stack([seq_ids, block_ids], axis=1)


# ============================================================
# Fixed-length mode: jnp.cumsum (optimal for head_first=False)
# ============================================================
def _chunk_local_cumsum_cumsum(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
    BT = chunk_size
    out_dtype = output_dtype or g.dtype

    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape

    NT = (T + BT - 1) // BT
    T_padded = NT * BT
    pad_t = T_padded - T

    if head_first:
        g_work = jnp.pad(g, ((0, 0), (0, 0), (0, pad_t), (0, 0))) if pad_t > 0 else g
        g_chunked = g_work.reshape(B, H, NT, BT, S).astype(jnp.float32)
        cum_axis = 3
    else:
        g_work = jnp.pad(g, ((0, 0), (0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else g
        g_chunked = g_work.reshape(B, NT, BT, H, S).astype(jnp.float32)
        cum_axis = 2

    if reverse:
        o_chunked = jnp.flip(jnp.cumsum(jnp.flip(g_chunked, axis=cum_axis), axis=cum_axis), axis=cum_axis)
    else:
        o_chunked = jnp.cumsum(g_chunked, axis=cum_axis)

    if head_first:
        o = o_chunked.reshape(B, H, T_padded, S)[:, :, :T, :]
    else:
        o = o_chunked.reshape(B, T_padded, H, S)[:, :T, :, :]

    if scale is not None:
        o = o * scale

    return o.astype(out_dtype)


# ============================================================
# Fixed-length mode: native JAX matmul (tril/triu mask)
# ============================================================
def _chunk_local_cumsum_matmul(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
) -> jax.Array:
    BT = chunk_size
    out_dtype = output_dtype or g.dtype

    if head_first:
        B, H, T, S = g.shape
    else:
        B, T, H, S = g.shape

    # Pad T to multiple of BT if needed
    NT = (T + BT - 1) // BT
    T_padded = NT * BT
    pad_t = T_padded - T

    if head_first:
        g_work = jnp.pad(g, ((0, 0), (0, 0), (0, pad_t), (0, 0))) if pad_t > 0 else g
        g_chunked = g_work.reshape(B, H, NT, BT, S).astype(jnp.float32)
    else:
        g_work = jnp.pad(g, ((0, 0), (0, pad_t), (0, 0), (0, 0))) if pad_t > 0 else g
        g_chunked = g_work.reshape(B, NT, BT, H, S).astype(jnp.float32)

    # Lower-triangular => forward cumsum, upper-triangular => reverse cumsum
    if reverse:
        cum_mask = jnp.triu(jnp.ones((BT, BT), dtype=jnp.float32))
    else:
        cum_mask = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))

    if head_first:
        o_chunked = jnp.einsum('ij,bhnjs->bhnis', cum_mask, g_chunked,
                               precision=jax.lax.Precision.HIGHEST)
        o = o_chunked.reshape(B, H, T_padded, S)[:, :, :T, :]
    else:
        o_chunked = jnp.einsum('ij,bnjhs->bnihs', cum_mask, g_chunked,
                               precision=jax.lax.Precision.HIGHEST)
        o = o_chunked.reshape(B, T_padded, H, S)[:, :T, :, :]

    if scale is not None:
        o = o * scale

    return o.astype(out_dtype)


# ============================================================
# Variable-length mode: Pallas kernel with Hillis-Steele + BB tiling
# ============================================================
def _chunk_cumsum_kernel_varlen(
    cu_seqlens_ref,
    chunk_indices_ref,
    s_ref,
    o_ref,
    *,
    BT: int,
    BS: int,
    BB: int,
    REVERSE: bool,
    HAS_SCALE: bool,
    scale: float,
):
    i_s, i_t, i_bb = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    i_n, local_i_t = chunk_indices_ref[i_t, 0], chunk_indices_ref[i_t, 1]
    bos, eos = cu_seqlens_ref[i_n], cu_seqlens_ref[i_n + 1]
    start_t = bos + local_i_t * BT
    start_s = i_s * BS
    start_bh = i_bb * BB

    # Load (BB, BT, BS) tile
    s = s_ref[dslice(start_bh, BB), dslice(start_t, BT), dslice(start_s, BS)]

    # Varlen masking: zero out positions beyond the sequence boundary
    T_seq = eos - bos
    valid_len = T_seq - local_i_t * BT
    valid_mask = (jnp.arange(BT) < valid_len).astype(jnp.float32)[None, :, None]
    s = s.astype(jnp.float32) * valid_mask

    # Hillis-Steele parallel prefix sum (fully unrolled at trace time)
    rows = [s[:, i, :] for i in range(BT)]  # list of (BB, BS) arrays
    num_steps = int(math.log2(BT))

    if REVERSE:
        for d in range(num_steps):
            stride = 1 << d
            new_rows = []
            for i in range(BT):
                if i + stride < BT:
                    new_rows.append(rows[i] + rows[i + stride])
                else:
                    new_rows.append(rows[i])
            rows = new_rows
    else:
        for d in range(num_steps):
            stride = 1 << d
            new_rows = []
            for i in range(BT):
                if i >= stride:
                    new_rows.append(rows[i] + rows[i - stride])
                else:
                    new_rows.append(rows[i])
            rows = new_rows

    o = jnp.stack(rows, axis=1)  # (BB, BT, BS)

    if HAS_SCALE:
        o = o * scale

    o_ref[dslice(start_bh, BB), dslice(start_t, BT), dslice(start_s, BS)] = o.astype(o_ref.dtype)


def _chunk_local_cumsum_pallas(
    g: jax.Array,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    head_first: bool = False,
    output_dtype: jnp.dtype | None = jnp.float32,
    chunk_indices: jax.Array | None = None,
) -> jax.Array:
    BT = chunk_size
    BS = 128
    BB = 16

    if head_first:
        B, H, T, S = g.shape
        g_flat = g.reshape(B * H, T, S)
    else:
        B, T, H, S = g.shape
        g_flat = jnp.transpose(g, (0, 2, 1, 3)).reshape(B * H, T, S)

    BH = B * H
    out_dtype = output_dtype or g.dtype
    HAS_SCALE = scale is not None
    scale_val = scale if scale is not None else 1.0

    interpret = jax.default_backend() != "tpu"

    # Pad S dimension to multiple of BS
    pad_S = (BS - (S % BS)) % BS
    if pad_S > 0:
        g_flat = jnp.pad(g_flat, ((0, 0), (0, 0), (0, pad_S)))
    S_padded = S + pad_S
    NS = S_padded // BS

    # Pad BH dimension to multiple of BB
    pad_BH = (BB - (BH % BB)) % BB
    if pad_BH > 0:
        g_flat = jnp.pad(g_flat, ((0, pad_BH), (0, 0), (0, 0)))
    NBH = (BH + pad_BH) // BB

    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices)

    # Pad T for varlen dslice overflow
    g_flat = jnp.pad(g_flat, ((0, 0), (0, BT), (0, 0)))

    grid = (NS, NT, NBH)

    kernel = functools.partial(
        _chunk_cumsum_kernel_varlen,
        BT=BT,
        BS=BS,
        BB=BB,
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

    # Remove all padding
    o_flat = o_flat[:BH, :T, :S]

    if head_first:
        return o_flat.reshape(B, H, T, S)
    else:
        return jnp.transpose(o_flat.reshape(B, H, T, S), (0, 2, 1, 3))


# ============================================================
# Main API: dispatches between matmul (fixed-len) and Pallas (varlen)
# ============================================================
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

    if cu_seqlens is None:
        # Fixed-length mode: dispatch based on layout
        # head_first=True → matmul is faster (conv output layout is free)
        # head_first=False → cumsum is faster (reduce-window preserves layout)
        if head_first:
            return _chunk_local_cumsum_matmul(
                g, chunk_size, reverse, scale, head_first, output_dtype
            )
        else:
            return _chunk_local_cumsum_cumsum(
                g, chunk_size, reverse, scale, head_first, output_dtype
            )
    else:
        # Variable-length mode: Pallas kernel with Hillis-Steele
        return _chunk_local_cumsum_pallas(
            g, chunk_size, reverse, scale, cu_seqlens, head_first, output_dtype, chunk_indices
        )


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
    # --- Fixed-length mode test (matmul path) ---
    B, T, H, S = 2, 64, 4, 128
    BT = 16

    key = jax.random.PRNGKey(42)
    g = jax.random.normal(key, (B, T, H, S), dtype=jnp.float32) * 0.1

    o_fixed = chunk_local_cumsum_vector(g, BT, head_first=False)
    o_fixed_ref = chunk_local_cumsum_ref(g, BT, head_first=False)
    diff_fixed = jnp.abs(o_fixed - o_fixed_ref).max()
    print(f"Fixed-length forward max diff: {diff_fixed:.2e}")

    o_fixed_rev = chunk_local_cumsum_vector(g, BT, reverse=True, head_first=False)
    o_fixed_rev_ref = chunk_local_cumsum_ref(g, BT, reverse=True, head_first=False)
    diff_fixed_rev = jnp.abs(o_fixed_rev - o_fixed_rev_ref).max()
    print(f"Fixed-length reverse max diff: {diff_fixed_rev:.2e}")

    # --- Variable-length mode test (Pallas + Hillis-Steele path) ---
    T_total = 64
    cu_seqlens = jnp.array([0, 48, 64], dtype=jnp.int32)
    g = jax.random.normal(key, (1, T_total, H, S), dtype=jnp.float32) * 0.1

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    print(f"\nchunk_indices:\n{chunk_indices}")

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
