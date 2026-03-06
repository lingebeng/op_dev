import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax import random
import jax.numpy as jnp
import numpy as np
import functools
from typing import Callable


def matmul_kernel(x_ref, y_ref, z_ref, acc_ref, *, nsteps, transpose_rhs, activation):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    if transpose_rhs:
        dims = ((1,), (1,)), ((), ())
    else:
        dims = ((1,), (0,)), ((), ())

    acc_ref[...] += jax.lax.dot_general(
        x_ref[...],
        y_ref[...],
        dims,
        preferred_element_type=jnp.float32,
    )

    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        z_ref[...] = activation(acc_ref[...]).astype(z_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=["bm", "bk", "bn", "transpose_rhs", "activation"]
)
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
    transpose_rhs: bool = False,
    activation: Callable[[jax.Array], jax.Array] = lambda x: x,
):
    if transpose_rhs:
        y = y.swapaxes(0, 1)
        y_block_spec = pl.BlockSpec((bn, bk), lambda i, j, k: (j, k))
    else:
        y_block_spec = pl.BlockSpec((bk, bn), lambda i, j, k: (k, j))
    m, k = x.shape
    n, _ = y.shape
    return pl.pallas_call(
        functools.partial(
            matmul_kernel,
            nsteps=k // bk,
            transpose_rhs=transpose_rhs,
            activation=activation,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                y_block_spec,
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
            grid=(m // bm, n // bn, k // bk),
        ),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
    )(x, y)


if __name__ == "__main__":
    m, k, n = 4096, 8192, 4096
    k1, k2 = random.split(random.key(0), 2)
    x = random.normal(k1, (m, k), dtype=jnp.bfloat16)
    y = random.normal(k2, (n, k), dtype=jnp.bfloat16)
    y = y.T
    activation = jax.nn.relu
    np.testing.assert_array_equal(
        activation(x @ y), matmul(x, y, transpose_rhs=True, activation=activation)
    )
