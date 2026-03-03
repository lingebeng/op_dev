from functools import partial

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import numpy as np


def add_vectors_kernel(x_ref, y_ref, o_ref):
    x, y = x_ref[...], y_ref[...]
    o_ref[...] = x + y


def add_sliced_kernel(x_ref, y_ref, o_ref):
    x = x_ref[...]
    y = y_ref[...]
    mid = x.shape[0] // 2

    x_l = x[:mid]
    x_r = x[mid:]
    y_l = y[:mid]
    y_r = y[mid:]

    # [x_l+y_l, x_l+y_r, x_r+y_l, x_r+y_r], total length = 4 * mid = 2 * n.
    o_ref[0:mid] = x_l + y_l
    o_ref[mid : 2 * mid] = x_l + y_r
    o_ref[2 * mid : 3 * mid] = x_r + y_l
    o_ref[3 * mid : 4 * mid] = x_r + y_r


@jax.jit
def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
    n = x.shape[0]
    return pl.pallas_call(
        # add_vectors_kernel,
        add_sliced_kernel,
        out_shape=jax.ShapeDtypeStruct((2 * n,), x.dtype),
        interpret=True,
    )(x, y)


output = add_vectors(jnp.arange(8), jnp.arange(8))


print(output)
