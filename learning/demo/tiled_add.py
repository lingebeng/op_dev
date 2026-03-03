import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp


def add_vectors_kernel(x_ref, y_ref, o_ref):
    # 显式读取和写入，增加稳定性
    x = x_ref[...]
    y = y_ref[...]
    o_ref[...] = x + y


def pallas_add_tiled(x, y):
    n = x.shape[0]
    b = 2  # block_size
    grid_size = n // b

    return pl.pallas_call(
        add_vectors_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(grid_size,),
        in_specs=[
            # 显式指定参数名 index_map 和 block_shape
            pl.BlockSpec(index_map=lambda i: (i * 2,), block_shape=(2,)),
            pl.BlockSpec(index_map=lambda i: (i,), block_shape=(2,)),
        ],
        out_specs=pl.BlockSpec(index_map=lambda i: (i * 2,), block_shape=(2,)),
        interpret=True,
    )(x, y)


x = jnp.arange(8, dtype=jnp.float32)
y = jnp.arange(8, dtype=jnp.float32)
output = pallas_add_tiled(x, y)

print(f"Output: {output}")
# Output: [ 0.  2.  4.  6.  8. 10. 12. 14.]
