import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def reduce_sum_kernel(x_ref, o_ref):
    @pl.when(pl.program_id(2) == 0)
    def _():
        o_ref[...] = jnp.zeros_like(o_ref)

    o_ref[...] += x_ref[...]


def reduce_sum(x: jax.Array, block_size: tuple[int, ...] = (256, 256)) -> jax.Array:
    reduction_size, *out_shape = x.shape
    # We moved the reduction to the last axis of the grid.
    grid = (*(out // blk for out, blk in zip(out_shape, block_size)), reduction_size)
    interpret = jax.default_backend() == "cpu"
    return pl.pallas_call(
        reduce_sum_kernel,
        grid=grid,
        # None in `block_shape` means we pick a size of 1 and squeeze it away
        in_specs=[pl.BlockSpec((None, *block_size), lambda i, j, k: (k, i, j))],
        out_specs=pl.BlockSpec(block_size, lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct(out_shape, x.dtype),
        interpret=interpret,
    )(x)


# 测试代码
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    x = jnp.ones((8, 1024, 1024))

    out_pallas = reduce_sum(x)
    out_jax = jnp.sum(x, axis=0)

    print("Pallas 输出形状:", out_pallas.shape)
    print("最大误差:", jnp.max(jnp.abs(out_pallas - out_jax)))
    print("结果是否一致?", jnp.allclose(out_pallas, out_jax, atol=1e-6))
    print(out_pallas)
