from functools import partial

import jax.numpy as jnp

# DEM experiments often involve extremely high priors, which do not work well
# with single precision float32
from jax import jit


@jit
def _fix_grad_shape(tensor):
    ndim = tensor.ndim
    if ndim == 6:
        batch_n = tensor.shape[0]
        batch_selection = jnp.arange(batch_n)
        # NOTE: The tensor includes all cross-batch derivatives too, which are
        # always zero hopefully this doesn't lead to unnecessary
        # computations...
        return tensor[batch_selection, :, 0, batch_selection, :, 0]
    elif ndim == 4:
        return tensor.reshape((tensor.shape[0], tensor.shape[2]))
    elif ndim == 2:
        return tensor
    else:
        raise ValueError(f"Unexpected shape: {tuple(tensor.shape)}")


@jit
def logdet(matr):
    """
    Returns log determinant of a matrix, assuming that the determinant is
    positive.
    """
    return jnp.linalg.slogdet(matr)[1]


@partial(
    jit,
    static_argnames=(
        "p",
        "n",
    ),
)
def deriv_mat(p, n):
    """
    Block derivative operator.

    Args:
        p: number of derivatives
        n: number of terms
    """
    return jnp.kron(jnp.eye(p + 1, k=1), jnp.eye(n))
