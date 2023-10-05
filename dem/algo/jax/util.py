from functools import partial

import jax.numpy as jnp
from math import ceil

# DEM experiments often involve extremely high priors, which do not work well
# with single precision float32
from jax import jacrev, jit, jvp, vjp, vmap


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


# shouldn't be JIT-compiled unfortunately due to the for loop
# the batching mechanism can't be nicely implemented in a JIT-compilable manner
# see commit 34e4ce767a7dce168e1dcf87d01577f31bc24354 for some alternative
# implementations
def jacfwd_low_memory(f, batch_size):
    def jacfun(x):
        def _jvp(s):
            return jvp(f, (x,), (s,))[1]

        basis = jnp.eye(x.size, dtype=x.dtype)
        Jtr = []
        for i in range(ceil(x.size / batch_size)):
            batch = basis[(i * batch_size) : ((i + 1) * batch_size)]
            Jtb = vmap(_jvp)(batch)
            Jtr.append(Jtb)
        Jt = jnp.concatenate(Jtr)
        return jnp.transpose(Jt)

    return jacfun


def hessian_low_memory(f, batch_size):
    return jacfwd_low_memory(jacrev(f), batch_size=batch_size)
