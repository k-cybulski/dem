from functools import partial
from math import ceil

import jax.numpy as jnp
# DEM experiments often involve extremely high priors, which do not work well
# with single precision float32
from jax import jacrev, jit, jvp, vjp, vmap
from jax.lax import fori_loop


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


# sometimes computing hessians takes incredible amounts of memory
# similar to here https://github.com/google/jax/issues/787#issuecomment-497146084
# the functions below move away from vectorizing the hessian computation in
# favour of serializing it, so that it takes less memory but takes longer


# the implementations below are variations on an example from the docs:
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#the-implementation-of-jacfwd-and-jacrev
def jacrev_low_memory(f, batch_size):
    """
    A low-memory but slower version of jacrev. In contrast to standard jacrev,
    it computes the jacobian in batches.
    """

    # this is slow, but takes less memory than the default method by not
    # computing *everything* in parallel
    def jacfun(x):
        # y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
        y, vjp_fun = vjp(f, x)
        basis = jnp.eye(x.size, dtype=x.dtype)
        Jr = []
        for i in range(ceil(x.size / batch_size)):
            batch = basis[(i * batch_size) : ((i + 1) * batch_size)]
            (Jb,) = vmap(vjp_fun, in_axes=0)(batch)
            Jr.append(Jb)
        J = jnp.concatenate(Jr)
        return J

    return jacfun


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


def jacfwd_low_memory_jit(f):
    # The batching mechanism used in low_memory functions above relies upon
    # dynamically sized slicing of jnp.eye, but JAX does not support dynamic
    # slices for jit compiled functions. Also it's not possible to call jnp.eye
    # with a dynamic integer input either.
    def jacfun(x):
        def _jvp(s):
            return jvp(f, (x,), (s,))[1]

        basis = jnp.eye(x.size, dtype=x.dtype)

        def body_loop(i, J):
            basis_vec = basis[i]
            Jtmp = _jvp(basis_vec)
            return J.at[i].set(Jtmp)

        Jinit = jnp.empty((x.size, x.size))
        Jt = fori_loop(0, x.size, body_loop, Jinit)
        return jnp.transpose(Jt)

    return jacfun


def hessian_low_memory(f, outer=partial(jacfwd_low_memory, batch_size=3)):
    return outer(jacrev(f))


def hessian_low_memory_jit(f):
    # just a special case of hessian_low_memory to ensure JIT-compatibility
    return jacfwd_low_memory_jit(jacrev(f))
