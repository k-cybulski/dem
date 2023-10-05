import jax.numpy as jnp
import numpy as np
import pytest
from dem.algo.jax.util import hessian_low_memory, jacfwd_low_memory
from jax import hessian as hessian_builtin
from jax import jacfwd as jacfwd_builtin


@pytest.fixture
def test_vec():
    rng = np.random.default_rng(1521)
    return rng.standard_normal(50)


def test_jacfwd_lm(test_vec):
    f = jnp.exp
    W = test_vec
    batch_size = 3
    assert jnp.allclose(
        jacfwd_builtin(f)(W), jacfwd_low_memory(f, batch_size=batch_size)(W)
    )


def test_hessian_lm(test_vec):
    f = jnp.linalg.norm
    W = test_vec
    batch_size = 3
    assert jnp.allclose(
        hessian_builtin(f)(W), hessian_low_memory(f, batch_size=batch_size)(W)
    )
