import pytest
import numpy as np
import jax.numpy as jnp
from jax import jacrev as jacrev_builtin
from jax import jacfwd as jacfwd_builtin
from jax import hessian as hessian_builtin
from hdm.dem.jax import jacrev_low_memory, jacfwd_low_memory, hessian_low_memory, jacfwd_low_memory_jittable

@pytest.fixture
def test_vec():
    rng = np.random.default_rng(1521)
    return rng.standard_normal(50)

def test_jacrev_lm(test_vec):
    f = jnp.exp
    W = test_vec
    batch_size = 3
    assert jnp.allclose(jacrev_builtin(f)(W), jacrev_low_memory(f, batch_size)(W))

def test_jacfwd_lm(test_vec):
    f = jnp.exp
    W = test_vec
    batch_size = 3
    assert jnp.allclose(jacfwd_builtin(f)(W), jacfwd_low_memory(f, batch_size)(W))

def test_jacfwd_lm_j(test_vec):
    f = jnp.exp
    W = test_vec
    batch_size = 3
    assert jnp.allclose(jacfwd_builtin(f)(W), jacfwd_low_memory_jittable(f, batch_size)(W))

def test_hessian_lm(test_vec):
    f = jnp.linalg.norm
    W = test_vec
    batch_size = 3
    assert jnp.allclose(hessian_builtin(f)(W), hessian_low_memory(f, batch_size=batch_size)(W))
    assert jnp.allclose(hessian_builtin(f)(W), hessian_low_memory(f, batch_size=batch_size, outer=jacfwd_low_memory)(W))
    assert jnp.allclose(hessian_builtin(f)(W), hessian_low_memory(f, batch_size=batch_size, outer=jacrev_low_memory)(W))
