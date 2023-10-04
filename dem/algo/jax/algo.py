"""
This module contains an implementation of DEM using JAX.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
# DEM experiments often involve extremely high priors, which do not work well
# with single precision float32
from jax import config, grad, hessian, jacfwd, jit, value_and_grad, vmap
from jax.lax import fori_loop, while_loop
from jaxlib.xla_extension import ArrayImpl

config.update("jax_enable_x64", True)

from ...core import iterate_generalized
from ...noise import autocorr_friston, noise_cov_gen_theoretical
from .util import _fix_grad_shape, deriv_mat, hessian_low_memory_jit, logdet

MATRIX_EXPM_MAX_SQUARINGS = 100

##
## Free action computation
##

# The functions below implement eq. (40) from [1], extended to arbitrary state
# transition and cause functions f and g following eq. (31) from [2].
#
#   [1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm
#   for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol.
#   23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
#   [2] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A
#   variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp.
#   849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.


def tilde_to_grad(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
    """Computes gradient and function value given x and v in generalized
    coordinates."""
    assert mu_x_tilde.shape == (m_x * (p + 1), 1)
    assert mu_v_tilde.shape == (m_v * (p + 1), 1)
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]

    func_appl = func(mu_x, mu_v, params)
    func_jac = jacfwd(lambda x, v: func(x, v, params), argnums=(0, 1))
    mu_x_grad, mu_v_grad = func_jac(mu_x, mu_v)
    # fix shapes
    mu_x_grad = _fix_grad_shape(mu_x_grad)
    mu_v_grad = _fix_grad_shape(mu_v_grad)
    return func_appl, mu_x_grad, mu_v_grad


def tildes_to_grads(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    # batched version of tilde_to_grad
    ttg_v = vmap(
        lambda mu_x_tilde, mu_v_tilde: tilde_to_grad(
            func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params
        )
    )
    func_appl, mu_x_grad, mu_v_grad = ttg_v(mu_x_tildes, mu_v_tildes)
    return func_appl, mu_x_grad, mu_v_grad


def generalized_func(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    func_appl, mu_x_grad, mu_v_grad = tildes_to_grads(
        func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params
    )

    n_batch = mu_x_tildes.shape[0]
    mu_x_tildes_r = mu_x_tildes.reshape((n_batch, p + 1, m_x))
    mu_v_tildes_r = mu_v_tildes.reshape((n_batch, p + 1, m_v))

    func_appl_d_x = jnp.einsum(
        "bkj,bdj->bdk", mu_x_grad, mu_x_tildes_r[:, 1:, :]
    ).reshape((n_batch, -1, 1))
    func_appl_d_v = jnp.einsum(
        "bkj,bdj->bdk", mu_v_grad, mu_v_tildes_r[:, 1:, :]
    ).reshape((n_batch, -1, 1))
    return jnp.concatenate((func_appl, func_appl_d_x + func_appl_d_v), axis=1)


@jit
def _int_eng_par_static(mu, eta, p):
    err = mu - eta
    return (-err.T @ p @ err + logdet(p)) / 2


def internal_energy_static(
    mu_theta, mu_lambda, eta_theta, eta_lambda, p_theta, p_lambda, low_memory=False
):
    """
    Computes static terms of the internal energy, along with necessary
    Hessians. These are the precision-weighted errors and precision log
    determinants on the parameters and hyperparameters.

    These terms are added in the internal energy formula only once,
    in contrast to dynamic terms which are a sum over all time.
    """
    # Computes some terms of the internal energy along with necessary Hessians
    u_c_theta = _int_eng_par_static(mu_theta, eta_theta, p_theta)
    u_c_lambda = _int_eng_par_static(mu_lambda, eta_lambda, p_lambda)
    u_c = u_c_theta + u_c_lambda

    if low_memory:
        hessian_func = hessian_low_memory_jit
    else:
        hessian_func = hessian

    # compute hessians used for mean-field terms in free action
    u_c_theta_dd = hessian_func(lambda mu: _int_eng_par_static(mu, eta_theta, p_theta))(
        mu_theta
    )
    u_c_lambda_dd = hessian_func(
        lambda mu: _int_eng_par_static(mu, eta_lambda, p_lambda)
    )(mu_lambda)
    return u_c, u_c_theta_dd, u_c_lambda_dd


def internal_energy_dynamic(
    gen_func_f,
    gen_func_g,
    mu_x_tildes,
    mu_v_tildes,
    y_tildes,
    m_x,
    m_v,
    p,
    d,
    mu_theta,
    eta_v_tildes,
    p_v_tildes,
    mu_lambda,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    low_memory=False,
    diagnostic=False,
):
    deriv_mat_x = deriv_mat(p, m_x)

    @partial(jit, static_argnames=("diagnostic",))
    def _int_eng_dynamic(
        mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda, diagnostic=False
    ):
        # Need to pad v_tilde with zeros to account for difference between
        # state embedding order `p` and causes embedding order `d`.
        mu_v_tildes_pad = jnp.pad(mu_v_tildes, ((0, 0), (0, p - d), (0, 0)))
        f_tildes = gen_func_f(mu_x_tildes, mu_v_tildes_pad, mu_theta)
        g_tildes = gen_func_g(mu_x_tildes, mu_v_tildes_pad, mu_theta)
        err_y = y_tildes - g_tildes
        err_v = mu_v_tildes - eta_v_tildes
        err_x = (
            vmap(lambda mu_x_tilde: jnp.matmul(deriv_mat_x, mu_x_tilde))(mu_x_tildes)
            - f_tildes
        )

        n_batch = mu_x_tildes.shape[0]

        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = jnp.exp(mu_lambda_z) * omega_z
        prec_w = jnp.exp(mu_lambda_w) * omega_w
        prec_z_tilde = jnp.kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = jnp.kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -vmap(lambda err: (err.T @ prec_z_tilde @ err))(err_y).reshape(
            n_batch
        ) + logdet(prec_z_tilde)
        u_t_v_ = -vmap(lambda err, p_v_tilde: (err.T @ p_v_tilde @ err))(
            err_v, p_v_tildes
        ).reshape(n_batch) + vmap(logdet)(p_v_tildes)
        u_t_x_ = -vmap(lambda err: (err.T @ prec_w_tilde @ err))(err_x).reshape(
            n_batch
        ) + logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        if diagnostic:
            extr = {
                "g_tildes": g_tildes,
                "f_tildes": f_tildes,
                "err_y": err_y,
                "err_v": err_v,
                "err_x": err_x,
                "prec_z_tilde": prec_z_tilde,
                "prec_w_tilde": prec_w_tilde,
                "u_t_y_": u_t_y_,
                "u_t_v_": u_t_v_,
                "u_t_x_": u_t_x_,
            }
            return u_t, extr
        else:
            return u_t

    out = _int_eng_dynamic(
        mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda, diagnostic=diagnostic
    )
    if diagnostic:
        u_t, extr = out
    else:
        u_t = out

    if low_memory:
        hessian_func = hessian_low_memory_jit
    else:
        hessian_func = hessian
    # NOTE: It seems more efficient to run `hessian` four times, once per each
    # argument, rather than just run it once for all arguments at once

    # FIXME: make it possible to use hessian_low_memory_jit below instead of jax.hessian
    # need to improve hessian_low_memory_jit to support the shapes of
    # mu_x_tildes and mu_v_tildes
    u_t_x_tilde_dd = hessian(
        lambda mu_x_tildes: jnp.sum(
            _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
        )
    )(mu_x_tildes)
    u_t_v_tilde_dd = hessian(
        lambda mu_v_tildes: jnp.sum(
            _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
        )
    )(mu_v_tildes)
    u_t_theta_dd = hessian_func(
        lambda mu_theta: jnp.sum(
            _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
        )
    )(mu_theta)
    u_t_lambda_dd = hessian_func(
        lambda mu_lambda: jnp.sum(
            _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
        )
    )(mu_lambda)

    u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
    u_t_theta_dd = _fix_grad_shape(u_t_theta_dd)
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)

    if diagnostic:
        return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd, extr
    else:
        return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd


def internal_action(
    # for static internal energy
    mu_theta,
    mu_lambda,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    # for dynamic internal energies
    gen_func_f,
    gen_func_g,
    m_x,
    m_v,
    p,
    d,
    mu_x_tildes,
    mu_v_tildes,
    sig_x_tildes,
    sig_v_tildes,
    y_tildes,
    eta_v_tildes,
    p_v_tildes,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    low_memory=False,
):
    """
    Computes internal energy/action, and hessians on parameter, hyperparameter,
    and state estimates. Used to update precisions at the end of a DEM iteration.
    """
    u, u_theta_dd, u_lambda_dd = internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,
        low_memory=low_memory,
    )
    (
        u_t,
        u_t_x_tilde_dds,
        u_t_v_tilde_dds,
        u_t_theta_dd,
        u_t_lambda_dd,
    ) = internal_energy_dynamic(
        gen_func_f,
        gen_func_g,
        mu_x_tildes,
        mu_v_tildes,
        y_tildes,
        m_x,
        m_v,
        p,
        d,
        mu_theta,
        eta_v_tildes,
        p_v_tildes,
        mu_lambda,
        omega_w,
        omega_z,
        noise_autocorr_inv,
        low_memory=low_memory,
    )
    u += jnp.sum(u_t)
    return u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds


@jit
def _batch_matmul_trace_sum(sig_tilde, u_t_tilde_dd):
    return vmap(
        lambda sig_tilde, u_t_tilde_dd: jnp.trace(jnp.matmul(sig_tilde, u_t_tilde_dd))
    )(sig_tilde, u_t_tilde_dd).sum()


@partial(
    jit,
    static_argnames=(
        "m_x",
        "m_v",
        "p",
        "d",
        "gen_func_f",
        "gen_func_g",
        "skip_constant",
        "diagnostic",
        "low_memory",
    ),
)
def free_action(
    m_x,
    m_v,
    p,
    d,
    mu_x_tildes,
    mu_v_tildes,
    sig_x_tildes,
    sig_v_tildes,
    y_tildes,
    eta_v_tildes,
    p_v_tildes,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    mu_theta,
    mu_lambda,
    sig_theta,
    sig_lambda,
    gen_func_f,
    gen_func_g,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    skip_constant=False,
    diagnostic=False,
    low_memory=False,
):
    """
    Computes the free action of a DEM model.

    It implements eq. (40) from [1], extended to arbitrary state transition and
    cause functions f and g following eq. (31) from [2].

    Args:
        m_x (int): number of dynamic states
        m_v (int): number of causes
        p (int): state embedding order, i.e. how many derivatives are tracked
        d (int): cause embedding order, i.e. how many derivatives are tracked
        mu_x_tildes (n_tilde, m_x * (p + 1), 1): array of estimated posterior
            means of states in generalized coordinates
        mu_v_tildes (n_tilde, m_v * (d + 1), 1): array of estimated posterior
            means of causes in generalized coordinates
        sig_x_tildes (n_tilde, m_x * (d + 1), m_x * (d + 1)): array of
            estimated posterior covariance matrices of states in generalized
            coordinates
        sig_v_tildes (n_tilde, m_x * (d + 1), m_x * (d + 1)): array of
            estimated posterior covariance matrices of causes in generalized
            coordinates
        y_tildes (n_tilde, m_y * (p + 1), 1): array of outputs in generalized
            coordinates
        eta_v_tildes (n_tilde, m_v * (d + 1), 1): array of prior means of
            causes in  generalized coordinates
        p_v_tildes (n_tilde, m_v * (d + 1), m_v * (d + 1)): array of prior
            precisions of causes in generalized coordinates
        eta_theta (m_theta,): array of prior means of parameters
        eta_theta (2,): array of prior means of hyperparameters. The two values
            correspond to output and state noise, respectively.
        p_theta (m_theta, m_theta): array of prior precisions of parameters
        p_lambda (2, 2): array of prior precisions of hyperparameters
        mu_theta (m_theta,): array of estimated posterior means of parameters
        mu_lambda (2,): array of estimated posterior means of hyperparameters
        sig_theta (m_theta, m_theta): array of estimated posterior covariances
            of parameters
        sig_lambda (2, 2): array of estimated posterior covariances of
            hyperparameters
        gen_func_f: jit-compiled state transition function in generalized
            coordinates, accepting argmuents (x_tilde, v_tilde). See
            DEMInputJAX code for example of how it's defined
        gen_func_g: jit-compiled  output function in generalized coordinates,
            accepting argmuents (x_tilde, v_tilde). See DEMInputJAX code for
            example of how it's defined
        omega_w (m_x, m_x): correlation matrix of state noises
        omega_z (m_y, m_y): correlation matrix of output noises
        noise_autocorr_inv (p + 1, p + 1): precision of noises in generalized
            coordinates, like eq. (7) of [1].
        skip_constant (bool): whether to skip constant terms of free action.
            and only compute free action on dynamic terms
        diagnostic (bool): whether to return an extra dictionary of detailed
            intermediate information
        low_memory (bool): whether to use a low-memory but slower method to
            compute hessians

      [1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm
        for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol.
        23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
      [2] K. J. Friston, N. Trujillo-Barreto, and J. Daunizeau, “DEM: A
        variational treatment of dynamic systems,” NeuroImage, vol. 41, no. 3, pp.
        849–885, Jul. 2008, doi: 10.1016/j.neuroimage.2008.02.054.
    """
    extr = {}
    # Constant terms of free action
    if not skip_constant:
        u_c, u_c_theta_dd, u_c_lambda_dd = internal_energy_static(
            mu_theta,
            mu_lambda,
            eta_theta,
            eta_lambda,
            p_theta,
            p_lambda,
            low_memory=low_memory,
        )

        sig_logdet_c = (logdet(sig_theta) + logdet(sig_lambda)) / 2
        f_c = u_c + sig_logdet_c
        if diagnostic:
            extr_c = {
                "sig_logdet_c": sig_logdet_c,
                "u_c": u_c,
                "u_c_theta_dd": u_c_theta_dd,
                "u_c_lambda_dd": u_c_lambda_dd,
            }
            extr = {**extr, **extr_c}
    else:
        f_c = 0

    # Dynamic terms of free action that vary with time
    out = internal_energy_dynamic(
        gen_func_f,
        gen_func_g,
        mu_x_tildes,
        mu_v_tildes,
        y_tildes,
        m_x,
        m_v,
        p,
        d,
        mu_theta,
        eta_v_tildes,
        p_v_tildes,
        mu_lambda,
        omega_w,
        omega_z,
        noise_autocorr_inv,
        diagnostic=diagnostic,
        low_memory=low_memory,
    )
    if diagnostic:
        u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd, extr_dt = out
        extr = {**extr, **extr_dt}
    else:
        u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd = out
    w_x_tilde_sum_ = _batch_matmul_trace_sum(sig_x_tildes, u_t_x_tilde_dd)
    w_v_tilde_sum_ = _batch_matmul_trace_sum(sig_v_tildes, u_t_v_tilde_dd)
    # w_theta and w_lambda are sums already, because u_t_theta_dd is a sum
    # because of how the batch Hessian is computed
    if not skip_constant:
        w_theta_sum_ = jnp.trace(sig_theta @ (u_c_theta_dd + u_t_theta_dd))
        w_lambda_sum_ = jnp.trace(sig_lambda @ (u_c_lambda_dd + u_t_lambda_dd))
    else:
        w_theta_sum_ = jnp.trace(sig_theta @ (u_t_theta_dd))
        w_lambda_sum_ = jnp.trace(sig_lambda @ (u_t_lambda_dd))

    sig_logdet_t = (
        jnp.sum(vmap(logdet)(sig_x_tildes)) + jnp.sum(vmap(logdet)(sig_v_tildes))
    ) / 2

    f_tsum = (
        jnp.sum(u_t)
        + sig_logdet_t
        + (w_x_tilde_sum_ + w_v_tilde_sum_ + w_theta_sum_ + w_lambda_sum_) / 2
    )

    if diagnostic:
        extr_t = {
            "u_t": u_t,
            "u_t_x_tilde_dd": u_t_x_tilde_dd,
            "u_t_v_tilde_dd": u_t_v_tilde_dd,
            "u_t_theta_dd": u_t_theta_dd,
            "u_t_lambda_dd": u_t_lambda_dd,
            "w_x_tilde_sum_": w_x_tilde_sum_,
            "w_v_tilde_sum_": w_v_tilde_sum_,
            "w_theta_sum_": w_theta_sum_,
            "w_lambda_sum_": w_lambda_sum_,
            "sig_logdet_t": sig_logdet_t,
        }
        extr = {**extr, **extr_t}

    f_bar = f_c + f_tsum
    if diagnostic:
        return f_bar, extr
    else:
        return f_bar


def _verify_attr_dtypes(parent, attributes, dtype):
    """Verifies that all of the"""
    for attr in attributes:
        obj = getattr(parent, attr)
        if not isinstance(obj, ArrayImpl) and not isinstance(obj, np.ndarray):
            raise ValueError(f"{attr} must be a numpy or jax.numpy array")
        if obj.dtype != dtype:
            raise ValueError(
                f"{attr} must be of dtype {dtype}, "
                "so that all data is in the same dtype"
            )


##
## DEM optimization steps
##

# Functions below implement steps necessary for Algorithm 1 of [1]. They are
# expected to be called in context of a DEMStateJAX object, since otherwise
# they're somewhat unergonomic.
#
# By being implemented as pure functions, they can be JIT-compiled by JAX.
#
# [1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm
#   for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol.
#   23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.


def _dynamic_free_energy(
    # time-dependent state and input estimates
    mu_x_tilde_t,
    mu_v_tilde_t,
    # extra time-dependent variables
    y_tilde,
    sig_x_tilde,
    sig_v_tilde,
    eta_v_tilde,
    p_v_tilde,
    # all of the other argmuents to free_action
    m_x,
    m_v,
    p,
    d,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    mu_theta,
    mu_lambda,
    sig_theta,
    sig_lambda,
    gen_func_g,
    gen_func_f,
    omega_w,
    omega_z,
    noise_autocorr_inv,
):
    """Computes the dynamic parts of free energy based on variables at a given
    point in time."""
    return free_action(
        m_x=m_x,
        m_v=m_v,
        p=p,
        d=d,
        mu_x_tildes=mu_x_tilde_t[None],
        mu_v_tildes=mu_v_tilde_t[None],
        sig_x_tildes=sig_x_tilde[None],
        sig_v_tildes=sig_v_tilde[None],
        y_tildes=y_tilde[None],
        eta_v_tildes=eta_v_tilde[None],
        p_v_tildes=p_v_tilde[None],
        eta_theta=eta_theta,
        eta_lambda=eta_lambda,
        p_theta=p_theta,
        p_lambda=p_lambda,
        mu_theta=mu_theta,
        mu_lambda=mu_lambda,
        sig_theta=sig_theta,
        sig_lambda=sig_lambda,
        gen_func_g=gen_func_g,
        gen_func_f=gen_func_f,
        omega_w=omega_w,
        omega_z=omega_z,
        noise_autocorr_inv=noise_autocorr_inv,
        skip_constant=True,
    )


@partial(jit, static_argnames=("m_x", "m_v", "p", "d", "gen_func_f", "gen_func_g"))
def _update_xv(
    m_x,
    m_v,
    p,
    d,
    mu_x_tildes,
    mu_v_tildes,
    sig_x_tildes,
    sig_v_tildes,
    y_tildes,
    eta_v_tildes,
    p_v_tildes,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    mu_theta,
    mu_lambda,
    sig_theta,
    sig_lambda,
    gen_func_g,
    gen_func_f,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    dt,
    lr_dynamic,
):
    dtype = y_tildes.dtype
    total_steps = len(y_tildes)
    mu_x0_tilde = mu_x_tildes[0]
    mu_v0_tilde = mu_v_tildes[0]
    mu_x_tildes = jnp.concatenate(
        [mu_x0_tilde[None], jnp.zeros((total_steps - 1, m_x * (p + 1), 1))], axis=0
    )
    mu_v_tildes = jnp.concatenate(
        [mu_v0_tilde[None], jnp.zeros((total_steps - 1, m_v * (d + 1), 1))], axis=0
    )
    deriv_mat_x = deriv_mat(p, m_x).astype(dtype)
    deriv_mat_v = deriv_mat(d, m_v).astype(dtype)

    def d_step_iter(t, dynamic_state):
        y_tilde = y_tildes[t]
        sig_x_tilde = sig_x_tildes[t]
        sig_v_tilde = sig_v_tildes[t]
        eta_v_tilde = eta_v_tildes[t]
        p_v_tilde = p_v_tildes[t]
        mu_x_tildes, mu_v_tildes = dynamic_state
        mu_x_tilde_t = mu_x_tildes[t]
        mu_v_tilde_t = mu_v_tildes[t]

        # free action on just a single timestep
        x_d_raw, v_d_raw = grad(_dynamic_free_energy, argnums=(0, 1))(
            mu_x_tilde_t,
            mu_v_tilde_t,
            y_tilde,
            sig_x_tilde,
            sig_v_tilde,
            eta_v_tilde,
            p_v_tilde,
            m_x=m_x,
            m_v=m_v,
            p=p,
            d=d,
            eta_theta=eta_theta,
            eta_lambda=eta_lambda,
            p_theta=p_theta,
            p_lambda=p_lambda,
            mu_theta=mu_theta,
            mu_lambda=mu_lambda,
            sig_theta=sig_theta,
            sig_lambda=sig_lambda,
            gen_func_g=gen_func_g,
            gen_func_f=gen_func_f,
            omega_w=omega_w,
            omega_z=omega_z,
            noise_autocorr_inv=noise_autocorr_inv,
        )
        # NOTE: In the original pseudocode, x and v are in one vector
        x_d = deriv_mat_x @ mu_x_tilde_t + lr_dynamic * x_d_raw
        v_d = deriv_mat_v @ mu_v_tilde_t + lr_dynamic * v_d_raw
        x_dd = lr_dynamic * hessian(_dynamic_free_energy, argnums=0)(
            mu_x_tilde_t,
            mu_v_tilde_t,
            y_tilde,
            sig_x_tilde,
            sig_v_tilde,
            eta_v_tilde,
            p_v_tilde,
            m_x=m_x,
            m_v=m_v,
            p=p,
            d=d,
            eta_theta=eta_theta,
            eta_lambda=eta_lambda,
            p_theta=p_theta,
            p_lambda=p_lambda,
            mu_theta=mu_theta,
            mu_lambda=mu_lambda,
            sig_theta=sig_theta,
            sig_lambda=sig_lambda,
            gen_func_g=gen_func_g,
            gen_func_f=gen_func_f,
            omega_w=omega_w,
            omega_z=omega_z,
            noise_autocorr_inv=noise_autocorr_inv,
        )
        v_dd = lr_dynamic * hessian(_dynamic_free_energy, argnums=1)(
            mu_x_tilde_t,
            mu_v_tilde_t,
            y_tilde,
            sig_x_tilde,
            sig_v_tilde,
            eta_v_tilde,
            p_v_tilde,
            m_x=m_x,
            m_v=m_v,
            p=p,
            d=d,
            eta_theta=eta_theta,
            eta_lambda=eta_lambda,
            p_theta=p_theta,
            p_lambda=p_lambda,
            mu_theta=mu_theta,
            mu_lambda=mu_lambda,
            sig_theta=sig_theta,
            sig_lambda=sig_lambda,
            gen_func_g=gen_func_g,
            gen_func_f=gen_func_f,
            omega_w=omega_w,
            omega_z=omega_z,
            noise_autocorr_inv=noise_autocorr_inv,
        )
        x_dd = deriv_mat_x + _fix_grad_shape(x_dd)
        v_dd = deriv_mat_v + _fix_grad_shape(v_dd)
        step_matrix_x = (
            jsp.linalg.expm(x_dd * dt, max_squarings=MATRIX_EXPM_MAX_SQUARINGS)
            - jnp.eye(x_dd.shape[0], dtype=dtype)
        ) @ jnp.linalg.inv(x_dd)
        step_matrix_v = (
            jsp.linalg.expm(v_dd * dt, max_squarings=MATRIX_EXPM_MAX_SQUARINGS)
            - jnp.eye(v_dd.shape[0], dtype=dtype)
        ) @ jnp.linalg.inv(v_dd)
        mu_x_tilde_tp1 = mu_x_tilde_t + step_matrix_x @ x_d
        mu_v_tilde_tp1 = mu_v_tilde_t + step_matrix_v @ v_d

        mu_x_tildes = mu_x_tildes.at[t + 1].set(mu_x_tilde_tp1)
        mu_v_tildes = mu_v_tildes.at[t + 1].set(mu_v_tilde_tp1)
        return (mu_x_tildes, mu_v_tildes)

    return fori_loop(0, len(y_tildes) - 1, d_step_iter, (mu_x_tildes, mu_v_tildes))


@partial(
    jit,
    static_argnames=("m_x", "m_v", "p", "d", "gen_func_f", "gen_func_g", "low_memory"),
)
def _update_precision(
    mu_theta,
    mu_lambda,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    gen_func_f,
    gen_func_g,
    m_x,
    m_v,
    p,
    d,
    mu_x_tildes,
    mu_v_tildes,
    sig_x_tildes,
    sig_v_tildes,
    y_tildes,
    eta_v_tildes,
    p_v_tildes,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    low_memory,
):
    u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds = internal_action(
        mu_theta=mu_theta,
        mu_lambda=mu_lambda,
        eta_theta=eta_theta,
        eta_lambda=eta_lambda,
        p_theta=p_theta,
        p_lambda=p_lambda,
        gen_func_g=gen_func_g,
        gen_func_f=gen_func_f,
        m_x=m_x,
        m_v=m_v,
        p=p,
        d=d,
        mu_x_tildes=mu_x_tildes,
        mu_v_tildes=mu_v_tildes,
        sig_x_tildes=sig_x_tildes,
        sig_v_tildes=sig_v_tildes,
        y_tildes=y_tildes,
        eta_v_tildes=eta_v_tildes,
        p_v_tildes=p_v_tildes,
        omega_w=omega_w,
        omega_z=omega_z,
        noise_autocorr_inv=noise_autocorr_inv,
        low_memory=low_memory,
    )
    sig_theta = jnp.linalg.inv(-u_theta_dd)
    sig_lambda = jnp.linalg.inv(-u_lambda_dd)
    sig_x_tildes = -jnp.linalg.inv(u_t_x_tilde_dds)
    sig_v_tildes = -jnp.linalg.inv(u_t_v_tilde_dds)
    return sig_theta, sig_lambda, sig_x_tildes, sig_v_tildes


@partial(
    jit,
    static_argnames=("m_x", "m_v", "p", "d", "gen_func_f", "gen_func_g", "iter_lambda"),
)
def _update_lambda(
    m_x,
    m_v,
    p,
    d,
    mu_x_tildes,
    mu_v_tildes,
    sig_x_tildes,
    sig_v_tildes,
    y_tildes,
    eta_v_tildes,
    p_v_tildes,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    mu_theta,
    mu_lambda,
    sig_theta,
    sig_lambda,
    gen_func_g,
    gen_func_f,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    lr_lambda,
    iter_lambda,
    min_improv,
):
    def lambda_free_action(mu_lambda):
        # free action as a function of lambda
        return free_action(
            m_x=m_x,
            m_v=m_v,
            p=p,
            d=d,
            mu_x_tildes=mu_x_tildes,
            mu_v_tildes=mu_v_tildes,
            sig_x_tildes=sig_x_tildes,
            sig_v_tildes=sig_v_tildes,
            y_tildes=y_tildes,
            eta_v_tildes=eta_v_tildes,
            p_v_tildes=p_v_tildes,
            eta_theta=eta_theta,
            eta_lambda=eta_lambda,
            p_theta=p_theta,
            p_lambda=p_lambda,
            mu_theta=mu_theta,
            mu_lambda=mu_lambda,
            sig_theta=sig_theta,
            sig_lambda=sig_lambda,
            gen_func_g=gen_func_g,
            gen_func_f=gen_func_f,
            omega_w=omega_w,
            omega_z=omega_z,
            noise_autocorr_inv=noise_autocorr_inv,
            skip_constant=False,
        )

    init_args = (0, -jnp.inf, -jnp.inf, mu_lambda)

    def cond_fun(args):
        step, f_bar, last_f_bar, mu_lambda = args
        return jnp.logical_and(last_f_bar + min_improv <= f_bar, step < iter_lambda)

    def body_fun(args):
        step, last_f_bar, _, mu_lambda = args
        f_bar, lambda_d_raw = value_and_grad(lambda_free_action)(mu_lambda)
        lambda_d = lr_lambda * lambda_d_raw
        lambda_dd = lr_lambda * hessian(lambda_free_action)(mu_lambda)
        step_matrix = (
            jsp.linalg.expm(lambda_dd, max_squarings=MATRIX_EXPM_MAX_SQUARINGS)
            - jnp.eye(lambda_dd.shape[0], dtype=mu_lambda.dtype)
        ) @ jnp.linalg.inv(lambda_dd)
        mu_lambda = mu_lambda + step_matrix @ lambda_d
        return step + 1, f_bar, last_f_bar, mu_lambda

    step, f_bar, last_f_bar, mu_lambda = while_loop(cond_fun, body_fun, init_args)
    return mu_lambda


@partial(
    jit,
    static_argnames=("m_x", "m_v", "p", "d", "gen_func_f", "gen_func_g", "low_memory"),
)
def _update_theta(
    m_x,
    m_v,
    p,
    d,
    mu_x_tildes,
    mu_v_tildes,
    sig_x_tildes,
    sig_v_tildes,
    y_tildes,
    eta_v_tildes,
    p_v_tildes,
    eta_theta,
    eta_lambda,
    p_theta,
    p_lambda,
    mu_theta,
    mu_lambda,
    sig_theta,
    sig_lambda,
    gen_func_g,
    gen_func_f,
    omega_w,
    omega_z,
    noise_autocorr_inv,
    lr_theta,
    low_memory,
):
    def theta_free_action(mu_theta):
        return free_action(
            m_x=m_x,
            m_v=m_v,
            p=p,
            d=d,
            mu_x_tildes=mu_x_tildes,
            mu_v_tildes=mu_v_tildes,
            sig_x_tildes=sig_x_tildes,
            sig_v_tildes=sig_v_tildes,
            y_tildes=y_tildes,
            eta_v_tildes=eta_v_tildes,
            p_v_tildes=p_v_tildes,
            eta_theta=eta_theta,
            eta_lambda=eta_lambda,
            p_theta=p_theta,
            p_lambda=p_lambda,
            mu_theta=mu_theta,
            mu_lambda=mu_lambda,
            sig_theta=sig_theta,
            sig_lambda=sig_lambda,
            gen_func_g=gen_func_g,
            gen_func_f=gen_func_f,
            omega_w=omega_w,
            omega_z=omega_z,
            noise_autocorr_inv=noise_autocorr_inv,
            skip_constant=False,
        )

    theta_d_raw = grad(theta_free_action)(mu_theta)
    theta_d = lr_theta * theta_d_raw
    # using standard jax.hessian causes an out-of-memory in even relatively
    # easy cases (tested on a 20 gb ram machine)
    if low_memory:
        hessian_func = hessian_low_memory_jit
    else:
        hessian_func = hessian
    theta_dd = lr_theta * hessian_func(theta_free_action)(mu_theta)
    step_matrix = (
        jsp.linalg.expm(theta_dd, max_squarings=MATRIX_EXPM_MAX_SQUARINGS)
        - jnp.eye(theta_dd.shape[0], dtype=mu_theta.dtype)
    ) @ jnp.linalg.inv(theta_dd)
    return mu_theta + step_matrix @ theta_d


##
## Objects for keeping DEM procedure state
##


@dataclass
class DEMInputJAX:
    """
    The input to DEM. It consists of data, priors, and transition functions,
    i.e. of all the things which remain fixed over the course of DEM
    optimization.

    Args:
        dt (float): sampling period
        m_x (int): number of dynamic states
        m_v (int): number of causes
        m_y (int): number of outputs
        p (int): state and output embedding order, i.e. how many derivatives
            are tracked
        d (int): cause embedding order, i.e. how many derivatives are tracked
        ys (n, m_y): timeseries of outputs
        eta_v (n, m_v): prior means of input timeseries
        p_v (m_v, m_v): prior precision of inputs
        eta_theta (m_theta,): array of prior means of parameters
        eta_theta (2,): array of prior means of hyperparameters. The two values
            correspond to output and state noise, respectively.
        p_theta (m_theta, m_theta): array of prior precisions of parameters
        p_lambda (2, 2): array of prior precisions of hyperparameters
        f: state transition function accepting arguments (x, v, params)
        g: output function accepting arguments (x, v, params)
        noise_temporal_sig (float): temporal noise smoothness, following eq.
            (7) of [1]. By default, the noise is assumed to come from a
            Gaussian process generated by convolving white noise with a
            Gaussian kernel. For alternative autocorrelation structures,
            override noise_autocorr_inv and v_autocorr_inv.
        y_tildes (n_tilde, m_y * (p + 1), 1): array of outputs in generalized
            coordinates. Computed from ys
        eta_v_tildes (n_tilde, m_v * (d + 1), 1): array of prior means of
            causes in  generalized coordinates. Computed from eta_v
        p_v_tildes (n_tilde, m_v * (d + 1), m_v * (d + 1)): array of prior
            precisions of causes in generalized coordinates. Computed from p_v
        dtype (np.dtype): dtype of data. Computed from ys
        omega_w (m_x, m_x): correlation matrix of state noises
        omega_z (m_y, m_y): correlation matrix of output noises
        noise_autocorr_inv (p + 1, p + 1): precision of noises in generalized
            coordinates, like eq. (7) of [1]. Computed based on
            noise_temporal_sig
        v_autocorr_inv (p + 1, p + 1): precision of inputs in generalized
            coordinates, like eq. (7) of [1]. Computed based on
            noise_temporal_sig

      [1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm
        for Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol.
        23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
    """

    # system information
    dt: float
    n: int = field(init=False)  # length of system will be determined from ys

    # how many terms are there in the states x, inputs v?
    m_x: int
    m_v: int
    m_y: int
    # how many derivatives to track
    p: int  # for states
    d: int  # for inputs

    # system output
    ys: ArrayImpl

    # prior on system input
    eta_v: ArrayImpl
    p_v: ArrayImpl

    # priors on parameters and hyperparameters
    eta_theta: ArrayImpl
    eta_lambda: ArrayImpl
    p_theta: ArrayImpl
    p_lambda: ArrayImpl

    # functions which take (x, y, params)
    f: Callable  # state transition
    g: Callable  # output

    # Noise temporal autocorrelation structure
    # Defines the correlation matrix of generalized noise vectors. By default,
    # the noise is assumed to come from a Gaussian process with a Gaussian
    # covariance kernel. For alternative temporal covariance structures,
    # overwrite v_autocorr_inv and noise_autocorr_inv.
    # TODO: Discussion of precise meaning of sigma.
    noise_temporal_sig: float

    # Precomputed values
    y_tildes: ArrayImpl = None
    eta_v_tildes: ArrayImpl = None
    p_v_tildes: ArrayImpl = None

    # Compute larger embedding vectors than used and truncate them to increase
    # accuracy
    p_comp: int = None  # >= p

    # Datatype for numerical precision
    dtype: jnp.dtype = None

    # Noise vector correlation. Usually it's just an identity matrix
    omega_w: ArrayImpl = None
    omega_z: ArrayImpl = None

    # Precision matrices of generalized input and noise vectors
    ## if not given, they are computed based on noise_temporal_sig
    v_autocorr_inv: ArrayImpl = None
    noise_autocorr_inv: ArrayImpl = None

    # JAX functions for precompilation
    gen_func_f: Callable = field(init=False)
    gen_func_g: Callable = field(init=False)

    def __post_init__(self):
        if self.ys.ndim == 1:
            self.ys = self.ys.reshape((-1, 1))
        self.n = self.ys.shape[0]
        if self.p_comp is None:
            self.p_comp = self.p
        self.d_comp = self.p_comp
        # Precomputed values
        if self.noise_temporal_sig is not None:
            if self.v_autocorr_inv is None:
                v_autocorr = noise_cov_gen_theoretical(
                    self.d, sig=self.noise_temporal_sig, autocorr=autocorr_friston()
                )
                self.v_autocorr_inv = jnp.linalg.inv(v_autocorr)
            if self.noise_autocorr_inv is None:
                noise_autocorr = noise_cov_gen_theoretical(
                    self.p, sig=self.noise_temporal_sig, autocorr=autocorr_friston()
                )
                self.noise_autocorr_inv = jnp.linalg.inv(noise_autocorr)
        elif self.v_autocorr_inv is None:
            raise ValueError(
                "v_autocorr_inv must be given if noise_temporal_sig is None"
            )
        elif self.noise_autocorr_inv is None:
            raise ValueError(
                "noise_autocorr_inv must be given if noise_temporal_sig is None"
            )
        if self.y_tildes is None:
            self.y_tildes = jnp.stack(
                list(iterate_generalized(self.ys, self.dt, self.p, p_comp=self.p_comp))
            ).astype(self.dtype)
        if self.eta_v_tildes is None:
            self.eta_v_tildes = jnp.stack(
                list(
                    iterate_generalized(self.eta_v, self.dt, self.d, p_comp=self.p_comp)
                )
            ).astype(self.dtype)
        if self.p_v_tildes is None:
            p_v_tilde = jnp.kron(self.v_autocorr_inv, self.p_v)
            self.p_v_tildes = jnp.tile(p_v_tilde, (len(self.eta_v_tildes), 1, 1))
        if self.dtype is None:
            self.dtype = self.ys.dtype
        if self.omega_w is None:
            self.omega_w = jnp.eye(self.m_x)
        if self.omega_z is None:
            self.omega_z = jnp.eye(self.m_y)

        self.gen_func_f = jit(
            lambda mu_x_tildes, mu_v_tildes, params: generalized_func(
                self.f, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params
            )
        )
        self.gen_func_g = jit(
            lambda mu_x_tildes, mu_v_tildes, params: generalized_func(
                self.g, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params
            )
        )

        _verify_attr_dtypes(
            self,
            [
                "ys",
                "eta_v",
                "p_v",
                "v_autocorr_inv",
                "eta_theta",
                "eta_lambda",
                "p_theta",
                "p_lambda",
                "omega_w",
                "omega_z",
                "noise_autocorr_inv",
            ],
            self.dtype,
        )

    # Remove unpicklable attributes (gen_func_f and gen_func_g)
    # see https://stackoverflow.com/a/2345953
    # and https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["gen_func_f"]
        del state["gen_func_g"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.gen_func_f = jit(
            lambda mu_x_tildes, mu_v_tildes, params: generalized_func(
                self.f, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params
            )
        )
        self.gen_func_g = jit(
            lambda mu_x_tildes, mu_v_tildes, params: generalized_func(
                self.g, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params
            )
        )


@dataclass
class DEMStateJAX:
    """
    Keeps track of the current state of a DEM model. Contains its state,
    parameter, and hyperparameter estimates.

    Args:
        input (DEMInputJAX): underlying input containing data and priors
        mu_x_tildes (n_tilde, m_x * (p + 1), 1): array of estimated posterior
            means of states in generalized coordinates
        mu_v_tildes (n_tilde, m_v * (d + 1), 1): array of estimated posterior
            means of causes in generalized coordinates
        sig_x_tildes (n_tilde, m_x * (d + 1), m_x * (d + 1)): array of
            estimated posterior covariance matrices of states in generalized
            coordinates
        sig_v_tildes (n_tilde, m_x * (d + 1), m_x * (d + 1)): array of
            estimated posterior covariance matrices of causes in generalized
            coordinates
        mu_theta (m_theta,): array of estimated posterior means of parameters
        mu_lambda (2,): array of estimated posterior means of hyperparameters
        sig_theta (m_theta, m_theta): array of estimated posterior covariances
            of parameters
        sig_lambda (2, 2): array of estimated posterior covariances of
            hyperparameters
    """

    # system input
    input: DEMInputJAX

    # dynamic state estimates
    mu_x_tildes: ArrayImpl
    mu_v_tildes: ArrayImpl
    sig_x_tildes: ArrayImpl
    sig_v_tildes: ArrayImpl

    # static parameter and hyperparameter estimates
    mu_theta: ArrayImpl
    mu_lambda: ArrayImpl
    sig_theta: ArrayImpl
    sig_lambda: ArrayImpl

    def __post_init__(self):
        _verify_attr_dtypes(
            self,
            [
                "mu_x_tildes",
                "mu_v_tildes",
                "sig_x_tildes",
                "sig_v_tildes",
                "mu_theta",
                "mu_lambda",
                "sig_theta",
                "sig_lambda",
            ],
            dtype=self.input.dtype,
        )

    def free_action(self, skip_constant=False, diagnostic=False):
        state = self
        return free_action(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.y_tildes,
            eta_v_tildes=state.input.eta_v_tildes,
            p_v_tildes=state.input.p_v_tildes,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            sig_theta=state.sig_theta,
            sig_lambda=state.sig_lambda,
            gen_func_g=state.input.gen_func_g,
            gen_func_f=state.input.gen_func_f,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
            skip_constant=skip_constant,
            diagnostic=diagnostic,
        )

    def internal_action(self):
        state = self
        return internal_action(
            # for static internal energy
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            # for dynamic internal energies
            gen_func_g=state.input.gen_func_g,
            gen_func_f=state.input.gen_func_f,
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.y_tildes,
            eta_v_tildes=state.input.eta_v_tildes,
            p_v_tildes=state.input.p_v_tildes,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
        )

    @classmethod
    def from_input(
        cls,
        input: DEMInputJAX,
        x0: ArrayImpl = None,
        **kwargs,
    ):
        """
        Initializes the DEM state with some sane defaults compatible with the
        input. Accepts all arguments as the constructor of DEMStateJAX.

        Args:
            x0 (m_x,): Array of initial state values.
        """
        if x0 is None:
            x0 = np.zeros(input.m_x)
        x0 = x0.reshape(-1)
        assert len(x0) == input.m_x

        # initialize all xs to 0
        mu_x0_tilde = jnp.concatenate([x0, jnp.zeros(input.p * input.m_x)]).reshape(
            (-1, 1)
        )
        mu_x_tildes = jnp.concatenate(
            [
                mu_x0_tilde[None],
                jnp.zeros((len(input.eta_v_tildes) - 1, (input.p + 1) * input.m_x, 1)),
            ]
        )
        mu_v_tildes = input.eta_v_tildes.copy()

        # TODO: What is a good value here?
        # this one shouldn't be *horrible*, but is very arbitrary
        # the noise autocorrelation structure given should be in some way
        # approximately ok for the xs as well...
        sig_x_tildes = jnp.tile(
            jnp.kron(input.noise_autocorr_inv, jnp.eye(input.m_x)),
            (len(mu_v_tildes), 1, 1),
        )

        kwargs_default = dict(
            input=input,
            mu_x_tildes=mu_x_tildes,
            mu_v_tildes=mu_v_tildes,
            sig_x_tildes=sig_x_tildes,
            sig_v_tildes=jnp.linalg.inv(input.p_v_tildes),
            mu_theta=input.eta_theta,
            mu_lambda=input.eta_lambda,
            sig_theta=jnp.linalg.inv(input.p_theta),
            sig_lambda=jnp.linalg.inv(input.p_lambda),
        )
        kwargs = {**kwargs_default, **kwargs}

        return cls(**kwargs)

    def step_d(state, lr_dynamic=1):
        """
        Performs D step of DEM, updating estimates of generalized states
        (mu_x_tildes) and generalized causes (mu_v_tildes).

        Args:
            lr_dynamic (float): learning rate
        """
        state.mu_x_tildes, state.mu_v_tildes = _update_xv(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.y_tildes,
            eta_v_tildes=state.input.eta_v_tildes,
            p_v_tildes=state.input.p_v_tildes,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            sig_theta=state.sig_theta,
            sig_lambda=state.sig_lambda,
            gen_func_g=state.input.gen_func_g,
            gen_func_f=state.input.gen_func_f,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
            dt=state.input.dt,
            lr_dynamic=lr_dynamic,
        )

    def step_e(state, lr_theta, low_memory=True):
        """
        Performs the E step of DEM, updating parameter estimates (mu_theta).

        Args:
            lr_theta (float): learning rate
            low_memory (bool): Whether to use a low-memory but slow method for
                computing hessians.
        """
        state.mu_theta = _update_theta(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.y_tildes,
            eta_v_tildes=state.input.eta_v_tildes,
            p_v_tildes=state.input.p_v_tildes,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            sig_theta=state.sig_theta,
            sig_lambda=state.sig_lambda,
            gen_func_g=state.input.gen_func_g,
            gen_func_f=state.input.gen_func_f,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
            lr_theta=lr_theta,
            low_memory=low_memory,
        )
        # TODO: should be an if statement comparing new f_bar with old

    def step_m(state, lr_lambda, iter_lambda=8, min_improv=0.01):
        """
        Performs the M step of DEM, updating hyperparameter estimates
        (mu_lambda).

        Args:
            lr_lambda (float): learning rate
            iter_lambda (int): maximum number of iterations
            min_improv (float): minimum improvement before assuming convergence
        """
        state.mu_lambda = _update_lambda(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.y_tildes,
            eta_v_tildes=state.input.eta_v_tildes,
            p_v_tildes=state.input.p_v_tildes,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            sig_theta=state.sig_theta,
            sig_lambda=state.sig_lambda,
            gen_func_g=state.input.gen_func_g,
            gen_func_f=state.input.gen_func_f,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
            lr_lambda=lr_lambda,
            iter_lambda=iter_lambda,
            min_improv=min_improv,
        )

    def step_precision(state, low_memory=False):
        """
        Performs the precision update step, updating posterior covariances
        (sig_x_tildes, sig_v_tildes, sig_theta, sig_lambda).

        Args:
            low_memory (bool): Whether to use a low-memory but slow method for
                computing hessians.
        """
        (
            state.sig_theta,
            state.sig_lambda,
            state.sig_x_tildes,
            state.sig_v_tildes,
        ) = _update_precision(
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            gen_func_g=state.input.gen_func_g,
            gen_func_f=state.input.gen_func_f,
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.y_tildes,
            eta_v_tildes=state.input.eta_v_tildes,
            p_v_tildes=state.input.p_v_tildes,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
            low_memory=low_memory,
        )

    def step(state, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=0.01):
        """
        Does one complete step of DEM, which includes the D, E, M, and
        precision update steps, in order following [1]

        [1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm
            for Estimation of Linear Systems with Colored Noise,” Entropy
            (Basel), vol. 23, no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
        """
        state.step_d(state, lr_dynamic)
        state.step_m(state, lr_lambda, iter_lambda, min_improv=m_min_improv)
        state.step_e(state, lr_theta)
        state.step_precision(state)


# def dem_step_d(state: DEMStateJAX, lr_dynamic):
#     """
#     Performs the D step of DEM.
#     """
#     state.step_d(lr_dynamic=lr_dynamic)


# def dem_step_precision(state: DEMStateJAX, low_memory=False):
#     """
#     Does a precision update of DEM.
#     """
#     state.step_precision(low_memory=low_memory)


# def dem_step_m(state: DEMStateJAX, lr_lambda, iter_lambda, min_improv):
#     """
#     Performs the noise hyperparameter update (step M) of DEM.
#     """
#     state.step_m(lr_lambda=lr_lambda, iter_lambda=iter_lambda, min_improv=min_improv)


# def dem_step_e(state: DEMStateJAX, lr_theta, low_memory=True):
#     """
#     Performs the parameter update (step E) of DEM.
#     """
#     state.step_e(lr_theta=lr_theta, low_memory=low_memory)


# def dem_step(
#     state: DEMStateJAX,
#     lr_dynamic,
#     lr_theta,
#     lr_lambda,
#     iter_lambda=8,
#     m_min_improv=0.01,
# ):
#     """
#     Does an iteration of DEM.
#     """
#     state(
#         lr_dynamic=lr_dynamic,
#         lr_theta=lr_theta,
#         lr_lambda=lr_lambda,
#         iter_lambda=iter_lambda,
#         m_min_improv=m_min_improv,
#     )


##
## Functions for inspecting DEM states
##


def generalized_batch_to_sequence(tensor, m, is2d=False):
    if not is2d:
        xs = jnp.stack([x_tilde[:m] for x_tilde in tensor], axis=0)[:, :, 0]
    else:
        xs = jnp.stack([jnp.diagonal(x_tilde)[:m] for x_tilde in tensor], axis=0)
    return xs


def extract_dynamic(state: DEMStateJAX):
    mu_xs = generalized_batch_to_sequence(state.mu_x_tildes, state.input.m_x)
    sig_xs = generalized_batch_to_sequence(
        state.sig_x_tildes, state.input.m_x, is2d=True
    )
    mu_vs = generalized_batch_to_sequence(state.mu_v_tildes, state.input.m_v)
    sig_vs = generalized_batch_to_sequence(
        state.sig_v_tildes, state.input.m_v, is2d=True
    )
    idx_first = int(state.input.p_comp // 2)
    idx_last = idx_first + len(mu_xs)
    ts_all = jnp.arange(state.input.n) * state.input.dt
    ts = ts_all[idx_first:idx_last]
    return mu_xs, sig_xs, mu_vs, sig_vs, ts
