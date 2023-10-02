"""
This module contains an implementation of DEM using JAX.
"""
from functools import partial
from dataclasses import dataclass, field, replace
from typing import Callable

from tqdm import tqdm
from jaxlib.xla_extension import ArrayImpl
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jacfwd, vmap, hessian, grad, value_and_grad, jit
from jax.lax import fori_loop

from ..noise import autocorr_friston, noise_cov_gen_theoretical
from ..core import iterate_generalized

##
## Free action computation
##

@jit
def _fix_grad_shape(tensor):
    ndim = tensor.ndim
    if ndim == 6:
        batch_n = tensor.shape[0]
        batch_selection = jnp.arange(batch_n)
        out_n = tensor.shape[1]
        in_n = tensor.shape[4]
        # NOTE: The tensor includes all cross-batch derivatives too, which are always zero
        # hopefully this doesn't lead to unnecessary computations...
        return tensor[batch_selection,:,0,batch_selection,:,0]
    elif ndim == 4:
        return tensor.reshape((tensor.shape[0], tensor.shape[2]))
    elif ndim == 2:
        return tensor
    else:
        raise ValueError(f"Unexpected shape: {tuple(tensor.shape)}")


def tilde_to_grad(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
    """Computes gradient and function value given x and v in generalized
    coordinates."""
    assert mu_x_tilde.shape == (m_x * (p + 1), 1)
    assert mu_v_tilde.shape == (m_v * (p + 1), 1)
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]

    func_appl = func(mu_x, mu_v, params)
    func_jac = jacfwd(lambda x, v: func(x, v, params), argnums=(0,1))
    mu_x_grad, mu_v_grad =  func_jac(mu_x, mu_v)
    # fix shapes
    mu_x_grad = _fix_grad_shape(mu_x_grad)
    mu_v_grad = _fix_grad_shape(mu_v_grad)
    return func_appl, mu_x_grad, mu_v_grad


def tildes_to_grads(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    # batched version of tilde_to_grad
    ttg_v = vmap(lambda mu_x_tilde, mu_v_tilde: tilde_to_grad(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params))
    func_appl, mu_x_grad, mu_v_grad = ttg_v(mu_x_tildes, mu_v_tildes)
    return func_appl, mu_x_grad, mu_v_grad


def generalized_func(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    func_appl, mu_x_grad, mu_v_grad = tildes_to_grads(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params)

    n_batch = mu_x_tildes.shape[0]
    mu_x_tildes_r = mu_x_tildes.reshape((n_batch, p + 1, m_x))
    mu_v_tildes_r = mu_v_tildes.reshape((n_batch, p + 1, m_v))

    func_appl_d_x = jnp.einsum('bkj,bdj->bdk', mu_x_grad, mu_x_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    func_appl_d_v = jnp.einsum('bkj,bdj->bdk', mu_v_grad, mu_v_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    return jnp.concatenate((func_appl, func_appl_d_x + func_appl_d_v), axis=1)

@jit
def logdet(matr):
    return jnp.linalg.slogdet(matr)[1]

@jit
def _int_eng_par_static(
        mu,
        eta,
        p
    ):
    err = mu - eta
    return (-err.T @ p @ err + logdet(p)) / 2

def internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,
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

    # compute hessians used for mean-field terms in free action
    u_c_theta_dd = hessian(lambda mu: _int_eng_par_static(mu, eta_theta, p_theta))(mu_theta)
    u_c_lambda_dd = hessian(lambda mu: _int_eng_par_static(mu, eta_lambda, p_lambda))(mu_lambda)
    return u_c, u_c_theta_dd, u_c_lambda_dd

@partial(jit, static_argnames=('p', 'n',))
def deriv_mat(p, n):
    """
    Block derivative operator.

    Args:
        p: number of derivatives
        n: number of terms
    """
    return jnp.kron(jnp.eye(p + 1, k=1), jnp.eye(n))

# gen_func_f = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(f_jax, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params))
def internal_energy_dynamic(
        gen_func_f, gen_func_g, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, d, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv, diagnostic=False):
    # FIXME: Unfinished in JAX

    deriv_mat_x = deriv_mat(p, m_x)

    @partial(jit, static_argnames=('diagnostic',))
    def _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda, diagnostic=False):
        # Need to pad v_tilde with zeros to account for difference between
        # state embedding number `p` and causes embedding number `d`.
        mu_v_tildes_pad = jnp.pad(mu_v_tildes, ((0, 0), (0, p - d), (0, 0)))
        f_tildes = gen_func_f(mu_x_tildes, mu_v_tildes_pad, mu_theta)
        g_tildes = gen_func_g(mu_x_tildes, mu_v_tildes_pad, mu_theta)
        err_y = y_tildes - g_tildes
        err_v = mu_v_tildes - eta_v_tildes
        err_x = vmap(lambda mu_x_tilde: jnp.matmul(deriv_mat_x, mu_x_tilde))(mu_x_tildes) - f_tildes

        n_batch = mu_x_tildes.shape[0]

        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = jnp.exp(mu_lambda_z) * omega_z
        prec_w = jnp.exp(mu_lambda_w) * omega_w
        prec_z_tilde = jnp.kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = jnp.kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -vmap(lambda err: (err.T @ prec_z_tilde @ err))(err_y).reshape(n_batch) \
                + logdet(prec_z_tilde)
        u_t_v_ = -vmap(lambda err, p_v_tilde: (err.T @ p_v_tilde @ err))(err_v, p_v_tildes).reshape(n_batch) \
                + vmap(logdet)(p_v_tildes)
        u_t_x_ = -vmap(lambda err: (err.T @ prec_w_tilde @ err))(err_x).reshape(n_batch) \
                + logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        if diagnostic:
            extr = {
                'g_tildes': g_tildes,
                'f_tildes': f_tildes,
                'err_y': err_y,
                'err_v': err_v,
                'err_x': err_x,
                'prec_z_tilde': prec_z_tilde,
                'prec_w_tilde': prec_w_tilde,
                'u_t_y_': u_t_y_,
                'u_t_v_': u_t_v_,
                'u_t_x_': u_t_x_
            }
            return u_t, extr
        else:
            return u_t

    # It's reused a bunch of times, but maybe it should just be precomputed outside of this function
    out = _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda, diagnostic=diagnostic)
    if diagnostic:
        u_t, extr = out
    else:
        u_t = out

    u_t_x_tilde_dd = hessian(lambda mu_x_tildes:
                             jnp.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)))(mu_x_tildes)
    u_t_v_tilde_dd = hessian(lambda mu_v_tildes:
                             jnp.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)))(mu_v_tildes)
    u_t_theta_dd = hessian(lambda mu_theta:
                           jnp.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)))(mu_theta)
    u_t_lambda_dd = hessian(lambda mu_lambda:
                            jnp.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)))(mu_lambda)

    # fix shapes
    # batch_n = tensor.shape[0]
    # batch_selection = jnp.arange(batch_n)
    # out_n = tensor.shape[1]
    # in_n = tensor.shape[4]
    # u_t_x_tilde_dd = u_t_x_tilde_dd[batch_selection,:,0,batch_selection,:,0]
    # u_t_v_tilde_dd = u_t_v_tilde_dd[batch_selection,:,0,batch_selection,:,0]
    # u_t_theta_dd = u_t_theta_dd.reshape((u_t_theta_dd.shape[0], u_t_theta_dd.shape[2]))
    # u_t_lambda_dd = u_t_lambda_dd.reshape((u_t_lambda_dd.shape[0], u_t_lambda_dd.shape[2]))

    u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd)
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
        gen_func_f, gen_func_g,
        m_x, m_v, p, d,

        mu_x_tildes, mu_v_tildes,
        sig_x_tildes, sig_v_tildes,
        y_tildes,
        eta_v_tildes, p_v_tildes,
        omega_w, omega_z, noise_autocorr_inv
        ):
    """
    Computes internal energy/action, and hessians. Used to update precisions at the end of a
    DEM iteration.
    """
    u, u_theta_dd, u_lambda_dd = internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda
    )
    u_t, u_t_x_tilde_dds, u_t_v_tilde_dds, u_t_theta_dd, u_t_lambda_dd = internal_energy_dynamic(
        gen_func_f, gen_func_g, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, d, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv)
    u += jnp.sum(u_t)
    return u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds

@jit
def _batch_matmul_trace_sum(sig_tilde, u_t_tilde_dd):
    return vmap(lambda sig_tilde, u_t_tilde_dd: jnp.trace(jnp.matmul(sig_tilde, u_t_tilde_dd)))(sig_tilde, u_t_tilde_dd).sum()


@partial(jit, static_argnames=('m_x', 'm_v', 'p', 'd', 'gen_func_f', 'gen_func_g', 'skip_constant', 'diagnostic',))
def free_action(
        # how many terms are there in mu_x and mu_v?
        m_x, m_v,

        # how many derivatives are we tracking in generalised vectors?
        p, # for states
        d, # for causes

        # dynamic terms
        mu_x_tildes, mu_v_tildes, # iterator of state and input mean estimates in generalized coordinates
        sig_x_tildes, sig_v_tildes, # as above but for covariance estimates
        y_tildes, # iterator of outputs in generalized coordinates
        eta_v_tildes, p_v_tildes, # iterator of input mean priors

        # prior means and precisions
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,

        # parameter estimate means and covariances
        mu_theta,
        mu_lambda,
        sig_theta,
        sig_lambda,

        # jit-compiled functions in generalized coordinates
        # see: generalized_func
        gen_func_f, # state transition function
        gen_func_g, # output function

        # noise precision matrices (to be scaled by hyperparameters)
        omega_w,
        omega_z,

        # generalized noise temporal autocorrelation inverse (precision)
        noise_autocorr_inv,

        # It's ok to skip the constant step if computing free action gradients
        # only on dynamic terms
        skip_constant=False,

        # Whether to return an extra dictionary of detailed information
        diagnostic=False
        ):
    extr = {}
    # Constant terms of free action
    if not skip_constant:
        u_c, u_c_theta_dd, u_c_lambda_dd = internal_energy_static(
            mu_theta,
            mu_lambda,
            eta_theta,
            eta_lambda,
            p_theta,
            p_lambda
        )

        sig_logdet_c = (logdet(sig_theta) + logdet(sig_lambda)) / 2
        f_c = u_c + sig_logdet_c
        if diagnostic:
            extr_c = {
                    'sig_logdet_c': sig_logdet_c,
                    'u_c': u_c,
                    'u_c_theta_dd': u_c_theta_dd,
                    'u_c_lambda_dd': u_c_lambda_dd
                    }
            extr = {**extr, **extr_c}
    else:
        f_c = 0

    # Dynamic terms of free action that vary with time
    out = internal_energy_dynamic(
        gen_func_f, gen_func_g, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, d, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv, diagnostic=diagnostic)
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

    sig_logdet_t = (jnp.sum(vmap(logdet)(sig_x_tildes)) + jnp.sum(vmap(logdet)(sig_v_tildes))) / 2

    f_tsum = jnp.sum(u_t) \
            + sig_logdet_t \
            + (w_x_tilde_sum_ + w_v_tilde_sum_ + w_theta_sum_ + w_lambda_sum_) / 2

    if diagnostic:
        extr_t = {
                'u_t': u_t,
                'u_t_x_tilde_dd': u_t_x_tilde_dd,
                'u_t_v_tilde_dd': u_t_v_tilde_dd,
                'u_t_theta_dd': u_t_theta_dd,
                'u_t_lambda_dd': u_t_lambda_dd,
                'w_x_tilde_sum_': w_x_tilde_sum_,
                'w_v_tilde_sum_': w_v_tilde_sum_,
                'w_theta_sum_': w_theta_sum_,
                'w_lambda_sum_': w_lambda_sum_,
                'sig_logdet_t': sig_logdet_t
                }
        extr = {**extr, **extr_t}

    f_bar = f_c + f_tsum
    if diagnostic:
        return f_bar, extr
    else:
        return f_bar


def _verify_attr_dtypes(parent, attributes, dtype):
    for attr in attributes:
        obj = getattr(parent, attr)
        if not isinstance(obj, ArrayImpl) and not isinstance(obj, np.ndarray):
            raise ValueError(f"{attr} must be a numpy or jax.numpy array")
        if obj.dtype != dtype:
            raise ValueError(f"{attr} must be of dtype {dtype}")

##
## Objects for keeping DEM procedure state
##

@dataclass
class DEMInputJAX:
    """
    The input to DEM. It consists of data, priors, starting values, and
    transition functions. It consists of all the things which remain fixed over
    the course of DEM optimization.
    """
    # system information
    dt: float
    n: int = field(init=False) # length of system will be determined from ys

    # how many terms are there in the states x, inputs v?
    m_x: int
    m_v: int
    m_y: int
    # how many derivatives to track
    p: int # for states
    d: int # for inputs

    # system output
    ys: ArrayImpl

    # prior on system input
    eta_v: ArrayImpl # input sequence (will be transformed to generalized coords by Taylor trick)
    p_v: ArrayImpl # precision (TODO: Should this accept a sequence of precisions?)

    # priors on parameters and hyperparameters
    eta_theta: ArrayImpl
    eta_lambda: ArrayImpl
    p_theta: ArrayImpl
    p_lambda: ArrayImpl

    # functions which take (x, y, params)
    f: Callable # state transition
    g: Callable # output

    # Noise temporal autocorrelation structure
    # Defines the correlation matrix of generalized noise vectors. By default,
    # the noise is assumed to come from a Gaussian process with a Gaussian
    # covariance kernel. For alternative temporal covariance structures,
    # overwrite v_autocorr_inv and noise_autocorr_inv.
    noise_temporal_sig: float

    # Precomputed values
    y_tildes: ArrayImpl = None
    eta_v_tildes: ArrayImpl = None
    p_v_tildes: ArrayImpl = None

    # Compute larger embeddings to truncate them?
    p_comp: int = None # >= p
    d_comp: int = None # >= d

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
        if self.d_comp is None:
            self.d_comp = self.d
        # Precomputed values
        if self.noise_temporal_sig is not None:
            if self.v_autocorr_inv is None:
                v_autocorr = noise_cov_gen_theoretical(self.d, sig=self.noise_temporal_sig, autocorr=autocorr_friston())
                self.v_autocorr_inv = jnp.linalg.inv(v_autocorr)
            if self.noise_autocorr_inv is None:
                noise_autocorr = noise_cov_gen_theoretical(self.p, sig=self.noise_temporal_sig, autocorr=autocorr_friston())
                self.noise_autocorr_inv = jnp.linalg.inv(noise_autocorr)
        elif self.v_autocorr_inv is None:
            raise ValueError("v_autocorr_inv must be given if noise_temporal_sig is None")
        elif self.noise_autocorr_inv is None:
            raise ValueError("noise_autocorr_inv must be given if noise_temporal_sig is None")
        if self.y_tildes is None:
            self.y_tildes = jnp.stack(list(iterate_generalized(self.ys, self.dt, self.p, p_comp=self.p_comp))).astype(self.dtype)
        if self.eta_v_tildes is None:
            self.eta_v_tildes = jnp.stack(list(iterate_generalized(self.eta_v, self.dt, self.d, p_comp=self.p_comp))).astype(self.dtype)
        if self.p_v_tildes is None:
            p_v_tilde = jnp.kron(self.v_autocorr_inv, self.p_v)
            self.p_v_tildes = jnp.tile(p_v_tilde, (len(self.eta_v_tildes), 1, 1))
        if self.dtype is None:
            self.dtype = self.ys.dtype
        if self.omega_w is None:
            self.omega_w = jnp.eye(self.m_x)
        if self.omega_z is None:
            self.omega_z = jnp.eye(self.m_y)

        self.gen_func_f = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(self.f, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params))
        self.gen_func_g = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(self.g, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params))

        _verify_attr_dtypes(self, [
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
            "noise_autocorr_inv"
        ], self.dtype)

    # Remove unpicklable attributes (gen_func_f and gen_func_g)
    # see https://stackoverflow.com/a/2345953
    # and https://docs.python.org/3/library/pickle.html#handling-stateful-objects
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['gen_func_f']
        del state['gen_func_g']
        return state


    def __setstate__(self, state):
        self.__dict__.update(state)
        self.gen_func_f = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(self.f, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params))
        self.gen_func_g = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(self.g, mu_x_tildes, mu_v_tildes, self.m_x, self.m_v, self.p, params))


@dataclass
class DEMStateJAX:
    """
    Keeps track of the current state of a DEM model. Contains its state,
    parameter, and hyperparameter estimates.
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

    # initial dynamic states
    mu_x0_tilde: ArrayImpl
    mu_v0_tilde: ArrayImpl

    def __post_init__(self):
        _verify_attr_dtypes(self, [
            "mu_x_tildes",
            "mu_v_tildes",
            "sig_x_tildes",
            "sig_v_tildes",
            "mu_theta",
            "mu_lambda",
            "sig_theta",
            "sig_lambda",
            "mu_x0_tilde",
            "mu_v0_tilde"
        ], dtype=self.input.dtype)

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
            diagnostic=diagnostic
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
                noise_autocorr_inv=state.input.noise_autocorr_inv)

    @classmethod
    def from_input(cls, input: DEMInputJAX, x0: ArrayImpl,  mu_theta: ArrayImpl=None, **kwargs):
        x0 = x0.reshape(-1)
        assert len(x0) == input.m_x

        mu_x0_tilde = jnp.concatenate([x0, jnp.zeros(input.p * input.m_x)]).reshape((-1, 1))
        mu_v_tildes = input.eta_v_tildes.copy()

        # TODO: What is a good value here?
        # this one shouldn't be *horrible*, but is very arbitrary
        # sig_x_tildes = _autocorr_inv(input.p, input.dt * 3).repeat(len(mu_v_tildes), 1, 1)
        # the noise autocorrelation structure given should be in some way
        # approximately ok for the xs as well...
        sig_x_tildes = jnp.tile(jnp.kron(input.noise_autocorr_inv, jnp.eye(input.m_x)), (len(mu_v_tildes), 1, 1))

        if mu_theta is None:
            mu_theta = input.eta_theta.copy()

        kwargs_default = dict(
                input=input,
                mu_x_tildes=mu_x0_tilde[None],
                mu_v_tildes=mu_v_tildes,
                sig_x_tildes=sig_x_tildes,
                sig_v_tildes=vmap(jnp.linalg.inv)(input.p_v_tildes),
                mu_theta=mu_theta,
                mu_lambda=input.eta_lambda,
                sig_theta=jnp.linalg.inv(input.p_theta),
                sig_lambda=jnp.linalg.inv(input.p_lambda),
                mu_x0_tilde=mu_x0_tilde,
                mu_v0_tilde=mu_v_tildes[0]
            )
        kwargs = {**kwargs_default, **kwargs}

        return cls(**kwargs)

##
## DEM algorithm itself
##

def _dynamic_free_energy(
        # time-dependent state and input estimates
        mu_x_tilde_t, mu_v_tilde_t,
        # extra time-dependent variables
        y_tilde, sig_x_tilde, sig_v_tilde, eta_v_tilde, p_v_tilde,
        # all of the other argmuents to free_action
        m_x, m_v,
        p, d,
        eta_theta, eta_lambda,
        p_theta, p_lambda,
        mu_theta, mu_lambda,
        sig_theta, sig_lambda,
        gen_func_g, gen_func_f,
        omega_w, omega_z,
        noise_autocorr_inv
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
        skip_constant=True)

def dem_step_d(state: DEMStateJAX, lr):
    """
    Performs the D step of DEM.
    """
    # In order to use fori_loop, we must precompute the shape of the output array
    total_steps = len(state.input.y_tildes)
    mu_x_tildes = jnp.concatenate([state.mu_x0_tilde[None],
                                   jnp.zeros((total_steps - 1, state.input.m_x * (state.input.p + 1), 1))],
                                  axis=0)
    mu_v_tildes = jnp.concatenate([state.mu_v0_tilde[None],
                                   jnp.zeros((total_steps - 1, state.input.m_v * (state.input.d + 1), 1))],
                                  axis=0)
    deriv_mat_x = deriv_mat(state.input.p, state.input.m_x).astype(state.input.dtype)
    deriv_mat_v = deriv_mat(state.input.d, state.input.m_v).astype(state.input.dtype)

    def d_step_iter(t, dynamic_state):
        y_tilde = state.input.y_tildes[t]
        sig_x_tilde = state.sig_x_tildes[t]
        sig_v_tilde = state.sig_v_tildes[t]
        eta_v_tilde = state.input.eta_v_tildes[t]
        p_v_tilde = state.input.p_v_tildes[t]
        mu_x_tildes, mu_v_tildes = dynamic_state
        mu_x_tilde_t = mu_x_tildes[t]
        mu_v_tilde_t = mu_v_tildes[t]

        # free action on just a single timestep
        x_d_raw, v_d_raw = grad(_dynamic_free_energy, argnums=(0,1))(mu_x_tilde_t, mu_v_tilde_t, y_tilde,
            sig_x_tilde, sig_v_tilde,
            eta_v_tilde, p_v_tilde,
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
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
            noise_autocorr_inv=state.input.noise_autocorr_inv)
        # NOTE: In the original pseudocode, x and v are in one vector
        x_d = deriv_mat_x @ mu_x_tilde_t + lr * x_d_raw
        v_d = deriv_mat_v @ mu_v_tilde_t + lr * v_d_raw
        x_dd = lr * hessian(_dynamic_free_energy, argnums=0)(mu_x_tilde_t, mu_v_tilde_t, y_tilde,
            sig_x_tilde, sig_v_tilde,
            eta_v_tilde, p_v_tilde,
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
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
            noise_autocorr_inv=state.input.noise_autocorr_inv)
        v_dd = lr * hessian(_dynamic_free_energy, argnums=1)(mu_x_tilde_t, mu_v_tilde_t, y_tilde,
            sig_x_tilde, sig_v_tilde,
            eta_v_tilde, p_v_tilde,
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            d=state.input.d,
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
            noise_autocorr_inv=state.input.noise_autocorr_inv)
        x_dd = deriv_mat_x + _fix_grad_shape(x_dd)
        v_dd = deriv_mat_v + _fix_grad_shape(v_dd)
        step_matrix_x = (jsp.linalg.expm(x_dd * state.input.dt) - jnp.eye(x_dd.shape[0], dtype=state.input.dtype)) @ jnp.linalg.inv(x_dd)
        step_matrix_v = (jsp.linalg.expm(v_dd * state.input.dt) - jnp.eye(v_dd.shape[0], dtype=state.input.dtype)) @ jnp.linalg.inv(v_dd)
        mu_x_tilde_tp1 = mu_x_tilde_t + step_matrix_x @ x_d
        mu_v_tilde_tp1 = mu_v_tilde_t + step_matrix_v @ v_d

        mu_x_tildes = mu_x_tildes.at[t+1].set(mu_x_tilde_tp1)
        mu_v_tildes = mu_v_tildes.at[t+1].set(mu_v_tilde_tp1)
        return (mu_x_tildes, mu_v_tildes)

    mu_x_tildes, mu_v_tildes = fori_loop(0, len(state.input.y_tildes) - 1, d_step_iter, (mu_x_tildes, mu_v_tildes))
    state.mu_x_tildes = mu_x_tildes
    state.mu_v_tildes = mu_v_tildes

# FIXME torch -> jax below this line

def dem_step_precision(state: DEMStateJAX):
    """
    Does a precision update of DEM.
    """
    u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds = state.internal_action()
    state.sig_theta = jnp.linalg.inv(-u_theta_dd)
    state.sig_lambda = jnp.linalg.inv(-u_lambda_dd)
    state.sig_x_tildes = -vmap(jnp.linalg.inv)(u_t_x_tilde_dds)
    state.sig_v_tildes = -vmap(jnp.linalg.inv)(u_t_v_tilde_dds)

def dem_step_m(state: DEMStateJAX, lr_lambda, iter_lambda, min_improv):
    """
    Performs the noise hyperparameter update (step M) of DEM.
    """
    last_f_bar = None
    def lambda_free_action(mu_lambda):
        # free action as a function of lambda
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
                    mu_lambda=mu_lambda,
                    sig_theta=state.sig_theta,
                    sig_lambda=state.sig_lambda,
                    gen_func_g=state.input.gen_func_g,
                    gen_func_f=state.input.gen_func_f,
                    omega_w=state.input.omega_w,
                    omega_z=state.input.omega_z,
                    noise_autocorr_inv=state.input.noise_autocorr_inv,
                    skip_constant=False)
    for i in tqdm(range(iter_lambda), desc="Step M"):
        f_bar, lambda_d_raw = value_and_grad(lambda_free_action)(state.mu_lambda)
        lambda_d = lr_lambda * lambda_d_raw
        lambda_dd = lr_lambda * hessian(lambda_free_action)(state.mu_lambda)
        step_matrix = (jsp.linalg.expm(lambda_dd) - jnp.eye(lambda_dd.shape[0], dtype=state.input.dtype)) @ jnp.linalg.inv(lambda_dd)
        state.mu_lambda = state.mu_lambda + step_matrix @ lambda_d
        # convergence check
        if last_f_bar is not None:
            if last_f_bar + min_improv > f_bar:
                break
        last_f_bar = f_bar.copy()


def dem_step_e(state: DEMStateJAX, lr_theta):
    """
    Performs the parameter update (step E) of DEM.
    """
    # TODO: should be an if statement comparing new f_bar with old
    def theta_free_action(mu_theta):
        # free action as a function of theta
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
                    mu_theta=mu_theta,
                    mu_lambda=state.mu_lambda,
                    sig_theta=state.sig_theta,
                    sig_lambda=state.sig_lambda,
                    gen_func_g=state.input.gen_func_g,
                    gen_func_f=state.input.gen_func_f,
                    omega_w=state.input.omega_w,
                    omega_z=state.input.omega_z,
                    noise_autocorr_inv=state.input.noise_autocorr_inv,
                    skip_constant=False)
    theta_d_raw = grad(theta_free_action)(state.mu_theta)
    theta_d = lr_theta * theta_d_raw
    theta_dd = lr_theta * hessian(theta_free_action)(state.mu_theta)
    step_matrix = (jsp.linalg.expm(theta_dd) - jnp.eye(theta_dd.shape[0], dtype=state.input.dtype)) @ jnp.linalg.inv(theta_dd)
    state.mu_theta = state.mu_theta + step_matrix @ theta_d

def dem_step(state: DEMStateJAX, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=0.01):
    """
    Does an iteration of DEM.
    """
    dem_step_d(state, lr_dynamic)
    dem_step_m(state, lr_lambda, iter_lambda, min_improv=m_min_improv)
    dem_step_e(state, lr_theta)
    dem_step_precision(state)
