from dataclasses import dataclass, field
from typing import Callable, Iterable
from itertools import repeat

import torch
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from hdm.core import deriv_mat, iterate_generalized
from hdm.noise import generate_noise_conv, autocorr_friston, noise_cov_gen_theoretical

plot = False

# Part 0: Implement DEM

def kron(A, B):
    """
    Kronecker product for matrices.

    This is a replacement for Torch's `torch.kron`, which is broken and crashes
    when called on an inverted matrix, similar to here:
        https://github.com/pytorch/pytorch/issues/54135

    Taken from Anton Obukhov's comment on GitHub:
        https://github.com/pytorch/pytorch/issues/74442#issuecomment-1111468515
    """
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

def _fix_grad_shape(tensor):
    """
    "Fixes" shape for outputs of torch.autograd.functional.jacobian or
    torch.autograd.functional.hessian. Transforms from dimension 4 to dimension
    2 just discarding two dimensions.
    """
    # FIXME: Is this necessary? I'm not sure I understand the output shape of
    # these functions.
    # Their output shapes are a bit peculiar for our case. It has dimension 4.
    # I'm guessing that this is because PyTorch can be very flexible in the
    # input/output shapes, also considering cases like minibatches. For now,
    # this solution *seems* to work (for no minibatches)
    return tensor.reshape((tensor.shape[0], tensor.shape[2]))

def generalized_func(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
    """
    Computes generalized application of a function, assuming local linearity.
    Used to find g_tilde and f_tilde.

    Args:
        func (function): function to apply (g or f)
        mu_x_tilde (Tensor): generalized vector of estimated state means
        mu_v_tilde (Tensor): generalized vector of estimated input means
        m_x (int): number of elements in state vector
        m_v (int): number of elements in input vector
        p (int): numer of derivatives in generalized vectors
        params (Tensor): parameters of func
    """
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad, mu_v_grad = torch.autograd.functional.jacobian(lambda x, v: func(x, v, params), (mu_x, mu_v), create_graph=True)
    mu_x_grad = _fix_grad_shape(mu_x_grad)
    mu_v_grad = _fix_grad_shape(mu_v_grad)
    func_appl_d = []
    for deriv in range(1, p):
        mu_x_d = mu_x_tilde[deriv*m_x:(deriv+1)*m_x]
        mu_v_d = mu_v_tilde[deriv*m_v:(deriv+1)*m_v]
        func_appl_d.append(mu_x_grad @ mu_x_d + mu_v_grad @ mu_v_d)
    return torch.vstack([func_appl] + func_appl_d)


def hessian_and_value(func, argnums):
    def func_double(*args, **kwargs):
        result = func(*args, **kwargs)
        return result, result
    return torch.func.jacfwd(torch.func.jacrev(func_double, argnums=argnums, has_aux=True), argnums=argnums, has_aux=True)

def _int_eng_par_static(
        mu_theta,
        eta_theta,
        p_theta
    ):
    err_theta = mu_theta - eta_theta
    return (-err_theta.T @ p_theta @ err_theta + torch.logdet(p_theta)) / 2

def internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda
        ):
    # Computes some terms of the internal energy along with necessary Hessians
    u_c_theta = _int_eng_par_static(mu_theta, eta_theta, p_theta)
    u_c_lambda = _int_eng_par_static(mu_lambda, eta_lambda, p_lambda)

    u_c_theta_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_par_static(mu, eta_theta, p_theta), mu_theta, create_graph=True)
    u_c_lambda_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_par_static(mu, eta_lambda, p_lambda), mu_lambda, create_graph=True)
    return u_c, u_c_theta_dd, u_c_lambda_dd

def internal_energy_dynamic(
        g, f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta, eta_v_tilde,
        mu_lambda, omega_w, omega_z, prec_w, prec_z):
    # make a temporary function which we can use to compute hessians w.r.t. the relevant parameters
    # for the computation of mean-field terms
    def _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda):
        g_tilde = generalized_func(g, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)
        f_tilde = generalized_func(f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)

        err_y = y_tilde - g_tilde
        err_v = mu_v_tilde - eta_v_tilde
        err_x = deriv_mat(p, m_x) @ mu_x_tilde - f_tilde

        # we need to split up mu_lambda into the hyperparameter for noise of states and of outputs
        # how many elements do the noise terms contain?
        # for z (state noise), it must be the same as number of states, so m_x
        # for w (output noise), it depends on the output of the function g, so
        # we could do
        #  m_w = g_tilde.shape[0] // p
        # but we can also take it as the remainder of mu_lambda after splitting
        # out mu_lambda_z
        mu_lambda_z = mu_lambda[:m_x]
        mu_lambda_w = mu_lambda[m_x:]
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)

        # TODO: Optimize this diagonalization?
        err_dynamic = torch.vstack([err_y, err_v, err_x])
        prec_dynamic = torch.block_diag(prec_z_tilde, p_v_tilde, prec_w_tilde)
        u_t = -(err_dynamic.T @ prec_dynamic @ err_dynamic + torch.logdet(prec_dynamic)) / 2
        return u_t
    # horribly inefficient way to go about this, but hey, at least it may work...
    # (so many unnecessary repeated computations)
    u_t = _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda)
    u_t_x_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu, mu_v_tilde, mu_theta, mu_lambda), mu_x_tilde, create_graph=True)
    u_t_v_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu, mu_theta, mu_lambda), mu_v_tilde, create_graph=True)
    u_t_theta_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu, mu_lambda), mu_theta, create_graph=True)
    u_t_lambda_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu), mu_lambda, create_graph=True)

    u_t_x_tilde_d = _fix_grad_shape(u_t_x_tilde_d)
    u_t_v_tilde_d = _fix_grad_shape(u_t_v_tilde_d)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd )
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)

    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd

def free_action(
        # how many terms are there in mu_x and mu_v?
        m_x, m_v,

        # how many derivatives are we tracking in generalised vectors?
        p,

        # dynamic terms
        mu_x_tildes, mu_v_tildes, # iterator of state and input mean estimates in generalized coordinates
        sig_x_tildes, sig_v_tildes, # like above, but covariance estimates
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

        # functions
        g, # output function
        f, # state transition function

        # noise precision matrices (to be scaled by hyperparameters)
        omega_w,
        omega_z,

        # generalized noise temporal autocorrelation inverse (precision)
        noise_autocorr_inv,
        ):

    u_c, u_c_theta_dd, u_c_lambda_dd = internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda
    )
    f_c = u_c + (torch.logdet(sig_theta) + torch.logdet(sig_lambda)) / 2
    f_t = []

    for t, (mu_x_tilde, mu_v_tilde,
            sig_x_tilde, sig_v_tilde,
            y_tilde,
            eta_v_tilde, p_v_tilde) in enumerate(
                zip(mu_x_tildes, mu_v_tildes,
                    sig_x_tildes, sig_v_tildes,
                    y_tildes,
                    eta_v_tildes, p_v_tildes,
                    strict=True)
            ):
        u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd = internal_energy_dynamic(
            g, f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta, eta_v_tilde,
            mu_lambda, omega_w, omega_z, prec_w, prec_z)

        # mean-field terms
        w_x_tilde, w_v_tilde, w_theta, w_lambda = [
            torch.trace(sig @ (u_c_dd + u_t_dd)) / 2
                for (sig, u_c_dd, u_t_dd) in [
                    (sig_x_tilde, 0, u_t_x_tilde_dd),
                    (sig_v_tilde, 0, u_t_v_tilde_dd),
                    (sig_theta, u_c_theta_dd, u_t_theta_dd),
                    (sig_lambda, u_c_lambda_dd, u_t_lambda_dd),
                ]
            ]
        f_t.append(
            u_t + (torch.logdet(sig_x_tilde) + torch.logdet(sig_v_tilde)) / 2
            )
    f_bar = f_c + sum(f_t)
    return f_bar

@dataclass
class DEMInput:
    """
    The input to DEM. It consists of data, priors, starting values, and transition functions.
    """
    # system information
    dt: float
    n: int = field(init=False) # length of system will be determined from ys

    # how many terms are there in the states x, inputs v?
    m_x: int
    m_v: int
    # how many derivatives to track
    p: int

    # system output
    ys: torch.Tensor

    # prior on system input
    eta_v: torch.Tensor # input sequence (will be transformed to generalized coords by Taylor trick)
    p_v: torch.Tensor # precision
    v_autocorr_inv: torch.Tensor # generalised input vector precision
    # TODO: How do we construct p_v_tildes? Probably kron(S, P_v)...

    # priors on parameters and hyperparameters
    eta_theta: torch.Tensor
    eta_lambda: torch.Tensor
    p_theta: torch.Tensor
    p_lambda: torch.Tensor

    # functions which take (x, y, params)
    g: Callable # output
    f: Callable # state transition

    # Noise correlation and temporal autocorrelation structure
    omega_w: torch.Tensor
    omega_z: torch.Tensor
    noise_autocorr_inv: torch.Tensor

    # Initial parameters
    # TODO: How to initialize parameters and x?
    x0: torch.Tensor
    theta0: torch.Tensor
    lambda0: torch.Tensor

    def __post_init__(self):
        if self.ys.ndim == 1:
            self.ys = self.ys.reshape((-1, 1))
        self.n = self.ys.shape[0]
        self.theta0 = self.eta_theta if self.theta0 is None else self.theta0
        self.lambda0 = self.eta_lambda if self.lambda0 is None else self.lambda0

    def iter_y_tildes(self):
        return iterate_generalized(self.ys, self.dt, self.p)

    def iter_eta_v_tildes(self):
        return iterate_generalized(self.eta_v, self.dt, self.p)

    def iter_p_v_tildes(self):
        p_v_tilde = kron(self.v_autocorr_inv, self.p_v)
        return repeat(p_v_tilde, self.n)


@dataclass
class DEMState:
    """
    Keeps track of the current state of a DEM model. Contains its state,
    parameter, and hyperparameter estimates.
    """
    # system input
    input: DEMInput

    # dynamic state estimates
    mu_x_tildes: Iterable[torch.Tensor]
    mu_v_tildes: Iterable[torch.Tensor]
    sig_x_tildes: Iterable[torch.Tensor]
    sig_v_tildes: Iterable[torch.Tensor]

    # static parameter and hyperparameter estimates
    mu_theta: torch.Tensor
    mu_lambda: torch.Tensor
    sig_theta: torch.Tensor
    sig_lambda: torch.Tensor

    def iter_mu_x_tildes(self):
        yield from self.mu_x_tildes

    def iter_mu_v_tildes(self):
        yield from self.mu_v_tildes

    def iter_sig_x_tildes(self):
        yield from self.sig_x_tildes

    def iter_sig_v_tildes(self):
        yield from self.sig_v_tildes

def free_action_from_state(state: DEMState):
    return free_action(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            mu_x_tildes=state.iter_mu_x_tildes(),
            mu_v_tildes=state.iter_mu_v_tildes(),
            sig_x_tildes=state.iter_sig_x_tildes(),
            sig_v_tildes=state.iter_sig_v_tildes(),
            y_tildes=state.input.iter_y_tildes(),
            eta_v_tildes=state.input.iter_eta_v_tildes(),
            p_v_tildes=state.input.iter_p_v_tildes(),
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            sig_theta=state.sig_theta,
            sig_lambda=state.sig_lambda,
            g=state.input.g,
            f=state.input.f,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv)

# Part 1: Simulate some data
# generate data with a simple model that we can invert with DEM

## Define the model
# x' = f(x, v) + w
# y  = g(x, v) + z

## Simple case
# x' = Ax + v + w
# y  = Ix + z

x0 = np.array([0, 1])
A = np.array([[0, 1], [-1, 0]])

noise_sd = 0.1
noise_temporal_sig = 1 # temporal smoothing kernel parameter

# Simulate the data
# NOTE: Data are in shape (number of samples, number of features)
t_start = 0
t_end = 50
t_span = (t_start, t_end)
dt = 0.1
ts = np.arange(start=t_start, stop=t_end, step=dt)
n = int((t_end - t_start) / dt)

# how many terms x and v contain
m_x = 2
m_v = 2

# Noises
# NOTE: correlations between noise terms are 0 with this construction
seed = 546
rng = np.random.default_rng(seed)
ws = np.vstack([
        generate_noise_conv(n, dt, noise_sd ** 2, noise_temporal_sig, rng=rng)
        for _ in range(m_x)
    ]).T
zs = np.vstack([
        generate_noise_conv(n, dt, noise_sd ** 2, noise_temporal_sig, rng=rng)
        for _ in range(m_x)
    ]).T

# Running the system with noise
def f(t, x):
    # need to interpolate noise at this t separately for each feature
    noises = []
    for col in range(ws.shape[1]):
        noise_col = np.interp(t, ts, ws[:,col])
        noises.append(noise_col)
    return A @ x + np.array(noises)

out = solve_ivp(f, t_span, x0, t_eval=ts)

# System
xs = out.y.T
ys = xs + zs

vs = np.zeros((ys.shape[0], 2))
v_temporal_sig = 1

if plot:
    plt.plot(ts, xs[:, 0], label="x0", linestyle="-", color="red")
    plt.plot(ts, ys[:, 0], label="y0", linestyle="--", color="red")
    plt.plot(ts, xs[:, 1], label="x1", linestyle="-", color="purple")
    plt.plot(ts, ys[:, 1], label="y1", linestyle="--", color="purple")
    plt.legend()
    plt.show()

def dem_f(x, v, params):
    params = params.reshape((2,2))
    return params @ x + v

def dem_g(x, v, params):
    return x

# Part 2: Define a DEM model
p = 3
p_comp = 8


v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()))
v_autocorr_inv = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()))
noise_autocorr_inv = torch.linalg.inv(noise_autocorr)

omega_w = torch.eye(2)
omega_z = torch.eye(2)

dem_input = DEMInput(
    dt=dt,
    m_x=2,
    m_v=2, ### <- will always be 0 anyway, just putting it in here to make it simple
    p=p,
    ys=torch.tensor(ys),
    eta_v=torch.tensor(vs),
    p_v=torch.eye(2),
    v_autocorr_inv=v_autocorr_inv,
    eta_theta=torch.tensor([0, 0, 0, 0]),
    eta_lambda=torch.tensor([[0, 0]]),
    p_theta=torch.eye(4),
    p_lambda=torch.eye(2),
    g=dem_g,
    f=dem_f,
    omega_w=torch.eye(2),
    omega_z=torch.eye(2),
    noise_autocorr_inv=noise_autocorr_inv,
    x0=torch.tensor([0,0]),
    theta0=torch.tensor([0, 0, 0, 0]),
    lambda0=torch.tensor([0, 0])
        )

# ideal parameters and states
ideal_mu_x_tildes = list(iterate_generalized(xs, dt, p, p_comp=p_comp))
ideal_mu_v_tildes = list(repeat(torch.tensor([0,0] * p), len(ideal_mu_x_tildes)))
ideal_sig_x_tildes = list(repeat(torch.eye(m_x * p), len(ideal_mu_x_tildes))) # uhh this probably isn't the ideal
ideal_sig_v_tildes = list(repeat(torch.eye(m_x * p), len(ideal_mu_x_tildes))) # uhh this probably isn't the ideal

if plot:
    l = np.hstack(ideal_x_tildes).T
    plt.plot(l[:, 0::2])
    plt.show()

ideal_mu_theta = A
ideal_sig_theta = torch.eye(2) * 0.01

ideal_mu_lambda = np.log(0.1) * torch.tensor([1, 1]) # idk
ideal_sig_lambda = torch.eye(2) * 0.01

# a well-fitted model with hopefully low free action
dem_state = DEMState(
        input=dem_input,
        mu_x_tildes=ideal_mu_x_tildes,
        mu_v_tildes=ideal_mu_v_tildes,
        sig_x_tildes=ideal_sig_x_tildes,
        sig_v_tildes=ideal_sig_v_tildes,
        mu_theta=ideal_mu_theta,
        mu_lambda=ideal_mu_lambda,
        sig_theta=ideal_sig_theta,
        sig_lambda=ideal_sig_lambda)

f_bar = free_action_from_state(dem_state)
