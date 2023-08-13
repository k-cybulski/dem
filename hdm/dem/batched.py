"""
This module contains an implementation of DEM which uses vectorized batch
operations instead of iterators when possible.
"""
from dataclasses import dataclass, field, replace
from copy import copy
from typing import Callable, Iterable
from itertools import repeat
from time import time

import torch

from ..core import deriv_mat, iterate_generalized
from ..noise import generate_noise_conv, autocorr_friston, noise_cov_gen_theoretical
from .util import kron, _fix_grad_shape, _fix_grad_shape_batch
from .naive import internal_energy_static, clear_gradients_on_state

def generalized_func(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    assert mu_x_tildes.shape[1:] == (m_x * (p + 1), 1)
    assert mu_v_tildes.shape[1:] == (m_v * (p + 1), 1)
    assert mu_x_tildes.shape[0] == mu_v_tildes.shape[0]
    mu_x = mu_x_tildes[:, :m_x]
    mu_v = mu_v_tildes[:, :m_v]
    n_batch = mu_x_tildes.shape[0]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad = torch.autograd.functional.jacobian(lambda x: func(x, mu_v, params), mu_x, create_graph=True)
    mu_v_grad = torch.autograd.functional.jacobian(lambda v: func(mu_x, v, params), mu_v, create_graph=True)
    mu_x_grad = _fix_grad_shape_batch(mu_x_grad)
    mu_v_grad = _fix_grad_shape_batch(mu_v_grad)

    mu_x_tildes_r = mu_x_tildes.reshape((n_batch, p + 1, m_x))
    mu_v_tildes_r = mu_v_tildes.reshape((n_batch, p + 1, m_v))

    func_appl_d_x = torch.einsum('bkj,bdj->bdk', mu_x_grad, mu_x_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    func_appl_d_v = torch.einsum('bkj,bdj->bdk', mu_v_grad, mu_v_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    return torch.concat((func_appl, func_appl_d_x + func_appl_d_v), dim=1)


def internal_energy_dynamic(
        g, f, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)

    def _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda):
        mu_theta = mu_theta
        mu_lambda = mu_lambda
        g_tildes = generalized_func(g, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)
        f_tildes = generalized_func(f, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)

        err_y = y_tildes - g_tildes
        err_v = mu_v_tildes - eta_v_tildes
        err_x = torch.matmul(deriv_mat_x, mu_x_tildes) - f_tildes

        n_batch = mu_x_tildes.shape[0]

        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -torch.bmm(err_y.mT, torch.matmul(prec_z_tilde, err_y)).reshape(n_batch) + torch.logdet(prec_z_tilde)
        u_t_v_ = -torch.bmm(err_v.mT, torch.bmm(p_v_tildes, err_v)).reshape(n_batch) + torch.logdet(p_v_tildes)
        u_t_x_ = -torch.bmm(err_x.mT, torch.matmul(prec_w_tilde, err_x)).reshape(n_batch) + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t

    u_t = _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
    # Note: Calling torch.autograd.functional.hessian four times is faster than
    # just doing it in one go
    u_t_x_tilde_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu, mu_v_tildes, mu_theta, mu_lambda)), mu_x_tildes, create_graph=True)
    u_t_v_tilde_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu_x_tildes, mu, mu_theta, mu_lambda)), mu_v_tildes, create_graph=True)
    u_t_theta_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu, mu_lambda)), mu_theta, create_graph=True)
    u_t_lambda_dd = torch.autograd.functional.hessian(lambda mu: torch.sum(_int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu)), mu_lambda, create_graph=True)

    u_t_x_tilde_dd = _fix_grad_shape_batch(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape_batch(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd)
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd

def _batch_diag(tensor):
    # As mentioned here:
    # https://discuss.pytorch.org/t/get-the-trace-for-a-batch-of-matrices/108504/2
    return tensor.diagonal(offset=0, dim1=-1, dim2=-2)

def free_action(
        # how many terms are there in mu_x and mu_v?
        m_x, m_v,

        # how many derivatives are we tracking in generalised vectors?
        p,

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

        # functions
        g, # output function
        f, # state transition function

        # noise precision matrices (to be scaled by hyperparameters)
        omega_w,
        omega_z,

        # generalized noise temporal autocorrelation inverse (precision)
        noise_autocorr_inv,

        # It's ok to skip the constant step if computing free action gradients
        # only on dynamic terms
        skip_constant=False
        ):
    # Constant terms of free action
    if not skip_constant:
        u_c, u_c_theta_dd, u_c_lambda_dd = internal_energy_static(
            mu_theta,
            mu_lambda,
            eta_theta,
            eta_lambda,
            p_theta,
            p_lambda,
            compute_dds=True
        )
        f_c = u_c + (torch.logdet(sig_theta) + torch.logdet(sig_lambda)) / 2
    else:
        f_c = 0

    # Dynamic terms of free action that vary with time
    u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd = internal_energy_dynamic(
        g, f, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv)
    w_x_tilde_sum_ = _batch_diag(torch.bmm(sig_x_tildes, u_t_x_tilde_dd)).sum()
    w_v_tilde_sum_ = _batch_diag(torch.bmm(sig_v_tildes, u_t_v_tilde_dd)).sum()
    # w_theta and w_lambda are sums already, because u_t_theta_dd is a sum
    # because of how the batch Hessian is computed
    if not skip_constant:
        w_theta_sum_ = torch.trace(sig_theta @ (u_c_theta_dd + u_t_theta_dd))
        w_lambda_sum_ = torch.trace(sig_lambda @ (u_c_lambda_dd + u_t_lambda_dd))
    else:
        w_theta_sum_ = torch.trace(sig_theta @ (u_t_theta_dd))
        w_lambda_sum_ = torch.trace(sig_lambda @ (u_t_lambda_dd))

    f_tsum = torch.sum(u_t) \
            + (torch.sum(torch.logdet(sig_x_tildes)) + torch.sum(torch.logdet(sig_v_tildes))) / 2 \
            + (w_x_tilde_sum_ + w_v_tilde_sum_ + w_theta_sum_ + w_lambda_sum_) / 2

    f_bar = f_c + f_tsum
    return f_bar


@dataclass
class DEMInput:
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
    p: int
    p_comp: int # can be equal to p or greater

    # system output
    ys: torch.Tensor

    # prior on system input
    eta_v: torch.Tensor # input sequence (will be transformed to generalized coords by Taylor trick)
    p_v: torch.Tensor # precision (TODO: Should this accept a sequence of precisions?)
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

    # Precomputed values
    y_tildes: torch.Tensor = None
    eta_v_tildes: torch.Tensor = None
    p_v_tildes: torch.Tensor = None

    def __post_init__(self):
        if self.ys.ndim == 1:
            self.ys = self.ys.reshape((-1, 1))
        self.n = self.ys.shape[0]
        if self.p_comp is None:
            self.p_comp = self.p
        # Precomputed values
        if self.y_tildes is None:
            self.y_tildes = torch.stack(list(iterate_generalized(self.ys, self.dt, self.p, p_comp=self.p_comp)))
        if self.eta_v_tildes is None:
            self.eta_v_tildes = torch.stack(list(iterate_generalized(self.eta_v, self.dt, self.p, p_comp=self.p_comp)))
        if self.p_v_tildes is None:
            p_v_tilde = kron(self.v_autocorr_inv, self.p_v)
            self.p_v_tildes = p_v_tilde.expand(len(self.eta_v_tildes), *p_v_tilde.shape)

@dataclass
class DEMState:
    """
    Keeps track of the current state of a DEM model. Contains its state,
    parameter, and hyperparameter estimates.
    """
    # system input
    input: DEMInput

    # dynamic state estimates
    mu_x_tildes: torch.Tensor
    mu_v_tildes: torch.Tensor
    sig_x_tildes: torch.Tensor
    sig_v_tildes: torch.Tensor

    # static parameter and hyperparameter estimates
    mu_theta: torch.Tensor
    mu_lambda: torch.Tensor
    sig_theta: torch.Tensor
    sig_lambda: torch.Tensor

    # initial dynamic states
    mu_x0_tilde: torch.Tensor
    mu_v0_tilde: torch.Tensor

    def free_action(self, skip_constant=False):
        state = self
        return free_action(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
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
            g=state.input.g,
            f=state.input.f,
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv,
            skip_constant=skip_constant
        )

    @classmethod
    def from_input(cls, input: DEMInput, x0: torch.Tensor):
        x0 = x0.reshape(-1)
        assert len(x0) == input.m_x

        mu_x0_tilde = torch.concat([x0, torch.zeros(input.p * input.m_x)]).reshape((-1, 1))
        mu_v_tildes = torch.stack(list(iterate_generalized(input.eta_v, input.dt, input.p, p_comp=input.p_comp)))

        return cls(
                input=input,
                mu_x_tildes=mu_x0_tilde[None],
                mu_v_tildes=mu_v_tildes,
                sig_x_tildes=torch.eye(x0.shape[0] * (input.p + 1)).repeat(len(mu_v_tildes), 1, 1), # TODO: What is a good value here?
                sig_v_tildes=torch.linalg.inv(input.p_v_tildes.clone().detach()),
                mu_theta=input.eta_theta.clone().detach(),
                mu_lambda=input.eta_lambda.clone().detach(),
                sig_theta=torch.linalg.inv(input.p_theta.clone().detach()),
                sig_lambda=torch.linalg.inv(input.p_lambda.clone().detach()),
                mu_x0_tilde=mu_x0_tilde,
                mu_v0_tilde=mu_v_tildes[0])


def dem_step_d(state: DEMState, lr):
    """
    Performs the D step of DEM.
    """
    # FIXME: How should we optimize x0 and v0?
    mu_x_tilde_t = state.mu_x0_tilde.detach()
    mu_v_tilde_t = state.mu_v0_tilde.detach()
    mu_x_tildes = [mu_x_tilde_t]
    mu_v_tildes = [mu_v_tilde_t]
    for t, (y_tilde,
         sig_x_tilde, sig_v_tilde,
         eta_v_tilde, p_v_tilde) in enumerate(zip(
                            state.input.y_tildes,
                            state.sig_x_tildes, state.sig_v_tildes,
                            state.input.eta_v_tildes,
                            state.input.p_v_tildes,
                            strict=True)):
        def dynamic_free_energy(mu_x_tilde_t, mu_v_tilde_t):
            # free action as a function of dynamic terms
            # NOTE: We can't just use free_action_from_state(replace(state, ...))
            # because we cannot easily override y_tildes, which comes
            # dynamically generated from state.input.iter_y_tildes()
            # Maybe we could precompute it?
            return free_action(
                m_x=state.input.m_x,
                m_v=state.input.m_v,
                p=state.input.p,
                mu_x_tildes=mu_x_tilde_t[None],
                mu_v_tildes=mu_v_tilde_t[None],
                sig_x_tildes=sig_x_tilde[None],
                sig_v_tildes=sig_v_tilde[None],
                y_tildes=y_tilde[None],
                eta_v_tildes=eta_v_tilde[None],
                p_v_tildes=p_v_tilde[None],
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
                noise_autocorr_inv=state.input.noise_autocorr_inv,
                skip_constant=True)
        # note that until the procedure is finished, these are a list and not a tensor
        state.mu_x_tildes = mu_x_tildes
        state.mu_v_tildes = mu_v_tildes
        # free action on just a single timestep
        mu_x_tilde_t = mu_x_tilde_t.clone().detach().requires_grad_()
        mu_v_tilde_t = mu_v_tilde_t.clone().detach().requires_grad_()
        f_eng = dynamic_free_energy(mu_x_tilde_t, mu_v_tilde_t)
        # NOTE: In the original pseudocode, x and v are in one vector
        x_d = lr * torch.autograd.grad(f_eng, mu_x_tilde_t, retain_graph=True)[0]
        v_d = lr * torch.autograd.grad(f_eng, mu_v_tilde_t)[0]
        x_dd = lr * torch.autograd.functional.hessian(lambda mu: dynamic_free_energy(mu, mu_v_tilde_t), mu_x_tilde_t)
        v_dd = lr * torch.autograd.functional.hessian(lambda mu: dynamic_free_energy(mu_x_tilde_t, mu), mu_v_tilde_t)
        x_dd = _fix_grad_shape(x_dd)
        v_dd = _fix_grad_shape(v_dd)
        step_matrix_x = (torch.matrix_exp(x_dd) - torch.eye(x_dd.shape[0])) @ torch.linalg.inv(x_dd)
        step_matrix_v = (torch.matrix_exp(v_dd) - torch.eye(v_dd.shape[0])) @ torch.linalg.inv(v_dd)
        mu_x_tilde_t = mu_x_tilde_t + step_matrix_x @ x_d
        mu_v_tilde_t = mu_v_tilde_t + step_matrix_v @ v_d
        mu_x_tilde_t = mu_x_tilde_t.clone().detach()
        mu_v_tilde_t = mu_v_tilde_t.clone().detach()
        mu_x_tildes.append(mu_x_tilde_t)
        mu_v_tildes.append(mu_v_tilde_t)
    mu_x_tildes = mu_x_tildes[:-1] # there is one too many
    mu_v_tildes = mu_v_tildes[:-1]
    state.mu_x_tildes = torch.stack(mu_x_tildes)
    state.mu_v_tildes = torch.stack(mu_v_tildes)


def dem_step_precision(state: DEMState):
    """
    Does a precision update of DEM.
    """
    clear_gradients_on_state(state)
    u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds = internal_action_from_state(state)
    state.sig_theta = torch.linalg.inv(-u_theta_dd)
    state.sig_lambda = torch.linalg.inv(-u_lambda_dd)
    state.sig_x_tildes = torch.stack([-torch.linalg.inv(u_t_x_tilde_dd) for u_t_x_tilde_dd in u_t_x_tilde_dds])
    state.sig_v_tildes = torch.stack([-torch.linalg.inv(u_t_v_tilde_dd) for u_t_v_tilde_dd in u_t_v_tilde_dds])


def dem_step_m(state: DEMState, lr_lambda, iter_lambda, min_improv):
    """
    Performs the noise hyperparameter update (step M) of DEM.
    """
    # FIXME: Do 'until convergence' rather than 'for some fixed number of steps'
    last_f_bar = None
    for i in range(iter_lambda):
        def lambda_free_action(mu_lambda):
            # free action as a function of lambda
            return replace(state, mu_lambda=mu_lambda).free_action()
        clear_gradients_on_state(state)
        f_bar = state.free_action()
        lambda_d = lr_lambda * torch.autograd.grad(f_bar, state.mu_lambda)[0]
        lambda_dd = lr_lambda * torch.autograd.functional.hessian(lambda_free_action, state.mu_lambda)
        step_matrix = (torch.matrix_exp(lambda_dd) - torch.eye(lambda_dd.shape[0])) @ torch.linalg.inv(lambda_dd)
        state.mu_lambda = state.mu_lambda + step_matrix @ lambda_d
        # convergence check
        if last_f_bar is not None:
            if last_f_bar + min_improv > f_bar:
                break
        last_f_bar = f_bar.clone().detach()


def dem_step_e(state: DEMState, lr_theta):
    """
    Performs the parameter update (step E) of DEM.
    """
    # TODO: should be an if statement comparing new f_bar with old
    def theta_free_action(mu_theta):
        # free action as a function of theta
        return replace(state, mu_theta=mu_theta).free_action()
    clear_gradients_on_state(state)
    f_bar = state.free_action()
    theta_d = lr_theta * torch.autograd.grad(f_bar, state.mu_theta)[0]
    theta_dd = lr_theta * torch.autograd.functional.hessian(theta_free_action, state.mu_theta)
    step_matrix = (torch.matrix_exp(theta_dd) - torch.eye(theta_dd.shape[0])) @ torch.linalg.inv(theta_dd)
    state.mu_theta = state.mu_theta + step_matrix @ theta_d


def dem_step(state: DEMState, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=0.01):
    """
    Does an iteration of DEM.
    """
    dem_step_d(state, lr_dynamic)
    dem_step_m(state, lr_lambda, iter_lambda, min_improv=m_min_improv)
    dem_step_e(state, lr_theta)
    dem_step_precision(state)