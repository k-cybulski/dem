from dataclasses import dataclass, field, replace
from copy import copy
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

    # It seems that if the parameters are a (n, 1) matrix, then the Hessian has
    # 4 dimensions (n, 1, n, 1)
    # if the parameters are a (n,) array, then the Hessian has 2 dimensions (n, n)
    if tensor.dim() == 2:
        return tensor
    elif tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[3] == 1:
        return tensor.reshape((tensor.shape[0], tensor.shape[2]))
    else:
        raise ValueError("Unexpected hessian shape")

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
    # TODO: Ensure correctness of shapes, e.g. shape (2, 1) instead of (2,). This causes lots of errors down the road.
    assert mu_x_tilde.shape == (m_x * (p + 1), 1)
    assert mu_v_tilde.shape == (m_v * (p + 1), 1)
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad, mu_v_grad = torch.autograd.functional.jacobian(lambda x, v: func(x, v, params), (mu_x, mu_v), create_graph=True)
    mu_x_grad = _fix_grad_shape(mu_x_grad)
    mu_v_grad = _fix_grad_shape(mu_v_grad)
    func_appl_d = []
    for deriv in range(1, p + 1):
        mu_x_d = mu_x_tilde[deriv*m_x:(deriv+1)*m_x]
        mu_v_d = mu_v_tilde[deriv*m_v:(deriv+1)*m_v]
        func_appl_d.append(mu_x_grad @ mu_x_d + mu_v_grad @ mu_v_d)
    return torch.vstack([func_appl] + func_appl_d)

def _int_eng_par_static(
        mu,
        eta,
        p
    ):
    err = mu - eta
    return (-err.T @ p @ err + torch.logdet(p)) / 2

def internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,
        # compute hessians used for mean-field terms in free action
        compute_dds: bool
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

    # FIXME OPT: Don't run torch.autograd.functional.hessian twice.
    if compute_dds:
        u_c_theta_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_par_static(mu, eta_theta, p_theta), mu_theta, create_graph=True)
        u_c_lambda_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_par_static(mu, eta_lambda, p_lambda), mu_lambda, create_graph=True)
        return u_c, u_c_theta_dd, u_c_lambda_dd
    else:
        return u_c

def internal_energy_dynamic(
        g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv, compute_dds):
    """
    Computes dynamic terms of the internal energy for a single timestep, along
    with necessary Hessians. These are the precision-weighted errors and
    precision log determinants on the dynamic states. Hessians are returned for
    parameters theta and hyperparameters lambda as well.
    """
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)
    # make a temporary function which we can use to compute hessians w.r.t. the relevant parameters
    # for the computation of mean-field terms
    def _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda):
        g_tilde = generalized_func(g, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)
        f_tilde = generalized_func(f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)

        err_y = y_tilde - g_tilde
        err_v = mu_v_tilde - eta_v_tilde
        err_x = deriv_mat_x @ mu_x_tilde - f_tilde

        # we need to split up mu_lambda into the hyperparameter for noise of states and of outputs
        # the hyperparameters are just a single lambda scalar, one for the states and one for the outputs
        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -err_y.T @ prec_z_tilde @ err_y + torch.logdet(prec_z_tilde)
        u_t_v_ = -err_v.T @ p_v_tilde @ err_v + torch.logdet(p_v_tilde)
        u_t_x_ = -err_x.T @ prec_w_tilde @ err_x + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t
    u_t = _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda)
    # horribly inefficient way to go about this, but hey, at least it may work...
    # (so many unnecessary repeated computations)

    # FIXME OPT: Optimize the code below. Don't run
    # torch.autograd.functional.hessian four times separately? Running it once
    # should allow for all the necessary outputs. But it might unnecessarily
    # compute Hessians _between_ the parameters, which might be slower?
    if compute_dds:
        u_t_x_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu, mu_v_tilde, mu_theta, mu_lambda), mu_x_tilde, create_graph=True)
        u_t_v_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu, mu_theta, mu_lambda), mu_v_tilde, create_graph=True)
        u_t_theta_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu, mu_lambda), mu_theta, create_graph=True)
        u_t_lambda_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu), mu_lambda, create_graph=True)

        u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
        u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
        u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd )
        u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)

        return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd
    else:
        return u_t

def internal_action(
        # for static internal energy
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,

        # for dynamic internal energies
        g, f,
        m_x, m_v, p,

        mu_x_tildes, mu_v_tildes,
        sig_x_tildes, sig_v_tildes,
        y_tildes,
        eta_v_tildes, p_v_tildes,
        omega_w, omega_z, noise_autocorr_inv
        ):
    """
    Computes internal energy/action, and Hessians. Used to update precisions at the end of a
    DEM iteration.
    """
    u, u_theta_dd, u_lambda_dd = internal_energy_static(
        mu_theta,
        mu_lambda,
        eta_theta,
        eta_lambda,
        p_theta,
        p_lambda,
        compute_dds=True
    )
    u_t_x_tilde_dds = []
    u_t_v_tilde_dds = []
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
            g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
            mu_lambda, omega_w, omega_z, noise_autocorr_inv, compute_dds=True)
        u_theta_dd += u_t_theta_dd
        u_lambda_dd += u_t_lambda_dd
        u += u_t.item()
        u_t_x_tilde_dds.append(u_t_x_tilde_dd)
        u_t_v_tilde_dds.append(u_t_v_tilde_dd)
    return u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds

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
        ):
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
    f_tsum = 0
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
            g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
            mu_lambda, omega_w, omega_z, noise_autocorr_inv, compute_dds=True)

        # mean-field terms
        # FIXME OPT: Section 11.1 of Anil Meera & Wisse shows that gradients
        # along w_lambda are 0, so it might be unnecessary to compute for
        # parameter updates.
        w_x_tilde, w_v_tilde, w_theta, w_lambda = [
            torch.trace(sig @ (u_c_dd + u_t_dd)) / 2
                for (sig, u_c_dd, u_t_dd) in [
                    (sig_x_tilde, 0, u_t_x_tilde_dd),
                    (sig_v_tilde, 0, u_t_v_tilde_dd),
                    (sig_theta, u_c_theta_dd, u_t_theta_dd),
                    (sig_lambda, u_c_lambda_dd, u_t_lambda_dd),
                ]
            ]
        f_tsum += u_t \
                + (torch.logdet(sig_x_tilde) + torch.logdet(sig_v_tilde)) / 2 \
                + w_x_tilde + w_v_tilde + w_theta + w_lambda
    f_bar = f_c + f_tsum
    return f_bar

@dataclass
class DEMInput:
    """
    The input to DEM. It consists of data, priors, starting values, and
    transition functions, so of all the things which remain fixed over
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
    p_v: torch.Tensor # precision
    v_autocorr_inv: torch.Tensor # generalised input vector precision
    # TODO: How do we construct p_v_tildes for all times?
    # For now let's use kron(S, P_v) for a fixed precision prior P_v over all times

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

    def __post_init__(self):
        if self.ys.ndim == 1:
            self.ys = self.ys.reshape((-1, 1))
        self.n = self.ys.shape[0]
        if self.p_comp is None:
            self.p_comp = self.p
        # FIXME: Do it more efficiently! It's easy
        self._iter_length = len(list(self.iter_y_tildes()))

    def iter_y_tildes(self):
        return iterate_generalized(self.ys, self.dt, self.p, p_comp=self.p_comp)

    def iter_eta_v_tildes(self):
        return iterate_generalized(self.eta_v, self.dt, self.p, p_comp=self.p_comp)

    def iter_p_v_tildes(self):
        p_v_tilde = kron(self.v_autocorr_inv, self.p_v)
        return repeat(p_v_tilde, self._iter_length)


@dataclass
class DEMState:
    """
    Keeps track of the current state of a DEM model. Contains its state,
    parameter, and hyperparameter estimates.
    """
    # system input
    input: DEMInput

    # dynamic state estimates
    # FIXME (OPT): These should be put in a single tensor for batched operations
    mu_x_tildes: list[torch.Tensor]
    mu_v_tildes: list[torch.Tensor]
    sig_x_tildes: list[torch.Tensor]
    sig_v_tildes: list[torch.Tensor]

    # static parameter and hyperparameter estimates
    mu_theta: torch.Tensor
    mu_lambda: torch.Tensor
    sig_theta: torch.Tensor
    sig_lambda: torch.Tensor

    # initial dynamic states
    mu_x0_tilde: torch.Tensor
    mu_v0_tilde: torch.Tensor


def free_action_from_state(state: DEMState):
    return free_action(
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,
            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
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

def internal_action_from_state(state: DEMState):
    return internal_action(
            # for static internal energy
            mu_theta=state.mu_theta,
            mu_lambda=state.mu_lambda,
            eta_theta=state.input.eta_theta,
            eta_lambda=state.input.eta_lambda,
            p_theta=state.input.p_theta,
            p_lambda=state.input.p_lambda,

            # for dynamic internal energies
            g=state.input.g,
            f=state.input.f,
            m_x=state.input.m_x,
            m_v=state.input.m_v,
            p=state.input.p,

            mu_x_tildes=state.mu_x_tildes,
            mu_v_tildes=state.mu_v_tildes,
            sig_x_tildes=state.sig_x_tildes,
            sig_v_tildes=state.sig_v_tildes,
            y_tildes=state.input.iter_y_tildes(),
            eta_v_tildes=state.input.iter_eta_v_tildes(),
            p_v_tildes=state.input.iter_p_v_tildes(),
            omega_w=state.input.omega_w,
            omega_z=state.input.omega_z,
            noise_autocorr_inv=state.input.noise_autocorr_inv)


# Functions for performing DEM optimization
def clear_gradients_on_state(state: DEMState):
    state.mu_theta = state.mu_theta.detach().clone().requires_grad_()
    state.mu_lambda = state.mu_lambda.detach().clone().requires_grad_()
    state.mu_x0_tilde = state.mu_x0_tilde.detach().clone().requires_grad_()
    state.mu_v0_tilde = state.mu_v0_tilde.detach().clone().requires_grad_()

def dem_step_d(state: DEMState, lr):
    """
    Performs the D step of DEM.
    """
    # FIXME: How should we optimize x0 and v0?
    mu_x_tilde_t = state.mu_x0_tilde.clone().detach()
    mu_v_tilde_t = state.mu_v0_tilde.clone().detach()
    mu_x_tildes = [mu_x_tilde_t]
    mu_v_tildes = [mu_v_tilde_t]
    for t, (y_tilde,
         sig_x_tilde, sig_v_tilde,
         eta_v_tilde, p_v_tilde) in enumerate(zip(
                            state.input.iter_y_tildes(),
                            state.sig_x_tildes, state.sig_v_tildes,
                            state.input.iter_eta_v_tildes(),
                            state.input.iter_p_v_tildes(),
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
                mu_x_tildes=(mu_x_tilde_t,),
                mu_v_tildes=(mu_v_tilde_t,),
                sig_x_tildes=(sig_x_tilde,),
                sig_v_tildes=(sig_v_tilde,),
                y_tildes=(y_tilde,),
                eta_v_tildes=(eta_v_tilde,),
                p_v_tildes=(p_v_tilde,),
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
    state.mu_x_tildes = mu_x_tildes
    state.mu_v_tildes = mu_v_tildes


def dem_step_m(state: DEMState, lr_lambda, iter_lambda, min_improv):
    """
    Performs the noise hyperparameter update (step M) of DEM.
    """
    last_f_bar = None
    for i in range(iter_lambda):
        def lambda_free_action(mu_lambda):
            # free action as a function of lambda
            return free_action_from_state(replace(state, mu_lambda=mu_lambda))
        clear_gradients_on_state(state)
        f_bar = free_action_from_state(state)
        print(f"  (M) {i}. f_bar={f_bar.item():.4f}, lambda={state.mu_lambda}")
        lambda_d = lr_lambda * torch.autograd.grad(f_bar, state.mu_lambda)[0]
        lambda_dd = lr_lambda * torch.autograd.functional.hessian(lambda_free_action, state.mu_lambda)
        step_matrix = (torch.matrix_exp(lambda_dd) - torch.eye(lambda_dd.shape[0])) @ torch.linalg.inv(lambda_dd)
        state.mu_lambda = state.mu_lambda + step_matrix @ lambda_d
        # convergence check
        if last_f_bar is not None:
            if last_f_bar + min_improv > f_bar:
                print("  (M) reached improvement threshold")
                break
        last_f_bar = f_bar.clone().detach()


def dem_step_e(state: DEMState, lr_theta):
    """
    Performs the parameter update (step E) of DEM.
    """
    # TODO: should be an if statement comparing new f_bar with old
    def theta_free_action(mu_theta):
        # free action as a function of theta
        return free_action_from_state(replace(state, mu_theta=mu_theta))
    clear_gradients_on_state(state)
    f_bar = free_action_from_state(state)
    theta_d = lr_theta * torch.autograd.grad(f_bar, state.mu_theta)[0]
    theta_dd = lr_theta * torch.autograd.functional.hessian(theta_free_action, state.mu_theta)
    step_matrix = (torch.matrix_exp(theta_dd) - torch.eye(theta_dd.shape[0])) @ torch.linalg.inv(theta_dd)
    state.mu_theta = state.mu_theta + step_matrix @ theta_d


def dem_step_ex0(state: DEMState, lr_theta):
    """
    Performs the parameter update (step E) of DEM together with an update of v0
    and x0 (not in the original algorithm).
    """
    # TODO: should be an if statement comparing new f_bar with old
    def param_free_action(mu_theta, mu_x0_tilde, mu_v0_tilde):
        mu_x_tildes = copy(state.mu_x_tildes)
        mu_v_tildes = copy(state.mu_v_tildes)
        mu_x_tildes[0] = mu_x0_tilde
        mu_v_tildes[0] = mu_v0_tilde
        # free action as a function of theta and initial state
        return free_action_from_state(replace(state,
                                              mu_theta=mu_theta,
                                              mu_x_tildes=mu_x_tildes,
                                              mu_v_tildes=mu_v_tildes))
    clear_gradients_on_state(state)
    mu_x0_tilde = state.mu_x0_tilde.clone().detach().requires_grad_()
    mu_v0_tilde = state.mu_v0_tilde.clone().detach().requires_grad_()
    state.mu_x_tildes[0] = mu_x0_tilde
    state.mu_v_tildes[0] = mu_v0_tilde
    f_bar = free_action_from_state(state)

    theta_d = lr_theta * torch.autograd.grad(f_bar, state.mu_theta, retain_graph=True)[0]
    out = torch.autograd.functional.hessian(param_free_action, (state.mu_theta, mu_x0_tilde, mu_v0_tilde))
    theta_dd = lr_theta * out[0][0]
    x0_tilde_dd = lr_theta * out[1][1]
    v0_tilde_dd = lr_theta * out[2][2]
    step_matrix_theta = (torch.matrix_exp(theta_dd) - torch.eye(theta_dd.shape[0])) @ torch.linalg.inv(theta_dd)

    x0_tilde_dd = _fix_grad_shape(x0_tilde_dd)
    x0_tilde_d = lr_theta * torch.autograd.grad(f_bar, mu_x0_tilde, retain_graph=True)[0]
    x0_tilde_d = _fix_grad_shape(x0_tilde_d)
    step_matrix_x0_tilde = (torch.matrix_exp(x0_tilde_dd) - torch.eye(x0_tilde_dd.shape[0])) @ torch.linalg.inv(x0_tilde_dd)

    v0_tilde_dd = _fix_grad_shape(v0_tilde_dd)
    v0_tilde_d = lr_theta * torch.autograd.grad(f_bar, mu_v0_tilde)[0]
    v0_tilde_d = _fix_grad_shape(v0_tilde_d)
    step_matrix_v0_tilde = (torch.matrix_exp(v0_tilde_dd) - torch.eye(v0_tilde_dd.shape[0])) @ torch.linalg.inv(v0_tilde_dd)

    state.mu_theta = state.mu_theta + step_matrix_theta @ theta_d
    state.mu_x0_tilde = state.mu_x0_tilde + step_matrix_x0_tilde @ x0_tilde_d
    state.mu_v0_tilde = state.mu_v0_tilde + step_matrix_v0_tilde @ v0_tilde_d


def dem_step_precision(state: DEMState):
    """
    Does a precision update of DEM.
    """
    clear_gradients_on_state(state)
    u, u_theta_dd, u_lambda_dd, u_t_x_tilde_dds, u_t_v_tilde_dds = internal_action_from_state(state)
    state.sig_theta = torch.linalg.inv(-u_theta_dd)
    state.sig_lambda = torch.linalg.inv(-u_lambda_dd)
    state.sig_x_tildes = [-torch.linalg.inv(u_t_x_tilde_dd) for u_t_x_tilde_dd in u_t_x_tilde_dds]
    state.sig_v_tildes = [-torch.linalg.inv(u_t_v_tilde_dd) for u_t_v_tilde_dd in u_t_v_tilde_dds]


def dem_step(state: DEMState, lr_dynamic, lr_theta, lr_lambda, iter_lambda, min_improv):
    """
    Does an iteration of DEM.
    """
    dem_step_d(state, lr_dynamic)
    dem_step_m(state, lr_lambda, iter_lambda, min_improv)
    dem_step_e(state, lr_theta)
    dem_step_precision(state)

# Part 1: Simulate some data
# generate data with a simple model that we can invert with DEM

## Define the model
# x' = f(x, v) + w
# y  = g(x, v) + z

## Simple case
# x' = Ax + w
# y  = Ix + z

x0 = np.array([0, 1])
A = np.array([[0, 1], [-1, 0]])

# noise standard deviations
w_sd = 0.01 # noise on states
z_sd = 0.05 # noise on outputs
noise_temporal_sig = 0.15 # temporal smoothing kernel parameter

# Simulate the data
# NOTE: Data are in shape (number of samples, number of features)
t_start = 0
t_end = 2
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
        generate_noise_conv(n, dt, w_sd ** 2, noise_temporal_sig, rng=rng)
        for _ in range(m_x)
    ]).T
zs = np.vstack([
        generate_noise_conv(n, dt, z_sd ** 2, noise_temporal_sig, rng=rng)
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
xs = torch.tensor(out.y.T, dtype=torch.float32)
ys = torch.tensor(xs + zs, dtype=torch.float32)

vs = torch.zeros((ys.shape[0], 2), dtype=torch.float32)
v_temporal_sig = 1

if plot:
    plt.plot(ts, xs[:, 0], label="x0", linestyle="-", color="red")
    plt.plot(ts, ys[:, 0], label="y0", linestyle="--", color="red")
    plt.plot(ts, xs[:, 1], label="x1", linestyle="-", color="purple")
    plt.plot(ts, ys[:, 1], label="y1", linestyle="--", color="purple")
    plt.legend()
    plt.show()

# The model is
# x' = Ax + w
# y  = Ix + z

def dem_f(x, v, params):
    params = params.reshape((2,2))
    return params @ x

def dem_g(x, v, params):
    return x

# Part 2: Define a DEM model
p = 4
p_comp = p


v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

omega_w = torch.eye(2)
omega_z = torch.eye(2)

dem_input = DEMInput(
    dt=dt,
    m_x=2,
    m_v=2, ### <- will always be 0 anyway, just putting it in here to make it simple
    m_y=2,
    p=p,
    p_comp=p_comp,
    ys=ys,
    eta_v=vs,
    p_v=torch.eye(2),
    v_autocorr_inv=v_autocorr_inv_,
    eta_theta=torch.tensor([0, 0, 0, 0], dtype=torch.float32),
    eta_lambda=torch.tensor([0, 0], dtype=torch.float32),
    p_theta=torch.eye(4, dtype=torch.float32),
    p_lambda=torch.eye(2, dtype=torch.float32),
    g=dem_g,
    f=dem_f,
    omega_w=torch.eye(2, dtype=torch.float32),
    omega_z=torch.eye(2, dtype=torch.float32),
    noise_autocorr_inv=noise_autocorr_inv_,
        )

# ideal states
## used during development as an "oracle" for what the states ought to be
ideal_mu_x_tildes = list(iterate_generalized(xs, dt, p, p_comp=p_comp))
ideal_mu_v_tildes = list(repeat(torch.zeros((2 * (p + 1), 1), dtype=torch.float32), len(ideal_mu_x_tildes)))
ideal_sig_x_tildes = list(repeat(torch.eye(m_x * (p + 1)), len(ideal_mu_x_tildes))) # uhh this probably isn't the ideal
ideal_sig_v_tildes = list(repeat(torch.eye(m_v * (p + 1)), len(ideal_mu_x_tildes))) # uhh this probably isn't the ideal

if plot:
    l = np.hstack(ideal_mu_x_tildes).T
    plt.plot(l[:, 0::2])
    plt.show()

ideal_mu_x0_tilde = ideal_mu_x_tildes[0].clone()
ideal_mu_v0_tilde = ideal_mu_v_tildes[0].clone()


# Starting parameters/hyperparameters, to see how well we can find the true ones
mu_lambda0 = torch.tensor([0, 0], dtype=torch.float32)
sig_lambda0 = torch.eye(2) * 0.01

mu_theta0 = torch.tensor([0,0,0,0], dtype=torch.float32)
sig_theta0 = torch.eye(4) * 0.01

sig_x_tilde0 = torch.eye(m_x * (p + 1))
sig_v_tilde0 = torch.eye(m_v * (p + 1))

def clean_state():
    """
    A DEMState initialized with xs and vs at zero.
    """
    dem_state = DEMState(
            input=dem_input,
            mu_x_tildes=[torch.zeros_like(mu_x_tilde) for mu_x_tilde in ideal_mu_x_tildes],
            mu_v_tildes=[torch.zeros_like(mu_v_tilde) for mu_v_tilde in ideal_mu_v_tildes],
            sig_x_tildes=[sig_x_tilde0.clone().detach() for _ in ideal_mu_x_tildes],
            sig_v_tildes=[sig_v_tilde0.clone().detach() for _ in ideal_mu_v_tildes],
            mu_theta=mu_theta0.clone().detach(),
            mu_lambda=mu_lambda0.clone().detach(),
            sig_theta=sig_theta0.clone().detach(),
            sig_lambda=sig_lambda0.clone().detach(),
            mu_x0_tilde=torch.zeros_like(ideal_mu_x0_tilde),
            mu_v0_tilde=torch.zeros_like(ideal_mu_v0_tilde))
    return dem_state

# Do a single step to find optimal sigmas
# dem_static_step(dem_state, lr_theta=0.01, lr_lambda0=0.01, iter_lambda=1)

def extract_dynamic(state: DEMState):
    mu_xs = torch.stack([mu_x_tilde[:state.input.m_x].clone().detach() for mu_x_tilde in state.mu_x_tildes], axis=0)[:,:,0]
    sig_xs = torch.stack([sig_x_tilde[:state.input.m_x, :state.input.m_x].clone().detach() for sig_x_tilde in state.sig_x_tildes], axis=0)[:,:,0]
    mu_vs = torch.stack([mu_v_tilde[:state.input.m_v].clone().detach() for mu_v_tilde in state.mu_v_tildes], axis=0)[:,:,0]
    idx_first = int(state.input.p_comp // 2)
    idx_last = idx_first + len(state.mu_x_tildes)
    return mu_xs, sig_xs, mu_vs, idx_first, idx_last


# Part 3: Run the DEM procedure

dem_state = clean_state()

mu_xss = []
sig_xss = []
mu_vss = []
mu_thetas = []
sig_thetas = []
lr_thetas = []
f_bars = []

lr_dynamic = 1
lr_theta0 =  5
lr_theta_rate_coeff = 20 # after how many steps does it get to half of lr_theta0
lr_lambda = 0.01
iter_lambda = 20
iter_dem = 200
m_min_improv = 0.01

# DEM procedure
for i in range(iter_dem):
    # heuristic for decreasing learning rate
    lr_theta = lr_theta_rate_coeff * lr_theta0 / (i + lr_theta_rate_coeff)
    f_bar = free_action_from_state(dem_state)
    print(f"{i}. {f_bar.item()}")
    print(f"""  theta = [ {dem_state.mu_theta[0]:.3f}, {dem_state.mu_theta[1]:.3f},
          {dem_state.mu_theta[2]:.3f}, {dem_state.mu_theta[3]:.3f} ]""")
    print(f"""  mu_lambda_z = {dem_state.mu_lambda[0]:.3f}, mu_lambda_w = {dem_state.mu_lambda[1]:.3f}""")
    print(f"""  x0 = [ {dem_state.mu_x0_tilde[0].item():.3f}, {dem_state.mu_x0_tilde[1].item():.3f} ]""")
    print(f"""  v0 = [ {dem_state.mu_v0_tilde[0].item():.3f}, {dem_state.mu_v0_tilde[1].item():.3f} ]""")
    dem_step_d(dem_state, lr=lr_dynamic)
    f_bar = free_action_from_state(dem_state)
    print(f"  (D). {f_bar.item()}")
    dem_step_m(dem_state, lr_lambda=lr_lambda, iter_lambda=iter_lambda, min_improv=m_min_improv)
    dem_step_ex0(dem_state, lr_theta=lr_theta)
    dem_step_precision(dem_state)

    mu_xs, sig_xs, mu_vs, idx_first, idx_last = extract_dynamic(dem_state)
    mu_xss.append(mu_xs)
    sig_xss.append(sig_xs)
    mu_vss.append(mu_vs)
    f_bars.append(f_bar.clone().detach())
    mu_thetas.append(dem_state.mu_theta.clone().detach())
    sig_thetas.append(dem_state.sig_theta.clone().detach())
    lr_thetas.append(lr_theta)
