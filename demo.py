import torch
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from hdm.core import deriv_mat

plot = False

# Part 1: Simulate some data
# generate data with a simple model that we can invert with DEM

## Define the model
# x' = f(x) + w
# y  = g(x) + z

## Simplest case
# x' = Ax
# y  = Ix + z
# for white noise z

x0 = np.array([0, 1])
A = np.array([[0, 1], [-1, 0]])

noise_sd = 0.1

def f(t, x):
    return A @ x

# Simulate the data
t_start = 0
t_end = 20
t_span = (t_start, t_end)
dt = 0.1
ts = np.arange(start=t_start, stop=t_end, step=dt)
out = solve_ivp(f, t_span, x0, t_eval=ts)

xs = out.y
ys = np.random.normal(xs, noise_sd)

if plot:
    plt.plot(ts, ys[0, :])
    plt.plot(ts, ys[1, :])
    plt.show()


# Part 2: Invert the model

## Extract generalized y, i.e. y_tilde
# we will use the inverted Taylor trick, using 6 derivatives

# say we focus on the 30th timestep
p = 6
t_step = 30


# x0 = np.array([[0], [1]])
# A = np.array([[0, 1], [-1, 0]])

# def evolve_system(x_0, A, iter=100):
#     # Evolves a system by Euler method
#     xs = x_0
#     x_i = x_0
#     for i in range(1, iter):
#         x_i = A @ x_i
#         xs = np.hstack([xs, x_i])
#     return xs

# xs = evolve_system(x_0, A)

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
    # FIXME: The Jacobian output shape is a bit peculiar for our case. It has dimension 4.
    # I'm guessing that this is because PyTorch can be very flexible in the
    # input/output shapes, also considering cases like minibatches. For now,
    # this solution *seems* to work (for no minibatches)
    mu_x_grad = mu_x_grad.reshape((mu_x_grad.shape[0], mu_x_grad.shape[2]))
    mu_v_grad = mu_v_grad.reshape((mu_v_grad.shape[0], mu_v_grad.shape[2]))
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
        prec_w_tilde = torch.kron(noise_autocorr_inv, prec_w)
        prec_z_tilde = torch.kron(noise_autocorr_inv, prec_z)

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
