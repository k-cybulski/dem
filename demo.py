import torch
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

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


def free_energy(
        # dynamic terms
        mu_x_tildes, mu_v_tildes, # iterator of state and input mean estimates in generalized coordinates
        sig_x_tildes, sig_v_tildes, # like above, but covariance estimates
        output_tildes, # iterator of outputs in generalized coordinates
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
    err_theta = mu_theta - eta_theta
    err_lambda = mu_lambda - eta_lambda
    u_c = (err_theta.T @ p_theta @ err_theta + torch.logdet(p_theta)
           - err_lambda.T @ p_lambda @ err_lambda + torch.logdet(p_lambda)) / 2
    f_c = u_c + (torch.logdet(sig_theta) + torch.logdet(sig_lambda)) / 2

    for t, (mu_x_tilde, mu_v_tilde,
            sig_x_tilde, sig_v_tilde,
            output_tilde,
            eta_v_tilde, p_v_tilde) in enumerate(
                zip(mu_x_tildes, mu_v_tildes,
                    sig_x_tildes, sig_v_tildes,
                    output_tildes,
                    eta_v_tildes, p_v_tildes,
                    strict=True)
            ):
