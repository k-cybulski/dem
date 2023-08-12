"""
An example which tries to align internal state estimates of DEM with a Lorenz
attractor. Based on an example from the dempy package.
"""
import numpy as np
import torch
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from hdm.core import len_generalized
from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import simulate_system
from hdm.dem.naive import DEMInput, DEMState, dem_step_d, extract_dynamic

## Prior expectations (and true values)
params_true = np.array([18., 18., 46.92, 2., 1., 2., 4., 1., 1., 1.])
x0_true = np.array([0.9,0.8,30])

def lorenz(x, v, P):
    x0 = P[0] * x[1] - P[1] * x[0]
    x1 = P[2] * x[0] - P[3] * x[2] * x[0] - P[4] * x[1]
    x2 = P[5] * x[0] * x[1] - P[6] * x[2]

    return np.array([x0, x1, x2]) / 128.

def obs(x, v, P):
    return np.array(x @ P[-3:])

def f_true(x, v):
    return lorenz(x, v, params_true)

def g_true(x, v):
    return obs(x, v, params_true)

rng = np.random.default_rng(215)
n = 1024
dt = 1.
vs = np.zeros((n, 1))
w_sd = 1
z_sd = 1
noise_temporal_sig = 5
ts, xs, ys, ws, zs = simulate_system(f_true, g_true, x0_true, dt, vs, w_sd, z_sd, noise_temporal_sig, rng=rng)


# Test DEM inversion
## We will use the real parameters with high precision
## our goal is to just find the proper states

p = 4
p_comp = p

x0_test = np.array([12,13,16])
p_v = torch.tensor(np.exp(64).reshape((1, 1)), dtype=torch.float32)

# doesn't matter, v is 0 anyway
v_temporal_sig = 5
v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

eta_theta = torch.tensor(params_true, dtype=torch.float32)
p_theta = torch.tensor(np.diag(np.ones(eta_theta.shape)) * np.exp(64), dtype=torch.float32)
eta_lambda = torch.zeros(2) # Not sure how to initialize these well
p_lambda = torch.eye(2) # Not sure how to initialize these well

mu_x0_tilde = torch.tensor(np.concatenate((x0_test, rng.normal([0] * 3 * p))), dtype=torch.float32).reshape((-1, 1))
mu_v0_tilde = torch.tensor(np.zeros((p + 1, 1)), dtype=torch.float32)

def lorenz_torch(x, v, P):
    x = x.reshape(-1)
    x0 = P[0] * x[1] - P[1] * x[0]
    x1 = P[2] * x[0] - P[3] * x[2] * x[0] - P[4] * x[1]
    x2 = P[5] * x[0] * x[1] - P[6] * x[2]

    return (torch.stack([x0, x1, x2]) / 128.).reshape((-1, 1))

def obs_torch(x, v, P):
    x = x.reshape(-1)
    return (x @ P[-3:]).reshape((-1, 1))

dem_input = DEMInput(dt=dt, m_x=3, m_v=1, m_y=1, p=p, p_comp=p_comp,
                     ys=torch.tensor(ys, dtype=torch.float32),
                     eta_v=torch.tensor(vs, dtype=torch.float32),
                     p_v=p_v,
                     v_autocorr_inv=v_autocorr_inv_,
                     eta_theta=eta_theta,
                     p_theta=p_theta,
                     eta_lambda=eta_lambda,
                     p_lambda=p_lambda,
                     f=lorenz_torch,
                     g=obs_torch,
                     noise_autocorr_inv=noise_autocorr_inv_)

dem_state = DEMState(input=dem_input,
                     mu_x_tildes=None,
                     mu_v_tildes=None,
                     sig_x_tildes=[torch.eye(3 * (p + 1)) for _ in range(len_generalized(n, p_comp))],
                     sig_v_tildes=[torch.eye(1 * (p + 1)) for _ in range(len_generalized(n, p_comp))],
                     mu_theta=eta_theta,
                     mu_lambda=eta_lambda,
                     sig_theta=torch.linalg.inv(p_theta),
                     sig_lambda=torch.linalg.inv(p_lambda),
                     mu_x0_tilde=mu_x0_tilde,
                     mu_v0_tilde=mu_v0_tilde)

lr_dynamic = 1
dem_step_d(dem_state, lr_dynamic)

mu_xs, sig_xs, mu_vs, idx_first, idx_last = extract_dynamic(dem_state)