from dataclasses import dataclass, field, replace
from copy import copy
from typing import Callable, Iterable
from itertools import repeat

from tabulate import tabulate
from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt

from hdm.dem.naive import DEMInput, DEMState, free_action_from_state, dem_step
from hdm.core import deriv_mat, iterate_generalized, len_generalized
from hdm.noise import generate_noise_conv, autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import dummy_lti



# Generate and simulate a little system
m_x = 3
m_v = 2
m_y = 2

n = 30
dt = 0.1

v_sd = 5 # standard deviation on inputs
v_temporal_sig = 1

w_sd = 0.1 # noise on states
z_sd = 0.05 # noise on outputs
noise_temporal_sig = 0.15 # temporal smoothing kernel parameter

seed = 546
rng = np.random.default_rng(seed)

A, B, C, D, x0, ts, vs, xs, ys, ws, zs = dummy_lti(
        m_x, m_v, m_y, n, dt, v_sd, v_temporal_sig, w_sd, z_sd, noise_temporal_sig, rng)

A, B, C, D, x0, ts, vs, xs, ys, ws, zs = [torch.tensor(obj, dtype=torch.float32)
                                          for obj in [A, B, C, D, x0, ts, vs, xs, ys, ws, zs]]


# Define a DEM model to invert it
# The model is
# x' = Ax + Bv + w
# y  = Cx + Dv + z

# params will be a vector which encodes the matrices A, B, C, and D
def ABCD_from_params(params):
    shapes = ((m_x, m_x),
              (m_x, m_v),
              (m_y, m_x),
              (m_y, m_v))
    cursor = 0
    matrs = []
    for rows, cols in shapes:
        size = rows * cols
        start = cursor
        end = cursor + size
        matr = params[start:end].reshape((rows, cols))
        cursor = end
        matrs.append(matr)

    A, B, C, D = matrs
    return A, B, C, D


def dem_f(x, v, params):
    A, B, C, D = ABCD_from_params(params)
    return A @ x + B @ v

def dem_g(x, v, params):
    A, B, C, D = ABCD_from_params(params)
    return C @ x + D @ v


rng_torch = torch.random.manual_seed(156)

p = 3
p_comp = 6

n_gen = len_generalized(n, p_comp)

v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

mu_x0_tilde = torch.normal(torch.zeros((m_x * (p + 1), 1)),
                           torch.ones((m_x * (p + 1)), 1),
                           generator=rng_torch)
sig_x_tilde0 = torch.eye(m_x * (p + 1))
sig_v_tilde0 = torch.eye(m_v * (p + 1))

params_size = m_x * m_x + m_x * m_v + m_y * m_x + m_y * m_v

mu_theta0 = torch.normal(torch.zeros(params_size), torch.ones(params_size), generator=rng_torch)
mu_lambda0 = torch.tensor(np.ones(2)* np.log(0.1), dtype=torch.float32)

sig_theta0 = torch.eye(params_size, dtype=torch.float32)
sig_lambda0 = torch.eye(2)

p_v = torch.eye(2) * np.exp(5).item() # high precision on inputs,

def clean_state():
    """
    A DEMState initialized with xs and vs at zero.
    """
    dem_input = DEMInput(
        dt=dt,
        m_x=m_x,
        m_v=m_v,
        p=p,
        p_comp=p_comp,
        ys=ys,
        eta_v=vs,
        p_v=torch.eye(m_v),
        v_autocorr_inv=v_autocorr_inv_,
        eta_theta=mu_theta0,
        eta_lambda=mu_lambda0,
        p_theta=torch.eye(params_size, dtype=torch.float32),
        p_lambda=torch.eye(2, dtype=torch.float32),
        g=dem_g,
        f=dem_f,
        omega_w=torch.eye(m_x, dtype=torch.float32),
        omega_z=torch.eye(m_y, dtype=torch.float32),
        noise_autocorr_inv=noise_autocorr_inv_,
    )
    dem_state = DEMState(
        input=dem_input,
        mu_x_tildes=None, # will be set in the D step
        mu_v_tildes=None,
        sig_x_tildes=[sig_x_tilde0.clone().detach() for _ in range(n_gen)],
        sig_v_tildes=[sig_v_tilde0.clone().detach() for _ in range(n_gen)],
        mu_theta=mu_theta0.clone().detach(),
        mu_lambda=mu_lambda0.clone().detach(),
        sig_theta=sig_theta0.clone().detach(),
        sig_lambda=sig_lambda0.clone().detach(),
        mu_x0_tilde=mu_x0_tilde.clone().detach(),
        mu_v0_tilde=next(iterate_generalized(vs, dt, p, p_comp=p_comp))
    )
    return dem_state

dem_state = clean_state()

params_hist = []
f_bar_hist = []
table_rows = []

lr_dynamic = 1
lr_theta =  0.1
lr_theta_rate_coeff = 20 # after how many steps does it get to half of lr_theta0
lr_lambda = 0.01
iter_lambda = 20
iter_dem = 50
m_min_improv = 0.01

for i in tqdm(range(iter_dem)):
    dem_step(dem_state, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=m_min_improv)
    params_now = [p.clone().detach() for p in ABCD_from_params(dem_state.mu_theta)]
    f_bar = free_action_from_state(dem_state)

    table_row = (i,
                 f_bar.detach().item(),
                 torch.linalg.matrix_norm(A - params_now[0]).item(),
                 torch.linalg.matrix_norm(B - params_now[1]).item(),
                 torch.linalg.matrix_norm(C - params_now[2]).item(),
                 torch.linalg.matrix_norm(D - params_now[3]).item())
    params_hist.append(params_now)
    f_bar_hist.append(f_bar.clone().detach().item())
    table_rows.append(table_roW)
    print(tabulate(table_rows, headers=('Iteration', 'F_bar', 'A', 'B', 'C', 'D')))