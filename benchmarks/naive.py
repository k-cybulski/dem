from dataclasses import dataclass, field, replace
from copy import copy, deepcopy
from typing import Callable, Iterable
from itertools import repeat
from time import time
import csv
from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt

from hdm.dem.naive import DEMInput, DEMState, free_action_from_state, dem_step, dem_step_d
from hdm.core import deriv_mat, iterate_generalized, len_generalized
from hdm.noise import generate_noise_conv, autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import dummy_lti

##
## Note: This file is very much a potentially irrelevant work in progress
##

# Generate and simulate a little system
m_x = 2
m_v = 1
m_y = 2

n = 50
dt = 0.1

v_sd = 5 # standard deviation on inputs
v_temporal_sig = 1

w_sd = 0.0 # noise on states
z_sd = 0.0 # noise on outputs
noise_temporal_sig = 0.15 # temporal smoothing kernel parameter

seed = 546
rng = np.random.default_rng(seed)

A, B, C, D, x0, ts, vs, xs, ys, ws, zs = dummy_lti(
        m_x, m_v, m_y, n, dt, None, None, w_sd, z_sd, noise_temporal_sig, rng, v_sd=v_sd, v_temporal_sig=v_temporal_sig)

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

ideal_mu_x_tildes = list(iterate_generalized(xs, dt, p, p_comp=p_comp))

n_gen = len_generalized(n, p_comp)

v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

mu_x0_tilde = torch.normal(torch.zeros((m_x * (p + 1), 1)),
                           torch.ones((m_x * (p + 1)), 1) / (m_x * (p + 1)),
                           generator=rng_torch)
sig_x_tilde0 = torch.eye(m_x * (p + 1))
sig_v_tilde0 = torch.eye(m_v * (p + 1))

params_size = m_x * m_x + m_x * m_v + m_y * m_x + m_y * m_v

# Assume that we confidently know C and D
eta_theta = torch.zeros(params_size, dtype=torch.float32)
# eta_theta[(m_x*m_x):(m_x*m_x + m_x*m_v)] = B.reshape(-1) # B
eta_theta[(m_x*m_x + m_x*m_v):(m_x*m_x + m_x*m_v + m_y * m_x)] = C.reshape(-1) # C
eta_theta[(m_x*m_x + m_x*m_v + m_y * m_x):(m_x*m_x + m_x*m_v + m_y * m_x + m_y * m_v)] = D.reshape(-1) # D
p_theta = torch.block_diag(
        torch.eye(m_x*m_x), torch.eye(m_x*m_v),
         torch.eye(m_y * m_x) * np.exp(6).item(), torch.eye(m_y * m_v) * np.exp(6).item())

eta_lambda = torch.zeros(2, dtype=torch.float32)
p_lambda = torch.eye(2, dtype=torch.float32),
mu_theta0 = torch.normal(torch.zeros(params_size), torch.ones(params_size) / params_size, generator=rng_torch)
mu_lambda0 = eta_lambda
sig_theta0 = torch.eye(params_size, dtype=torch.float32)
sig_lambda0 = torch.eye(2)
p_v = torch.eye(2) * np.exp(6).item() # high precision on inputs,

def clean_state():
    """
    A DEMState initialized with xs and vs at zero.
    """
    dem_input = DEMInput(
        dt=dt,
        m_x=m_x,
        m_v=m_v,
        m_y=m_y,
        p=p,
        p_comp=p_comp,
        ys=ys,
        eta_v=vs,
        p_v=torch.eye(m_v) * np.exp(5).item(), # high precision on the inputs
        v_autocorr_inv=v_autocorr_inv_,
        eta_theta=eta_theta.clone(),
        eta_lambda=mu_lambda0,
        p_theta=p_theta.clone(),
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

lr_dynamic = 0.25
lr_theta = 0.05
lr_lambda = 0.01
iter_lambda = 20
iter_dem = 200
m_min_improv = 0.01

params_hist = []
f_bar_hist = []
table_rows = []
dem_states = [copy(dem_state)]

ideal_mu_x_tildes = list(iterate_generalized(xs, dt, p, p_comp=p_comp))

# It seems like the LTI system cannot converge with wrong initial state
dem_state.mu_x0_tilde = ideal_mu_x_tildes[0]

dem_step_d(dem_state, 1)

f_bar = free_action_from_state(dem_state)
params_now = [p.clone().detach() for p in ABCD_from_params(dem_state.mu_theta)]

table_row = {
        'Iteration': 0,
        'F_bar': f_bar.detach().item(),
        'A': torch.linalg.matrix_norm(A - params_now[0]).item(),
        'B': torch.linalg.matrix_norm(B - params_now[1]).item(),
        'C': torch.linalg.matrix_norm(C - params_now[2]).item(),
        'D': torch.linalg.matrix_norm(D - params_now[3]).item(),
        'x0_tilde': torch.linalg.matrix_norm(dem_state.mu_x0_tilde - ideal_mu_x_tildes[0]).item(),
        'xT_tilde': torch.linalg.matrix_norm(dem_state.mu_x_tildes[-1] - ideal_mu_x_tildes[-1]).item(),
        'Total (s)': None,
        'D (s)': None,
        'D (iter)': None,
        'D (s/iter)': None,
        'E (s)': None,
        'M (s)': None,
        'M (iter)':  None,
        'M (s/iter)':  None,
        'Prec (s)': None,
             }
table_rows.append(table_row)
print(tabulate(table_rows, headers='keys', floatfmt='.2f'))

for i in tqdm(range(iter_dem)):
    t0 = time()
    benchmark_ts = dem_step(dem_state, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=m_min_improv, benchmark=True)
    tstep = time() - t0
    f_bar = free_action_from_state(dem_state)
    params_now = [p.clone().detach() for p in ABCD_from_params(dem_state.mu_theta)]

    table_row = {
            'Iteration': i + 1,
            'F_bar': f_bar.detach().item(),
            'A': torch.linalg.matrix_norm(A - params_now[0]).item(),
            'B': torch.linalg.matrix_norm(B - params_now[1]).item(),
            'C': torch.linalg.matrix_norm(C - params_now[2]).item(),
            'D': torch.linalg.matrix_norm(D - params_now[3]).item(),
            'x0_tilde': torch.linalg.matrix_norm(dem_state.mu_x0_tilde - ideal_mu_x_tildes[0]).item(),
            'xT_tilde': torch.linalg.matrix_norm(dem_state.mu_x_tildes[-1] - ideal_mu_x_tildes[-1]).item(),
            'Total (s)': tstep,
            'D (s)': sum(benchmark_ts['ts_d']),
            'D (iter)': len(benchmark_ts['ts_d']),
            'D (s/iter)': sum(benchmark_ts['ts_d'])/len(benchmark_ts['ts_d']),
            'E (s)': benchmark_ts['t_e'],
            'M (s)': sum(benchmark_ts['ts_m']),
            'M (iter)':  len(benchmark_ts['ts_m']),
            'M (s/iter)':  sum(benchmark_ts['ts_m'])/len(benchmark_ts['ts_m']),
            'Prec (s)': benchmark_ts['t_prec'],
                 }
    params_hist.append(params_now)
    f_bar_hist.append(f_bar.clone().detach().item())
    table_rows.append(table_row)
    dem_states.append(copy(dem_state))
    print(tabulate(table_rows, headers='keys', floatfmt='.2f'))

table_headers = list(table_rows[0].keys())

outpath = Path('out/benchmark-naive.csv')
outpath.parent.mkdir(exist_ok=True)
with open(outpath, 'w') as file_:
    writer = csv.DictWriter(file_, table_headers)
    writer.writeheader()
    writer.writerows(table_rows)

# Plotting


def extract_dynamic(state: DEMState):
    mu_xs = torch.stack([mu_x_tilde[:state.input.m_x].clone().detach() for mu_x_tilde in state.mu_x_tildes], axis=0)[:,:,0]
    sig_xs = torch.stack([sig_x_tilde[:state.input.m_x, :state.input.m_x].clone().detach() for sig_x_tilde in state.sig_x_tildes], axis=0)[:,:,0]
    mu_vs = torch.stack([mu_v_tilde[:state.input.m_v].clone().detach() for mu_v_tilde in state.mu_v_tildes], axis=0)[:,:,0]
    idx_first = int(state.input.p_comp // 2)
    idx_last = idx_first + len(state.mu_x_tildes)
    return mu_xs, sig_xs, mu_vs, idx_first, idx_last


plt.plot(ts, xs[:, 0], label='Target')
for i in range(1, 8):
    mu_xs, sig_xs, mu_vs, idx_first, idx_last = extract_dynamic(dem_states[i])
    plt.plot(ts[idx_first:idx_last], mu_xs[:, 0], label=f'{i}')
plt.legend()
plt.show()