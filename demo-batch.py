from dataclasses import dataclass, field, replace
from copy import copy, deepcopy
from typing import Callable, Iterable
from itertools import repeat
from time import time
import csv
from pathlib import Path
import math

from tabulate import tabulate
from tqdm import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt

from hdm.dummy import dummy_lti
from matplotlib import pyplot as plt
import numpy as np


from hdm.dem.util import extract_dynamic
from hdm.dem.batched import DEMInput, DEMState, dem_step, dem_step_d, dem_step_precision, free_action
from hdm.core import deriv_mat, iterate_generalized, len_generalized
from hdm.noise import generate_noise_conv, autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import dummy_lti

# Generate and simulate a little system
m_x = 2
m_v = 1
m_y = 2
n = 30
dt = 0.5
x0 = np.zeros(m_x)
vs = np.zeros((n, m_v))
vs[5:10,0] = np.arange(5) # triangular impulse
vs[10:15,0] = np.arange(5)[::-1]
w_sd = 0
z_sd = 0.
noise_temporal_sig = 0.1
rng = np.random.default_rng(70)

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


A, B, C, D, x0, ts, vs, xs, ys, ws, zs = dummy_lti(m_x=m_x, m_v=m_v, m_y=m_y, n=n, dt=dt,
              x0=x0, vs=vs,
              w_sd=w_sd, z_sd=z_sd, noise_temporal_sig=noise_temporal_sig, rng=rng)

A, B, C, D, x0, ts, vs, xs, ys, ws, zs = [torch.tensor(obj, dtype=torch.float32)
                                          for obj in [A, B, C, D, x0, ts, vs, xs, ys, ws, zs]]

def dem_state_from_lti(A, B, C, D, x0, ts, vs, xs, ys, ws, zs,
                       p, p_comp, w_sd, z_sd, noise_temporal_sig, v_temporal_sig,
                       seed, known_matrices=('B', 'C', 'D')):
    """
    Returns a hdm.dem.batched.DEMState for a dummy LTI inversion problem.
    """
    v_sd = 1 # standard deviation on inputs

    A, B, C, D, x0, ts, vs, xs, ys, ws, zs = [torch.tensor(obj, dtype=torch.float32)
                                              for obj in [A, B, C, D, x0, ts, vs, xs, ys, ws, zs]]
    n = len(ts)
    n_gen = len_generalized(n, p_comp)
    m_x = xs.shape[1]
    m_v = vs.shape[1]
    m_y = ys.shape[1]
    dt = ts[1] - ts[0]

    rng_torch = torch.random.manual_seed(seed)

    # Prior precisions on known or unknown parameters
    precision_unknown = 1 # weak prior on unknown parmaeters
    precision_known = math.exp(8) # strong prior on known parameters

    # Temporal autocorrelation structure
    v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    v_autocorr_inv_ = torch.linalg.inv(v_autocorr)
    noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)


    # Initial state estimates
    mu_x0_tilde = x0.reshape((-1, 1))
    ## quite uninformative guess, also doesn't take into account expected temporal covariance
    sig_x_tilde0 = torch.eye(m_x * (p + 1))
    sig_x_tildes=[sig_x_tilde0.clone().detach() for _ in range(n_gen)]

    # Assume we know the inputs with high confidence
    p_v = torch.eye(m_v) * precision_known

    # Initial and prior parameters
    ## we construct a vector for theta with true values and high prior
    ## precisions on 'known' matrices
    params_size = m_x * m_x + m_x * m_v + m_y * m_x + m_y * m_v

    eta_thetas = []
    p_thetas = []
    mu_thetas = []
    for matrix, matrix_name in ((A, 'A'), (B, 'B'), (C, 'C'), (D, 'D')):
        if matrix_name not in known_matrices:
            eta = torch.zeros(matrix.shape[0] * matrix.shape[1], dtype=torch.float32)
            p_ = torch.ones(matrix.shape[0] * matrix.shape[1], dtype=torch.float32) * precision_unknown # precision
            mu = torch.normal(torch.zeros(matrix.shape[0] * matrix.shape[1]),
                              torch.ones(matrix.shape[0] * matrix.shape[1]) / params_size,
                              generator=rng_torch)
        else:
            eta = matrix.clone().to(dtype=torch.float32).reshape(-1)
            p_ = torch.ones(matrix.shape[0] * matrix.shape[1], dtype=torch.float32) * precision_known
            mu = eta.clone()
        eta_thetas.append(eta)
        p_thetas.append(p_)
        mu_thetas.append(mu)

    eta_theta = torch.concat(eta_thetas)
    p_theta = torch.diag(torch.concat(p_thetas))
    eta_lambda = torch.zeros(2, dtype=torch.float32)
    p_lambda = torch.eye(2, dtype=torch.float32)

    mu_theta = torch.concat(mu_thetas)
    mu_lambda = torch.tensor([math.exp(0), math.exp(-8)])

    # Covariance structure of noises
    ## the way we generate them now, they are necessarily independent
    omega_w=torch.eye(m_x, dtype=torch.float32)
    omega_z=torch.eye(m_y, dtype=torch.float32)

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
        return torch.matmul(A, x) + torch.matmul(B, v)

    def dem_g(x, v, params):
        A, B, C, D = ABCD_from_params(params)
        return torch.matmul(C, x) + torch.matmul(D, v)

    dem_input = DEMInput(
        dt=dt,
        m_x=m_x,
        m_v=m_v,
        m_y=m_y,
        p=p,
        p_comp=p_comp,
        ys=ys,
        eta_v=vs,
        p_v=p_v,
        v_autocorr_inv=v_autocorr_inv_,
        eta_theta=eta_theta,
        eta_lambda=eta_lambda,
        p_theta=p_theta,
        p_lambda=p_lambda,
        g=dem_g,
        f=dem_f,
        omega_w=omega_w,
        omega_z=omega_z,
        noise_autocorr_inv=noise_autocorr_inv_,
    )
    dem_state = DEMState.from_input(dem_input, x0, mu_theta=mu_theta, mu_lambda=mu_lambda)
    return dem_state


p = 4
p_comp = p
v_temporal_sig = 1

dem_state = dem_state_from_lti(A, B, C, D, x0, ts, vs, xs, ys, ws, zs,
                       p, p_comp, w_sd, z_sd, noise_temporal_sig, v_temporal_sig,
                       seed=2532, known_matrices=('A', 'B', 'C', 'D'))
# Do an initial run to get precision estimates
dem_step_d(dem_state, 1) # Do an initial run
dem_step_precision(dem_state)

# Interlude for diagnostics
f_bar, extr = dem_state.free_action(diagnostic=True)
pdict = {
    key: (item.norm().detach().item(), item.max().detach().item(), item.min().detach().item())
    for key, item in extr.items()
    if isinstance(item, torch.Tensor)
}
print(tabulate([(key, *item) for key, item in pdict.items()], headers=('variable', 'norm', 'max', 'min'), floatfmt='.3f'))
# End interlude

mu_xs, sig_xs, mu_vs, sig_vs, tsm = extract_dynamic(dem_state)
plt.plot(ts, xs[:, 0], label='Target 1')
plt.plot(tsm, mu_xs[:, 0], label=f'Model 1')
plt.plot(ts, xs[:, 1], label='Target 2')
plt.plot(tsm, mu_xs[:, 1], label=f'Model 2')
plt.plot(ts, vs[:, 0], label='Input')
plt.plot(tsm, sig_xs[:, 0], label=f'Variance 1')
plt.legend()
plt.show()

# Parameters for the optimization procedure
lr_dynamic = 1
lr_theta = 0.0001
lr_lambda = 0.001
iter_lambda = 20
iter_dem = 200
m_min_improv = 0.01

f_bar = dem_state.free_action()

## Some helper lists to track how the model develops over time
params_hist = []
f_bar_hist = []
table_rows = []
dem_states = [copy(dem_state)]

ideal_mu_x_tildes = list(iterate_generalized(xs, dt, p, p_comp=p_comp))

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
        'Total (s)': None
             }
table_rows.append(table_row)
print(tabulate(table_rows, headers='keys', floatfmt='.2f'))

for i in tqdm(range(iter_dem)):
    t0 = time()
    dem_step(dem_state, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=m_min_improv)
    tstep = time() - t0
    f_bar = dem_state.free_action().detach()
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
            'Total (s)': tstep
                 }
    params_hist.append(params_now)
    f_bar_hist.append(f_bar.clone().detach().item())
    table_rows.append(table_row)
    dem_states.append(copy(dem_state))
    print(tabulate(table_rows, headers='keys', floatfmt='.2f'))

table_headers = list(table_rows[0].keys())

outpath = Path('out/benchmark-batched.csv')
outpath.parent.mkdir(exist_ok=True)
with open(outpath, 'w') as file_:
    writer = csv.DictWriter(file_, table_headers)
    writer.writeheader()
    writer.writerows(table_rows)

# Plotting

plt.plot(ts, xs[:, 0], label='Target')
for i in range(1, 8):
    mu_xs, sig_xs, mu_vs, sig_vs, tsm = extract_dynamic(dem_states[i])
    plt.plot(tsm, mu_xs[:, 0], label=f'{i}')
plt.legend()
plt.show()
