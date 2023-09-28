"""
An LTI example with the setup as given in Section 15 of [1].

[1] A. Anil Meera and M. Wisse, “Dynamic Expectation Maximization Algorithm for
    Estimation of Linear Systems with Colored Noise,” Entropy (Basel), vol. 23,
    no. 10, p. 1306, Oct. 2021, doi: 10.3390/e23101306.
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from tabulate import tabulate
from pathlib import Path

from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
from hdm.dem.batched import (DEMInput, DEMState, dem_step_d,
                             dem_step_precision, dem_step_m, dem_step_e,
                             dem_step_precision, dem_step,
                             free_action)
from hdm.dem.util import extract_dynamic
from hdm.dummy import simulate_colored_lti

OUTPUT_DIR = Path('out/lti')

# lti model definition
A = np.array([[0.0484, 0.7535],
              [-0.7617, -0.2187]])
B = np.array([[0.3604], [0.0776]])
C = np.array([[0.2265, -0.4786],
              [0.4066, -0.2641],
              [0.3871, 0.3817],
              [-0.1630, -0.9290]])
D = np.array([[0], [0], [0], [0]])

x0 = np.zeros(2)
t_max = 32
dt = 0.1
# input pulse
vs = np.exp(-0.25 * (np.arange(0, t_max, dt) - 12)**2).reshape((-1, 1))
# noises
rng = np.random.default_rng(215)
noise_temporal_sig = 0.5
## noise precisions in the paper were exp(8)
lam_w = 8
lam_z = 8
noise_prec = np.exp(8)
noise_var = 1/noise_prec
noise_sd = np.sqrt(noise_var)
w_sd = noise_sd
z_sd = noise_sd
# simulate
ts, xs, ys, ws, zs = simulate_colored_lti(A, B, C, D, x0, dt, vs, w_sd, z_sd, noise_temporal_sig, rng)
## the outputs look similar to Figure 5.1
# plt.plot(ts, xs)

# Now we define the model
# embedding order
p = 6 # for states
d = 2 # for inputs

m_x = 2
m_v = 1
m_y = 4

def ABC_from_params(params):
    shapes = ((m_x, m_x),
              (m_x, m_v),
              (m_y, m_x))
    cursor = 0
    matrs = []
    for rows, cols in shapes:
        size = rows * cols
        start = cursor
        end = cursor + size
        matr = params[start:end].reshape((rows, cols))
        cursor = end
        matrs.append(matr)

    A, B, C = matrs
    return A, B, C

TORCH_DTYPE = torch.float64
TORCH_DEVICE = 'cuda:0'

# Anil Meera & Wisse use 32, but that seems way too high for our algorithm to work well
known_value_exp = 20

true_params = np.concatenate([A.reshape(-1), B.reshape(-1), C.reshape(-1)])

# Priors for estimation
p_v = torch.tensor(np.exp(known_value_exp), dtype=TORCH_DTYPE, device=TORCH_DEVICE).reshape((1,1))
eta_v = torch.tensor(vs, dtype=TORCH_DTYPE, device=TORCH_DEVICE)

eta_theta = torch.tensor(np.concatenate([rng.uniform(-2, 2, m_x * m_x + m_x * m_v), C.reshape(-1)]), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
p_theta_diag = torch.tensor(np.concatenate([np.full(m_x * m_x + m_x * m_v, np.exp(6)), np.full(m_y * m_x, np.exp(known_value_exp))]), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
p_theta = torch.tensor(torch.diag(p_theta_diag), dtype=TORCH_DTYPE, device=TORCH_DEVICE)

eta_lambda = torch.tensor(np.zeros(2), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
p_lambda = torch.tensor(np.eye(2) * np.exp(3), dtype=TORCH_DTYPE, device=TORCH_DEVICE)

## Some extras due to my implementation
v_autocorr = torch.tensor(noise_cov_gen_theoretical(d, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=TORCH_DTYPE, device=TORCH_DEVICE)
noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

def dem_f(x, v, params):
    A = params[0:(m_x * m_x)].reshape((m_x, m_x))
    B = params[(m_x * m_x):(m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
    return torch.matmul(A, x) + torch.matmul(B, v)

def dem_g(x, v, params):
    C = params[(m_x * m_x + m_x * m_v):(m_x * m_x + m_x * m_v + m_y * m_x)].reshape((m_y, m_x))
    return torch.matmul(C, x)

dem_input = DEMInput(dt=dt, m_x=m_x, m_v=m_v, m_y=m_y, p=p, d=d, d_comp=p,
                     ys=torch.tensor(ys, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
                     eta_v=torch.tensor(vs, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
                     p_v=p_v,
                     v_autocorr_inv=v_autocorr_inv_,
                     eta_theta=eta_theta,
                     p_theta=p_theta,
                     eta_lambda=eta_lambda,
                     p_lambda=p_lambda,
                     f=dem_f,
                     g=dem_g,
                     noise_autocorr_inv=noise_autocorr_inv_,
                     omega_w=torch.eye(m_x, dtype=TORCH_DTYPE, device=TORCH_DEVICE),
                     omega_z=torch.eye(m_y, dtype=TORCH_DTYPE, device=TORCH_DEVICE))

dem_state = DEMState.from_input(dem_input, torch.tensor(x0, dtype=TORCH_DTYPE, device=TORCH_DEVICE))

# Let's see if it works
dem_step_d(dem_state, 1)


# Plot it
# mu_xs, sig_xs, mu_vs, sig_vs, ts_model = extract_dynamic(dem_state)

# fig, ax = plt.subplots()
# ax.set_prop_cycle(color=['red',  'blue'])
# ax.plot(ts_model, mu_xs, label="states", linestyle='--')
# ax.plot(ts, xs, label="target states")
# # plt.plot(ts_model, mu_vs, label="causes")
# ax.legend()
# fig.show()

# Now do more DEM steps
lr_dynamic = 1
lr_theta = 10 # from the matlab code
lr_lambda = 1
iter_lambda = 8 # from the matlab code
m_min_improv = 0.01
num_iter = 50

trajectories = [[np.array(v.cpu()) for v in extract_dynamic(dem_state)]]
param_estimates = [dem_state.mu_theta.clone().detach().cpu().numpy()]
f_bars = []
f_bar_diagnostics = []

f_bar, extr = dem_state.free_action(diagnostic=True)
pdict = {
    key: (item.norm().detach().item(), item.max().detach().item(), item.min().detach().item())
    for key, item in extr.items()
    if isinstance(item, torch.Tensor)
}
print(tabulate([(key, *item) for key, item in pdict.items()], headers=('variable', 'norm', 'max', 'min'), floatfmt='.3f'))

f_bars.append(f_bar)
f_bar_diagnostics.append(extr)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_parameter_comparison(A, B, C, mu_thetas):
    rows = []
    with np.printoptions(precision=3, suppress=True):
        for iter, mu_theta in enumerate(mu_thetas):
            A_est, B_est, C_est = ABC_from_params(mu_theta)
            A_diff = A - A_est
            B_diff = B - B_est
            C_diff = C - C_est
            row = [
                    iter,
                    str(A_diff),
                    str(B_diff),
                    str(C_diff),
                    str(np.linalg.norm(A_diff)),
                    str(np.linalg.norm(B_diff)),
                    str(np.linalg.norm(C_diff))
            ]
            rows.append(row)
    print(tabulate(rows, headers=('Iter', 'A err', 'B err', 'C err', 'A err norm', 'B err norm', 'C err norm')))

for i in tqdm(range(num_iter), desc="Running DEM..."):
    dem_step(dem_state, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=m_min_improv)
    param_estimates.append(dem_state.mu_theta.clone().detach().cpu().numpy())
    trajectories.append([np.array(v.cpu()) for v in extract_dynamic(dem_state)])
    f_bar, extr = dem_state.free_action(diagnostic=True)

    f_bars.append(f_bar.detach().cpu().item())
    extr = { key: item.detach().cpu() for key, item in extr.items()}
    f_bar_diagnostics.append(extr)

    with open(OUTPUT_DIR / f'traj{i:02}.pkl', 'wb') as file_:
        pickle.dump(trajectories, file_)

    with open(OUTPUT_DIR / f'params{i:02}.pkl', 'wb') as file_:
        pickle.dump(param_estimates, file_)

    with open(OUTPUT_DIR / f'states{i:02}.pkl', 'wb') as file_:
        pickle.dump(dem_state, file_)

    with open(OUTPUT_DIR / f'fbars{i:02}.pkl', 'wb') as file_:
        pickle.dump(f_bars, file_)

    pdict = {
        key: (item.norm().detach().item(), item.max().detach().item(), item.min().detach().item())
        for key, item in extr.items()
        if isinstance(item, torch.Tensor)
    }
    print(tabulate([(key, *item) for key, item in pdict.items()], headers=('variable', 'norm', 'max', 'min'), floatfmt='.3f'))
    print_parameter_comparison(A, B, C, param_estimates)
    print('fbar:',f_bars)
