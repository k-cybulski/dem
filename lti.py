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

from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
from hdm.dem.batched import (DEMInput, DEMState, dem_step_d,
                             dem_step_precision, dem_step_m, dem_step_e,
                             dem_step_precision, dem_step,
                             free_action)
from hdm.dem.util import extract_dynamic
from hdm.dummy import simulate_colored_lti

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
t_max = 16
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
p = 4 # for states
d = p # for inputs
# In the paper d = 2, but our implementation assumes equal number of entries

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

# Anil Meera & Wisse use 32, but that seems way too high for float32 which we
# use (they use double precision)
known_value_exp = 12

true_params = np.concatenate([A.reshape(-1), B.reshape(-1), C.reshape(-1)])

# Priors for estimation
p_v = torch.tensor(np.exp(known_value_exp), dtype=torch.float32).reshape((1,1))
eta_v = torch.tensor(vs, dtype=torch.float32)

eta_theta = torch.tensor(np.concatenate([rng.uniform(-2, 2, m_x * m_x + m_x * m_v), C.reshape(-1)]), dtype=torch.float32)
p_theta_diag = torch.tensor(np.concatenate([np.full(m_x * m_x + m_x * m_v, np.exp(6)), np.full(m_y * m_x, np.exp(known_value_exp))]), dtype=torch.float32)
p_theta = torch.tensor(np.diag(p_theta_diag), dtype=torch.float32)

eta_lambda = torch.tensor(np.zeros(2), dtype=torch.float32)
p_lambda = torch.tensor(np.eye(2) * np.exp(3), dtype=torch.float32)

## Some extras due to my implementation
v_autocorr = torch.tensor(noise_cov_gen_theoretical(d, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

def dem_f(x, v, params):
    A = params[0:(m_x * m_x)].reshape((m_x, m_x))
    B = params[(m_x * m_x):(m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
    return torch.matmul(A, x) + torch.matmul(B, v)

def dem_g(x, v, params):
    C = params[(m_x * m_x + m_x * m_v):(m_x * m_x + m_x * m_v + m_y * m_x)].reshape((m_y, m_x))
    return torch.matmul(C, x)

dem_input = DEMInput(dt=dt, m_x=m_x, m_v=m_v, m_y=m_y, p=p, p_comp=p, d=d,
                     ys=torch.tensor(ys, dtype=torch.float32),
                     eta_v=torch.tensor(vs, dtype=torch.float32),
                     p_v=p_v,
                     v_autocorr_inv=v_autocorr_inv_,
                     eta_theta=eta_theta,
                     p_theta=p_theta,
                     eta_lambda=eta_lambda,
                     p_lambda=p_lambda,
                     f=dem_f,
                     g=dem_g,
                     noise_autocorr_inv=noise_autocorr_inv_,
                     omega_w=torch.eye(m_x),
                     omega_z=torch.eye(m_y))

dem_state = DEMState.from_input(dem_input, torch.tensor(x0, dtype=torch.float32))

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

trajectories = [[np.array(v) for v in extract_dynamic(dem_state)]]
param_estimates = [dem_state.mu_theta.clone().detach().numpy()]
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

for i in tqdm(range(num_iter), desc="Running DEM..."):
    dem_step(dem_state, lr_dynamic, lr_theta, lr_lambda, iter_lambda, m_min_improv=m_min_improv)
    param_estimates.append(dem_state.mu_theta.clone().detach().numpy())
    trajectories.append([np.array(v) for v in extract_dynamic(dem_state)])
    f_bar, extr = dem_state.free_action(diagnostic=True)

    f_bars.append(f_bar)
    f_bar_diagnostics.append(extr)

    with open(f'out/ntraj{i:02}.pkl', 'wb') as file_:
        pickle.dump(trajectories, file_)

    with open(f'out/nparams{i:02}.pkl', 'wb') as file_:
        pickle.dump(param_estimates, file_)

    with open(f'out/nstates{i:02}.pkl', 'wb') as file_:
        pickle.dump(dem_state, file_)

    with open(f'out/nfbars{i:02}.pkl', 'wb') as file_:
        pickle.dump(f_bars, file_)

    pdict = {
        key: (item.norm().detach().item(), item.max().detach().item(), item.min().detach().item())
        for key, item in extr.items()
        if isinstance(item, torch.Tensor)
    }
    print(tabulate([(key, *item) for key, item in pdict.items()], headers=('variable', 'norm', 'max', 'min'), floatfmt='.3f'))
    print('fbar:',f_bars)
