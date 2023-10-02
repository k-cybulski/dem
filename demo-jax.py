from time import time

import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

from hdm.dem.jax import (DEMInputJAX, DEMStateJAX, dem_step_d,
                         dem_step_precision, dem_step_m, dem_step_e,
                         dem_step_precision, dem_step,
                         free_action)

##########
########## Test setup
##########

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import pickle
from tabulate import tabulate
from pathlib import Path

from hdm.core import iterate_generalized
from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
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
p_comp = p

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

true_params = np.concatenate([A.reshape(-1), B.reshape(-1), C.reshape(-1)])

# JAX doesn't seem to be able to support a value as high as 20
known_value_exp = 12


######### Utils

# https://github.com/google/jax/pull/762#issuecomment-1002267121
import functools
import jax
import jax.numpy as jnp

def value_and_jacfwd(f, x):
  pushfwd = functools.partial(jax.jvp, f, (x,))
  basis = jnp.eye(x.size, dtype=x.dtype)
  y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
  return y, jac

def value_and_jacrev(f, x):
  y, pullback = jax.vjp(f, x)
  basis = jnp.eye(y.size, dtype=y.dtype)
  jac = jax.vmap(pullback)(basis)
  return y, jac

##########
########## Making DEM work
##########

# for verification
import torch
from hdm.dem.batched import generalized_func as generalized_func_torch

# for implementation
from jax import vjp, jacfwd
from functools import partial

def tilde_to_grad(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
    """Computes gradient and function value given x and v in generalized
    coordinates."""
    assert mu_x_tilde.shape == (m_x * (p + 1), 1)
    assert mu_v_tilde.shape == (m_v * (p + 1), 1)
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]

    func_appl = func(mu_x, mu_v, params)
    func_jac = jacfwd(lambda x, v: func(x, v, params), argnums=(0,1))
    mu_x_grad, mu_v_grad =  func_jac(mu_x, mu_v)
    mu_x_grad = mu_x_grad.reshape((mu_x_grad.shape[0], mu_x_grad.shape[2]))
    mu_v_grad = mu_v_grad.reshape((mu_v_grad.shape[0], mu_v_grad.shape[2]))
    return func_appl, mu_x_grad, mu_v_grad


def tildes_to_grads(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    # batched version of tilde_to_grad
    ttg_v = vmap(lambda mu_x_tilde, mu_v_tilde: tilde_to_grad(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params))
    func_appl, mu_x_grad, mu_v_grad = ttg_v(mu_x_tildes, mu_v_tildes)
    return func_appl, mu_x_grad, mu_v_grad


def f_jax(x, v, params):
    A = params[0:(m_x * m_x)].reshape((m_x, m_x))
    B = params[(m_x * m_x):(m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
    return jnp.matmul(A, x) + jnp.matmul(B, v)

def g_jax(x, v, params):
    C = params[(m_x * m_x + m_x * m_v):(m_x * m_x + m_x * m_v + m_y * m_x)].reshape((m_y, m_x))
    return jnp.matmul(C, x)

gen_func_f = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(f_jax, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params))
gen_func_g = jit(lambda mu_x_tildes, mu_v_tildes, params: generalized_func(g_jax, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params))

x_tildes = jnp.stack(list(iterate_generalized(xs, dt, p, p_comp=p_comp))).astype(jnp.float64)
v_tildes = jnp.stack(list(iterate_generalized(vs, dt, d, p_comp=p_comp))).astype(jnp.float64)

params = true_params

## SCRIB
func = f_jax
params = true_params
mu_x_tildes = x_tildes
mu_v_tildes = v_tildes
mu_x_tilde = mu_x_tildes[0]
mu_v_tilde = mu_v_tildes[0]

## Priors
p_v = np.exp(known_value_exp).reshape((1,1))
eta_v = vs.copy()

eta_theta = np.concatenate([rng.uniform(-2, 2, m_x * m_x + m_x * m_v), C.reshape(-1)])
p_theta_diag = np.concatenate([np.full(m_x * m_x + m_x * m_v, np.exp(6)), np.full(m_y * m_x, np.exp(known_value_exp))])
p_theta = np.diag(p_theta_diag)

eta_lambda = np.zeros(2)
p_lambda = np.eye(2) * np.exp(3)

v_autocorr = noise_cov_gen_theoretical(d, sig=noise_temporal_sig, autocorr=autocorr_friston())
v_autocorr_inv = np.linalg.inv(v_autocorr)

noise_autocorr = noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston())
noise_autocorr_inv = np.linalg.inv(noise_autocorr)

omega_w = np.eye(m_x)
omega_z = np.eye(m_y)

### Priors depending on the other priors

y_tildes =  np.stack(list(iterate_generalized(ys, dt, p, p_comp=p_comp)))
eta_v_tildes = np.stack(list(iterate_generalized(eta_v, dt, d, p_comp=p_comp)))

p_v_tilde = np.kron(v_autocorr_inv, p_v)
p_v_tildes = np.tile(p_v_tilde, (len(eta_v_tildes), 1, 1))

## Parameters
mu_theta = eta_theta.copy()
mu_lambda = eta_lambda.copy()

# mu_x0_tilde = np.concatenate([x0, np.zeros(p * m_x)]).reshape((-1, 1))
mu_x_tildes = x_tildes
mu_v_tildes = v_tildes
mu_x0_tilde = x_tildes[0] # groundtruth

sig_x_tildes = np.tile(np.kron(noise_autocorr_inv, np.eye(m_x)), (len(mu_v_tildes), 1, 1))
sig_v_tildes = vmap(jnp.linalg.inv)(p_v_tildes)

sig_theta = jnp.linalg.inv(p_theta)
sig_lambda = jnp.linalg.inv(p_lambda)

mu_v0_tilde = mu_v_tildes[0]

# f_bar = free_action(         m_x, m_v,         p,         d,         mu_x_tildes, mu_v_tildes,         sig_x_tildes, sig_v_tildes,         y_tildes,         eta_v_tildes, p_v_tildes,         eta_theta,         eta_lambda,         p_theta,         p_lambda,         mu_theta,         mu_lambda,         sig_theta,         sig_lambda,         gen_func_f,         gen_func_g,         omega_w,         omega_z,         noise_autocorr_inv)

## END SCRIB


dem_input = DEMInputJAX(dt=dt, m_x=m_x, m_v=m_v, m_y=m_y, p=p, d=d, d_comp=p,
                     ys=jnp.array(ys).astype(jnp.float64),
                     eta_v=jnp.array(vs).astype(jnp.float64),
                     p_v=p_v,
                     eta_theta=eta_theta,
                     p_theta=p_theta,
                     eta_lambda=eta_lambda,
                     p_lambda=p_lambda,
                     f=f_jax,
                     g=g_jax,
                     noise_temporal_sig=noise_temporal_sig)

dem_state = DEMStateJAX.from_input(dem_input, x0)

# These are groundtruths:
# dem_state.mu_x_tildes = mu_x_tildes
# dem_state.mu_v_tildes = mu_v_tildes
# dem_state.mu_x0_tilde = mu_x_tildes[0]
# dem_state.mu_v0_tilde = mu_v_tildes[0]

## As a validity check, make a copy of the above but with torch instead
# TODO: Compare validity!
from hdm.dem.batched import DEMInputBatched, DEMStateBatched
from hdm.dem.batched import dem_step_d as dem_step_d_b
from hdm.dem.batched import dem_step_e as dem_step_e_b
from hdm.dem.batched import dem_step_m as dem_step_m_b
from hdm.dem.batched import dem_step_precision as dem_step_precision_b

def f_torch(x, v, params):
    A = params[0:(m_x * m_x)].reshape((m_x, m_x))
    B = params[(m_x * m_x):(m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
    return torch.matmul(A, x) + torch.matmul(B, v)

def g_torch(x, v, params):
    C = params[(m_x * m_x + m_x * m_v):(m_x * m_x + m_x * m_v + m_y * m_x)].reshape((m_y, m_x))
    return torch.matmul(C, x)

TORCH_DTYPE = torch.float64
TORCH_DEVICE = 'cpu' # 'cuda:0'
batched_input_dict = {
        'f': f_torch,
        'g': g_torch,
        **{attr: dem_input.__dict__[attr] for attr in ['dt', 'm_x', 'm_v', 'm_y', 'p', 'd', 'd_comp']},
        **{attr: torch.from_numpy(np.array(dem_input.__dict__[attr])).to(dtype=TORCH_DTYPE, device=TORCH_DEVICE)
           for attr in ['ys', 'eta_v', 'p_v', 'eta_theta', 'eta_lambda', 'p_theta', 'p_lambda', 'noise_autocorr_inv', 'v_autocorr_inv', 'omega_w', 'omega_z']}}

dem_input_b = DEMInputBatched(**batched_input_dict)
dem_state_b = DEMStateBatched.from_input(dem_input_b, torch.from_numpy(np.array(x0)).to(dtype=TORCH_DTYPE, device=TORCH_DEVICE))

dem_state_b.mu_x_tildes = torch.from_numpy(np.array(mu_x_tildes)).to(dtype=TORCH_DTYPE, device=TORCH_DEVICE)
dem_state_b.mu_v_tildes = torch.from_numpy(np.array(mu_v_tildes)).to(dtype=TORCH_DTYPE, device=TORCH_DEVICE)
dem_state_b.mu_x0_tilde = torch.from_numpy(np.array(mu_x_tildes[0])).to(dtype=TORCH_DTYPE, device=TORCH_DEVICE)
dem_state_b.mu_v0_tilde = torch.from_numpy(np.array(mu_v_tildes[0])).to(dtype=TORCH_DTYPE, device=TORCH_DEVICE)

def compare_states_jax_torch(state_jax, state_torch):
    equivalent = True
    # attributes of state
    for attr in [
            "mu_x_tildes",
            "mu_v_tildes",
            "sig_x_tildes",
            "sig_v_tildes",
            "mu_theta",
            "mu_lambda",
            "sig_theta",
            "sig_lambda",
            "mu_x0_tilde",
            "mu_v0_tilde"
            ]:
        if not np.isclose(getattr(state_jax, attr), getattr(state_torch, attr).detach().numpy()).all():
            print(f"{attr} differ")
            equivalent = False
    # attributes of input
    for attr in [
            "ys",
            "eta_v",
            "p_v",
            "v_autocorr_inv",
            "eta_theta",
            "eta_lambda",
            "p_theta",
            "p_lambda",
            "omega_w",
            "omega_z",
            "noise_autocorr_inv"
            ]:
        if not np.isclose(getattr(state_jax.input, attr), getattr(state_torch.input, attr).detach().numpy()).all():
            print(f"input.{attr} differ")
            equivalent = False
    return equivalent

# Are free actions the same?
# assert np.isclose(dem_state.free_action(), dem_state_b.free_action().detach().numpy())

# Are their hessians the same?
from hdm.dem.jax import free_action
from jax import hessian
from dataclasses import replace

# def free_action_from_state(state, mu_x_tildes, mu_v_tildes, skip_constant=False, diagnostic=False):
#     return free_action(
#         m_x=state.input.m_x,
#         m_v=state.input.m_v,
#         p=state.input.p,
#         d=state.input.d,
#         mu_x_tildes=mu_x_tildes,
#         mu_v_tildes=mu_v_tildes,
#         # mu_x_tildes=state.mu_x_tildes,
#         # mu_v_tildes=state.mu_v_tildes,
#         sig_x_tildes=state.sig_x_tildes,
#         sig_v_tildes=state.sig_v_tildes,
#         y_tildes=state.input.y_tildes,
#         eta_v_tildes=state.input.eta_v_tildes,
#         p_v_tildes=state.input.p_v_tildes,
#         eta_theta=state.input.eta_theta,
#         eta_lambda=state.input.eta_lambda,
#         p_theta=state.input.p_theta,
#         p_lambda=state.input.p_lambda,
#         mu_theta=state.mu_theta,
#         mu_lambda=state.mu_lambda,
#         sig_theta=state.sig_theta,
#         sig_lambda=state.sig_lambda,
#         gen_func_g=state.input.gen_func_g,
#         gen_func_f=state.input.gen_func_f,
#         omega_w=state.input.omega_w,
#         omega_z=state.input.omega_z,
#         noise_autocorr_inv=state.input.noise_autocorr_inv,
#         skip_constant=skip_constant,
#         diagnostic=diagnostic
#     )

# state = dem_state
# hesses_messes = hessian(free_action, argnums=(4,5))(
#         state.input.m_x,
#         state.input.m_v,
#         state.input.p,
#         state.input.d,
#         mu_x_tildes,
#         mu_v_tildes,
#         # mu_x_tildes=state.mu_x_tildes,
#         # mu_v_tildes=state.mu_v_tildes,
#         sig_x_tildes=state.sig_x_tildes,
#         sig_v_tildes=state.sig_v_tildes,
#         y_tildes=state.input.y_tildes,
#         eta_v_tildes=state.input.eta_v_tildes,
#         p_v_tildes=state.input.p_v_tildes,
#         eta_theta=state.input.eta_theta,
#         eta_lambda=state.input.eta_lambda,
#         p_theta=state.input.p_theta,
#         p_lambda=state.input.p_lambda,
#         mu_theta=state.mu_theta,
#         mu_lambda=state.mu_lambda,
#         sig_theta=state.sig_theta,
#         sig_lambda=state.sig_lambda,
#         gen_func_g=state.input.gen_func_g,
#         gen_func_f=state.input.gen_func_f,
#         omega_w=state.input.omega_w,
#         omega_z=state.input.omega_z,
#         noise_autocorr_inv=state.input.noise_autocorr_inv,
#         skip_constant=False,
#         diagnostic=False
#     )


# def lambda_free_action(mu_x_tildes, mu_v_tildes):
#     # free action as a function of lambda
#     return replace(dem_state_b, mu_x_tildes=mu_x_tildes, mu_v_tildes=mu_v_tildes).free_action()
# hesses_messes_torch = torch.autograd.functional.hessian(lambda_free_action, (dem_state_b.mu_x_tildes, dem_state_b.mu_v_tildes))

# assert np.isclose(hesses_messes[0][0], np.array(hesses_messes_torch[0][0])).all()
# assert np.isclose(hesses_messes[1][1], np.array(hesses_messes_torch[1][1])).all()

# Let's see if it works

# # Let's see if the JAX and PyTorch implementation follow the same trajectories
# lr_dynamic = 1
# lr_theta = 10 # from the matlab code
# lr_lambda = 1
# iter_lambda = 8 # from the matlab code
# m_min_improv = 0.01
# num_iter = 2
# for i in tqdm(range(num_iter), desc="Comparing JAX and PyTorch DEM trajectories"):
#     print(f"{i}. Step D")
#     dem_step_d(dem_state, lr_dynamic)
#     dem_step_d_b(dem_state_b, lr_dynamic)
#     compare_states_jax_torch(dem_state, dem_state_b)

#     print(f"{i}. Step M")
#     dem_step_m(dem_state, lr_lambda, iter_lambda, min_improv=m_min_improv)
#     dem_step_m_b(dem_state_b, lr_lambda, iter_lambda, min_improv=m_min_improv)
#     compare_states_jax_torch(dem_state, dem_state_b)

#     print(f"{i}. Step E")
#     dem_step_e(dem_state, lr_theta)
#     dem_step_e_b(dem_state_b, lr_theta)
#     compare_states_jax_torch(dem_state, dem_state_b)

#     print(f"{i}. Step precision")
#     dem_step_precision(dem_state)
#     dem_step_precision_b(dem_state_b)
#     compare_states_jax_torch(dem_state, dem_state_b)


def generalized_batch_to_sequence(tensor, m, is2d=False):
    if not is2d:
        xs = jnp.stack([x_tilde[:m] for x_tilde in tensor], axis=0)[:,:,0]
    else:
        xs = jnp.stack([jnp.diagonal(x_tilde)[:m] for x_tilde in tensor], axis=0)
    return xs

def extract_dynamic(state):
    mu_xs = generalized_batch_to_sequence(state.mu_x_tildes, state.input.m_x)
    sig_xs = generalized_batch_to_sequence(state.sig_x_tildes, state.input.m_x, is2d=True)
    mu_vs = generalized_batch_to_sequence(state.mu_v_tildes, state.input.m_v)
    sig_vs = generalized_batch_to_sequence(state.sig_v_tildes, state.input.m_v, is2d=True)
    idx_first = int(state.input.p_comp // 2)
    idx_last = idx_first + len(mu_xs)
    ts_all = torch.arange(state.input.n) * state.input.dt
    ts = ts_all[idx_first:idx_last]
    return mu_xs, sig_xs, mu_vs, sig_vs, ts

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

print("Running initial D step")
dem_step_d(dem_state, lr_dynamic)

trajectories = [[np.array(v) for v in extract_dynamic(dem_state)]]
param_estimates = [dem_state.mu_theta]
f_bars = []
f_bar_diagnostics = []

f_bar, extr = dem_state.free_action(diagnostic=True)
pdict = {
    key: (jnp.linalg.norm(item), item.max(), item.min())
    for key, item in extr.items()
    if isinstance(item, jnp.ndarray)
}
print(tabulate([(key, *item) for key, item in pdict.items()], headers=('variable', 'norm', 'max', 'min'), floatfmt='.3f'))

f_bars.append(f_bar)
extr = { key: item for key, item in extr.items()}
f_bar_diagnostics.append(extr)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def print_convergence_table(A, B, C, mu_thetas, f_bars):
    rows = []
    with np.printoptions(precision=3, suppress=True):
        for iter, (mu_theta, f_bar) in enumerate(zip(mu_thetas, f_bars)):
            A_est, B_est, C_est = ABC_from_params(mu_theta)
            A_diff = A - A_est
            B_diff = B - B_est
            C_diff = C - C_est
            row = [
                    iter,
                    f_bar.item(),
                    str(A_diff),
                    str(B_diff),
                    str(C_diff),
                    str(np.linalg.norm(A_diff)),
                    str(np.linalg.norm(B_diff)),
                    str(np.linalg.norm(C_diff))
            ]
            rows.append(row)
    print(tabulate(rows, headers=('Iter', 'Free action', 'A err', 'B err', 'C err', 'A err norm', 'B err norm', 'C err norm')))

times = {k: [] for k in ['D', 'E', 'M', 'Precision']}

for i in tqdm(range(num_iter), desc="Running DEM..."):

    print("Step D")
    t0 = time()
    dem_step_d(dem_state, lr_dynamic)
    times['D'].append(time() - t0)
    print("Step M")
    t0 = time()
    dem_step_m(dem_state, lr_lambda, iter_lambda, min_improv=m_min_improv)
    times['M'].append(time() - t0)
    print("Step E")
    t0 = time()
    dem_step_e(dem_state, lr_theta)
    times['E'].append(time() - t0)
    print("Step precision")
    t0 = time()
    dem_step_precision(dem_state)
    times['Precision'].append(time() - t0)

    param_estimates.append(dem_state.mu_theta)
    trajectories.append([np.array(v) for v in extract_dynamic(dem_state)])
    f_bar, extr = dem_state.free_action(diagnostic=True)

    f_bars.append(f_bar)
    extr = { key: item for key, item in extr.items()}
    f_bar_diagnostics.append(extr)

    with open(OUTPUT_DIR / f'jtraj{i:02}.pkl', 'wb') as file_:
        pickle.dump(trajectories, file_)

    with open(OUTPUT_DIR / f'jparams{i:02}.pkl', 'wb') as file_:
        pickle.dump(param_estimates, file_)

    with open(OUTPUT_DIR / f'jstates{i:02}.pkl', 'wb') as file_:
        pickle.dump(dem_state, file_)

    with open(OUTPUT_DIR / f'jfbars{i:02}.pkl', 'wb') as file_:
        pickle.dump(f_bars, file_)

    with open(OUTPUT_DIR / f'jtimes{i:02}.pkl', 'wb') as file_:
        pickle.dump(times, file_)

    pdict = {
        key: (jnp.linalg.norm(item), item.max(), item.min())
        for key, item in extr.items()
        if isinstance(item, jnp.ndarray)
    }
    print(tabulate([(key, *item) for key, item in pdict.items()], headers=('variable', 'norm', 'max', 'min'), floatfmt='.3f'))
    print_convergence_table(A, B, C, param_estimates, f_bars)
    print(tabulate(times, headers='keys', floatfmt='.3f'))


### FOR COMPARISON WITH TORCH

# Check equivalence between torch and jax outputs
def f_torch(x, v, params):
    A = params[0:(m_x * m_x)].reshape((m_x, m_x))
    B = params[(m_x * m_x):(m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
    return torch.matmul(A, x) + torch.matmul(B, v)

def g_torch(x, v, params):
    C = params[(m_x * m_x + m_x * m_v):(m_x * m_x + m_x * m_v + m_y * m_x)].reshape((m_y, m_x))
    return torch.matmul(C, x)

x_tildes_t = torch.stack(list(iterate_generalized(torch.from_numpy(xs), dt, p)))
v_tildes_t = torch.stack(list(iterate_generalized(torch.from_numpy(vs), dt, p)))

params_t = torch.from_numpy(true_params).to(dtype=torch.float64)

f_tildes_t = generalized_func_torch(f_torch, x_tildes_t, v_tildes_t, m_x, m_v, p, params_t)
g_tildes_t = generalized_func_torch(g_torch, x_tildes_t, v_tildes_t, m_x, m_v, p, params_t)

## SCRIB
func = f_torch
params = params_t
mu_x_tildes = x_tildes
mu_v_tildes = v_tildes
mu_x_tilde = mu_x_tildes[0]
mu_v_tilde = mu_v_tildes[0]
## END SCRIB
