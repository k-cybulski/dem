"""
An example which tries to align internal state estimates of DEM with a Lorenz
attractor. Reproduction of this notebook from the dempy package:

    https://github.com/johmedr/dempy/blob/8689cfa99a472bfdbd78c1901e950a563ec1b7c1/notebooks/Lorenz%20model.ipynb

It's also very similar to the 'virtual birdsong' experiment from

    K. Friston and S. Kiebel, “Predictive coding under the free-energy
    principle,” Philos Trans R Soc Lond B Biol Sci, vol. 364, no. 1521, pp.
    1211–1221, May 2009, doi: 10.1098/rstb.2008.0300.
"""
import math
import numpy as np
import jax.numpy as jnp
from jax import jit
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from pathlib import Path
from dataclasses import replace

from hdm.core import len_generalized
from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import simulate_system, assert_system_func_equivalence
from hdm.dem.jax import (DEMInputJAX, DEMStateJAX, dem_step_d,
                             dem_step_precision, dem_step_m, dem_step_e,
                             free_action, extract_dynamic)

## Prior expectations (and true values)
params_true = np.array([18., 18., 46.92, 2., 1., 2., 4., 1., 1., 1.])
x0_true = np.array([0.9,0.8,30])

# Lorenz attractor transition function f
@jit
def lorenz(x, v, P):
    x0 = P[0] * x[1] - P[1] * x[0]
    x1 = P[2] * x[0] - P[3] * x[2] * x[0] - P[4] * x[1]
    x2 = P[5] * x[0] * x[1] - P[6] * x[2]

    return jnp.array([x0, x1, x2]) / 128.

# Observation function g
@jit
def obs(x, v, P):
    return x @ P[-3:]

def f_true(x, v):
    return lorenz(x, v, params_true)

def g_true(x, v):
    return obs(x, v, params_true)

# the DEM optimizer expects f and g to take and return shape (-1, 1)
def f_dem(x, v, P):
    x = x.reshape(-1)
    v = v.reshape(-1)
    return lorenz(x, v, P).reshape(-1, 1)

def g_dem(x, v, P):
    x = x.reshape(-1)
    v = v.reshape(-1)
    return obs(x, v, P).reshape(-1, 1)


rng = np.random.default_rng(215)
n = 1024
dt = 1
vs = np.zeros((n, 1))
noise_prec = np.exp(32)
noise_var = 1/noise_prec
noise_sd = np.sqrt(noise_var)
w_sd = noise_sd
z_sd = 1
noise_temporal_sig = 1/8
ts, xs, ys, ws, zs = simulate_system(f_true, g_true, x0_true, dt, vs, w_sd, z_sd, noise_temporal_sig, rng=rng)


p = 4
d = 2 # d is irrelevant, because the inputs v are all zero

x0_test = np.array([12,13,16]).reshape(-1, 1)
p_v = np.exp(5).reshape((1, 1))

eta_theta = params_true # prior equal to parameters, so we assume that they are known
p_theta = np.diag(np.ones(eta_theta.shape)) * np.exp(64) # high precision prior
eta_lambda = np.zeros(2)
p_lambda = np.eye(2) * np.exp(3)

m_x = 3
m_v = 1 # v are all zero in any case
m_y = 1

dem_input = DEMInputJAX(dt=dt, m_x=m_x, m_v=m_v, m_y=m_y, p=p, d=d,
                     ys=ys,
                     eta_v=vs,
                     p_v=p_v,
                     eta_theta=eta_theta,
                     p_theta=p_theta,
                     eta_lambda=eta_lambda,
                     p_lambda=p_lambda,
                     f=f_dem,
                     g=g_dem,
                     noise_temporal_sig=noise_temporal_sig)

dem_state = DEMStateJAX.from_input(dem_input, x0=x0_test)

# Run only the D step of DEM, since we assume the parameters to be already known
lr_dynamic = 1
dem_step_d(dem_state, lr_dynamic)

# Extract and plot the trajectories
mu_xs1, sig_xs1, mu_vs1, sig_vs1, ts1 = extract_dynamic(dem_state)

Path('out').mkdir(exist_ok=True) # make a path for output images
fig, ax = plt.subplots()
ax.plot(ts, xs[:, 0], label='Target 1', color='blue', linewidth=0.5)
ax.plot(ts1, mu_xs1[:, 0], label='Model 1', color='blue', linestyle='--', linewidth=0.75)
ax.plot(ts, xs[:, 1], label='Target 2', color='red', linewidth=0.5)
ax.plot(ts1, mu_xs1[:, 1], label='Model 2', color='red', linestyle='--', linewidth=0.75)
ax.plot(ts, xs[:, 2], label='Target 3', color='violet', linewidth=0.5)
ax.plot(ts1, mu_xs1[:, 2], label='Model 3', color='violet', linestyle='--', linewidth=0.75)
ax.legend()
fig.savefig('out/lorenz-01.pdf')
plt.close(fig)

# Unfortunately, the current implementation is not able to do the E, M, and
# precision steps, because the memory usage explodes :(
# I'm guessing that the parallelized computations of hessians are to blame. They could be made parallelized,

print("Computing free action...")
f_bar = dem_state.free_action()
