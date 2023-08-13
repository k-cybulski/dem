"""
Benchmarks for hdm.dem.batched

Can be used for profiling if pytest-profiling is installed.
"""

import pytest
import numpy as np
import torch
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

from hdm.core import len_generalized
from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import simulate_system
from hdm.dem.batched import DEMInput, DEMState, dem_step_d, dem_step_precision
from hdm.dem.naive import extract_dynamic
from hdm.dem.batched import DEMState, DEMInput, dem_step_d


@pytest.fixture
def dummy_dem_lorenz():
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
    # n = 1024
    n = 30
    dt = 0.5
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
    p_v = torch.tensor(np.exp(5).reshape((1, 1)), dtype=torch.float32)

    # doesn't matter, v is 0 anyway
    v_temporal_sig = 5
    v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    v_autocorr_inv_ = torch.linalg.inv(v_autocorr)

    noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

    eta_theta = torch.tensor(params_true, dtype=torch.float32)
    p_theta = torch.tensor(np.diag(np.ones(eta_theta.shape)) * np.exp(5), dtype=torch.float32)
    eta_lambda = torch.zeros(2) # Not sure how to initialize these well
    p_lambda = torch.eye(2) # Not sure how to initialize these well

    mu_x0 = torch.tensor(x0_test, dtype=torch.float32)
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

    def lorenz_torch_b(x, v, P):
        b_num = x.shape[0]
        x = x.reshape((b_num, -1))
        x0 = P[0] * x[:, 1] - P[1] * x[:,0]
        x1 = P[2] * x[:, 0] - P[3] * x[:,2] * x[:,0] - P[4] * x[:,1]
        x2 = P[5] * x[:, 0] * x[:,1] - P[6] * x[:,2]
        return (torch.concat([x0, x1, x2]) / 128.).reshape((b_num, -1, 1))

    def obs_torch_b(x, v, P):
        b_num = x.shape[0]
        x = x.reshape((b_num, -1))
        return torch.matmul(x, P[-3:]).reshape((b_num, -1, 1))

    m_x = 3
    m_v = 1
    m_y = 1

    dem_input = DEMInput(dt=dt, m_x=m_x, m_v=m_v, m_y=m_y, p=p, p_comp=p_comp,
                         ys=torch.tensor(ys, dtype=torch.float32),
                         eta_v=torch.tensor(vs, dtype=torch.float32),
                         p_v=p_v,
                         v_autocorr_inv=v_autocorr_inv_,
                         eta_theta=eta_theta,
                         p_theta=p_theta,
                         eta_lambda=eta_lambda,
                         p_lambda=p_lambda,
                         f=lorenz_torch_b,
                         g=obs_torch_b,
                         noise_autocorr_inv=noise_autocorr_inv_,
                         omega_w=torch.eye(m_x),
                         omega_z=torch.eye(m_y))
    dem_state = DEMState.from_input(dem_input, mu_x0)
    return dem_state

# Tests for crash and for profiling
def test_step_d(dummy_dem_lorenz):
    dem_state = dummy_dem_lorenz
    lr_dynamic = 0.01
    dem_step_d(dem_state, lr_dynamic)
