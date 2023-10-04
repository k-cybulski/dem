"""
Tests for checking equivalence between implementations.
"""

# This script is quite dirty, but it does the job...

import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random
import jax
import torch

from jax import config
config.update("jax_enable_x64", True)

from dem.algo.jax.algo import (DEMInputJAX, DEMStateJAX)

from dem.algo.torch.batched import DEMInputBatched, DEMStateBatched


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
            print(f"JAX: {getattr(state_jax, attr)}")
            print(f"PyTorch: {getattr(state_torch, attr).detach().numpy()}")
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
            print(f"JAX: {getattr(state_jax.input, attr)}")
            print(f"PyTorch: {getattr(state_torch.input, attr).detach().numpy()}")
            equivalent = False
    return equivalent



def test_equivalence_jax_batched():


    ##########
    ########## Test setup
    ##########

    import numpy as np
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import pickle
    from tabulate import tabulate
    from pathlib import Path

    from dem.core import iterate_generalized
    from dem.noise import autocorr_friston, noise_cov_gen_theoretical
    from dem.algo.torch.util import extract_dynamic
    from dem.dummy import simulate_colored_lti

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
    t_max = 2
    dt = 0.1
    # input pulse
    vs = np.exp(-0.25 * (np.arange(0, t_max, dt) - 4)**2).reshape((-1, 1))
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
    known_value_exp = 20


    ######### Utils

    # https://github.com/google/jax/pull/762#issuecomment-1002267121
    import functools

    ##########
    ########## Making DEM work
    ##########

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

    dem_input = DEMInputJAX(dt=dt, m_x=m_x, m_v=m_v, m_y=m_y, p=p, d=d,
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

    dem_state.mu_x_tildes = mu_x_tildes
    dem_state.mu_v_tildes = mu_v_tildes
    dem_state.mu_x0_tilde = mu_x_tildes[0]
    dem_state.mu_v0_tilde = mu_v_tildes[0]

    ## As a validity check, make a copy of the above but with torch instead

    def f_torch(x, v, params):
        A = params[0:(m_x * m_x)].reshape((m_x, m_x))
        B = params[(m_x * m_x):(m_x * m_x + m_x * m_v)].reshape((m_x, m_v))
        return torch.matmul(A, x) + torch.matmul(B, v)

    def g_torch(x, v, params):
        C = params[(m_x * m_x + m_x * m_v):(m_x * m_x + m_x * m_v + m_y * m_x)].reshape((m_y, m_x))
        return torch.matmul(C, x)

    TORCH_DTYPE = torch.float64
    TORCH_DEVICE = 'cpu'
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

    # Let's see if it works

    # Let's see if the JAX and PyTorch implementation follow the same trajectories
    lr_dynamic = 1
    lr_theta = 10 # from the matlab code
    lr_lambda = 1
    iter_lambda = 8 # from the matlab code
    m_min_improv = 0.01
    num_iter = 2
    for i in tqdm(range(num_iter), desc="Comparing JAX and PyTorch DEM trajectories"):
        print(f"{i}. Step D")
        dem_state.step_d(lr_dynamic)
        dem_state_b.step_d(lr_dynamic)
        assert compare_states_jax_torch(dem_state, dem_state_b)

        print(f"{i}. Step M")
        dem_state.step_m(lr_lambda, iter_lambda, min_improv=m_min_improv)
        dem_state_b.step_m(lr_lambda, iter_lambda, min_improv=m_min_improv)
        assert compare_states_jax_torch(dem_state, dem_state_b)

        print(f"{i}. Step E")
        dem_state.step_e(lr_theta)
        dem_state_b.step_e(lr_theta)
        assert compare_states_jax_torch(dem_state, dem_state_b)

        print(f"{i}. Step precision")
        dem_state.dem_step_precision()
        dem_state_b.dem_step_precision_b()
        assert compare_states_jax_torch(dem_state, dem_state_b)
