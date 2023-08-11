"""
This module tests DEM implementations.
"""
import torch
import numpy as np
import math

from hdm.dummy import dummy_lti

from hdm.dem import naive, batched
from hdm.core import iterate_generalized, len_generalized
from hdm.noise import noise_cov_gen_theoretical, autocorr_friston

## Checking equivalence between batched and naive DEM

def dummy_lti_system(m_x, m_v, m_y, n, dt, v_sd, v_temporal_sig, w_sd, z_sd, noise_temporal_sig, seed=546):

    rng = np.random.default_rng(seed)

    A, B, C, D, x0, ts, vs, xs, ys, ws, zs = dummy_lti(
            m_x, m_v, m_y, n, dt, v_sd, v_temporal_sig, w_sd, z_sd, noise_temporal_sig, rng)

    A, B, C, D, x0, ts, vs, xs, ys, ws, zs = [torch.tensor(obj, dtype=torch.float32)
                                              for obj in [A, B, C, D, x0, ts, vs, xs, ys, ws, zs]]
    return A, B, C, D, x0, ts, vs, xs, ys, ws, zs

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

def dummy_dem_state(m_x, m_v, m_y, n, dt, p, p_comp, w_sd, z_sd,
                    noise_temporal_sig, v_temporal_sig,
                    seed, known_matrices=('B', 'C', 'D')):
    """
    Returns a hdm.dem.naive.DEMState for a dummy LTI inversion problem, along
    with target parameters.
    """
    v_sd = 1 # standard deviation on inputs

    A, B, C, D, x0, ts, vs, xs, ys, ws, zs = dummy_lti_system(
        m_x, m_v, m_y, n, dt, v_sd, v_temporal_sig, w_sd, z_sd, noise_temporal_sig, seed=seed)
    n_gen = len_generalized(n, p_comp)

    rng_torch = torch.random.manual_seed(seed + 53)

    # Prior precisions on known or unknown parameters
    precision_unknown = 1 # weak prior on unknown parmaeters
    precision_known = math.exp(6) # strong prior on known parameters

    # Temporal autocorrelation structure
    v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    v_autocorr_inv_ = torch.linalg.inv(v_autocorr)
    noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)


    # Initial state estimates
    mu_x0_tilde = torch.normal(torch.zeros((m_x * (p + 1), 1)),
                               torch.ones((m_x * (p + 1)), 1) / (m_x * (p + 1)),
                               generator=rng_torch)
    ## quite uninformative guess, also doesn't take into account expected temporal covariance
    sig_x_tilde0 = torch.eye(m_x * (p + 1))
    sig_x_tildes=[sig_x_tilde0.clone().detach() for _ in range(n_gen)]

    # Assume we know the inputs with high confidence
    mu_v_tildes = list(iterate_generalized(vs, dt, p, p_comp=p_comp))
    p_v = torch.eye(m_v) * precision_known
    sig_v_tildes = [
        naive.kron(v_autocorr_inv_, torch.linalg.inv(p_v)) for _ in range(n_gen)
    ]

    # Initial and prior parameters
    ## we construct a vector for theta with true values and high prior
    ## precisions on 'known' matrices
    params_size = m_x * m_x + m_x * m_v + m_y * m_x + m_y * m_v

    eta_thetas = []
    p_thetas = []
    mu_thetas = []
    sig_thetas =[]
    for matrix, matrix_name in ((A, 'A'), (B, 'B'), (C, 'C'), (D, 'D')):
        if matrix_name not in known_matrices:
            eta = torch.zeros(matrix.shape[0] * matrix.shape[1], dtype=torch.float32)
            p_ = torch.ones(matrix.shape[0] * matrix.shape[1], dtype=torch.float32) * precision_unknown # precision
            mu = torch.normal(torch.zeros(matrix.shape[0] * matrix.shape[1]),
                              torch.ones(matrix.shape[0] * matrix.shape[1]) / params_size,
                              generator=rng_torch)
            sig = 1 / p_
        else:
            eta = matrix.clone().to(dtype=torch.float32).reshape(-1)
            p_ = torch.ones(matrix.shape[0] * matrix.shape[1], dtype=torch.float32) * precision_known
            mu = eta.clone()
            sig = 1 / p_
        eta_thetas.append(eta)
        p_thetas.append(p_)
        mu_thetas.append(mu)
        sig_thetas.append(sig)
    eta_theta = torch.concat(eta_thetas)
    p_theta = torch.diag(torch.concat(p_thetas))
    mu_theta = torch.concat(mu_thetas)
    sig_theta = torch.diag(torch.concat(sig_thetas))

    # Initial and prior hyperparameters
    ## not much philosophy behind these
    eta_lambda = torch.zeros(2, dtype=torch.float32)
    p_lambda = torch.eye(2, dtype=torch.float32)
    mu_lambda = eta_lambda
    sig_lambda = torch.linalg.inv(p_lambda)

    # Covariance structure of noises
    ## the way we generate them now, they are necessarily independent
    omega_w=torch.eye(m_x, dtype=torch.float32)
    omega_z=torch.eye(m_y, dtype=torch.float32)

    def dem_f(x, v, params):
        A, B, C, D = ABCD_from_params(params)
        return A @ x + B @ v

    def dem_g(x, v, params):
        A, B, C, D = ABCD_from_params(params)
        return C @ x + D @ v

    dem_input_naive = naive.DEMInput(
        dt=dt,
        m_x=m_x,
        m_v=m_v,
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
    dem_state_naive = naive.DEMState(
        input=dem_input_naive,
        mu_x_tildes=None, # will be set in the D step
        mu_v_tildes=mu_v_tildes,
        sig_x_tildes=sig_x_tildes,
        sig_v_tildes=sig_v_tildes,
        mu_theta=mu_theta.clone(),
        mu_lambda=mu_lambda.clone(),
        sig_theta=sig_theta.clone(),
        sig_lambda=sig_lambda.clone(),
        mu_x0_tilde=mu_x0_tilde.clone(),
        mu_v0_tilde=mu_v_tildes[0].clone()
    )

    groundtruth = {
            'A': A.clone(),
            'B': B.clone(),
            'C': C.clone(),
            'D': D.clone(),
            'x0': x0.clone(),
            'xs': xs.clone(),
            'vs': vs.clone(),
            'ws': ws.clone(),
            'zs': zs.clone()
    }
    return dem_state_naive, groundtruth



def test_dem_eqiuv_naive_batched():
    m_x = 2
    m_v = 1
    m_y = 2

    n = 15
    dt = 0.1

    p = 4
    p_comp = 6

    w_sd = 0.1 # noise on states
    z_sd = 0.05 # noise on outputs
    noise_temporal_sig = 0.15 # temporal smoothing kernel parameter
    v_temporal_sig = 1 # temporal smoothness of inputs

    dem_state_naive, groundtruth = dummy_dem_state(
            m_x, m_v, m_y, n, dt, p, p_comp, w_sd, z_sd,
            noise_temporal_sig, v_temporal_sig,
            seed=513, known_matrices=('B', 'C', 'D'))

    def dem_f_batch(x, v, params):
        A, B, C, D = ABCD_from_params(params)
        return A @ x + B @ v

    def dem_g_batch(x, v, params):
        A, B, C, D = ABCD_from_params(params)
        return C @ x + D @ v

    batch_mu_v_tildes = torch.stack(dem_state_naive.mu_v_tildes)
    batch_sig_x_tildes = torch.stack(dem_state_naive.sig_x_tildes)
    batch_sig_v_tildes = torch.stack(dem_state_naive.sig_v_tildes)

    dem_input_batched = batched.DEMInput(
        dt=dt,
        m_x=m_x,
        m_v=m_v,
        p=p,
        p_comp=p_comp,
        ys=dem_state_naive.input.ys.clone(),
        eta_v=dem_state_naive.input.eta_v.clone(),
        p_v=dem_state_naive.input.p_v.clone(),
        v_autocorr_inv=dem_state_naive.input.v_autocorr_inv_.clone(),
        eta_theta=dem_state_naive.input.eta_theta.clone(),
        eta_lambda=dem_state_naive.input.eta_lambda.clone(),
        p_theta=dem_state_naive.input.p_theta.clone(),
        p_lambda=dem_state_naive.input.p_lambda.clone(),
        g=dem_g_batch, # FIXME
        f=dem_f_batch,
        omega_w=dem_state_naive.input.omega_w.clone(),
        omega_z=dem_state_naive.input.omega_z.clone(),
        noise_autocorr_inv=dem_state_naive.input.noise_autocorr_inv_.clone(),
    )
    dem_state_batched = batched.DEMState(
        input=dem_input_batched,
        mu_x_tildes=None,
        mu_v_tildes=batch_mu_v_tildes,
        sig_x_tildes=batch_sig_x_tildes,
        sig_v_tildes=batch_sig_v_tiles,
        mu_theta=dem_state_naive.mu_theta.clone(),
        mu_lambda=dem_state_naive.mu_lambda.clone(),
        sig_theta=dem_state_naive.sig_theta.clone(),
        sig_lambda=dem_state_naive.sig_lambda.clone(),
        mu_x0_tilde=dem_state_naive.mu_x0_tilde.clone(),
        mu_v0_tilde=dem_state_naive.mu_v0_tilde.clone()
    )

    naive.dem_step(dem_state_naive)

    dem_step_naive(dem_state_naive)
    dem_step_batched(dem_state_batched)

    assert torch.isclose(dem_step_naive.mu_theta, dem_step_batched.mu_theta)

if __name__ == '__main__':
    # test_dem_eqiuv_naive_batched()
    m_x = 2
    m_v = 1
    m_y = 2

    n = 15
    dt = 0.1

    p = 4
    p_comp = 6

    w_sd = 0.1 # noise on states
    z_sd = 0.05 # noise on outputs
    noise_temporal_sig = 0.15 # temporal smoothing kernel parameter
    v_temporal_sig = 1 # temporal smoothness of inputs

    dem_state_naive, groundtruth = dummy_dem_state(
            m_x, m_v, m_y, n, dt, p, p_comp, w_sd, z_sd,
            noise_temporal_sig, v_temporal_sig,
            seed=513, known_matrices=('B', 'C', 'D'))

    naive.dem_step_d(dem_state_naive, 1) # initialize x

    naive.internal_energy_static(
            dem_state_naive.mu_theta, dem_state_naive.mu_lambda,
            dem_state_naive.input.eta_theta, dem_state_naive.input.eta_lambda,
            dem_state_naive.input.p_theta, dem_state_naive.input.p_lambda, True)
    naive.internal_energy_dynamic(
            dem_state_naive.input.g, dem_state_naive.input.f,
            dem_state_naive.mu_x_tildes[0],
            dem_state_naive.mu_v_tildes[0],
            list(dem_state_naive.input.iter_y_tildes())[0], m_x, m_v, p,
            dem_state_naive.mu_theta,
            list(dem_state_naive.input.iter_eta_v_tildes())[0],
            list(dem_state_naive.input.iter_p_v_tildes())[0],
            dem_state_naive.mu_lambda,
            dem_state_naive.input.omega_w,
            dem_state_naive.input.omega_z,
            dem_state_naive.input.noise_autocorr_inv, True)
    naive.free_action_from_state(dem_state_naive).item()
