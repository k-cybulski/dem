"""
This little script is used to compare implementations of various functions used in computation of free action.
"""

import numpy as np
import torch
import timeit
from tabulate import tabulate
from itertools import repeat

from matplotlib import pyplot as plt

from hdm.core import iterate_generalized, deriv_mat
from hdm.noise import autocorr_friston, noise_cov_gen_theoretical
from hdm.dummy import sin_gen, cos_gen, combine_gen, simulate_system
from hdm.dem.naive import _fix_grad_shape, generalized_func, DEMState, DEMInput, dem_step_d, kron

from benchmarks.implementations.generalized_func import (
        generalized_func_naive, generalized_func_batched,
        generalized_func_batched_manyhess, generalized_func_einsum)
from benchmarks.implementations.internal_energy import (
        internal_energy_dynamic_naive,
        internal_energy_dynamic_onehess,
        internal_energy_dynamic_batched,
        internal_energy_dynamic_batched_manyhess)

##
## Helper functions
##
def dummy_dem_simple(n, rng=None):
    # Very simple DEM model
    # x' = Ax + w
    # y  = Ix + z
    if rng is None:
        rng = np.random.default_rng()
    elif not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    dt = 0.1
    m_x = 2
    m_v = 1
    m_y = 2

    # Dynamics definition
    x0 = np.array([0, 1])
    A = np.array([[0, 1], [-1, 0]])
    def f(x, v):
        return A @ x
    def g(x, v):
        return x

    # noise standard deviations
    w_sd = 0.01 # noise on states
    z_sd = 0.05 # noise on outputs
    noise_temporal_sig = 0.15 # temporal smoothing kernel parameter

    vs = torch.zeros((n, 1))
    v_temporal_sig = 1

    ts, xs, ys, ws, zs = simulate_system(f, g, x0, dt, vs, w_sd, z_sd, noise_temporal_sig, rng=rng)
    ts, xs, ys, ws, zs = [torch.tensor(m, dtype=torch.float32) for m in [ts, xs, ys, ws, zs]]

    # Define a DEM model
    def dem_f(x, v, params):
        params = params.reshape((2,2))
        return params @ x

    def dem_g(x, v, params):
        return x

    p = 4
    p_comp = p
    v_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=v_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    v_autocorr_inv_ = torch.linalg.inv(v_autocorr)
    noise_autocorr = torch.tensor(noise_cov_gen_theoretical(p, sig=noise_temporal_sig, autocorr=autocorr_friston()), dtype=torch.float32)
    noise_autocorr_inv_ = torch.linalg.inv(noise_autocorr)

    dem_input = DEMInput(
        dt=dt,
        m_x=m_x,
        m_v=m_v,
        m_y=m_y,
        p=p,
        p_comp=p_comp,
        ys=ys,
        eta_v=vs,
        p_v=torch.eye(m_v),
        v_autocorr_inv=v_autocorr_inv_,
        eta_theta=torch.tensor([0, 0, 0, 0], dtype=torch.float32),
        eta_lambda=torch.tensor([0, 0], dtype=torch.float32),
        p_theta=torch.eye(4, dtype=torch.float32),
        p_lambda=torch.eye(2, dtype=torch.float32),
        g=dem_g,
        f=dem_f,
        noise_autocorr_inv=noise_autocorr_inv_,
    )

    ideal_mu_x_tildes = list(iterate_generalized(xs, dt, p, p_comp=p_comp))
    ideal_mu_v_tildes = list(repeat(torch.zeros((m_v * (p + 1), 1), dtype=torch.float32), len(ideal_mu_x_tildes)))
    ideal_sig_x_tildes = list(repeat(torch.eye(m_x * (p + 1)), len(ideal_mu_x_tildes))) # uhh this probably isn't the ideal
    ideal_sig_v_tildes = list(repeat(torch.eye(m_v * (p + 1)), len(ideal_mu_x_tildes))) # uhh this probably isn't the ideal

    ideal_mu_x0_tilde = ideal_mu_x_tildes[0].clone()
    ideal_mu_v0_tilde = ideal_mu_v_tildes[0].clone()
    mu_lambda0 = torch.tensor([0, 0], dtype=torch.float32)
    sig_lambda0 = torch.eye(2) * 0.01
    mu_theta0 = torch.tensor([0,0,0,0], dtype=torch.float32)
    sig_theta0 = torch.eye(4) * 0.01
    sig_x_tilde0 = torch.eye(m_x * (p + 1))
    sig_v_tilde0 = torch.eye(m_v * (p + 1))

    dem_state = DEMState(
            input=dem_input,
            mu_x_tildes=[torch.zeros_like(mu_x_tilde) for mu_x_tilde in ideal_mu_x_tildes],
            mu_v_tildes=[torch.zeros_like(mu_v_tilde) for mu_v_tilde in ideal_mu_v_tildes],
            sig_x_tildes=[sig_x_tilde0.clone().detach() for _ in ideal_mu_x_tildes],
            sig_v_tildes=[sig_v_tilde0.clone().detach() for _ in ideal_mu_v_tildes],
            mu_theta=mu_theta0.clone().detach(),
            mu_lambda=mu_lambda0.clone().detach(),
            sig_theta=sig_theta0.clone().detach(),
            sig_lambda=sig_lambda0.clone().detach(),
            mu_x0_tilde=torch.zeros_like(ideal_mu_x0_tilde),
            mu_v0_tilde=torch.zeros_like(ideal_mu_v0_tilde))

    lr_dynamic = 1
    dem_step_d(dem_state, lr_dynamic)
    return dem_state



##
## generalized_func
##

def test_generalized_func_correctness():
    # Compares the outputs of batched generalized_func to standard
    p = 4
    x_tilde = torch.tensor(combine_gen(sin_gen(p, 0), cos_gen(p, 0)), requires_grad=True, dtype=torch.float32)
    v_tilde = torch.zeros((p + 1, 1), requires_grad=True, dtype=torch.float32)
    params = torch.tensor([[0, 1], [-1, 0], [1, 1]], dtype=torch.float32).reshape(-1).requires_grad_()

    m_x = 2
    m_v = 1

    def f(x, v, params):
        A = params.reshape((3,2))
        return torch.matmul(A, x)

    x_tildes_n = [
        torch.tensor(combine_gen(sin_gen(p, 0), cos_gen(p, 0)), requires_grad=True, dtype=torch.float32),
        torch.tensor(combine_gen(sin_gen(p, 0.5), cos_gen(p, 0.5)), requires_grad=True, dtype=torch.float32)
    ]
    v_tildes_n = [
        torch.zeros((p + 1, 1), requires_grad=True, dtype=torch.float32),
        torch.zeros((p + 1, 1), requires_grad=True, dtype=torch.float32)
    ]

    x_tildes_b = torch.stack(x_tildes_n).clone().detach().requires_grad_()
    v_tildes_b = torch.stack(v_tildes_n).clone().detach().requires_grad_()

    out_naive = [generalized_func_naive(f, x_tilde, v_tilde, m_x, m_v, p, params)
                 for x_tilde, v_tilde in zip(x_tildes_n, v_tildes_n)]

    out_einsum = [generalized_func_einsum(f, x_tilde, v_tilde, m_x, m_v, p, params)
                 for x_tilde, v_tilde in zip(x_tildes_n, v_tildes_n)]

    out_batched = generalized_func_batched(f, x_tildes_b, v_tildes_b, m_x, m_v, p, params)
    out_batched_manyhess = generalized_func_batched_manyhess(f, x_tildes_b, v_tildes_b, m_x, m_v, p, params)

    # Do the batched and naive functions get the same result?
    assert torch.isclose(torch.stack(out_naive), out_batched).all()
    assert torch.isclose(torch.stack(out_naive), torch.stack(out_einsum)).all()
    assert torch.isclose(torch.stack(out_naive), out_batched_manyhess).all()


def test_generalized_func_speed():
    nrun = 100
    ns = [10, 100, 250, 500, 1000]
    dt = 0.1

    m_x = 2
    m_v = 1
    p = 4
    params = torch.tensor([[0, 1], [-1, 0], [1, 1]], dtype=torch.float32).reshape(-1).requires_grad_()

    def f(x, v, params):
        A = params.reshape((3,2))
        return torch.matmul(A, x)


    t_naives = []
    t_einsums = []
    t_batcheds = []
    t_batched_manyhesses = []

    table_rows = []
    for n in ns:
        ts = np.arange(n) * dt
        x_tildes_n = [
            torch.tensor(combine_gen(sin_gen(p, 0), cos_gen(p, t)), requires_grad=True, dtype=torch.float32)
            for t in ts
        ]
        v_tildes_n = [
            torch.zeros((p + 1, 1), requires_grad=True, dtype=torch.float32)
            for t in ts
        ]
        x_tildes_b = torch.stack(x_tildes_n).clone().detach().requires_grad_()
        v_tildes_b = torch.stack(v_tildes_n).clone().detach().requires_grad_()

        t_naives.append(timeit.timeit(lambda: [generalized_func_naive(f, x_tilde, v_tilde, m_x, m_v, p, params)
                     for x_tilde, v_tilde in zip(x_tildes_n, v_tildes_n)],
                                      number=nrun))
        t_einsums.append(timeit.timeit(lambda: [generalized_func_einsum(f, x_tilde, v_tilde, m_x, m_v, p, params)
                     for x_tilde, v_tilde in zip(x_tildes_n, v_tildes_n)],
                                       number=nrun))
        t_batcheds.append(timeit.timeit(lambda:generalized_func_batched(f, x_tildes_b, v_tildes_b, m_x, m_v, p, params),
                                        number=nrun))
        t_batched_manyhesses.append(timeit.timeit(lambda:generalized_func_batched_manyhess(f, x_tildes_b, v_tildes_b, m_x, m_v, p, params),
                                        number=nrun))
        table_rows.append({
            'n': n,
            't_naive (sec per 100 runs)': t_naives[-1],
            't_einsums (sec per 100 runs)': t_einsums[-1],
            't_batcheds (sec per 100 runs)': t_batcheds[-1],
            't_batched_manyhesses (sec per 100 runs)': t_batched_manyhesses[-1],
            })
    print(tabulate(table_rows, headers='keys'))
    plt.plot(ns, t_naives, label='Naive')
    plt.plot(ns, t_einsums, label='Einsum')
    plt.plot(ns, t_batcheds, label='Batched')
    plt.legend()
    plt.show()

##
## internal_energy_dynamic
##

def _clone_list_of_tensors(ls):
    return [l.clone().detach() for l in ls]

def test_internal_energy_dynamic_correctness():
    n = 30
    rng = np.random.default_rng(25)
    dem_state = dummy_dem_simple(n, rng=rng)

    y_tildes_n = list(dem_state.input.iter_y_tildes())
    eta_v_tildes_n = list(dem_state.input.iter_eta_v_tildes())
    p_v_tildes_n = list(dem_state.input.iter_p_v_tildes())
    mu_x_tildes_n = dem_state.mu_x_tildes
    mu_v_tildes_n = dem_state.mu_v_tildes
    omega_w = dem_state.input.omega_w
    omega_z = dem_state.input.omega_z
    m_x = dem_state.input.m_x
    m_v = dem_state.input.m_v
    p = dem_state.input.p
    mu_theta = dem_state.mu_theta
    mu_lambda = dem_state.mu_lambda

    y_tildes_b, eta_v_tildes_b, p_v_tildes_b, mu_x_tildes_b, mu_v_tildes_b = [
            torch.stack(ls) for ls in
            [y_tildes_n, eta_v_tildes_n, p_v_tildes_n, mu_x_tildes_n, mu_v_tildes_n]]

    out_naive = [
        internal_energy_dynamic_naive(
            dem_state.input.g,
            dem_state.input.f,
            mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p,
            dem_state.mu_theta,
            eta_v_tilde,
            p_v_tilde,
            dem_state.mu_lambda,
            dem_state.input.omega_w,
            dem_state.input.omega_z,
            dem_state.input.noise_autocorr_inv)
        for mu_x_tilde, mu_v_tilde, y_tilde, eta_v_tilde, p_v_tilde in zip(
            mu_x_tildes_n,
            mu_v_tildes_n,
            y_tildes_n,
            eta_v_tildes_n,
            p_v_tildes_n)
    ]

    out_onehess = [
            internal_energy_dynamic_onehess(
            dem_state.input.g,
            dem_state.input.f,
            mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p,
            dem_state.mu_theta,
            eta_v_tilde,
            p_v_tilde,
            dem_state.mu_lambda,
            dem_state.input.omega_w,
            dem_state.input.omega_z,
            dem_state.input.noise_autocorr_inv)
        for mu_x_tilde, mu_v_tilde, y_tilde, eta_v_tilde, p_v_tilde in zip(
            mu_x_tildes_n,
            mu_v_tildes_n,
            y_tildes_n,
            eta_v_tildes_n,
            p_v_tildes_n)
    ]
    assert all(all(torch.isclose(t1, t2).all() for t1, t2 in zip(tup1, tup2)) for tup1, tup2 in zip(out_naive, out_onehess))

    out_batched = internal_energy_dynamic_batched(
            dem_state.input.g,
            dem_state.input.f,
            mu_x_tildes_b, mu_v_tildes_b, y_tildes_b, m_x, m_v, p,
            dem_state.mu_theta,
            eta_v_tildes_b,
            p_v_tildes_b,
            dem_state.mu_lambda,
            dem_state.input.omega_w,
            dem_state.input.omega_z,
            dem_state.input.noise_autocorr_inv)

    out_batched_manyhess = internal_energy_dynamic_batched_manyhess(
            dem_state.input.g,
            dem_state.input.f,
            mu_x_tildes_b, mu_v_tildes_b, y_tildes_b, m_x, m_v, p,
            dem_state.mu_theta,
            eta_v_tildes_b,
            p_v_tildes_b,
            dem_state.mu_lambda,
            dem_state.input.omega_w,
            dem_state.input.omega_z,
            dem_state.input.noise_autocorr_inv)

    assert all(torch.isclose(t1, t2).all() for t1, t2 in zip(out_batched, out_batched_manyhess))

    # need to do an awkward transpose
    out_naive_listed = list(zip(*out_naive))

    # Compare dynamic terms
    for name, tup1, tup2 in zip(['u_t', 'x_tilde', 'v_tilde'], out_naive_listed[:3], out_batched[:3]):
        for t, (t1, t2) in enumerate(zip(tup1, tup2)):
            assert torch.isclose(t1, t2).all()

    # We expect the sum of static terms in naive to be the same as the output in out_batched
    # these terms are only ever used as a sum in any case, so we don't lose
    # important information by using a batched sum
    for name, tup1, tensor2 in zip(['t_theta', 't_lambda'], out_naive_listed[3:], out_batched[3:]):
        # numerical errors become larger here, so we need to increase tolerance
        assert torch.isclose(torch.sum(torch.stack(tup1), axis=0), tensor2,
                             rtol=1e-04).all()

    # Scribbly scrib
    def dem_f_b(x, v, params):
        params = params.reshape((2,2))
        return torch.matmul(params, x)

    def dem_g_b(x, v, params):
        return x

    f = dem_f_b
    g = dem_g_b
    mu_x_tildes = mu_x_tildes_b
    mu_v_tildes = mu_v_tildes_b
    y_tildes = y_tildes_b
    eta_v_tildes = eta_v_tildes_b
    p_v_tildes = p_v_tildes_b
    noise_autocorr_inv = dem_state.input.noise_autocorr_inv

    n_batch = mu_x_tildes.shape[0]

    # We're in the function now

def test_internal_energy_dynamic_speed():
    ns = [10, 25, 50, 100, 250, 500, 1000]
    nrun = 20

    times = []
    dem_states = {}
    for idx, n in enumerate(ns):
        rng = np.random.default_rng(25)
        dem_state = dummy_dem_simple(n)
        dem_states[n] = dem_state
        times.append({
            't_naive': [],
            't_onehess': [],
            't_batched': [],
            't_batched_manyhess': []
        })

    table_rows = [{}] * len(times)

    for run_num in range(nrun):
        for idx, n in enumerate(ns):
            dem_state = dem_states[n]

            y_tildes = list(dem_state.input.iter_y_tildes())
            eta_v_tildes = list(dem_state.input.iter_eta_v_tildes())
            p_v_tildes = list(dem_state.input.iter_p_v_tildes())
            mu_x_tildes = dem_state.mu_x_tildes
            mu_v_tildes = dem_state.mu_v_tildes
            omega_w = dem_state.input.omega_w
            omega_z = dem_state.input.omega_z
            m_x = dem_state.input.m_x
            m_v = dem_state.input.m_v
            p = dem_state.input.p
            mu_theta = dem_state.mu_theta
            mu_lambda = dem_state.mu_lambda

            y_tildes_b, eta_v_tildes_b, p_v_tildes_b, mu_x_tildes_b, mu_v_tildes_b = [
                    torch.stack(ls) for ls in
                    [y_tildes, eta_v_tildes, p_v_tildes, mu_x_tildes, mu_v_tildes]]

            t_naive = timeit.timeit(lambda: [
                internal_energy_dynamic_naive(
                    dem_state.input.g,
                    dem_state.input.f,
                    mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p,
                    dem_state.mu_theta,
                    eta_v_tilde,
                    p_v_tilde,
                    dem_state.mu_lambda,
                    dem_state.input.omega_w,
                    dem_state.input.omega_z,
                    dem_state.input.noise_autocorr_inv)
                for mu_x_tilde, mu_v_tilde, y_tilde, eta_v_tilde, p_v_tilde in zip(
                    mu_x_tildes,
                    mu_v_tildes,
                    y_tildes,
                    eta_v_tildes,
                    p_v_tildes)
            ], number=nrun)
            times[idx]['t_naive'].append(t_naive)
            t_onehess = timeit.timeit(lambda: [
                internal_energy_dynamic_onehess(
                    dem_state.input.g,
                    dem_state.input.f,
                    mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p,
                    dem_state.mu_theta,
                    eta_v_tilde,
                    p_v_tilde,
                    dem_state.mu_lambda,
                    dem_state.input.omega_w,
                    dem_state.input.omega_z,
                    dem_state.input.noise_autocorr_inv)
                for mu_x_tilde, mu_v_tilde, y_tilde, eta_v_tilde, p_v_tilde in zip(
                    mu_x_tildes,
                    mu_v_tildes,
                    y_tildes,
                    eta_v_tildes,
                    p_v_tildes)
            ], number=nrun)
            times[idx]['t_onehess'].append(t_onehess)
            t_batched = timeit.timeit(lambda :internal_energy_dynamic_batched(
                    dem_state.input.g,
                    dem_state.input.f,
                    mu_x_tildes_b, mu_v_tildes_b, y_tildes_b, m_x, m_v, p,
                    dem_state.mu_theta,
                    eta_v_tildes_b,
                    p_v_tildes_b,
                    dem_state.mu_lambda,
                    dem_state.input.omega_w,
                    dem_state.input.omega_z,
                    dem_state.input.noise_autocorr_inv), number=nrun)
            times[idx]['t_batched'].append(t_batched)
            t_batched_manyhess = timeit.timeit(lambda :internal_energy_dynamic_batched_manyhess(
                    dem_state.input.g,
                    dem_state.input.f,
                    mu_x_tildes_b, mu_v_tildes_b, y_tildes_b, m_x, m_v, p,
                    dem_state.mu_theta,
                    eta_v_tildes_b,
                    p_v_tildes_b,
                    dem_state.mu_lambda,
                    dem_state.input.omega_w,
                    dem_state.input.omega_z,
                    dem_state.input.noise_autocorr_inv), number=nrun)
            times[idx]['t_batched_manyhess'].append(t_batched_manyhess)
            table_rows[idx] = {
                'n': n,
                't_manyhess': np.mean(times[idx]['t_naive']),
                't_onehess': np.mean(times[idx]['t_onehess']),
                't_batched_manyhess': np.mean(times[idx]['t_batched_manyhess']),
                't_batched_onehess ': np.mean(times[idx]['t_batched']),
                'as of': run_num + 1,
                }
            print("Interim table:")
            print(tabulate(table_rows, headers='keys'))
    print(tabulate(table_rows, headers='keys'))
    plt.plot(ns, t_naives, label='Naive')
    plt.plot(ns, t_onehesses, label='One Hess')
    plt.plot(ns, t_batcheds, label='Batched')
    plt.plot(ns, t_batched_manyhesses, label='Batched Many Hess')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print("Checking correctness...")
    test_generalized_func_correctness()
    test_internal_energy_dynamic_correctness()

    print("Running speed benchmarks...")
    test_generalized_func_speed()
    test_internal_energy_dynamic_speed()
    pass
