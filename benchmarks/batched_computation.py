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

def _fix_grad_shape_batch(tensor):
    ndim = tensor.dim()
    if ndim == 6:
        batch_n = tensor.shape[0]
        batch_selection = range(batch_n)
        out_n = tensor.shape[1]
        in_n = tensor.shape[4]
        # NOTE: The tensor includes all cross-batch derivatives too, which are always zero
        # hopefully this doesn't lead to unnecessary computations...
        return tensor[batch_selection,:,0,batch_selection,:,0]
    else:
        raise ValueError(f"Unexpected hessian shape: {tuple(tensor.shape)}")

def generalized_func_naive(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
    """
    Computes generalized application of a function, assuming local linearity.
    Used to find g_tilde and f_tilde.

    Args:
        func (function): function to apply (g or f)
        mu_x_tilde (Tensor): generalized vector of estimated state means
        mu_v_tilde (Tensor): generalized vector of estimated input means
        m_x (int): number of elements in state vector
        m_v (int): number of elements in input vector
        p (int): numer of derivatives in generalized vectors
        params (Tensor): parameters of func
    """
    # TODO: Ensure correctness of shapes, e.g. shape (2, 1) instead of (2,). This causes lots of errors down the road.
    assert mu_x_tilde.shape == (m_x * (p + 1), 1)
    assert mu_v_tilde.shape == (m_v * (p + 1), 1)
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad, mu_v_grad = torch.autograd.functional.jacobian(lambda x, v: func(x, v, params), (mu_x, mu_v), create_graph=True)
    mu_x_grad = _fix_grad_shape(mu_x_grad)
    mu_v_grad = _fix_grad_shape(mu_v_grad)

    func_appl_d = []
    for deriv in range(1, p + 1):
        mu_x_d = mu_x_tilde[deriv*m_x:(deriv+1)*m_x]
        mu_v_d = mu_v_tilde[deriv*m_v:(deriv+1)*m_v]
        func_appl_d.append(mu_x_grad @ mu_x_d + mu_v_grad @ mu_v_d)
    return torch.vstack([func_appl] + func_appl_d)

def generalized_func_batched(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    """
    Computes generalized application of a function, assuming local linearity.
    Used to find g_tilde and f_tilde.

    Args:
        func (function): function to apply (g or f)
        mu_x_tildes (Tensor): batch of generalized vectors of estimated state means
        mu_v_tildes (Tensor): batch of generalized vectors of estimated input means
        m_x (int): number of elements in state vector
        m_v (int): number of elements in input vector
        p (int): numer of derivatives in generalized vectors
        params (Tensor): parameters of func
    """
    # TODO: Ensure correctness of shapes, e.g. shape (2, 1) instead of (2,). This causes lots of errors down the road.
    assert mu_x_tildes.shape[1:] == (m_x * (p + 1), 1)
    assert mu_v_tildes.shape[1:] == (m_v * (p + 1), 1)
    assert mu_x_tildes.shape[0] == mu_v_tildes.shape[0]
    mu_x = mu_x_tildes[:, :m_x]
    mu_v = mu_v_tildes[:, :m_v]
    n_batch = mu_x_tildes.shape[0]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad, mu_v_grad = torch.autograd.functional.jacobian(lambda x, v: func(x, v, params), (mu_x, mu_v), create_graph=True)
    mu_x_grad = _fix_grad_shape_batch(mu_x_grad)
    mu_v_grad = _fix_grad_shape_batch(mu_v_grad)

    mu_x_tildes_r = mu_x_tildes.reshape((n_batch, p + 1, m_x))
    mu_v_tildes_r = mu_v_tildes.reshape((n_batch, p + 1, m_v))

    func_appl_d_x = torch.einsum('bkj,bdj->bdk', mu_x_grad, mu_x_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    func_appl_d_v = torch.einsum('bkj,bdj->bdk', mu_v_grad, mu_v_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    return torch.concat((func_appl, func_appl_d_x + func_appl_d_v), dim=1)


def generalized_func_einsum(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
    """
    Computes generalized application of a function, assuming local linearity.
    Used to find g_tilde and f_tilde.

    Args:
        func (function): function to apply (g or f)
        mu_x_tilde (Tensor): generalized vector of estimated state means
        mu_v_tilde (Tensor): generalized vector of estimated input means
        m_x (int): number of elements in state vector
        m_v (int): number of elements in input vector
        p (int): numer of derivatives in generalized vectors
        params (Tensor): parameters of func
    """
    # TODO: Ensure correctness of shapes, e.g. shape (2, 1) instead of (2,). This causes lots of errors down the road.
    assert mu_x_tilde.shape == (m_x * (p + 1), 1)
    assert mu_v_tilde.shape == (m_v * (p + 1), 1)
    mu_x = mu_x_tilde[:m_x]
    mu_v = mu_v_tilde[:m_v]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad, mu_v_grad = torch.autograd.functional.jacobian(lambda x, v: func(x, v, params), (mu_x, mu_v), create_graph=True)
    mu_x_grad = _fix_grad_shape(mu_x_grad)
    mu_v_grad = _fix_grad_shape(mu_v_grad)

    mu_x_tilde_r = mu_x_tilde.reshape((p + 1, m_x))
    mu_v_tilde_r = mu_v_tilde.reshape((p + 1, m_v))
    func_appl_d_x = torch.einsum('kj,dj->dk', mu_x_grad, mu_x_tilde_r[1:,:]).reshape((-1, 1))
    func_appl_d_v = torch.einsum('kj,dj->dk', mu_v_grad, mu_v_tilde_r[1:,:]).reshape((-1, 1))
    return torch.concat((func_appl, func_appl_d_x + func_appl_d_v), dim=0)

def test_generalized_func_correctness():
    # Compares the outputs of batched generalized_func to standard
    p = 5
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

    # Do the batched and naive functions get the same result?
    assert torch.isclose(torch.stack(out_naive), out_batched).all()
    assert torch.isclose(torch.stack(out_naive), torch.stack(out_einsum)).all()


def test_generalized_func_speed():
    nrun = 100
    ns = [10, 100, 250, 500, 1000]
    dt = 0.1

    t_naives = []
    t_einsums = []
    t_batcheds = []

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
        table_rows.append({
            'n': n,
            't_naive (sec per 100 runs)': t_naives[-1],
            't_einsums (sec per 100 runs)': t_einsums[-1],
            't_batcheds (sec per 100 runs)': t_batcheds[-1]})
    print(tabulate(table_rows, headers='keys'))
    plt.plot(ns, t_naives, label='Naive')
    plt.plot(ns, t_einsums, label='Einsum')
    plt.plot(ns, t_batcheds, label='Batched')
    plt.legend()
    plt.show()

##
## internal_energy_dynamic
##

def internal_energy_dynamic_naive(
        g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    """
    Computes dynamic terms of the internal energy for a single timestep, along
    with necessary Hessians. These are the precision-weighted errors and
    precision log determinants on the dynamic states. Hessians are returned for
    parameters theta and hyperparameters lambda as well.
    """
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)
    # make a temporary function which we can use to compute hessians w.r.t. the relevant parameters
    # for the computation of mean-field terms
    def _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda):
        g_tilde = generalized_func(g, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)
        f_tilde = generalized_func(f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)

        err_y = y_tilde - g_tilde
        err_v = mu_v_tilde - eta_v_tilde
        err_x = deriv_mat_x @ mu_x_tilde - f_tilde

        # we need to split up mu_lambda into the hyperparameter for noise of states and of outputs
        # the hyperparameters are just a single lambda scalar, one for the states and one for the outputs
        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -err_y.T @ prec_z_tilde @ err_y + torch.logdet(prec_z_tilde)
        u_t_v_ = -err_v.T @ p_v_tilde @ err_v + torch.logdet(p_v_tilde)
        u_t_x_ = -err_x.T @ prec_w_tilde @ err_x + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t
    u_t = _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda)
    # horribly inefficient way to go about this, but hey, at least it may work...
    # (so many unnecessary repeated computations)

    # FIXME OPT: Optimize the code below. Don't run
    # torch.autograd.functional.hessian four times separately? Running it once
    # should allow for all the necessary outputs. But it might unnecessarily
    # compute Hessians _between_ the parameters, which might be slower?
    u_t_x_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu, mu_v_tilde, mu_theta, mu_lambda), mu_x_tilde, create_graph=True)
    u_t_v_tilde_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu, mu_theta, mu_lambda), mu_v_tilde, create_graph=True)
    u_t_theta_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu, mu_lambda), mu_theta, create_graph=True)
    u_t_lambda_dd = torch.autograd.functional.hessian(lambda mu: _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu), mu_lambda, create_graph=True)

    u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd )
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd

def internal_energy_dynamic_onehess(
        g, f, mu_x_tilde, mu_v_tilde, y_tilde, m_x, m_v, p, mu_theta, eta_v_tilde, p_v_tilde,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)
    # make a temporary function which we can use to compute hessians w.r.t. the relevant parameters
    # for the computation of mean-field terms
    def _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda):
        g_tilde = generalized_func(g, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)
        f_tilde = generalized_func(f, mu_x_tilde, mu_v_tilde, m_x, m_v, p, mu_theta)

        err_y = y_tilde - g_tilde
        err_v = mu_v_tilde - eta_v_tilde
        err_x = deriv_mat_x @ mu_x_tilde - f_tilde

        # we need to split up mu_lambda into the hyperparameter for noise of states and of outputs
        # the hyperparameters are just a single lambda scalar, one for the states and one for the outputs
        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -err_y.T @ prec_z_tilde @ err_y + torch.logdet(prec_z_tilde)
        u_t_v_ = -err_v.T @ p_v_tilde @ err_v + torch.logdet(p_v_tilde)
        u_t_x_ = -err_x.T @ prec_w_tilde @ err_x + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t
    u_t = _int_eng_dynamic(mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda)
    # horribly inefficient way to go about this, but hey, at least it may work...
    # (so many unnecessary repeated computations)

    # FIXME OPT: Optimize the code below. Don't run
    # torch.autograd.functional.hessian four times separately? Running it once
    # should allow for all the necessary outputs. But it might unnecessarily
    # compute Hessians _between_ the parameters, which might be slower?
    dds = torch.autograd.functional.hessian(
            lambda mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda: _int_eng_dynamic(
                mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda),
            (mu_x_tilde, mu_v_tilde, mu_theta, mu_lambda),
            create_graph=True)
    u_t_x_tilde_dd = dds[0][0]
    u_t_v_tilde_dd = dds[1][1]
    u_t_theta_dd = dds[2][2]
    u_t_lambda_dd = dds[3][3]

    u_t_x_tilde_dd = _fix_grad_shape(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd )
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd

def internal_energy_dynamic_batched(
        g, f, mu_x_tildes, mu_v_tildes, y_tildes, m_x, m_v, p, mu_theta, eta_v_tildes, p_v_tildes,
        mu_lambda, omega_w, omega_z, noise_autocorr_inv):
    deriv_mat_x = torch.from_numpy(deriv_mat(p, m_x)).to(dtype=torch.float32)

    def _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda):
        mu_theta = mu_theta
        mu_lambda = mu_lambda
        g_tildes = generalized_func_batched(g, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)
        f_tildes = generalized_func_batched(f, mu_x_tildes, mu_v_tildes, m_x, m_v, p, mu_theta)

        err_y = y_tildes - g_tildes
        err_v = mu_v_tildes - eta_v_tildes
        err_x = torch.matmul(deriv_mat_x, mu_x_tildes) - f_tildes

        n_batch = mu_x_tildes.shape[0]

        mu_lambda_z = mu_lambda[0]
        mu_lambda_w = mu_lambda[1]
        prec_z = torch.exp(mu_lambda_z) * omega_z
        prec_w = torch.exp(mu_lambda_w) * omega_w
        prec_z_tilde = kron(noise_autocorr_inv, prec_z)
        prec_w_tilde = kron(noise_autocorr_inv, prec_w)

        u_t_y_ = -torch.bmm(err_y.mT, torch.matmul(prec_z_tilde, err_y)).reshape(n_batch) + torch.logdet(prec_z_tilde)
        u_t_v_ = -torch.bmm(err_v.mT, torch.bmm(p_v_tildes, err_v)).reshape(n_batch) + torch.logdet(p_v_tildes)
        u_t_x_ = -torch.bmm(err_x.mT, torch.matmul(prec_w_tilde, err_x)).reshape(n_batch) + torch.logdet(prec_w_tilde)

        u_t = (u_t_y_ + u_t_v_ + u_t_x_) / 2
        return u_t

    u_t = _int_eng_dynamic(mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)
    dds = torch.autograd.functional.hessian(
            lambda mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda: torch.sum(_int_eng_dynamic(
                mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda)),
            (mu_x_tildes, mu_v_tildes, mu_theta, mu_lambda),
            create_graph=True)
    u_t_x_tilde_dd = dds[0][0]
    u_t_v_tilde_dd = dds[1][1]
    u_t_theta_dd = dds[2][2]
    u_t_lambda_dd = dds[3][3]

    u_t_x_tilde_dd = _fix_grad_shape_batch(u_t_x_tilde_dd)
    u_t_v_tilde_dd = _fix_grad_shape_batch(u_t_v_tilde_dd)
    u_t_theta_dd  = _fix_grad_shape(u_t_theta_dd)
    u_t_lambda_dd = _fix_grad_shape(u_t_lambda_dd)
    return u_t, u_t_x_tilde_dd, u_t_v_tilde_dd, u_t_theta_dd, u_t_lambda_dd

def _clone_list_of_tensors(ls):
    return [l.clone().detach() for l in ls]


# def test_internal_energy_dynamic_correctness():
if __name__ == '__main__':
    n = 30
    nrun = 20
    rng = np.random.default_rng(25)
    dem_state = dummy_dem_simple(n)

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
    ns = [10, 50, 100]
    nrun = 20

    t_naives = []
    t_onehess = []
    t_batcheds = []

    table_rows = []

    for n in ns:
        rng = np.random.default_rng(25)
        dem_state = dummy_dem_simple(n)

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
                [y_tildes_n, eta_v_tildes_n, p_v_tildes_n, mu_x_tildes_n, mu_v_tildes_n]]

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
        t_oneness = timeit.timeit(lambda: [
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
        t_naives.append(t_naive)
        t_onehess.append(t_onehess)
        t_batcheds.append(t_batched)
        table_rows.append({
            'n': n,
            't_naive (sec per 20 runs)': t_naives[-1],
            't_onehess (sec per 20 runs)': t_onehess[-1],
            't_batched (sec per 20 runs)': t_batcheds[-1]})
    print(tabulate(table_rows, headers='keys'))
    plt.plot(ns, t_naives, label='Naive')
    plt.plot(ns, t_onehess, label='One Hess')
    plt.plot(ns, t_batcheds, label='Batched')
    plt.legend()
    plt.show()


# if __name__ == '__main__':
    # test_generalized_func_correctness()
    # test_generalized_func_speed()
    pass
