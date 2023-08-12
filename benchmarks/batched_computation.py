"""
This little script is used to compare implementations of various functions used in computation of free action.
"""

import numpy as np
import torch
import timeit
from tabulate import tabulate

from matplotlib import pyplot as plt

from hdm.dummy import sin_gen, cos_gen, combine_gen
from hdm.dem.naive import _fix_grad_shape

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


if __name__ == '__main__':
    test_generalized_func_correctness()
    test_generalized_func_speed()
