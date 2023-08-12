import torch

from .util import _fix_grad_shape, _fix_grad_shape_batch

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


def generalized_func_batched_manyhess(func, mu_x_tildes, mu_v_tildes, m_x, m_v, p, params):
    assert mu_x_tildes.shape[1:] == (m_x * (p + 1), 1)
    assert mu_v_tildes.shape[1:] == (m_v * (p + 1), 1)
    assert mu_x_tildes.shape[0] == mu_v_tildes.shape[0]
    mu_x = mu_x_tildes[:, :m_x]
    mu_v = mu_v_tildes[:, :m_v]
    n_batch = mu_x_tildes.shape[0]
    func_appl = func(mu_x, mu_v, params)
    mu_x_grad = torch.autograd.functional.jacobian(lambda x: func(x, mu_v, params), mu_x, create_graph=True)
    mu_v_grad = torch.autograd.functional.jacobian(lambda v: func(mu_x, v, params), mu_v, create_graph=True)
    mu_x_grad = _fix_grad_shape_batch(mu_x_grad)
    mu_v_grad = _fix_grad_shape_batch(mu_v_grad)

    mu_x_tildes_r = mu_x_tildes.reshape((n_batch, p + 1, m_x))
    mu_v_tildes_r = mu_v_tildes.reshape((n_batch, p + 1, m_v))

    func_appl_d_x = torch.einsum('bkj,bdj->bdk', mu_x_grad, mu_x_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    func_appl_d_v = torch.einsum('bkj,bdj->bdk', mu_v_grad, mu_v_tildes_r[:, 1:,:]).reshape((n_batch, -1, 1))
    return torch.concat((func_appl, func_appl_d_x + func_appl_d_v), dim=1)



def generalized_func_einsum(func, mu_x_tilde, mu_v_tilde, m_x, m_v, p, params):
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
