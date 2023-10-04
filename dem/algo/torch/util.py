import torch

def kron(A, B):
    """
    Kronecker product for matrices.

    This is a replacement for Torch's `torch.kron`, which is broken and crashes
    when called on an inverted matrix, similar to here:
        https://github.com/pytorch/pytorch/issues/54135

    Taken from Anton Obukhov's comment on GitHub:
        https://github.com/pytorch/pytorch/issues/74442#issuecomment-1111468515
    """
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

def _fix_grad_shape(tensor):
    """
    "Fixes" shape for outputs of torch.autograd.functional.jacobian or
    torch.autograd.functional.hessian. Transforms from dimension 4 to dimension
    2 just discarding two dimensions.
    """
    if tensor.dim() == 2:
        return tensor
    elif tensor.dim() == 4 and tensor.shape[1] == 1 and tensor.shape[3] == 1:
        return tensor.reshape((tensor.shape[0], tensor.shape[2]))
    else:
        raise ValueError(f"Unexpected hessian shape: {tuple(tensor.shape)}")

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

def generalized_batch_to_sequence(tensor, m, is2d=False):
    if not is2d:
        xs = torch.stack([x_tilde[:m].clone().detach() for x_tilde in tensor], axis=0)[:,:,0]
    else:
        xs = torch.stack([torch.diagonal(x_tilde)[:m].clone().detach() for x_tilde in tensor], axis=0)
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

def clear_gradients_on_state(state):
    state.mu_theta = state.mu_theta.detach().clone().requires_grad_()
    state.mu_lambda = state.mu_lambda.detach().clone().requires_grad_()
    state.mu_x0_tilde = state.mu_x0_tilde.detach().clone().requires_grad_()
    state.mu_v0_tilde = state.mu_v0_tilde.detach().clone().requires_grad_()
