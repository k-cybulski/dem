


def _fix_grad_shape(tensor):
    """
    "Fixes" shape for outputs of torch.autograd.functional.jacobian or
    torch.autograd.functional.hessian. Transforms from dimension 4 to dimension
    2 just discarding two dimensions.
    """
    # FIXME: Is this necessary? I'm not sure I understand the output shape of
    # these functions.
    # Their output shapes are a bit peculiar for our case. It has dimension 4.
    # I'm guessing that this is because PyTorch can be very flexible in the
    # input/output shapes, also considering cases like minibatches. For now,
    # this solution *seems* to work (for no minibatches)

    # It seems that if the parameters are a (n, 1) matrix, then the Hessian has
    # 4 dimensions (n, 1, n, 1)
    # if the parameters are a (n,) array, then the Hessian has 2 dimensions (n, n)
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
