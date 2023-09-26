import numpy as np
import torch
import pytest

import hdm.core

@pytest.fixture
def ex_sincos():
    t_start = 0
    t_end = 20
    t_span = (t_start, t_end)
    dt = 0.1
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    x1s = np.sin(ts)
    x2s = np.cos(ts)
    ys = ys = np.stack([x1s, x2s]).T
    return ys, ts, dt

@pytest.mark.parametrize("array_type", [np.array, torch.tensor])
def test_iterate_generalized_padding(ex_sincos, array_type):
    ys, ts, dt = ex_sincos
    m = ys.shape[1]

    ys = array_type(ys)

    # First test the numpy case
    t1 = list(hdm.core.iterate_generalized(ys, dt, p=4))
    t2 = list(hdm.core.iterate_generalized(ys, dt, p=2, p_pad=4))
    t3 = list(hdm.core.iterate_generalized(ys, dt, p=2, p_comp=4, p_pad=4))

    for c1, c2, c3 in zip(t1, t2, t3):
        # c1 and c3 should be computed in the same way, except for the padding
        # at the end of c3
        stem = m * (2 + 1)
        assert (c1[:stem] == c3[:stem]).all()
        # c2 and c3 should both end in zeros
        assert (c2[stem:] == 0).all()
        assert (c2[stem:] == c3[stem:]).all()
