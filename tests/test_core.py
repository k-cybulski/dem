import dem.core
import numpy as np
import pytest


@pytest.fixture
def ex_sincos():
    t_start = 0
    t_end = 20
    dt = 0.1
    ts = np.arange(start=t_start, stop=t_end, step=dt)
    x1s = np.sin(ts)
    x2s = np.cos(ts)
    ys = ys = np.stack([x1s, x2s]).T
    return ys, ts, dt


@pytest.mark.parametrize("array_type", [np.array])
def test_iterate_generalized_padding(ex_sincos, array_type):
    ys, ts, dt = ex_sincos

    ys = array_type(ys)

    t1 = list(dem.core.iterate_generalized(ys, dt, p=4))
    t2 = list(dem.core.iterate_generalized(ys, dt, p=2, p_comp=4))

    assert len(t1) == len(t2)
