import numpy as np
import pytest

from signalweave import windows


def test_hann_window_endpoints() -> None:
    w = windows.window("hann", 8)
    assert w[0] == 0
    assert w[-1] == 0


def test_unknown_window() -> None:
    with pytest.raises(ValueError, match="unknown window"):
        windows.window("triangle", 8)


def test_frame_shape() -> None:
    s = np.arange(20)
    f = windows.frame(s, size=8, hop=4)
    assert f.shape == (4, 8)
    assert f[0].tolist() == list(range(0, 8))
    assert f[1].tolist() == list(range(4, 12))
