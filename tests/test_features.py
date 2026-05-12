import numpy as np

from signalweave import features


def test_rms_sine() -> None:
    x = np.sin(np.linspace(0, 2 * np.pi, 100))
    assert 0.68 <= features.rms(x) <= 0.72


def test_zero_crossing() -> None:
    x = np.array([1, -1, 1, -1, 1])
    rate = features.zero_crossing_rate(x)
    assert rate == 4.0 / 4.0


def test_sliding_shape() -> None:
    s = np.arange(100.0)
    mat, keys = features.sliding_features(s, size=20, hop=10)
    assert mat.shape == (9, len(keys))
