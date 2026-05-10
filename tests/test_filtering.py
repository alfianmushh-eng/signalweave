import numpy as np

from signalweave import filtering, generators


def test_lowpass_sine() -> None:
    s = generators.sine(frequency=50, sample_rate=1000, duration=1.0)
    filtered = filtering.lowpass(s, cutoff=30, sample_rate=1000)
    # 50 Hz above 30 Hz cutoff -> heavily attenuated
    assert float(np.abs(filtered).mean()) < 0.5 * float(np.abs(s).mean())


def test_moving_average_shape() -> None:
    x = np.arange(100.0)
    out = filtering.moving_average(x, window=5)
    assert len(out) == 100
    assert np.isnan(out[:4]).all()
