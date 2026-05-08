import numpy as np

from signalweave import generators


def test_sine_length() -> None:
    s = generators.sine(frequency=10, sample_rate=1000, duration=1.0)
    assert s.shape == (1000,)
    assert -1.001 <= s.min() and s.max() <= 1.001


def test_square_duty() -> None:
    s = generators.square(frequency=5, sample_rate=1000, duration=1.0, duty=0.25)
    # 25 percent of samples should be high
    high_ratio = float((s > 0).mean())
    assert 0.22 <= high_ratio <= 0.28


def test_chirp_monotonic_phase() -> None:
    s = generators.chirp(f0=10, f1=100, sample_rate=2000, duration=1.0)
    # Sanity: signal stays bounded
    assert -1.01 <= s.min() and s.max() <= 1.01
