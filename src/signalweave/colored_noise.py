Noise coloring filters.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def pink_noise(*, sample_rate: float, duration: float, rng: np.random.Generator | None = None) -> np.ndarray:
    white = (rng or np.random.default_rng()).standard_normal(int(sample_rate * duration))
    sos = _signal.iirfilter(2, 100, fs=sample_rate, btype="low", output="sos")
    return _signal.sosfilt(sos, white)

def brown_noise(*, sample_rate: float, duration: float, rng: np.random.Generator | None = None) -> np.ndarray:
    n = int(sample_rate * duration)
    white = (rng or np.random.default_rng()).standard_normal(n)
    brown = np.cumsum(white)
    return brown / (np.abs(brown).max() + 1e-10)

def blue_noise(*, sample_rate: float, duration: float, rng: np.random.Generator | None = None) -> np.ndarray:
    white = (rng or np.random.default_rng()).standard_normal(int(sample_rate * duration))
    sos = _signal.iirfilter(2, 2000, fs=sample_rate, btype="high", output="sos")
    return _signal.sosfilt(sos, white)
