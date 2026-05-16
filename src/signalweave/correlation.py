Cross-correlation and convolution helpers.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def cross_correlate(x: np.ndarray, y: np.ndarray, *, mode: str = "full") -> tuple[np.ndarray, np.ndarray]:
    corr = _signal.correlate(x, y, mode=mode)
    lags = _signal.correlation_lags(len(x), len(y), mode=mode)
    return lags, corr

def auto_correlate(x: np.ndarray, *, mode: str = "full") -> tuple[np.ndarray, np.ndarray]:
    return cross_correlate(x, x, mode=mode)

def convolution(x: np.ndarray, y: np.ndarray, *, mode: str = "full") -> np.ndarray:
    return _signal.convolve(x, y, mode=mode)

def match_filter(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    return _signal.correlate(x, template[::-1], mode="same")
