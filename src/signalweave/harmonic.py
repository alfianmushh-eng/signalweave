Harmonic analysis utilities.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def harmonic_product_spectrum(data: np.ndarray, *, sample_rate: float, n_harmonics: int = 5) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = _signal.periodogram(data, fs=sample_rate)
    hps = Pxx.copy()
    for h in range(2, n_harmonics + 1):
        decimated = Pxx[::h][:len(Pxx) // h]
        hps_dec = np.zeros(len(Pxx))
        hps_dec[:len(decimated)] = decimated
        hps = np.concatenate([np.ones(len(Pxx) - len(hps)), hps]) if len(hps) > len(decimated) else hps[:len(decimated)] * decimated
    return f, hps

def fundamental_frequency(data: np.ndarray, *, sample_rate: float, fmin: float = 50.0, fmax: float = 500.0) -> float:
    f, Pxx = _signal.periodogram(data, fs=sample_rate)
    mask = (f >= fmin) & (f <= fmax)
    if not mask.any(): return 0.0
    return f[mask][Pxx[mask].argmax()]
