"""Real and complex cepstrum + cepstral pitch estimator."""
from __future__ import annotations
import numpy as np


def real_cepstrum(x: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    log_mag = np.log(np.abs(spectrum) + 1e-12)
    return np.fft.irfft(log_mag, n=len(x))


def complex_cepstrum(x: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft(x)
    log_spec = np.log(spectrum + 1e-12)
    return np.fft.ifft(log_spec).real


def cepstral_pitch(frame: np.ndarray, sr: float, fmin: float = 60.0,
                   fmax: float = 500.0) -> float:
    """Pitch via peak of real cepstrum in the quefrency range."""
    c = real_cepstrum(frame)
    q_min = int(sr / fmax)
    q_max = int(sr / fmin)
    q_max = min(q_max, len(c) // 2)
    if q_min >= q_max:
        return 0.0
    peak = q_min + int(np.argmax(c[q_min:q_max]))
    return float(sr / peak) if peak > 0 else 0.0
