"""Wiener filtering in the frequency domain."""
from __future__ import annotations
import numpy as np


def wiener_filter(noisy: np.ndarray, noise_estimate: np.ndarray) -> np.ndarray:
    """Frequency-domain Wiener filter using a noise PSD estimate."""
    N = len(noisy)
    X = np.fft.rfft(noisy)
    Pn = np.abs(np.fft.rfft(noise_estimate, n=N)) ** 2
    Px = np.abs(X) ** 2
    H = np.maximum(Px - Pn, 0.0) / np.maximum(Px, 1e-12)
    return np.fft.irfft(H * X, n=N)


def spectral_subtraction(noisy: np.ndarray, noise_estimate: np.ndarray,
                         alpha: float = 1.0, beta: float = 0.01) -> np.ndarray:
    """Classic spectral subtraction with floor."""
    N = len(noisy)
    X = np.fft.rfft(noisy)
    Pn = np.abs(np.fft.rfft(noise_estimate, n=N))
    mag = np.abs(X)
    phase = np.angle(X)
    sub = mag - alpha * Pn
    floor = beta * mag
    new_mag = np.maximum(sub, floor)
    return np.fft.irfft(new_mag * np.exp(1j * phase), n=N)
