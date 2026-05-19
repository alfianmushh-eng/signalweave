"""Chirp-Z Transform for arbitrary spectral zoom."""
from __future__ import annotations
import numpy as np


def chirp_z(x: np.ndarray, m: int, w: complex, a: complex = 1.0 + 0j) -> np.ndarray:
    """Chirp-Z transform of length-m at points a * w^k."""
    n = len(x)
    k = np.arange(max(m, n))
    nn = np.arange(n)
    yn = x * (a ** -nn) * (w ** (nn ** 2 / 2.0))
    L = 1
    while L < n + m - 1:
        L *= 2
    Y = np.fft.fft(yn, L)
    vk = w ** (-(k ** 2) / 2.0)
    V = np.fft.fft(vk, L)
    G = np.fft.ifft(Y * V)[:m]
    return G * (w ** (np.arange(m) ** 2 / 2.0))


def zoom_fft(x: np.ndarray, sr: float, f0: float, f1: float, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Zoom FFT between f0 and f1 with m bins."""
    a = np.exp(2j * np.pi * f0 / sr)
    w = np.exp(-2j * np.pi * (f1 - f0) / (m * sr))
    spec = chirp_z(x, m, w, a)
    freqs = np.linspace(f0, f1, m, endpoint=False)
    return freqs, spec
