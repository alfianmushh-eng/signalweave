"""Discrete wavelet transform helpers (Haar, Daubechies via pywt if available)."""
from __future__ import annotations
import numpy as np

try:
    import pywt  # type: ignore
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False


def haar_dwt(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single-level Haar DWT. Returns (approx, detail)."""
    x = np.asarray(x, dtype=float)
    if x.size % 2:
        x = np.append(x, x[-1])
    even = x[0::2]
    odd = x[1::2]
    cA = (even + odd) / np.sqrt(2.0)
    cD = (even - odd) / np.sqrt(2.0)
    return cA, cD


def haar_idwt(cA: np.ndarray, cD: np.ndarray) -> np.ndarray:
    even = (cA + cD) / np.sqrt(2.0)
    odd = (cA - cD) / np.sqrt(2.0)
    out = np.empty(even.size + odd.size)
    out[0::2] = even
    out[1::2] = odd
    return out


def dwt(x: np.ndarray, wavelet: str = "db4", level: int | None = None):
    """Multi-level DWT via pywt when available."""
    if not _HAS_PYWT:
        raise RuntimeError("pywt not installed; install PyWavelets for non-Haar wavelets")
    return pywt.wavedec(x, wavelet, level=level)


def idwt(coeffs, wavelet: str = "db4") -> np.ndarray:
    if not _HAS_PYWT:
        raise RuntimeError("pywt not installed")
    return pywt.waverec(coeffs, wavelet)
