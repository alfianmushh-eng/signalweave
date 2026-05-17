"""Wavelet-based denoising via soft/hard thresholding."""
from __future__ import annotations
import numpy as np
from .wavelet import dwt, idwt, _HAS_PYWT


def _threshold(x: np.ndarray, value: float, mode: str = "soft") -> np.ndarray:
    if mode == "hard":
        return np.where(np.abs(x) > value, x, 0.0)
    return np.sign(x) * np.maximum(np.abs(x) - value, 0.0)


def universal_threshold(detail: np.ndarray) -> float:
    """Donoho universal threshold sigma*sqrt(2*log(n))."""
    sigma = np.median(np.abs(detail)) / 0.6745
    return sigma * np.sqrt(2.0 * np.log(max(detail.size, 2)))


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int | None = None,
                    mode: str = "soft") -> np.ndarray:
    if not _HAS_PYWT:
        raise RuntimeError("pywt required for wavelet denoising")
    coeffs = dwt(x, wavelet=wavelet, level=level)
    thr = universal_threshold(coeffs[-1])
    new = [coeffs[0]] + [_threshold(c, thr, mode) for c in coeffs[1:]]
    return idwt(new, wavelet=wavelet)[: len(x)]
