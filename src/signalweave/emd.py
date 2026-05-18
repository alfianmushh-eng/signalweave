"""Empirical Mode Decomposition (EMD) for non-stationary signals."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline


def _find_extrema(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.diff(x)
    sign = np.sign(d)
    change = np.diff(sign)
    maxima = np.where(change < 0)[0] + 1
    minima = np.where(change > 0)[0] + 1
    return maxima, minima


def _envelope(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    if len(idx) < 2:
        return np.full_like(x, x.mean(), dtype=float)
    cs = CubicSpline(idx, x[idx], extrapolate=True)
    return cs(np.arange(len(x)))


def sift(x: np.ndarray, max_iter: int = 30, tol: float = 0.05) -> np.ndarray:
    h = x.astype(float).copy()
    for _ in range(max_iter):
        maxima, minima = _find_extrema(h)
        if len(maxima) < 2 or len(minima) < 2:
            break
        upper = _envelope(h, maxima)
        lower = _envelope(h, minima)
        m = 0.5 * (upper + lower)
        new = h - m
        if np.linalg.norm(h - new) / (np.linalg.norm(h) + 1e-12) < tol:
            h = new
            break
        h = new
    return h


def emd(x: np.ndarray, max_imfs: int = 6) -> list[np.ndarray]:
    residue = np.asarray(x, dtype=float).copy()
    imfs = []
    for _ in range(max_imfs):
        imf = sift(residue)
        imfs.append(imf)
        residue = residue - imf
        if np.std(residue) < 1e-6:
            break
    imfs.append(residue)
    return imfs
