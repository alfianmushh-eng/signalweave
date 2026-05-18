"""Pitch estimation: autocorrelation and YIN-lite."""
from __future__ import annotations
import numpy as np


def autocorr_pitch(frame: np.ndarray, sr: float, fmin: float = 60.0,
                   fmax: float = 800.0) -> float:
    """Returns dominant pitch in Hz via autocorrelation peak picking."""
    frame = frame - frame.mean()
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    if lag_max >= len(corr):
        lag_max = len(corr) - 1
    if lag_min >= lag_max:
        return 0.0
    region = corr[lag_min:lag_max]
    peak = lag_min + int(np.argmax(region))
    return float(sr / peak) if peak > 0 else 0.0


def yin_lite(frame: np.ndarray, sr: float, fmin: float = 60.0,
             fmax: float = 800.0, threshold: float = 0.1) -> float:
    """Simplified YIN difference function pitch detector."""
    n = len(frame)
    tau_max = int(sr / fmin)
    tau_min = int(sr / fmax)
    tau_max = min(tau_max, n - 1)
    if tau_min >= tau_max:
        return 0.0
    d = np.zeros(tau_max + 1)
    for tau in range(1, tau_max + 1):
        diff = frame[: n - tau] - frame[tau:]
        d[tau] = float((diff * diff).sum())
    cmnd = np.zeros_like(d)
    cmnd[0] = 1.0
    running = 0.0
    for tau in range(1, tau_max + 1):
        running += d[tau]
        cmnd[tau] = d[tau] * tau / (running + 1e-12)
    for tau in range(tau_min, tau_max + 1):
        if cmnd[tau] < threshold:
            return float(sr / tau)
    tau_best = tau_min + int(np.argmin(cmnd[tau_min:tau_max + 1]))
    return float(sr / tau_best) if tau_best > 0 else 0.0
