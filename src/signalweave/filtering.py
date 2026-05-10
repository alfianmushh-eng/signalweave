Digital filter design and application.
from __future__ import annotations

import numpy as np
from scipy import signal as _signal


def lowpass(
    data: np.ndarray,
    *,
    cutoff: float,
    sample_rate: float,
    order: int = 4,
    ftype: str = "butter",
) -> np.ndarray:
    sani = _signal.iirfilter(order, cutoff, fs=sample_rate, btype="low", ftype=ftype)
    return _signal.sosfiltfilt(sani, data)


def highpass(
    data: np.ndarray,
    *,
    cutoff: float,
    sample_rate: float,
    order: int = 4,
    ftype: str = "butter",
) -> np.ndarray:
    sani = _signal.iirfilter(order, cutoff, fs=sample_rate, btype="high", ftype=ftype)
    return _signal.sosfiltfilt(sani, data)


def bandpass(
    data: np.ndarray,
    *,
    lo: float,
    hi: float,
    sample_rate: float,
    order: int = 4,
    ftype: str = "butter",
) -> np.ndarray:
    sani = _signal.iirfilter(order, [lo, hi], fs=sample_rate, btype="band", ftype=ftype)
    return _signal.sosfiltfilt(sani, data)


def notch(
    data: np.ndarray,
    *,
    freq: float,
    sample_rate: float,
    quality: float = 30.0,
) -> np.ndarray:
    b, a = _signal.iirnotch(freq, quality, fs=sample_rate)
    return _signal.filtfilt(b, a, data)


def moving_average(data: np.ndarray, *, window: int) -> np.ndarray:
    conv = np.concatenate([np.full(window - 1, np.nan), np.convolve(data, np.ones(window) / window, mode="valid")])  # noqa: NPY201
    return conv
