Spectral analysis utilities.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as _signal

from signalweave import windows


@dataclass
class SpectrogramResult:
    f: np.ndarray  
    t: np.ndarray  
    S: np.ndarray  


def spectrogram(
    data: np.ndarray,
    *,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window_name: str = "hann",
) -> SpectrogramResult:
    win = windows.window(window_name, nperseg)
    f, t, Sxx = _signal.spectrogram(data, fs=sample_rate, window=win, nperseg=nperseg, noverlap=noverlap or nperseg // 2)
    return SpectrogramResult(f=f, t=t, S=Sxx)


def psd(
    data: np.ndarray,
    *,
    sample_rate: float,
    nperseg: int = 256,
    window_name: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    f, Pxx = _signal.periodogram(data, fs=sample_rate, window=windows.window(window_name, len(data)) if len(data) > 0 else None)  # noqa: E501
    return f, Pxx


@dataclass
class Peak:
    frequency: float
    magnitude: float


def find_peaks(
    f: np.ndarray,
    Pxx: np.ndarray,
    *,
    min_freq: float = 0.0,
    max_freq: float | None = None,
    min_height: float | None = None,
    min_distance: float = 1.0,
) -> list[Peak]:
    idx_max = len(f) - 1
    lo = int(np.searchsorted(f, min_freq))
    hi = int(np.searchsorted(f, max_freq)) if max_freq is not None else idx_max + 1
    segment = Pxx[lo:hi]
    if min_height is None:
        min_height = float(segment.mean()) + float(segment.std())
    peaks, props = _signal.find_peaks(segment, height=min_height, distance=int(min_distance / (f[1] - f[0])) if len(f) > 1 else 1)  # noqa: E501
    return [Peak(frequency=f[lo + p], magnitude=props["peak_heights"][i]) for i, p in enumerate(peaks)]
