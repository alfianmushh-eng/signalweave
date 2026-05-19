Onset/transient detection.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def spectral_onset(data: np.ndarray, *, sample_rate: float, nperseg: int = 256) -> np.ndarray:
    f, t, Zxx = _signal.stft(data, fs=sample_rate, nperseg=nperseg, noverlap=nperseg // 2)
    magnitude = np.abs(Zxx)
    diff = np.diff(magnitude, axis=1)
    diff = np.maximum(diff, 0)
    onset_strength = np.sum(diff, axis=0)
    onset_strength = np.concatenate([[0], onset_strength])
    return onset_strength

def peak_onsets(onset_strength: np.ndarray, *, threshold: float = 0.5, min_distance: int = 5) -> np.ndarray:
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(onset_strength, height=threshold * onset_strength.max(), distance=min_distance)
    return peaks
