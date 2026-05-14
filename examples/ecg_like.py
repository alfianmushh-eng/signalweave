"""Synthetic ECG-like signal walkthrough.

Demonstrates generators, filtering, framing, spectral analysis, and features.
"""
from __future__ import annotations

import numpy as np

from signalweave import features, filtering, generators, spectral, windows


def _ecg_like(sample_rate: int = 500) -> np.ndarray:
    """Simulate a few heartbeats using harmonic sinusoids + noise."""
    dur = 4.0
    n = int(sample_rate * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    ecg = np.zeros(n)
    for beat_peak in np.arange(0.2, dur, 0.8):
        idx = int(beat_peak * sample_rate)
        if idx >= n:
            break
        ecg[idx] = 1.2
    ecg += 0.5 * np.sin(2 * np.pi * 1.2 * t) * np.cos(2 * np.pi * 0.5 * t)
    ecg += generators.white_noise(sample_rate=float(sample_rate), duration=dur) * 0.05
    return filtering.lowpass(ecg, cutoff=40, sample_rate=float(sample_rate))


def main() -> None:
    sr = 500
    ecg = _ecg_like(sr)
    print(f"Signal length: {len(ecg)} samples ({len(ecg)/sr:.1f} sec)")

    f32 = windows.frame(ecg, size=128, hop=64)
    print(f"Frames: {f32.shape[0]} ({f32.shape[1]} samples each)")

    mat, keys = features.sliding_features(ecg, size=128, hop=64, feature_names=["rms", "zcr", "energy"])
    print(f"Feature matrix: {mat.shape}")
    print(f"Features: {keys}")
    print(f"First 3 rows:\n{mat[:3]}")

    f, Pxx = spectral.psd(ecg, sample_rate=sr, nperseg=256)
    peaks = spectral.find_peaks(f, Pxx, min_height=2 * Pxx.mean())
    print(f"Dominant PSD peaks: {[round(p.frequency, 1) for p in peaks[:3]]} Hz")

    r = spectral.spectrogram(ecg, sample_rate=sr, nperseg=128)
    print(f"Spectrogram shape: {r.S.shape} (freq={len(r.f)} x time={len(r.t)})")
    print("Done.")


if __name__ == "__main__":
    main()
