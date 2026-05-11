import numpy as np

from signalweave import generators, spectral


def test_spectrogram_shape() -> None:
    s = generators.chirp(f0=10, f1=100, sample_rate=1000, duration=2.0)
    r = spectral.spectrogram(s, sample_rate=1000, nperseg=128)
    assert r.S.ndim == 2
    assert len(r.f) > 0
    assert len(r.t) > 0


def test_find_peaks_sine() -> None:
    s = generators.sine(frequency=100, sample_rate=2000, duration=0.5)
    f, Pxx = spectral.psd(s, sample_rate=2000)
    peaks = spectral.find_peaks(f, Pxx, min_height=2 * Pxx.mean())
    assert any(90 <= p.frequency <= 110 for p in peaks)
