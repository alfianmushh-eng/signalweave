import numpy as np
from signalweave import spectral_extra as sx, generators

def test_stft_shape():
    s = generators.sine(frequency=100, sample_rate=1000, duration=0.5)
    r = sx.stft(s, sample_rate=1000, nperseg=128)
    assert r.Zxx.ndim == 2
