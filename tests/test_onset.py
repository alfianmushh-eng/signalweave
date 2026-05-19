import numpy as np
from signalweave import onset, generators

def test_spectral_onset():
    s = generators.sine(frequency=100, sample_rate=1000, duration=0.5)
    o = onset.spectral_onset(s, sample_rate=1000, nperseg=128)
    assert len(o) > 0
