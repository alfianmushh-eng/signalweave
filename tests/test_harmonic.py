import numpy as np
from signalweave import harmonic, generators

def test_f0_sine():
    s = generators.sine(frequency=100, sample_rate=2000, duration=0.5)
    f0 = harmonic.fundamental_frequency(s, sample_rate=2000, fmin=50, fmax=500)
    assert abs(f0 - 100) < 5
