import numpy as np
from signalweave import generators_extra as gx

def test_sawtooth_range():
    s = gx.sawtooth(frequency=10, sample_rate=1000, duration=0.1)
    assert -1.01 <= s.min() and s.max() <= 1.01

def test_triangle_symmetry():
    s = gx.triangle(frequency=10, sample_rate=1000, duration=0.1)
    assert abs(s.mean()) < 0.1
