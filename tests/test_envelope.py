import numpy as np
from signalweave import envelope, generators

def test_hilbert_envelope():
    s = generators.sine(frequency=50, sample_rate=1000, duration=1.0)
    env = envelope.hilbert_envelope(s)
    assert len(env) == len(s)
    assert env.min() >= 0
