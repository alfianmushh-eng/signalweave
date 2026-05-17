import numpy as np
from signalweave import resample

def test_resample_length():
    x = np.arange(100)
    out = resample.resample(x, orig_rate=100, target_rate=200)
    assert len(out) == 200
