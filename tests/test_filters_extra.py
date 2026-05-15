import numpy as np
from signalweave import filters_extra as fx

def test_median_filter_shape():
    x = np.arange(100.0)
    out = fx.median_filter(x, size=5)
    assert len(out) == 100

def test_fir_lowpass():
    x = np.sin(2 * np.pi * 50 * np.arange(1000) / 1000)
    out = fx.fir_lowpass(x, cutoff=30, sample_rate=1000, numtaps=31)
    assert len(out) == 1000
