import numpy as np
from signalweave import correlation

def test_auto_correlate_peak():
    x = np.sin(2 * np.pi * 10 * np.arange(100) / 100)
    lags, c = correlation.auto_correlate(x)
    assert c.argmax() == len(c) // 2
