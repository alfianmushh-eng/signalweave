import numpy as np
from signalweave import statistics

def test_crest_factor_sine():
    s = np.sin(np.linspace(0, 2*np.pi, 1000))
    cf = statistics.crest_factor(s)
    assert abs(cf - np.sqrt(2)) < 0.1
