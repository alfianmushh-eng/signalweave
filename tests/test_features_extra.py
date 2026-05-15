import numpy as np
from signalweave import features_extra as fx

def test_spectral_centroid():
    f = np.array([0, 100, 200])
    S = np.array([[1, 0], [0, 0], [0, 1]])
    c = fx.spectral_centroid(f, S)
    assert c[0] == 0 and c[1] == 200
