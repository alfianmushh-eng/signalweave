import numpy as np
from signalweave import segment

def test_silence_detect():
    x = np.concatenate([np.zeros(200), np.ones(100), np.zeros(200)])
    sil = segment.silence_detect(x, threshold=0.5, min_silence=50)
    assert len(sil) >= 1
