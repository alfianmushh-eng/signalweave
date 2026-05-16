import numpy as np
from signalweave import audio_io, generators
import tempfile, os

def test_write_read_wav():
    s = generators.sine(frequency=440, sample_rate=16000, duration=0.1)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        p = f.name
    try:
        audio_io.write_wav(p, s, 16000)
        data, sr = audio_io.read_wav(p)
        assert sr == 16000
        assert len(data) > 0
    finally:
        os.unlink(p)
