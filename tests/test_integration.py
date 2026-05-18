Integration test for signalweave pipeline.
from signalweave import generators, filtering, features, spectral

def test_end_to_end():
    sig = generators.sine(frequency=100, sample_rate=1000, duration=2.0)
    filt = filtering.lowpass(sig, cutoff=80, sample_rate=1000)
    assert len(filt) == 2000
    mat, keys = features.sliding_features(filt, size=256, hop=128)
    assert mat.shape[1] == len(keys)
    f, Pxx = spectral.psd(filt, sample_rate=1000)
    assert len(f) == len(Pxx)
