"""Tests for extended DSP modules."""
import numpy as np
from signalweave.kalman import Kalman1D
from signalweave.kalman_cv import KalmanCV
from signalweave.peaks import find_peaks
from signalweave.zerocross import zero_crossings, zero_crossing_rate
from signalweave.cepstrum import real_cepstrum
from signalweave.hilbert_tools import envelope, instantaneous_frequency
from signalweave.wavelet import haar_dwt, haar_idwt


def test_kalman_1d_converges():
    rng = np.random.default_rng(0)
    truth = 5.0
    measurements = truth + rng.normal(0, 0.5, 200)
    kf = Kalman1D(process_var=1e-4, measure_var=0.25)
    out = kf.filter(measurements)
    assert abs(out[-1] - truth) < 0.3


def test_kalman_cv_tracks_motion():
    sr = 100
    t = np.arange(sr)
    truth = 0.1 * t
    rng = np.random.default_rng(1)
    measurements = truth + rng.normal(0, 0.2, len(t))
    kf = KalmanCV(dt=1.0, process_var=1e-3, measure_var=0.04)
    out = kf.run(measurements)
    assert abs(out[-1, 0] - truth[-1]) < 1.0


def test_find_peaks_simple():
    x = np.array([0, 1, 0, 2, 0, 1, 0])
    peaks = find_peaks(x)
    assert set(peaks.tolist()) == {1, 3, 5}


def test_zero_crossings_sine():
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t)
    zc = zero_crossings(x)
    assert 9 <= len(zc) <= 11


def test_zcr_shape():
    x = np.random.randn(1024)
    zcr = zero_crossing_rate(x, 256, 128)
    assert zcr.ndim == 1


def test_real_cepstrum_length():
    x = np.random.randn(512)
    c = real_cepstrum(x)
    assert len(c) == len(x)


def test_envelope_positive():
    sr = 1000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 10 * t) * (1 + 0.5 * np.sin(2 * np.pi * 1 * t))
    env = envelope(x)
    assert np.all(env >= 0)


def test_instantaneous_frequency_close():
    sr = 2000
    t = np.linspace(0, 1, sr, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t)
    f = instantaneous_frequency(x, sr)
    assert 45 < np.median(f) < 55


def test_haar_roundtrip():
    rng = np.random.default_rng(2)
    x = rng.normal(size=64)
    cA, cD = haar_dwt(x)
    rec = haar_idwt(cA, cD)[: len(x)]
    assert np.allclose(rec, x, atol=1e-10)
