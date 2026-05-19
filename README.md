# signalweave

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, dependency-light signal-processing toolkit built on numpy and scipy.
Designed for quick prototyping of time-series filtering, spectral analysis, and
feature extraction without wiring up a full audio DSP chain.

## Modules

| Module                          | Purpose                                                    |
| ------------------------------- | ---------------------------------------------------------- |
| `signalweave.generators`         | Test signal synthesis: sine, square, chirp, white noise    |
| `signalweave.generators_extra`   | Sawtooth, triangle, pulse train, AM, FM modulation         |
| `signalweave.windows`            | Hann / Hamming / Blackman / Bartlett windows + framing     |
| `signalweave.filtering`          | Butterworth IIR lowpass/highpass/bandpass/notch, MA filter |
| `signalweave.filters_extra`      | FIR, median, Savitzky-Golay, bandstop                      |
| `signalweave.filter_design`      | Frequency response, group delay                            |
| `signalweave.spectral`           | Spectrogram, periodogram PSD, peak finding                 |
| `signalweave.spectral_extra`     | STFT, mel-spectrogram, MFCC, cepstrum                      |
| `signalweave.features`           | Sliding-window RMS, ZCR, energy, peak-to-peak, stats       |
| `signalweave.features_extra`     | Spectral centroid, bandwidth, rolloff, flatness            |
| `signalweave.time_features`      | Peak-to-peak, energy ratio, ZC density                     |
| `signalweave.envelope`           | Hilbert, peak, RMS envelope detection                      |
| `signalweave.correlation`        | Cross-correlation, auto-correlation, convolution           |
| `signalweave.resample`           | Sample rate conversion, decimate, interpolate              |
| `signalweave.streaming`          | RingBuffer, OnlineRMS for online processing                |
| `signalweave.audio_io`           | WAV read/write, resampling load                            |
| `signalweave.normalize`          | Peak, RMS, DC removal normalization                        |
| `signalweave.segment`            | Silence detection, split on silence                        |
| `signalweave.harmonic`           | HPS, fundamental frequency estimation                      |
| `signalweave.onset`              | Spectral flux onset detection                              |
| `signalweave.phase`              | Phase unwrap, instantaneous frequency, time stretch        |
| `signalweave.colored_noise`      | Pink, brown, blue noise generators                         |
| `signalweave.dither`             | Dithering, quantization                                    |
| `signalweave.statistics`         | Crest factor, form factor, skewness, kurtosis              |
| `signalweave.viz`                | Waveform, spectrogram, filter response plots               |
| `signalweave.cli`                | `signalweave gen`, `frame`, `features`, `spectrogram`      |
| `signalweave.cli_extra`          | `correlate`, `envelope`, `info` commands                   |

## Install

```bash
pip install -e .
```

With plotting support:

```bash
pip install -e ".[viz]"
```

## Quickstart

```python
from signalweave import generators, filtering, features

sig = generators.sine(frequency=100, sample_rate=1000, duration=2.0)
sig_filt = filtering.lowpass(sig, cutoff=80, sample_rate=1000)
mat, keys = features.sliding_features(sig_filt, size=256, hop=128)
```

```bash
# Generate a 50 Hz sine, lowpass at 30 Hz, print samples to stdout
signalweave gen --type sine --freq 50 --lp 30 > out.txt

# Split a file into overlapping frames
signalweave frame signal.txt --size 128 --hop 64

# Extract sliding-window RMS and ZCR
signalweave features signal.txt --size 256 --hop 128 --names rms,zcr
```

## Example

```bash
python examples/ecg_like.py
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## License

MIT - see [LICENSE](LICENSE).
