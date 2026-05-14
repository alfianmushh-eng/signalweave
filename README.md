# signalweave

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, dependency-light signal-processing toolkit built on numpy and scipy.
Designed for quick prototyping of time-series filtering, spectral analysis, and
feature extraction without wiring up a full audio DSP chain.

## Modules

| Module                  | Purpose                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `signalweave.generators` | Test signal synthesis: sine, square, chirp, white noise    |
| `signalweave.windows`    | Hann / Hamming / Blackman / Bartlett windows + framing     |
| `signalweave.filtering`  | Butterworth IIR lowpass/highpass/bandpass/notch, MA filter |
| `signalweave.spectral`   | Spectrogram, periodogram PSD, peak finding                 |
| `signalweave.features`   | Sliding-window RMS, ZCR, energy, peak-to-peak, stats       |
| `signalweave.cli`        | `signalweave gen`, `frame`, `features`, `spectrogram`      |

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
