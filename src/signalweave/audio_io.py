Audio file I/O utilities.
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import wavfile

def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    sr, data = wavfile.read(str(path))
    if data.dtype in (np.int16, np.int32):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data, int(sr)

def write_wav(path: str | Path, data: np.ndarray, sample_rate: int, *, normalize: bool = True) -> Path:
    p = Path(path)
    out = data.astype(np.float32)
    if normalize and out.max() > 1.0:
        out = out / out.max()
    wavfile.write(str(p), sample_rate, (out * 32767).astype(np.int16))
    return p

def load_audio(path: str | Path, *, target_rate: int | None = None) -> tuple[np.ndarray, int]:
    from signalweave import resample
    data, sr = read_wav(path)
    if target_rate and target_rate != sr:
        data = resample.resample(data, orig_rate=float(sr), target_rate=float(target_rate))
        sr = target_rate
    return data, sr
