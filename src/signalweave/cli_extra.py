Extended CLI commands for signalweave.
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from signalweave import audio_io, envelope, correlation, viz

def _cmd_correlate(args):
    x = np.loadtxt(args.x); y = np.loadtxt(args.y)
    lags, c = correlation.cross_correlate(x, y)
    np.savetxt(args.output, np.column_stack([lags, c]), fmt="%.6f", header="lag,correlation")
    return 0

def _cmd_envelope(args):
    data = np.loadtxt(args.input)
    env = envelope.hilbert_envelope(data) if args.type == "hilbert" else envelope.peak_envelope(data, window=args.window)
    np.savetxt(args.output, env, fmt="%.6f")
    return 0

def _cmd_info(args):
    data, sr = audio_io.read_wav(args.path)
    info = {"path": str(args.path), "sample_rate": sr, "samples": len(data), "duration": len(data)/sr, "dtype": str(data.dtype)}
    for k, v in info.items(): print(f"{k}: {v}")
    return 0

def build_extra_parser(sub):
    for pdef in [
        ("correlate", "Cross-correlate two signals", _cmd_correlate, [("x", type=Path), ("y", type=Path), ("--output", type=Path, default=Path("correlation.txt"))]),
        ("envelope", "Extract signal envelope", _cmd_envelope, [("input", type=Path), ("--type", choices=["hilbert", "peak"], default="hilbert"), ("--window", type=int, default=100), ("--output", type=Path, default=Path("envelope.txt"))]),
        ("info", "Show audio file info", _cmd_info, [("path", type=Path)]),
    ]:
        p = sub.add_parser(pdef[0], help=pdef[1])
        for arg in pdef[3]:
            kwargs = {k: v for k, v in arg.items() if k != "dest"}; p.add_argument(arg.get("dest") or next(iter(k for k in arg if k != "help")), **kwargs)
        p.set_defaults(func=pdef[2])
    return sub
