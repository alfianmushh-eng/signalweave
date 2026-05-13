signalweave command-line entry point.
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from signalweave import __version__, features, filtering, generators, spectral, windows


def _cmd_gen(args: argparse.Namespace) -> int:
    if args.type == "sine":
        sig = generators.sine(frequency=args.freq, sample_rate=args.rate, duration=args.dur)
    elif args.type == "chirp":
        sig = generators.chirp(f0=args.f0, f1=args.f1, sample_rate=args.rate, duration=args.dur)
    else:
        print(f"unknown type: {args.type}", file=sys.stderr)
        return 1
    out: np.ndarray = filtering.lowpass(sig, cutoff=args.lp, sample_rate=args.rate) if args.lp else sig
    np.savetxt(sys.stdout, out)
    return 0


def _cmd_frame(args: argparse.Namespace) -> int:
    data = np.loadtxt(args.input)
    f = windows.frame(data, size=args.size, hop=args.hop)
    np.savetxt(args.output, f, fmt="%.6f")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    data = np.loadtxt(args.input)
    names = args.names.split(",") if args.names else None
    mat, keys = features.sliding_features(data, size=args.size, hop=args.hop, feature_names=names)
    out = args.output
    with out.open("w", newline="") if isinstance(out, str) else open(str(out), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        w.writerows(mat)
    return 0


def _cmd_spectrogram(args: argparse.Namespace) -> int:
    data = np.loadtxt(args.input)
    r = spectral.spectrogram(data, sample_rate=args.rate, nperseg=args.nperseg)
    np.savez(args.output_npz, f=r.f, t=r.t, S=r.S)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="signalweave", description="Signal processing toolkit")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="Generate and optionally filter a test signal")
    g.add_argument("--type", default="sine", choices=["sine", "chirp"])
    g.add_argument("--rate", type=float, default=1000, help="Sample rate (Hz)")
    g.add_argument("--dur", type=float, default=1.0, help="Duration (sec)")
    g.add_argument("--freq", type=float, default=50, help="Frequency (Hz, sine only)")
    g.add_argument("--f0", type=float, default=10, help="Start frequency (chirp)")
    g.add_argument("--f1", type=float, default=100, help="End frequency (chirp)")
    g.add_argument("--lp", type=float, default=None, help="Optional lowpass cutoff")
    g.set_defaults(func=_cmd_gen)

    fr = sub.add_parser("frame", help="Split a signal into overlapping frames")
    fr.add_argument("input", type=Path)
    fr.add_argument("--size", type=int, default=256)
    fr.add_argument("--hop", type=int, default=128)
    fr.add_argument("--output", type=Path, default="-")
    fr.set_defaults(func=_cmd_frame)

    fx = sub.add_parser("features", help="Extract sliding-window features")
    fx.add_argument("input", type=Path)
    fx.add_argument("--size", type=int, default=256)
    fx.add_argument("--hop", type=int, default=128)
    fx.add_argument("--names", default=None, help="Comma-separated feature names")
    fx.add_argument("--output", type=Path, default=Path("features.csv"))
    fx.set_defaults(func=_cmd_features)

    sp = sub.add_parser("spectrogram", help="Compute and save a spectrogram (.npz)")
    sp.add_argument("input", type=Path)
    sp.add_argument("--rate", type=float, default=1000)
    sp.add_argument("--nperseg", type=int, default=256)
    sp.add_argument("--output-npz", type=Path, default=Path("spectrogram.npz"))
    sp.set_defaults(func=_cmd_spectrogram)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
