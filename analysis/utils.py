"""Loaders for the alpha_*.bin / lambda_*.bin outputs of the StreamCause pipeline."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def load_alpha(path: Path) -> np.ndarray:
    """Load one alpha matrix: [int32 rows][int32 cols][float64 column-major]."""
    with open(path, "rb") as f:
        n_rows, n_cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(n_rows * n_cols * 8), dtype=np.float64)
        return data.reshape((n_rows, n_cols), order="F")


def load_lambda(path: Path) -> np.ndarray:
    """Load a single lambda vector (same binary format as alpha, n_cols=1)."""
    with open(path, "rb") as f:
        n_rows, n_cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(n_rows * n_cols * 8), dtype=np.float64)
        return data.reshape((n_rows, n_cols), order="F").flatten()


def _session_ns(date_str: str):
    """Return (open_ns, close_ns) for the ET regular session of a YYYY-MM-DD date."""
    import datetime
    y, m, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    # EDT (UTC-4) before Nov 2 2025, EST (UTC-5) after DST ends.
    date = datetime.date(y, m, d)
    offset_s = 4 * 3600 if date < datetime.date(2025, 11, 2) else 5 * 3600
    epoch = datetime.datetime(1970, 1, 1)
    open_s = (datetime.datetime(y, m, d, 9, 30) - epoch).total_seconds() + offset_s
    close_s = (datetime.datetime(y, m, d, 16, 0) - epoch).total_seconds() + offset_s
    return int(open_s * 1_000_000_000), int(close_s * 1_000_000_000)


def _filter_session(paths: list, date_str: str) -> list:
    """Keep only paths whose embedded timestamp is within the ET session."""
    open_ns, close_ns = _session_ns(date_str)
    return [p for p in paths if open_ns <= int(p.stem.split("_")[1]) < close_ns]


def load_alpha_series(directory: Path, date: str, session_filter: bool = False) -> np.ndarray:
    """Load the alpha matrix time series for a date. Shape (S, N, N)."""
    date_dir = directory / date
    if not date_dir.exists():
        return np.empty((0, 0, 0))
    paths = sorted(date_dir.glob("alpha_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    if session_filter:
        paths = _filter_session(paths, date)
    matrices = [load_alpha(p) for p in paths]
    if not matrices:
        return np.empty((0, 0, 0))
    return np.stack(matrices, axis=0)


def load_lambda_series(directory: Path, date: str, session_filter: bool = False) -> np.ndarray:
    """Load the per-instrument intensity time series for a date. Shape (S, N)."""
    date_dir = directory / date
    if not date_dir.exists():
        return np.empty((0, 0))
    paths = sorted(date_dir.glob("lambda_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    if session_filter:
        paths = _filter_session(paths, date)
    vecs = [load_lambda(p) for p in paths]
    return np.stack(vecs, axis=0) if vecs else np.empty((0, 0))
