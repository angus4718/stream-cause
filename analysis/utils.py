"""Shared utilities for StreamCause Python analysis layer."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz


# -- Mark and Edge types (mirror C++ enums) ----

class Mark(IntEnum):
    TAIL = 0
    ARROW = 1
    CIRCLE = 2


@dataclass
class Edge:
    i: int
    j: int
    mark_i: Mark
    mark_j: Mark
    weight: float = 1.0

    def is_directed(self) -> bool:
        """True if this is a -> edge (TAIL at i, ARROW at j)."""
        return self.mark_i == Mark.TAIL and self.mark_j == Mark.ARROW

    def is_bidirected(self) -> bool:
        """True if this is a <-> edge (both ARROWs)."""
        return self.mark_i == Mark.ARROW and self.mark_j == Mark.ARROW


@dataclass
class PAGSnapshot:
    n_nodes: int
    timestamp_ns: int
    edges: List[Edge]

    def has_directed_edge(self, i: int, j: int) -> bool:
        """Return True if there is a directed edge i -> j."""
        for e in self.edges:
            if e.i == i and e.j == j and e.is_directed():
                return True
        return False

    def density(self) -> float:
        if self.n_nodes <= 1:
            return 0.0
        return len(self.edges) / (self.n_nodes * (self.n_nodes - 1))


# -- Instrument universe ----

# Top 50 S&P 500 equities by 2022 dollar volume, + 3 ETFs + 6 futures = 59 instruments
ID_TO_SYMBOL: Dict[int, str] = {
    # 0-49: top 50 Nasdaq-listed equities by 2022 dollar volume
    0: "AAPL", 1: "MSFT", 2: "NVDA", 3: "AMZN", 4: "GOOGL"
    5: "META", 6: "TSLA", 7: "AVGO", 8: "COST", 9: "ADBE"
    10: "ASML", 11: "AMD", 12: "QCOM", 13: "TXN", 14: "INTU"
    15: "AMAT", 16: "MU", 17: "LRCX", 18: "KLAC", 19: "MRVL"
    20: "PANW", 21: "CDNS", 22: "SNPS", 23: "NXPI", 24: "ADI"
    25: "MCHP", 26: "FTNT", 27: "REGN", 28: "BIIB", 29: "VRTX"
    30: "GILD", 31: "IDXX", 32: "ISRG", 33: "ALGN", 34: "DXCM"
    35: "ILMN", 36: "SGEN", 37: "MRNA", 38: "PYPL", 39: "EBAY"
    40: "NFLX", 41: "CMCSA", 42: "CHTR", 43: "ATVI", 44: "EA"
    45: "TTWO", 46: "WBA", 47: "FAST", 48: "ODFL", 49: "PAYX"
    # 50-52: ETFs
    50: "SPY"
    51: "QQQ"
    52: "IWM"
    # 53-58: futures
    53: "ES"
    54: "NQ"
    55: "ZN"
    56: "ZB"
    57: "6E"
    58: "CL"
}
SYMBOL_TO_ID: Dict[str, int] = {v: k for k, v in ID_TO_SYMBOL.items()}


# -- Snapshot I/O ----

def load_snapshot(path: Path) -> PAGSnapshot:
    """Deserialize a single PAG JSON snapshot from GraphStore."""
    with open(path) as f:
        d = json.load(f)
    edges = []
    for e in d.get("edges", []):
        edges.append(Edge(
            i=e["i"], j=e["j"]
            mark_i=Mark(e["mark_i"]), mark_j=Mark(e["mark_j"])
            weight=e.get("weight", 1.0)
        ))
    return PAGSnapshot(n_nodes=d["n_nodes"], timestamp_ns=d["timestamp_ns"], edges=edges)


def load_snapshots(directory: Path, date: str) -> List[PAGSnapshot]:
    """Load all PAG snapshots for a given date (YYYY-MM-DD), sorted by time."""
    date_dir = directory / date
    if not date_dir.exists():
        return []
    paths = sorted(date_dir.glob("snapshot_*.json")
                   key=lambda p: int(p.stem.split("_")[1]))
    return [load_snapshot(p) for p in paths]


def load_alpha(path: Path) -> np.ndarray:
    """Load a binary alpha matrix (format written by GraphStore::append)."""
    with open(path, "rb") as f:
        # Expected: [int32 n_rows][int32 n_cols][float64* column-major data].
        n_rows, n_cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(n_rows * n_cols * 8), dtype=np.float64)
        return data.reshape((n_rows, n_cols), order="F")


def _session_ns(date_str: str):
    """Return (open_ns, close_ns) for the ET regular session of a YYYY-MM-DD date."""
    import datetime
    y, m, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    # ET = UTC-4 (EDT summer) or UTC-5 (EST winter); Oct-Dec 2025 uses EST (UTC-5)
    # after DST ends first Sunday of November (Nov 2, 2025).
    # Oct 1 - Nov 1: EDT (UTC-4); Nov 2 onwards: EST (UTC-5)
    date = datetime.date(y, m, d)
    dst_end = datetime.date(2025, 11, 2)
    offset_s = 4 * 3600 if date < dst_end else 5 * 3600
    epoch = datetime.datetime(1970, 1, 1)
    open_s = (datetime.datetime(y, m, d, 9, 30) - epoch).total_seconds() + offset_s
    close_s = (datetime.datetime(y, m, d, 16, 0) - epoch).total_seconds() + offset_s
    return int(open_s * 1_000_000_000), int(close_s * 1_000_000_000)


def _filter_session(paths: list, date_str: str) -> list:
    """Keep only paths whose embedded timestamp falls within the ET session."""
    open_ns, close_ns = _session_ns(date_str)
    return [p for p in paths
            if open_ns <= int(p.stem.split("_")[1]) < close_ns]


def load_alpha_series(directory: Path, date: str
                      session_filter: bool = False) -> np.ndarray:
    """Load the time series of alpha matrices for a date. Shape: (S, N, N)."""
    date_dir = directory / date
    if not date_dir.exists():
        return np.empty((0, 0, 0))
    paths = sorted(date_dir.glob("alpha_*.bin")
                   key=lambda p: int(p.stem.split("_")[1]))
    if session_filter:
        paths = _filter_session(paths, date)
    matrices = [load_alpha(p) for p in paths]
    if not matrices:
        return np.empty((0, 0, 0))
    return np.stack(matrices, axis=0)


def load_lambda(path: Path) -> np.ndarray:
    """Load a single lambda vector from binary (same format as alpha_*.bin, n_cols=1)."""
    with open(path, "rb") as f:
        n_rows, n_cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(n_rows * n_cols * 8), dtype=np.float64)
        return data.reshape((n_rows, n_cols), order="F").flatten()


def load_lambda_series(directory: Path, date: str
                       session_filter: bool = False) -> np.ndarray:
    """Load time series of per-instrument intensity vectors. Shape: (S, N)."""
    date_dir = directory / date
    if not date_dir.exists():
        return np.empty((0, 0))
    paths = sorted(date_dir.glob("lambda_*.bin")
                   key=lambda p: int(p.stem.split("_")[1]))
    if session_filter:
        paths = _filter_session(paths, date)
    vecs = [load_lambda(p) for p in paths]
    return np.stack(vecs, axis=0) if vecs else np.empty((0, 0))


# -- Macro calendar ----

def load_macro_calendar(path: Path) -> pd.DataFrame:
    """Load macro announcement calendar.

    Returns DataFrame with columns [timestamp_ns, event_type].
    """
    return pd.DataFrame(columns=["timestamp_ns", "event_type"])


# -- Graph metrics ----

def edge_persistence_rate(snapshots: List[PAGSnapshot]) -> Dict[Tuple[int, int], float]:
    """Compute rho_e = fraction of consecutive snapshot pairs where edge e is present
    with the same orientation. Paper"""
    if len(snapshots) < 2:
        return {}

    counts = {} # (i,j) -> count of snapshots where edge is present with same orientation
    present = {} # (i,j) -> total snapshots where edge is present (in either snapshot)

    for a, b in zip(snapshots, snapshots[1:]):
        edge_map_b = {(e.i, e.j): e for e in b.edges}
        for e in a.edges:
            key = (e.i, e.j)
            present[key] = present.get(key, 0) + 1
            eb = edge_map_b.get(key)
            if eb and eb.mark_i == e.mark_i and eb.mark_j == e.mark_j:
                counts[key] = counts.get(key, 0) + 1

    return {k: counts.get(k, 0) / v for k, v in present.items()}


def graph_edit_distance(a: PAGSnapshot, b: PAGSnapshot) -> float:
    """Weighted graph edit distance. Accounts for add/remove/mark change."""
    dist = 0.0

    # Edges in a
    edge_map_a = {(e.i, e.j): e for e in a.edges}
    edge_map_b = {(e.i, e.j): e for e in b.edges}

    # Edges in a but not b, or with changed marks
    for e in a.edges:
        eb = edge_map_b.get((e.i, e.j))
        if not eb:
            dist += e.weight # edge removed
        elif eb.mark_i != e.mark_i or eb.mark_j != e.mark_j:
            dist += e.weight # mark changed

    # Edges in b but not a
    for e in b.edges:
        if (e.i, e.j) not in edge_map_a:
            dist += e.weight # edge added

    return dist


def classify_regime(timestamp_ns: int
                    macro_calendar: Optional[pd.DataFrame] = None
                    macro_window_minutes: int = 30) -> str:
    """Map a nanosecond UTC timestamp to an intraday regime label.

    Returns one of: "pre-open", "open-auction", "regular", "close-auction"
    "announcement", "unknown".
    """
    # 1. Check macro announcement window first (overrides time-of-day).
    if macro_calendar is not None and not macro_calendar.empty:
        window_ns = macro_window_minutes * 60 * 1_000_000_000
        nearby = macro_calendar[
            (macro_calendar.timestamp_ns >= timestamp_ns - window_ns) &
            (macro_calendar.timestamp_ns <= timestamp_ns + window_ns)
        ]
        if not nearby.empty:
            return "announcement"

    # 2. Convert to ET and apply time-of-day boundaries.
    ET = pytz.timezone("America/New_York")
    dt_utc = datetime.fromtimestamp(timestamp_ns / 1e9, tz=timezone.utc)
    dt_et = dt_utc.astimezone(ET)
    t = dt_et.hour * 60 + dt_et.minute # minutes since midnight ET

    if t < 8 * 60: return "unknown"
    if t < 9 * 60 + 28: return "pre-open"
    if t < 9 * 60 + 32: return "open-auction"
    if t < 15 * 60 + 45: return "regular"
    if t < 16 * 60: return "close-auction"
    return "unknown"


# -- Return computation ----

def mid_price(alpha_events: pd.DataFrame) -> pd.Series:
    """Estimate mid-price from MBO events. Used in Experiment 4."""
    # mid = (best_bid + best_ask) / 2.
    raise NotImplementedError


def log_return(prices: pd.Series, delta_ns: int) -> pd.Series:
    """Compute forward log-return at horizon delta_ns nanoseconds."""
    raise NotImplementedError
