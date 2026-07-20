#!/usr/bin/env python3
"""Multi-day parallel LOB preprocessor: one worker per symbol, merged into per-day binaries."""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pytz
from sortedcontainers import SortedDict

RAW_DIR = "/ocean/projects/cis260122p/shared/data/raw"
LOB_FMT = "<qIBBxxqqqI4x"
LOB_SZ = struct.calcsize(LOB_FMT)
assert LOB_SZ == 48

ET = pytz.timezone("America/New_York")
US_HOLIDAYS = {
    datetime.date(2025, 11, 27),
    datetime.date(2025, 11, 28),
    datetime.date(2025, 12, 25),
}


def trading_days(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    days, d = [], start
    while d < end:
        if d.weekday() < 5 and d not in US_HOLIDAYS:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def session_ns(date: datetime.date) -> tuple[int, int]:
    open_dt = ET.localize(datetime.datetime(date.year, date.month, date.day, 9, 30))
    close_dt = ET.localize(datetime.datetime(date.year, date.month, date.day, 16, 0))
    return (int(open_dt.timestamp() * 1_000_000_000),
            int(close_dt.timestamp() * 1_000_000_000))



def _norm_action(mbo) -> str:
    a = mbo.action
    return a.value if hasattr(a, "value") else str(a)

def _norm_side(mbo) -> str:
    s = mbo.side
    return s.value if hasattr(s, "value") else str(s)


@dataclass(slots=True)
class PriceLevel:
    price: int
    size: int = 0
    count: int = 0


@dataclass(slots=True)
class LevelOrders:
    price: int
    orders: list = field(default_factory=list, compare=False)

    def __bool__(self) -> bool:
        return bool(self.orders)

    @property
    def level(self) -> PriceLevel:
        import databento as db
        return PriceLevel(
            price=self.price,
            count=sum(1 for o in self.orders if not (o.flags & db.RecordFlags.F_TOB)),
            size=sum(o.size for o in self.orders),
        )


@dataclass(slots=True)
class Book:
    orders_by_id: dict = field(default_factory=dict)
    offers: SortedDict = field(default_factory=SortedDict)
    bids: SortedDict = field(default_factory=SortedDict)

    def get_bid_level(self):
        if self.bids:
            return self.bids.peekitem(-1)[1].level
        return None

    def get_ask_level(self):
        if self.offers:
            return self.offers.peekitem(0)[1].level
        return None

    def apply(self, mbo) -> None:
        import databento as db
        action = _norm_action(mbo)
        side = _norm_side(mbo)
        if action in ("T", "F", "N"):
            return
        if action == "R":
            self._clear(); return
        assert side in ("A", "B")
        if mbo.price == db.UNDEF_PRICE and (mbo.flags & db.RecordFlags.F_TOB):
            self._side_levels(side).clear(); return
        if action == "A": self._add(mbo, side)
        elif action == "C": self._cancel(mbo, side)
        elif action == "M": self._modify(mbo, side)

    def _clear(self):
        self.orders_by_id.clear(); self.offers.clear(); self.bids.clear()

    def _add(self, mbo, side):
        import databento as db
        if mbo.flags & db.RecordFlags.F_TOB:
            lvls = self._side_levels(side)
            lvls.clear()
            lvls[mbo.price] = LevelOrders(price=mbo.price, orders=[mbo])
        else:
            lv = self._get_or_insert(mbo.price, side)
            self.orders_by_id[mbo.order_id] = mbo
            lv.orders.append(mbo)

    def _cancel(self, mbo, side):
        order = self.orders_by_id.get(mbo.order_id)
        if order is None: return
        lv = self._side_levels(side).get(mbo.price)
        if lv is None: return
        order.size -= mbo.size
        if order.size <= 0:
            self.orders_by_id.pop(mbo.order_id, None)
            try: lv.orders.remove(order)
            except ValueError: pass
            if not lv: self._side_levels(side).pop(mbo.price, None)

    def _modify(self, mbo, side):
        order = self.orders_by_id.get(mbo.order_id)
        if order is None:
            self._add(mbo, side); return
        lvls = self._side_levels(side)
        lv = lvls.get(order.price)
        if lv is None:
            self._add(mbo, side); return
        if order.price != mbo.price:
            try: lv.orders.remove(order)
            except ValueError: pass
            if not lv: lvls.pop(order.price, None)
            self._get_or_insert(mbo.price, side).orders.append(mbo)
        elif order.size < mbo.size:
            try: lv.orders.remove(order)
            except ValueError: pass
            lv.orders.append(mbo)
        else:
            idx = next((i for i, o in enumerate(lv.orders)
                        if o.order_id == mbo.order_id), None)
            if idx is not None: lv.orders[idx] = mbo
        self.orders_by_id[mbo.order_id] = mbo

    def _side_levels(self, side): return self.offers if side == "A" else self.bids

    def _get_or_insert(self, price, side):
        lvls = self._side_levels(side)
        if price not in lvls: lvls[price] = LevelOrders(price=price)
        return lvls[price]



def process_symbol(sym: str, inst_id: int,
                   target_days: list[str],
                   day_sessions: dict[str, tuple[int, int]],
                   tmp_dir: str) -> dict[str, int]:
    """
Worker: replay order book for one symbol across all monthly files.
Writes per-day LOB temp files to tmp_dir/<sym>/<YYYYMMDD>.bin.
Returns {date_str: n_records_emitted}.
"""
    import databento as db

    sym_tmp = os.path.join(tmp_dir, sym)
    os.makedirs(sym_tmp, exist_ok=True)

    file_list = []
    sym_dir = os.path.join(RAW_DIR, sym)
    for f in sorted(glob.glob(f"{sym_dir}/*.dbn.zst")):
        fname = os.path.basename(f)
        parts = fname.replace(".dbn.zst", "").split("_")
        if len(parts) >= 2:
            file_list.append((parts[-2], parts[-1], f))
    file_list.sort()

    counts: dict[str, int] = {d: 0 for d in target_days}

    for file_start, file_end, filepath in file_list:
        days_in_file = sorted(
            d for d in target_days
            if file_start <= d.replace("-", "") < file_end
        )
        if not days_in_file:
            continue

        sessions = [(d, *day_sessions[d]) for d in days_in_file]
        last_close = sessions[-1][2] + 60_000_000_000

        # Open per-day temp output files for this symbol
        day_handles: dict[str, object] = {}
        for d in days_in_file:
            path = os.path.join(sym_tmp, f"{d.replace('-','')}.bin")
            day_handles[d] = open(path, "wb")

        store = db.DBNStore.from_file(filepath)
        book = Book()

        for mbo in store:
            ts = mbo.ts_event
            action = _norm_action(mbo)

            if ts > last_close and action == "T":
                break

            if action == "T":
                for (date_str, open_ns, close_ns) in sessions:
                    if open_ns <= ts < close_ns:
                        bid_lvl = book.get_bid_level()
                        ask_lvl = book.get_ask_level()
                        best_bid = bid_lvl.price if bid_lvl else 0
                        best_ask = ask_lvl.price if ask_lvl else 0
                        side_b = ord(_norm_side(mbo)[0])
                        day_handles[date_str].write(struct.pack(
                            LOB_FMT, ts, inst_id, side_b, 0,
                            mbo.price, best_bid, best_ask, mbo.size,
                        ))
                        counts[date_str] += 1
                        break

            book.apply(mbo)

        for fh in day_handles.values():
            fh.close()

    return counts



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-10-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--out", default="data/lob")
    parser.add_argument("--id_map", default="data/allstock_id_map.json")
    parser.add_argument("--tmp", default="data/lob_tmp")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only_syms", nargs="*", default=None,
                        help="Only run per-symbol workers for these symbols; "
                             "still merges all symbols from lob_tmp")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)

    with open(args.id_map) as f:
        id_map = json.load(f)
    sym_to_id = {v: int(k) for k, v in id_map.items()}

    start_date = datetime.date.fromisoformat(args.start)
    end_date = datetime.date.fromisoformat(args.end)
    days = trading_days(start_date, end_date)

    target_days = [d.strftime("%Y-%m-%d") for d in days]
    day_sessions = {d.strftime("%Y-%m-%d"): session_ns(d) for d in days}

    # Skip days already finalized
    todo_days = [
        d for d in target_days
        if not os.path.exists(
            os.path.join(args.out, d.replace("-", "") + "_allstock_lob.bin")
        ) or os.path.getsize(
            os.path.join(args.out, d.replace("-", "") + "_allstock_lob.bin")
        ) == 0
    ]
    if not todo_days:
        return

    # Parallel: one worker per symbol (optionally restricted to --only_syms)
    worker_syms = sym_to_id if args.only_syms is None else {
        s: sym_to_id[s] for s in args.only_syms if s in sym_to_id
    }
    all_counts: dict[str, dict[str, int]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_symbol, sym, inst_id, todo_days, day_sessions, args.tmp): sym
            for sym, inst_id in worker_syms.items()
        }
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                all_counts[sym] = fut.result()
            except Exception as e:
                print(f" [FAIL] {sym}: ERROR {e}", flush=True)
                import traceback; traceback.print_exc()

    # Merge per-symbol temp files into per-day LOB binaries
    for d in todo_days:
        out_path = os.path.join(args.out, d.replace("-", "") + "_allstock_lob.bin")
        n_total = 0
        with open(out_path, "wb") as fout:
            for sym in sorted(sym_to_id.keys(), key=lambda s: sym_to_id[s]):
                tmp_path = os.path.join(args.tmp, sym, f"{d.replace('-','')}.bin")
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    with open(tmp_path, "rb") as fin:
                        data = fin.read()
                    fout.write(data)
                    n_total += len(data) // LOB_SZ

    print(f"\nDone. LOB binaries written to {args.out}/")


if __name__ == "__main__":
    main()
