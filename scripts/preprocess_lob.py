#!/usr/bin/env python3
"""
Preprocess MBO data into a LOB binary with best bid/ask at each trade.

Replays the full order book using the official databento Book/Market
reconstruction logic, and emits one 48-byte record per trade containing:
  (ts_ns, inst_id, side, price, best_bid, best_ask, size)

Usage:
    /ocean/projects/cis260122p/ccheung1/.conda_envs/lob/bin/python3 \
        scripts/preprocess_lob.py
"""

from __future__ import annotations

import glob
import json
import os
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import takewhile

import databento as db
from sortedcontainers import SortedDict

# -- Session window (2025-12-01 09:30-16:00 ET = UTC-5) ----
OPEN_NS  = 1_764_599_400_000_000_000
CLOSE_NS = 1_764_622_800_000_000_000

RAW_DIR     = "/ocean/projects/cis260122p/shared/data/raw"
TARGET_DATE = "20251201"
OUTPUT_PATH = "data/test_20251201_allstock_lob.bin"
ID_MAP_PATH = "data/test_20251201_allstock_id_map.json"

# 48-byte LOB record
LOB_FMT = "<qIBBxxqqqI4x"
assert struct.calcsize(LOB_FMT) == 48

# -- Official databento order book reconstruction ----
# Ported from the official databento Python example, with action/side
# normalization for compatibility across library versions.

def _norm_action(mbo: db.MBOMsg) -> str:
    a = mbo.action
    return a.value if hasattr(a, "value") else str(a)

def _norm_side(mbo: db.MBOMsg) -> str:
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

    def bbo(self) -> tuple:
        return self.get_bid_level(), self.get_ask_level()

    def get_bid_level(self, idx: int = 0) -> PriceLevel | None:
        if self.bids and len(self.bids) > idx:
            return self.bids.peekitem(-(idx + 1))[1].level
        return None

    def get_ask_level(self, idx: int = 0) -> PriceLevel | None:
        if self.offers and len(self.offers) > idx:
            return self.offers.peekitem(idx)[1].level
        return None

    def apply(self, mbo: db.MBOMsg) -> None:
        action = _norm_action(mbo)
        side   = _norm_side(mbo)

        if action in ("T", "F", "N"):
            return
        if action == "R":
            self._clear()
            return
        assert side in ("A", "B"), f"unexpected side {side!r} for action {action!r}"
        if mbo.price == db.UNDEF_PRICE and (mbo.flags & db.RecordFlags.F_TOB):
            self._side_levels(side).clear()
            return
        if action == "A":
            self._add(mbo, side)
        elif action == "C":
            self._cancel(mbo, side)
        elif action == "M":
            self._modify(mbo, side)
        else:
            raise ValueError(f"Unknown action={action!r}")

    def _clear(self) -> None:
        self.orders_by_id.clear()
        self.offers.clear()
        self.bids.clear()

    def _add(self, mbo: db.MBOMsg, side: str) -> None:
        if mbo.flags & db.RecordFlags.F_TOB:
            levels = self._side_levels(side)
            levels.clear()
            levels[mbo.price] = LevelOrders(price=mbo.price, orders=[mbo])
        else:
            level = self._get_or_insert_level(mbo.price, side)
            self.orders_by_id[mbo.order_id] = mbo
            level.orders.append(mbo)

    def _cancel(self, mbo: db.MBOMsg, side: str) -> None:
        order = self.orders_by_id.get(mbo.order_id)
        if order is None:
            return  # may have been cleared by a Reset
        level = self._side_levels(side).get(mbo.price)
        if level is None:
            return
        order.size -= mbo.size
        if order.size <= 0:
            self.orders_by_id.pop(mbo.order_id, None)
            try:
                level.orders.remove(order)
            except ValueError:
                pass
            if not level:
                self._side_levels(side).pop(mbo.price, None)

    def _modify(self, mbo: db.MBOMsg, side: str) -> None:
        order = self.orders_by_id.get(mbo.order_id)
        if order is None:
            self._add(mbo, side)
            return
        levels = self._side_levels(side)
        level = levels.get(order.price)
        if level is None:
            self._add(mbo, side)
            return
        if order.price != mbo.price:
            try:
                level.orders.remove(order)
            except ValueError:
                pass
            if not level:
                levels.pop(order.price, None)
            new_level = self._get_or_insert_level(mbo.price, side)
            new_level.orders.append(mbo)
        elif order.size < mbo.size:
            try:
                level.orders.remove(order)
            except ValueError:
                pass
            level.orders.append(mbo)
        else:
            idx = next((i for i, o in enumerate(level.orders) if o.order_id == mbo.order_id), None)
            if idx is not None:
                level.orders[idx] = mbo
        self.orders_by_id[mbo.order_id] = mbo

    def _side_levels(self, side: str) -> SortedDict:
        return self.offers if side == "A" else self.bids

    def _get_or_insert_level(self, price: int, side: str) -> LevelOrders:
        levels = self._side_levels(side)
        if price not in levels:
            levels[price] = LevelOrders(price=price)
        return levels[price]


# -- Main processing ----

def discover_stocks() -> dict[str, str]:
    stocks = {}
    for sym_dir in sorted(glob.glob(f"{RAW_DIR}/*/")):
        sym = os.path.basename(sym_dir.rstrip("/"))
        for f in sorted(glob.glob(f"{sym_dir}*.dbn.zst")):
            fname = os.path.basename(f)
            parts = fname.replace(".dbn.zst", "").split("_")
            if len(parts) >= 2:
                start, end = parts[-2], parts[-1]
                if start <= TARGET_DATE < end:
                    stocks[sym] = f
                    break
    return stocks


def process_symbol(sym: str, inst_id: int, filepath: str,
                   out_file) -> int:
    print(f"  {sym} (id={inst_id}): {os.path.basename(filepath)}", flush=True)
    store  = db.DBNStore.from_file(filepath)
    book   = Book()
    n_emit = 0

    for mbo in store:
        ts = mbo.ts_event
        action = _norm_action(mbo)

        if action == "T" and OPEN_NS <= ts < CLOSE_NS:
            bid_lvl = book.get_bid_level()
            ask_lvl = book.get_ask_level()
            best_bid = bid_lvl.price if bid_lvl else 0
            best_ask = ask_lvl.price if ask_lvl else 0
            side_byte = ord(_norm_side(mbo)[0])  # 65='A', 66='B'
            out_file.write(struct.pack(
                LOB_FMT,
                ts, inst_id, side_byte, 0,  # ts, id, side, 1-byte pad
                mbo.price, best_bid, best_ask, mbo.size
            ))
            n_emit += 1

        book.apply(mbo)

        # Stop reading past close + 60s buffer (file may span months)
        if ts > CLOSE_NS + 60_000_000_000 and action == "T":
            break

    print(f"    -> {n_emit:,} LOB trade records", flush=True)
    return n_emit


def main():
    with open(ID_MAP_PATH) as f:
        id_map = json.load(f)
    sym_to_id = {v: int(k) for k, v in id_map.items()}

    stocks = discover_stocks()
    stocks = {sym: path for sym, path in stocks.items() if sym in sym_to_id}
    print(f"Processing {len(stocks)} stocks for {TARGET_DATE}:")

    os.makedirs("data", exist_ok=True)
    total = 0
    with open(OUTPUT_PATH, "wb") as out_file:
        for sym, path in stocks.items():
            total += process_symbol(sym, sym_to_id[sym], path, out_file)

    mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"\nWrote {OUTPUT_PATH}: {total:,} records * 48 bytes = {mb:.1f} MB")


if __name__ == "__main__":
    main()
