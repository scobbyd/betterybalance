"""Live minute prices: TenneT running imbalance price per minute, from the TenEnergy page.

Source: https://services.tenergy.nl/public.aspx/actualimbalanceprices, an HTML page that embeds
the chart as `var data = {...};`. The series PRICES_TAKE (buy) and PRICES_FEED (sell) hold points
keyed `/Date(ms)/`. The page holds about 650 kB of points. The research report names no other
tokenless live route: docs/research/2026-10-29-source-hunt.md, question 3.

The page can freeze. A frozen run of identical prices is stale: see stale_mask.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from ..timegrid import ISP_S, Table
from .base import http_get

TENERGY_URL = "https://services.tenergy.nl/public.aspx/actualimbalanceprices"
BUY_ID = "PRICES_TAKE"
SELL_ID = "PRICES_FEED"
_ELAPSED_MINUTES = (1, 5, 10)
_DATA_BLOCK = re.compile(r"var data = ({.*?});\s*\n", flags=re.S)


def key_ms(point: dict) -> int:
    """Point keys look like '/Date(1760000000000)/'."""
    return int(re.search(r"-?\d+", point.get("Key", "0")).group())


def parse_tenergy(html: str) -> list[dict]:
    """Every timestamp with both prices, as {"ts": epoch_s, "buy_live", "sell_live"}, sorted.

    The returned ts values are the page's point times, not observation times. The page publishes
    a point after its time. A recorder must label a sample by the minute it saw it, which is known
    at t + 60 (the convention of the archive), and not by the point key.
    """
    match = _DATA_BLOCK.search(html)
    if not match:
        return []
    try:
        chart = json.loads(match.group(1))
    except ValueError:
        return []
    series: dict[str, dict[int, float]] = {}
    for entry in chart.get("ChartSeries", []):
        if entry.get("Id") in (BUY_ID, SELL_ID):
            series[entry["Id"]] = {
                key_ms(p) // 1000: float(p["Value"])
                for p in entry.get("Data") or []
                if p.get("Value") is not None
            }
    buy = series.get(BUY_ID, {})
    sell = series.get(SELL_ID, {})
    return [
        {"ts": ts, "buy_live": buy[ts], "sell_live": sell[ts]} for ts in sorted(buy.keys() & sell)
    ]


def poll_live() -> list[dict]:
    """Fetch the TenEnergy page and parse it.

    The timestamps are the page's point times, not observation times. Label a recorded sample by
    the minute of the poll that saw it, known at t + 60, as the archive does.
    """
    return parse_tenergy(http_get(TENERGY_URL).decode("utf-8", "replace"))


poll_tenergy = poll_live


def load_live_minutes(path: Path) -> Table:
    """Read a CSV `ts,buy_live,sell_live` (one row per minute, gaps allowed). Empty means NaN."""
    ts: list[int] = []
    buy: list[float] = []
    sell: list[float] = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            ts.append(int(row["ts"]))
            buy.append(float(row["buy_live"]) if row["buy_live"] != "" else np.nan)
            sell.append(float(row["sell_live"]) if row["sell_live"] != "" else np.nan)
    return {
        "ts": np.asarray(ts, dtype=np.int64),
        "buy_live": np.asarray(buy, dtype=float),
        "sell_live": np.asarray(sell, dtype=float),
    }


def stale_mask(table: Table, max_unchanged_s: int = 1800) -> np.ndarray:
    """True where both prices stayed exactly unchanged for longer than max_unchanged_s.

    The run starts at the first minute with the frozen value. NaN rows are missing, not stale,
    and they end a run.
    """
    ts = np.asarray(table["ts"], dtype=np.int64)
    buy = np.asarray(table["buy_live"], dtype=float)
    sell = np.asarray(table["sell_live"], dtype=float)
    n = len(ts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    valid = ~(np.isnan(buy) | np.isnan(sell))
    same = np.zeros(n, dtype=bool)
    same[1:] = valid[1:] & valid[:-1] & (buy[1:] == buy[:-1]) & (sell[1:] == sell[:-1])
    start = np.where(same, 0, np.arange(n))
    start = np.maximum.accumulate(start)
    return valid & (ts - ts[start] > max_unchanged_s)


def minute_to_isp(table: Table, grid: np.ndarray) -> Table:
    """Per ISP: last valid minute and the mean of the valid minutes among the first 1, 5, 10.

    Minute convention: a minute labelled t holds the samples taken in [t, t + 60) and is known at
    t + 60. So the e_m mean of an ISP covers its first m minutes and is known at start + m * 60 s,
    and the end value is the ISP's true last minute, known at the ISP end.
    """
    ts = np.asarray(table["ts"], dtype=np.int64)
    stale = stale_mask(table)
    out: Table = {"ts": np.asarray(grid, dtype=np.int64)}
    for name in ("buy_live", "sell_live"):
        vals = np.asarray(table[name], dtype=float)
        ok = ~np.isnan(vals) & ~stale
        end = np.full(len(grid), np.nan)
        elapsed = {m: np.full(len(grid), np.nan) for m in _ELAPSED_MINUTES}
        for i, start in enumerate(out["ts"].tolist()):
            lo = np.searchsorted(ts, start)
            hi = np.searchsorted(ts, start + ISP_S)
            good = ok[lo:hi]
            if not good.any():
                continue
            seg = vals[lo:hi]
            end[i] = seg[good][-1]
            age = ts[lo:hi] - start
            for m in _ELAPSED_MINUTES:
                sel = good & (age < m * 60)
                if sel.any():
                    elapsed[m][i] = seg[sel].mean()
        out[f"{name}_end"] = end
        for m in _ELAPSED_MINUTES:
            out[f"{name}_e{m}"] = elapsed[m]
    return out
