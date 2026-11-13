"""ISP time grid, table alignment and the Dutch local calendar."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

ISP_S = 900
ZONE = ZoneInfo("Europe/Amsterdam")

Table = dict[str, np.ndarray]


def isp_start(ts: np.ndarray) -> np.ndarray:
    """Floor epoch seconds to the start of their ISP."""
    return (np.asarray(ts, dtype=np.int64) // ISP_S) * ISP_S


def isp_grid(start: int, end: int) -> np.ndarray:
    """Every ISP start in [start, end), with start floored to 900 s."""
    first = (int(start) // ISP_S) * ISP_S
    return np.arange(first, int(end), ISP_S, dtype=np.int64)


def align(table: Table, grid: np.ndarray, cols: list[str]) -> Table:
    """Place columns on the grid by exact timestamp match, NaN where missing."""
    ts = np.asarray(table["ts"], dtype=np.int64)
    idx = np.searchsorted(ts, grid)
    idx_c = np.minimum(idx, max(len(ts) - 1, 0))
    hit = (idx < len(ts)) & (ts[idx_c] == grid) if len(ts) else np.zeros(len(grid), bool)
    out: Table = {"ts": grid}
    for col in cols:
        vals = np.asarray(table[col], dtype=float)
        res = np.full(len(grid), np.nan)
        if len(ts):
            res[hit] = vals[idx_c[hit]]
        out[col] = res
    return out


def _easter(year: int) -> dt.date:
    """Easter Sunday by the anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    el = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * el) // 451
    month, day = divmod(h + el - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def nl_holidays(year: int) -> set[dt.date]:
    """Dutch public holidays of a year."""
    easter = _easter(year)
    king = dt.date(year, 4, 27)
    if king.weekday() == 6:
        king = dt.date(year, 4, 26)
    offsets = (-2, 0, 1, 39, 49, 50)
    days = {easter + dt.timedelta(days=o) for o in offsets}
    days |= {
        dt.date(year, 1, 1),
        king,
        dt.date(year, 5, 5),
        dt.date(year, 12, 25),
        dt.date(year, 12, 26),
    }
    return days


def local_calendar(ts: np.ndarray) -> Table:
    """Local Europe/Amsterdam calendar features per timestamp."""
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    slot = np.zeros(n, dtype=np.int64)
    hour = np.zeros(n, dtype=np.int64)
    dow = np.zeros(n, dtype=np.int64)
    month = np.zeros(n, dtype=np.int64)
    holiday = np.zeros(n, dtype=bool)
    cache: dict[int, set[dt.date]] = {}
    for i, t in enumerate(ts.tolist()):
        local = dt.datetime.fromtimestamp(t, ZONE)
        midnight = dt.datetime(local.year, local.month, local.day, tzinfo=ZONE)
        slot[i] = (t - int(midnight.timestamp())) // ISP_S
        hour[i] = local.hour
        dow[i] = local.weekday()
        month[i] = local.month
        if local.year not in cache:
            cache[local.year] = nl_holidays(local.year)
        holiday[i] = local.date() in cache[local.year]
    return {
        "ts": ts,
        "slot_of_day": slot,
        "hour": hour,
        "dow": dow,
        "month": month,
        "is_weekend": dow >= 5,
        "is_holiday": holiday,
    }
