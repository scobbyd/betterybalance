"""NL day-ahead prices from the public energy-charts.info API."""

from __future__ import annotations

import calendar
import datetime as dt
import time

import numpy as np

from ..timegrid import ISP_S, ZONE, Table, align, isp_grid
from .base import DayCache, http_get, json_object, local_days

API = "https://api.energy-charts.info/price"
BZN = "NL"
PACING_S = 1.0
_HOUR_S = 3600

_last_request = [0.0]


def _pace() -> None:
    wait = PACING_S - (time.monotonic() - _last_request[0])
    if wait > 0 and _last_request[0] > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def parse_price_json(payload: dict) -> list[tuple[int, float]]:
    """Return (epoch_s, eur_mwh) pairs from an API payload. Null prices become NaN."""
    stamps = payload.get("unix_seconds") or []
    prices = payload.get("price") or []
    return [
        (int(ts), float("nan") if price is None else float(price))
        for ts, price in zip(stamps, prices, strict=False)
    ]


def to_quarter_table(points: list[tuple[int, float]]) -> Table:
    """Expand points to the 900 s grid, holding hourly values for four quarters."""
    first: dict[int, float] = {}
    for ts, price in points:
        first.setdefault(ts, price)
    stamps = sorted(first)
    ts_out: list[int] = []
    val_out: list[float] = []
    for i, ts in enumerate(stamps):
        if i + 1 < len(stamps):
            gap = stamps[i + 1] - ts
        elif i > 0:
            gap = ts - stamps[i - 1]
        else:
            gap = ISP_S
        quarters = 4 if gap == _HOUR_S else 1
        for q in range(quarters):
            ts_out.append(ts + q * ISP_S)
            val_out.append(first[ts])
    return {
        "ts": np.array(ts_out, dtype=np.int64),
        "da_eur_mwh": np.array(val_out, dtype=float),
    }


def _day_bounds(day: dt.date) -> tuple[int, int]:
    start = dt.datetime(day.year, day.month, day.day, tzinfo=ZONE)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), tzinfo=ZONE)
    return int(start.timestamp()), int(end.timestamp())


def _fetch_month(year: int, month: int) -> Table:
    last = calendar.monthrange(year, month)[1]
    params = {
        "bzn": BZN,
        "start": dt.date(year, month, 1).isoformat(),
        "end": dt.date(year, month, last).isoformat(),
    }
    _pace()
    payload = json_object(http_get(API, params=params), "energy-charts")
    return to_quarter_table(parse_price_json(payload))


def _split_into_days(table: Table, days: list[dt.date], cache: DayCache) -> None:
    ts = table["ts"]
    vals = table["da_eur_mwh"]
    for day in days:
        lo, hi = _day_bounds(day)
        mask = (ts >= lo) & (ts < hi)
        if int(mask.sum()) < (hi - lo) // ISP_S:
            continue
        rows = [
            {"ts": int(t), "da_eur_mwh": None if np.isnan(v) else float(v)}
            for t, v in zip(ts[mask], vals[mask], strict=True)
        ]
        cache.put(day, rows)


def fetch_day_ahead(
    start: dt.date, end: dt.date, cache: DayCache, offline: bool = False
) -> Table:
    """Day-ahead price (EUR/MWh) on the 15 minute grid for the local days start..end."""
    days = local_days(start, end)
    cached: dict[dt.date, list[dict]] = {}
    missing: list[dt.date] = []
    for day in days:
        rows = cache.get(day)
        if rows is None:
            missing.append(day)
        else:
            cached[day] = rows
    if missing and not offline:
        months = sorted({(d.year, d.month) for d in missing})
        for year, month in months:
            table = _fetch_month(year, month)
            month_days = [d for d in missing if (d.year, d.month) == (year, month)]
            _split_into_days(table, month_days, cache)
        for day in missing:
            rows = cache.get(day)
            if rows is not None:
                cached[day] = rows
    ts_list: list[int] = []
    val_list: list[float] = []
    for day in days:
        for row in cached.get(day, []):
            ts_list.append(int(row["ts"]))
            val_list.append(float("nan") if row["da_eur_mwh"] is None else row["da_eur_mwh"])
    source: Table = {
        "ts": np.array(ts_list, dtype=np.int64),
        "da_eur_mwh": np.array(val_list, dtype=float),
    }
    grid = isp_grid(_day_bounds(days[0])[0], _day_bounds(days[-1])[1])
    return align(source, grid, ["da_eur_mwh"])
