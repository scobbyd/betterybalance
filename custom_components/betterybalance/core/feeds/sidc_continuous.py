"""NL SIDC continuous intraday market statistics from the Nord Pool data portal.

Source: https://dataportal-api.nordpoolgroup.com/api/IntradayMarketStatistics
Request: ?date=YYYY-MM-DD&deliveryArea=NL (date is the delivery day).
The route needs no key and no cookie. Research: docs/research/2026-10-29-source-hunt.md, Q2.

The route answers only for a rolling window of about 7 days back to 1 day ahead. An older
date returns HTTP 401, which http_get raises as FeedError. History exists only if a recorder
(tools/record_continuous.py) fetches the window on a timer. Later fetches of a day replace
earlier ones in the cache, because the statistics of a delivery day keep changing until it ends.

Terms of use (Nord Pool website terms, same as the IDA feed): content may be used "for analysis
or research of your firm or company"; the terms say Nord Pool "does not permit automatic
extraction of data" and forbid redistribution. Publish nothing derived from this feed without
checking the terms first.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np

from ..timegrid import ZONE, Table, align, isp_grid
from .base import DayCache, http_get, json_object

API = "https://dataportal-api.nordpoolgroup.com/api/IntradayMarketStatistics"
AREA = "NL"
PACING_S = 1.0

_FIELDS = {
    "vwap": "averagePrice",
    "vwap_last3h": "averagePriceLast3H",
    "vwap_last1h": "averagePriceLast1H",
    "high": "highPrice",
    "low": "lowPrice",
    "close": "closePrice",
    "volume_mwh": "volume",
}

_last_request = [0.0]


def _pace() -> None:
    wait = PACING_S - (time.monotonic() - _last_request[0])
    if wait > 0 and _last_request[0] > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _epoch(stamp: str) -> int:
    parsed = dt.datetime.fromisoformat(stamp.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return int(parsed.timestamp())


def _num(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def parse_statistics(payload: dict) -> list[dict]:
    """Return one row per contract, sorted by start. Null fields stay NaN."""
    update = payload.get("updateTime")
    updated = _epoch(update) if update else 0
    rows = []
    for contract in payload.get("contracts") or []:
        start = _epoch(contract["deliveryStart"])
        end = _epoch(contract["deliveryEnd"])
        row = {"ts": start, "duration_min": (end - start) // 60}
        for name, source in _FIELDS.items():
            row[name] = _num(contract.get(source))
        row["update_time"] = updated
        rows.append(row)
    rows.sort(key=lambda row: (row["ts"], row["duration_min"]))
    return rows


def _to_json(row: dict) -> dict:
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}


def _from_json(row: dict) -> dict:
    return {k: (float("nan") if v is None else v) for k, v in row.items()}


def fetch_continuous_day(delivery_day: dt.date, cache: DayCache, offline: bool = False) -> Table:
    """Quarter-hour contract statistics of a delivery day on the 900 s ISP grid.

    A fetch writes every contract of the day to the cache, and the last write wins.
    With offline=True the function reads the cache only. NaN where a quarter hour has
    no contract or no trade.
    """
    rows = cache.get(delivery_day)
    if not offline:
        _pace()
        body = http_get(API, params={"date": delivery_day.isoformat(), "deliveryArea": AREA})
        payload = json_object(body, "nord pool continuous")
        rows = [_to_json(row) for row in parse_statistics(payload)]
        cache.put(delivery_day, rows)
    quarter = sorted(
        (_from_json(r) for r in rows or [] if r["duration_min"] == 15), key=lambda r: r["ts"]
    )
    names = [*_FIELDS, "update_time"]
    source: Table = {"ts": np.array([r["ts"] for r in quarter], dtype=np.int64)}
    for name in names:
        source[name] = np.array([r[name] for r in quarter], dtype=float)
    start = dt.datetime(delivery_day.year, delivery_day.month, delivery_day.day, tzinfo=ZONE)
    end = dt.datetime.combine(delivery_day + dt.timedelta(days=1), dt.time(), tzinfo=ZONE)
    grid = isp_grid(int(start.timestamp()), int(end.timestamp()))
    return align(source, grid, names)
