"""NL intraday auction prices (IDA1, IDA2, IDA3) from the Nord Pool data portal.

Source: https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices
Request: ?date=YYYY-MM-DD&market=SIDC_IntradayAuction1|2|3&deliveryArea=NL&currency=EUR
The route needs no key and no cookie. Research: docs/research/2026-10-29-source-hunt.md, Q1.

Terms of use (Nord Pool website terms, quoted in the research report): content may be used
"for analysis or research of your firm or company"; the terms say Nord Pool "does not permit
automatic extraction of data" and forbid redistribution. Publish nothing derived from this feed
without checking the terms first.

Auction clearing, local Europe/Amsterdam time: IDA1 15:00 on D-1, IDA2 22:00 on D-1, IDA3
10:00 on D (slots from 12:00 only). Nord Pool publishes about 17 minutes after each gate.
A feature from this feed must be NaN before the result is published: see available_at.
"""

from __future__ import annotations

import datetime as dt
import json
import time

import numpy as np

from ..timegrid import ZONE, Table, align, isp_grid
from .base import DayCache, http_get, local_days

API = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"
AREA = "NL"
PACING_S = 1.0
AUCTIONS = ("IDA1", "IDA2", "IDA3")
MARKET = {
    "IDA1": "SIDC_IntradayAuction1",
    "IDA2": "SIDC_IntradayAuction2",
    "IDA3": "SIDC_IntradayAuction3",
}
HISTORY_START = {
    "IDA1": dt.date(2024, 6, 15),
    "IDA2": dt.date(2024, 6, 14),
    "IDA3": dt.date(2024, 6, 14),
}
_IDA3_FIRST_HOUR = 12
_SETTLED_AFTER_DAYS = 2

_last_request = [0.0]


def _pace() -> None:
    wait = PACING_S - (time.monotonic() - _last_request[0])
    if wait > 0 and _last_request[0] > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _local(day: dt.date, hour: int) -> int:
    return int(dt.datetime(day.year, day.month, day.day, hour, tzinfo=ZONE).timestamp())


def clearing_time(auction: str, delivery_day: dt.date) -> int:
    """Epoch s of the auction gate: IDA1 15:00 and IDA2 22:00 local on D-1, IDA3 10:00 on D."""
    prev = delivery_day - dt.timedelta(days=1)
    if auction == "IDA1":
        return _local(prev, 15)
    if auction == "IDA2":
        return _local(prev, 22)
    if auction == "IDA3":
        return _local(delivery_day, 10)
    raise ValueError(f"unknown auction: {auction}")


def _local_dt(ts: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=ZONE)


def covers(auction: str, ts: int) -> bool:
    """True when the auction prices the slot starting at ts. IDA3 covers local hour >= 12."""
    if auction not in AUCTIONS:
        raise ValueError(f"unknown auction: {auction}")
    if auction == "IDA3":
        return _local_dt(ts).hour >= _IDA3_FIRST_HOUR
    return True


def available_at(
    auction: str, ts_target: int, t_forecast: int, publish_delay_s: int = 1200
) -> bool:
    """True when the auction result for the slot ts_target is public at time t_forecast."""
    if not covers(auction, ts_target):
        return False
    day = _local_dt(ts_target).date()
    return t_forecast >= clearing_time(auction, day) + publish_delay_s


def available_mask(
    auction: str, ts_target: np.ndarray, t_forecast: np.ndarray, publish_delay_s: int = 1200
) -> np.ndarray:
    """``available_at`` over arrays, element by element, with one clearing time per local day.

    Each target maps to its local Amsterdam day by the day's local midnight; IDA3 covers a
    target at or after the day's local 12:00 (the wall-clock hour is 12 or later, since no
    DST change falls between 12:00 and midnight).
    """
    if auction not in AUCTIONS:
        raise ValueError(f"unknown auction: {auction}")
    ts = np.asarray(ts_target, dtype=np.int64)
    tf = np.asarray(t_forecast, dtype=np.int64)
    if len(ts) == 0:
        return np.zeros(0, dtype=bool)
    first = _local_dt(int(ts.min())).date()
    last = _local_dt(int(ts.max())).date()
    days = [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]
    midnight = np.array([_local(d, 0) for d in days], dtype=np.int64)
    k = np.searchsorted(midnight, ts, side="right") - 1
    clear = np.array([clearing_time(auction, d) for d in days], dtype=np.int64)
    avail = tf >= clear[k] + int(publish_delay_s)
    if auction == "IDA3":
        noon = np.array([_local(d, _IDA3_FIRST_HOUR) for d in days], dtype=np.int64)
        avail &= ts >= noon[k]
    return avail


def _epoch(stamp: str) -> int:
    text = stamp.strip().replace("Z", "+00:00")
    if "." in text:
        head, rest = text.split(".", 1)
        frac = "".join(ch for ch in rest if ch.isdigit())[:6]
        tail = rest[len("".join(ch for ch in rest if ch.isdigit())) :]
        text = f"{head}.{frac}{tail}"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return int(parsed.timestamp())


def parse_results(payload: bytes, auction: str, delivery_day: dt.date) -> list[dict]:
    """Return one row per slot in a Nord Pool document. A Missing document gives []."""
    if not payload.strip():
        return []
    doc = json.loads(payload)
    entries = doc.get("multiAreaEntries") if isinstance(doc, dict) else None
    if not entries:
        return []
    updated = _epoch(doc["updatedAt"]) if doc.get("updatedAt") else 0
    rows = []
    for entry in entries:
        price = (entry.get("entryPerArea") or {}).get(AREA)
        rows.append(
            {
                "ts": _epoch(entry["deliveryStart"]),
                "ida_eur_mwh": float("nan") if price is None else float(price),
                "ida_volume_mwh": float("nan"),
                "updated_at": updated,
            }
        )
    rows.sort(key=lambda row: row["ts"])
    return rows


def _day_bounds(day: dt.date) -> tuple[int, int]:
    start = dt.datetime(day.year, day.month, day.day, tzinfo=ZONE)
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), tzinfo=ZONE)
    return int(start.timestamp()), int(end.timestamp())


def _nan_to_none(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


def _none_to_nan(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _fetch_day(auction: str, day: dt.date) -> list[dict]:
    params = {
        "date": day.isoformat(),
        "market": MARKET[auction],
        "deliveryArea": AREA,
        "currency": "EUR",
    }
    _pace()
    return parse_results(http_get(API, params=params), auction, day)


def fetch_ida(
    auction: str, start: dt.date, end: dt.date, cache: DayCache, offline: bool = False
) -> Table:
    """Auction price (EUR/MWh) on the 900 s grid for the local days start..end.

    The caller passes a DayCache that belongs to this auction alone, for example
    <cache dir>/ida_IDA1.jsonl, so the cache key is the delivery day. NaN where the auction
    has no slot (IDA3 mornings, days before HISTORY_START, days without a result). Empty
    results are cached only for days that are settled, so a day awaiting its auction is asked again.
    """
    if auction not in AUCTIONS:
        raise ValueError(f"unknown auction: {auction}")
    days = local_days(start, end)
    settled_before = dt.datetime.now(ZONE).date() - dt.timedelta(days=_SETTLED_AFTER_DAYS)
    ts_list: list[int] = []
    cols: dict[str, list[float]] = {"ida_eur_mwh": [], "ida_volume_mwh": [], "updated_at": []}
    for day in days:
        if day < HISTORY_START[auction]:
            continue
        rows = cache.get(day)
        if rows is None and not offline:
            fetched = _fetch_day(auction, day)
            rows = [
                {
                    "ts": r["ts"],
                    "ida_eur_mwh": _nan_to_none(r["ida_eur_mwh"]),
                    "ida_volume_mwh": _nan_to_none(r["ida_volume_mwh"]),
                    "updated_at": r["updated_at"],
                }
                for r in fetched
            ]
            if rows or day < settled_before:
                cache.put(day, rows)
        for row in rows or []:
            ts_list.append(int(row["ts"]))
            cols["ida_eur_mwh"].append(_none_to_nan(row["ida_eur_mwh"]))
            cols["ida_volume_mwh"].append(_none_to_nan(row["ida_volume_mwh"]))
            cols["updated_at"].append(_none_to_nan(row["updated_at"]))
    source: Table = {"ts": np.array(ts_list, dtype=np.int64)}
    source.update({name: np.array(vals, dtype=float) for name, vals in cols.items()})
    grid = isp_grid(_day_bounds(days[0])[0], _day_bounds(days[-1])[1])
    return align(source, grid, list(cols))
