"""Store-writing fetchers the coordinator runs in the executor (sync, blocking).

Each fetcher writes what it got to the store and raises ``FeedError`` when the feed failed or
the data is not there yet; the coordinator updates the ``live.feeds.FeedState``. Every request
goes through ``core.feeds.base`` (``http_get`` and ``http_post``) to the hosts of the global
network list. Prices stay EUR/MWh; a missing price is NaN (NULL in the store), never 0.
"""

from __future__ import annotations

import datetime as dt
import math
import time

import numpy as np

from ..core.feeds import base
from ..core.feeds import energy_charts as EC
from ..core.feeds import ida as IDA
from ..core.feeds import live as LV
from ..core.feeds import openmeteo as OM
from ..core.feeds import tennet_live as TL
from ..core.feeds import tennet_settlement as TS
from ..core.site.contract import round_location
from ..core.store import Store
from ..core.timegrid import ISP_S, ZONE

FeedError = base.FeedError
MINUTE_S = 60
# a TenneT `latest` window whose newest point ended longer ago than this is a stalled feed
# (the publication delay is 2 minutes in the press; my call: five times that)
TENNET_MAX_AGE_S = 600
SETTLEMENT_SOURCE = "tennet_ords"
FORECAST_SOURCE = "forecast"


def _day_bounds(day: dt.date) -> tuple[int, int]:
    """Local midnight of ``day`` and of the day after (92, 96 or 100 quarters apart)."""
    lo = dt.datetime(day.year, day.month, day.day, tzinfo=ZONE)
    nxt = day + dt.timedelta(days=1)
    hi = dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=ZONE)
    return int(lo.timestamp()), int(hi.timestamp())


def _finite(v) -> bool:
    return v is not None and math.isfinite(float(v))


def fetch_day_ahead(store: Store, day: dt.date) -> int:
    """Day-ahead prices of the local day ``day`` from energy-charts into ``features.da_eur_mwh``.

    Returns the number of priced quarters. Raises FeedError when the day is not complete (the
    priced quarters are written all the same, so a retry only adds what is new).
    """
    params = {"bzn": EC.BZN, "start": day.isoformat(), "end": day.isoformat()}
    payload = base.json_object(base.http_get(EC.API, params=params), "energy-charts")
    table = EC.to_quarter_table(EC.parse_price_json(payload))
    lo, hi = _day_bounds(day)
    priced = 0
    for ts, price in zip(table["ts"].tolist(), table["da_eur_mwh"].tolist(), strict=True):
        if lo <= ts < hi and ts % ISP_S == 0 and _finite(price):
            store.upsert_features(ts, da_eur_mwh=price)
            priced += 1
    expected = (hi - lo) // ISP_S
    if priced < expected:
        raise FeedError(f"day-ahead {day}: {priced} of {expected} quarters priced")
    return priced


def fetch_ida(store: Store, auction: str, day: dt.date) -> int:
    """The result of ``auction`` for the delivery day ``day`` (Nord Pool) into
    ``features.ida<n>_eur_mwh``. Returns the number of priced slots; raises FeedError with none.
    """
    if auction not in IDA.AUCTIONS:
        raise ValueError(f"unknown auction: {auction}")
    params = {
        "date": day.isoformat(),
        "market": IDA.MARKET[auction],
        "deliveryArea": IDA.AREA,
        "currency": "EUR",
    }
    try:
        rows = IDA.parse_results(base.http_get(IDA.API, params=params), auction, day)
    except (ValueError, KeyError, TypeError) as exc:
        raise FeedError(f"{auction} {day}: unreadable result: {exc!r}") from exc
    lo, hi = _day_bounds(day)
    column = f"{auction.lower()}_eur_mwh"
    priced = 0
    for r in rows:
        ts = int(r["ts"])
        if lo <= ts < hi and IDA.covers(auction, ts) and _finite(r["ida_eur_mwh"]):
            store.upsert_features(ts, **{column: r["ida_eur_mwh"]})
            priced += 1
    if priced == 0:
        raise FeedError(f"{auction} {day}: no result yet")
    return priced


def fetch_settlement(store: Store, day: dt.date, now: int | None = None) -> list[dict]:
    """Settled prices of the local day ``day`` (TenneT ORDS) appended to ``settlements``.

    A quarter is appended when it has both prices and differs from its newest stored
    settlement (the first publication or a TenneT correction), with ``published_at = now``
    (``None``: the wall time after the response). Returns the appended rows (store columns),
    not yet booked: ``LiveSite.settle`` books them at a minute the live loop processes, so
    ``published_at`` must be the time the rows were known, never before. Raises FeedError
    when the report has no rows.
    """
    rows = TS._fetch_day(TS._Session(), day)
    now = int(time.time()) if now is None else int(now)
    if not rows:
        raise FeedError(f"settlement {day}: no rows published")
    lo, hi = _day_bounds(day)
    newest = {int(r["quarter_ts"]): r for r in store.settlements(lo, hi)}
    out: list[dict] = []
    for r in rows:
        take, feed = r["buy_eur_mwh"], r["sell_eur_mwh"]
        if not (_finite(take) and _finite(feed)):
            continue
        rs = r["regulation_state"]
        state = None if isinstance(rs, float) and math.isnan(rs) else int(rs)
        qts = int(r["ts"])
        old = newest.get(qts)
        if old is not None and (old["take"], old["feed"], old["regulation_state"]) == (
            float(take), float(feed), state,
        ):
            continue
        row = {
            "quarter_ts": qts,
            "take": float(take),
            "feed": float(feed),
            "regulation_state": state,
            "published_at": now,
            "source": SETTLEMENT_SOURCE,
        }
        if store.put_settlement(row):
            out.append(row)
    return out


def _tenergy_rows() -> list[dict]:
    html = base.http_get(LV.TENERGY_URL).decode("utf-8", "replace")
    return [{**r, "state": None} for r in LV.parse_tenergy(html)]


def poll_live(store: Store, now: int | None, tennet_key: str | None) -> int:
    """One live minute: the newest priced point of the feed, labelled by the minute of ``now``
    (the minute that saw it, known at its end), into ``live_minutes``. Returns rows written.

    ``now`` None (the coordinator) reads the wall clock after the response arrived, so the label
    is never earlier than the poll that saw the price; an explicit ``now`` stands for that clock.
    The TenneT publications API with ``tennet_key``, else the TenEnergy page. Raises FeedError
    when the feed fails, has no point, its newest point has no price, or (TenneT) its newest
    point ended more than ``TENNET_MAX_AGE_S`` before ``now``.
    """
    if tennet_key:
        source, rows = TL.SOURCE, TL.poll_tennet(tennet_key)
    else:
        source, rows = "tenergy", _tenergy_rows()
    now = int(time.time()) if now is None else int(now)
    if not rows:
        raise FeedError(f"{source}: no point")
    newest = rows[-1]
    if tennet_key:
        age = now - (int(newest["ts"]) + MINUTE_S)
        if age > TENNET_MAX_AGE_S:
            raise FeedError(f"{source}: newest point is {age} s old")
    if not (_finite(newest["buy_live"]) and _finite(newest["sell_live"])):
        raise FeedError(f"{source}: newest point has no price")
    row = {
        "ts": now - now % MINUTE_S,
        "buy_live": float(newest["buy_live"]),
        "sell_live": float(newest["sell_live"]),
        "state": newest.get("state"),
        "source": source,
    }
    store.put_live_minutes([row])
    return 1


def fetch_irradiance(store: Store, lat: float, lon: float, now: int) -> int:
    """The open-meteo forecast at the location rounded to 0,1 degree, stored as the fetch
    ``(issue_ts=now, source="forecast")``. Returns the number of quarters."""
    lat, lon = round_location(lat, lon)
    table = OM.OpenMeteo().forecast(lat, lon, int(now))
    if not len(table["ts"]) or not np.isfinite(table["ghi_w_m2"]).any():
        raise FeedError("open-meteo: forecast has no irradiance")
    store.put_fetch(issue_ts=int(now), source=FORECAST_SOURCE, fetched_at=int(now), table=table)
    return int(len(table["ts"]))
