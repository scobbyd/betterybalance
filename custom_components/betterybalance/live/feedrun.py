"""The feed runs of the coordinator: which feeds are due, the fetchers they run, the seed.

``FeedRunner`` holds no Home Assistant object and no clock: every method takes ``now`` and is
blocking (store reads, network), so the coordinator calls it in the executor. The
``FeedState`` objects live in the coordinator, which changes them on the event loop only.

- ``seed(states, now)``: ``last_ok`` from the newest stored row per feed.
- ``due(states, now)``: the cadences of ``live.feeds``, plus at start (never tried) the feeds
  whose data is missing now.
- ``fetch(feed, now, first)``: the fetcher(s) of one feed. The live poll labels its row by its
  own wall time (``poll_live`` with ``now=None``); the irradiance forecast is stamped with the
  wall time of its call, the settlement rows with the wall time after the response.
- ``pull_settlement(day)`` keeps the rows it appended until the coordinator takes them
  (``take_pulled``) to book them under the minute lock.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections.abc import Callable

import numpy as np

from ..const import FEEDS
from ..core.feeds import ida as IDA
from ..core.feeds import tennet_live as TL
from ..core.store import Store
from ..core.timegrid import ISP_S, ZONE
from . import feeds as FD
from . import fetch as FX
from .feeds import FeedState

_LOGGER = logging.getLogger(__name__)

MINUTE_S = 60
DAY_S = 86400
IRRADIANCE_FIRST_S = 3 * 3600  # a stored forecast older than this is fetched at start
DA_GATE_HOUR = 13
SETTLEMENT_HOUR = 9
SOURCES = {
    "day_ahead": "energy-charts",
    "ida": "nordpool",
    "settlement": FX.SETTLEMENT_SOURCE,
    "irradiance": "open-meteo",
    "pack": "",
}


def _local(now: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(now), ZONE)


def _wall() -> int:
    return int(time.time())


def _each(fetch, calls: list[tuple]) -> list[str]:
    """Run ``fetch(*args)`` for every call; the messages of the calls that raised FeedError."""
    errors = []
    for args in calls:
        try:
            fetch(*args)
        except FX.FeedError as err:
            errors.append(str(err))
    return errors


def new_states() -> dict[str, FeedState]:
    return {name: FeedState(name, source=SOURCES.get(name, "")) for name in FEEDS}


class FeedRunner:
    def __init__(self, store: Store, lat: float, lon: float, key: Callable[[], str | None]):
        self.store, self.lat, self.lon, self.key = store, lat, lon, key
        self._pulled: list[dict] = []  # appended settlements not yet taken for booking
        self._pulled_lock = threading.Lock()  # the feeds and the nightly job both pull

    def day_complete(self, day: dt.date) -> bool:
        lo, hi = FX._day_bounds(day)
        f = self.store.features(lo, hi)
        return int(np.isfinite(f["da_eur_mwh"]).sum()) >= (hi - lo) // ISP_S

    def settled(self, day: dt.date) -> bool:
        lo, hi = FX._day_bounds(day)
        quarters = {r["quarter_ts"] for r in self.store.settlements(lo, hi)}
        return len(quarters) >= (hi - lo) // ISP_S

    def seed(self, states: dict[str, FeedState], now: int) -> dict[str, int]:
        """``last_ok`` per feed from the newest stored row (the coordinator applies it)."""
        store, out = self.store, {}
        lm = store.live_minutes(now - 2 * DAY_S, now + MINUTE_S)
        if len(lm["ts"]):
            out["live"] = int(lm["ts"][-1]) + MINUTE_S  # known at the minute's end
        got = store.fetch_at(now, FX.FORECAST_SOURCE)
        if got is not None:
            out["irradiance"] = int(got[0])
        f = store.features(now - 2 * DAY_S, now + 2 * DAY_S)
        ida = [f"{a.lower()}_eur_mwh" for a in IDA.AUCTIONS]
        for name, cols in (("day_ahead", ["da_eur_mwh"]), ("ida", ida)):
            ok = np.zeros(len(f["ts"]), dtype=bool)
            for c in cols:
                ok |= np.isfinite(np.asarray(f[c], dtype=float))
            if ok.any():
                out[name] = min(int(f["ts"][ok][-1]), now)
        rows = store.settlements(now - 3 * DAY_S, now)
        if rows:
            out["settlement"] = max(int(r["published_at"]) for r in rows)
        return out

    def _first_due(self, feed: str, now: int, st: FeedState) -> bool:
        """At start (never tried): the feeds whose data is missing now are due at once."""
        local = _local(now)
        today = local.date()
        if feed == "day_ahead":
            return not self.day_complete(today) or (
                local.hour >= DA_GATE_HOUR and not self.day_complete(today + dt.timedelta(days=1))
            )
        if feed == "ida":
            return FD.problem("ida", now, st, False, True, True)
        if feed == "settlement":
            return local.hour >= SETTLEMENT_HOUR and not self.settled(today - dt.timedelta(days=1))
        if feed == "irradiance":
            return st.last_ok is None or now - st.last_ok > IRRADIANCE_FIRST_S
        return False

    def due(self, states: dict[str, FeedState], now: int) -> list[str]:
        have = self.day_complete(_local(now).date() + dt.timedelta(days=1))
        out = []
        for feed in FEEDS:
            if feed == "pack":
                continue  # the pack updater (Task 20) runs its own fetch
            st = states[feed]
            if not st.enabled:
                continue  # off by choice: never fetched
            if FD.due(feed, now, st, have) or (
                st.last_try is None and self._first_due(feed, now, st)
            ):
                out.append(feed)
        return out

    def pull_settlement(self, day: dt.date) -> list[dict]:
        """The settled prices of one local day appended to the store, kept for ``take_pulled``.

        ``published_at`` is the wall time after the response (``fetch_settlement`` with
        ``now=None``), so a booking minute at or after it knew the rows.
        """
        rows = FX.fetch_settlement(self.store, day, None)
        with self._pulled_lock:
            self._pulled.extend(rows)
        return rows

    def take_pulled(self) -> list[dict]:
        """The settlement rows pulled since the last call, in pull order."""
        with self._pulled_lock:
            rows, self._pulled = self._pulled, []
        return rows

    def fetch(self, feed: str, now: int, first: bool) -> str | None:
        """Run the fetcher(s) of ``feed`` (blocking); the source label, raises on failure."""
        store, local = self.store, _local(now)
        today = local.date()
        if feed == "day_ahead":
            days = [] if self.day_complete(today) else [today]
            if local.hour >= DA_GATE_HOUR:
                days.append(today + dt.timedelta(days=1))
            errors = _each(FX.fetch_day_ahead, [(store, day) for day in days])
            if errors:  # each day is tried; any incomplete day is a failure
                raise FX.FeedError("; ".join(errors))
            return None
        if feed == "ida":
            pairs = FD.due_auctions(now)
            if first:
                pairs = [
                    (a, d) for d in (today, today + dt.timedelta(days=1)) for a in IDA.AUCTIONS
                    if IDA.clearing_time(a, d) + FD.IDA_PUBLISH_DELAY_S <= now
                ]
            errors = _each(FX.fetch_ida, [(store, a, d) for a, d in pairs])
            if errors and len(errors) == len(pairs):  # one auction result is a success
                raise FX.FeedError("; ".join(errors))
            return None
        if feed == "settlement":
            self.pull_settlement(today - dt.timedelta(days=1))
            return None
        if feed == "live":
            key = self.key()
            FX.poll_live(store, None, key)  # labelled by the poll's own wall time
            return TL.SOURCE if key else "tenergy"
        FX.fetch_irradiance(store, self.lat, self.lon, _wall())
        return None

    def run(self, feeds: list[str], now: int, first: set[str]) -> dict:
        """Every feed of ``feeds``: ``{feed: (ok, source, error)}``; a failure never raises."""
        out = {}
        for feed in feeds:
            try:
                out[feed] = (True, self.fetch(feed, now, feed in first), None)
            except Exception as err:  # a feed failure never stops the loop
                if not isinstance(err, FX.FeedError):
                    _LOGGER.warning("feed %s failed: %r", feed, err)
                out[feed] = (False, None, str(err) or repr(err))
        return out
