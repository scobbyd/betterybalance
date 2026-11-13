"""Feed cadences and problem rules (design section 7). Pure; every gate in Europe/Amsterdam.

The coordinator keeps one ``FeedState`` per feed of ``const.FEEDS``, asks ``due`` each minute,
runs the fetcher of ``live.fetch`` in the executor and updates the state. ``problem`` drives
the feed binary sensors.

A slot is a local wall-clock minute S. It is due while ``now`` lies in the catch-up window
``[S, S + 45 min)`` and the last try is before S, so a tick that misses the slot minute (a
restart, a long job) still fetches. With several slots (day-ahead) the latest slot at or
before ``now`` counts. IDA slots are ``clearing_time(auction, day) + 20 min``; after a failed
try in the window, IDA retries every 5 minutes until the gate plus 60 minutes. The live feed
has no slots: it is due when the backoff since the last try has passed. Retry intervals
compare minute indices (``ts // 60``), so a ``last_try`` stored after the fetch, a few
seconds into the minute, does not push the next try one minute late.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from ..const import FEEDS
from ..core.feeds.ida import AUCTIONS, clearing_time
from ..core.timegrid import ZONE

MINUTE_S = 60
CATCH_UP_S = 45 * 60
LIVE_PERIOD_S = 60
LIVE_BACKOFF_MAX_S = 600
IDA_PUBLISH_DELAY_S = 1200
IDA_RETRY_S = 300
IDA_GIVE_UP_S = 3600  # after the auction gate, not after the slot
DAY_AHEAD_SLOTS = ((13, 5), (13, 35), (14, 5))
SETTLEMENT_SLOTS = ((9, 0), (15, 0))
IRRADIANCE_SLOTS = tuple(sorted({(h, 7) for h in range(0, 24, 3)} | {(5, 7)}))
PACK_SLOTS = ((4, 10),)
DAY_AHEAD_LATE = dt.time(14, 30)
SETTLEMENT_LATE = dt.time(16, 0)
LIVE_STALE_S = 180
IDA_STALE_S = 26 * 3600
IRRADIANCE_STALE_S = 86400
PACK_MAX_FAILURES = 3


@dataclass
class FeedState:
    name: str
    last_ok: int | None = None
    last_try: int | None = None
    last_error: str | None = None
    failures: int = 0
    source: str = ""
    enabled: bool = True  # False: off by choice (the IDA feed without the fetch_ida option)


def _check(feed: str) -> None:
    if feed not in FEEDS:
        raise ValueError(f"unknown feed: {feed}")


def _local(now: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(now), ZONE)


def _minutes_since(now: int, then: int) -> int:
    """Whole minutes between the minutes of ``then`` and ``now``."""
    return int(now) // MINUTE_S - int(then) // MINUTE_S


def _tried_before(st: FeedState, slot_start: int) -> bool:
    return st.last_try is None or st.last_try < slot_start


def _latest_slot(now: int, slots) -> int | None:
    """Epoch s of the latest local slot at or before ``now``, today or yesterday."""
    today = _local(now).date()
    starts = [
        int(dt.datetime(d.year, d.month, d.day, h, m, tzinfo=ZONE).timestamp())
        for d in (today - dt.timedelta(days=1), today)
        for h, m in slots
    ]
    past = [s for s in starts if s <= int(now)]
    return max(past) if past else None


def _slot_due(now: int, st: FeedState, slots) -> bool:
    slot = _latest_slot(now, slots)
    return slot is not None and int(now) < slot + CATCH_UP_S and _tried_before(st, slot)


def _ida_slots(now: int) -> list[tuple[str, dt.date, int]]:
    """(auction, delivery day, slot) with ``slot = gate + 20 min`` and ``now`` in the slot's
    catch-up window. IDA1 and IDA2 price tomorrow, IDA3 today."""
    today = _local(now).date()
    out = []
    for day in (today, today + dt.timedelta(days=1)):
        for auction in AUCTIONS:
            slot = clearing_time(auction, day) + IDA_PUBLISH_DELAY_S
            if slot <= int(now) < slot + CATCH_UP_S:
                out.append((auction, day, slot))
    return out


def due_auctions(now: int) -> list[tuple[str, dt.date]]:
    """The (auction, delivery day) pairs a due ``ida`` slot at ``now`` means: ``now`` lies in
    ``[gate + 20 min, gate + 65 min)``. The gates are hours apart, so at most one pair."""
    return [(auction, day) for auction, day, _ in _ida_slots(now)]


def next_backoff(st: FeedState) -> int:
    """Seconds between tries: 60 while ``failures == 0``, then ``min(60 * 2 ** failures, 600)``."""
    if st.failures <= 0:
        return LIVE_PERIOD_S
    return min(LIVE_PERIOD_S * 2 ** st.failures, LIVE_BACKOFF_MAX_S)


def _ida_due(now: int, st: FeedState) -> bool:
    for _, _, slot in _ida_slots(now):
        if _tried_before(st, slot):
            return True
        gate = slot - IDA_PUBLISH_DELAY_S
        if (
            st.failures > 0
            and int(now) < gate + IDA_GIVE_UP_S
            and _minutes_since(now, st.last_try) >= IDA_RETRY_S // MINUTE_S
        ):
            return True
    return False


def due(feed: str, now: int, st: FeedState, have_tomorrow_da: bool = False) -> bool:
    """True when the coordinator runs the fetcher of ``feed`` at ``now`` (module docstring)."""
    _check(feed)
    if feed == "live":
        return (
            st.last_try is None
            or _minutes_since(now, st.last_try) >= next_backoff(st) // MINUTE_S
        )
    if feed == "ida":
        return _ida_due(now, st)
    slots = {
        "day_ahead": DAY_AHEAD_SLOTS[:1] if have_tomorrow_da else DAY_AHEAD_SLOTS,
        "settlement": SETTLEMENT_SLOTS,
        "irradiance": IRRADIANCE_SLOTS,
        "pack": PACK_SLOTS,
    }[feed]
    return _slot_due(now, st, slots)


def _older(now: int, last_ok: int | None, limit_s: int) -> bool:
    """No success yet, or the last success more than ``limit_s`` before ``now``."""
    return last_ok is None or int(now) - last_ok > limit_s


def problem(
    feed: str,
    now: int,
    st: FeedState,
    have_tomorrow_da: bool,
    yesterday_settled: bool,
    pack_installed: bool,
) -> bool:
    """True when the feed's binary sensor reports a problem. A feed that never succeeded has
    no data, so the age rules count ``last_ok is None`` as a problem."""
    _check(feed)
    if not st.enabled:  # off by choice is not a problem
        return False
    t = _local(now).time()
    if feed == "day_ahead":
        return t >= DAY_AHEAD_LATE and not have_tomorrow_da
    if feed == "ida":
        return _older(now, st.last_ok, IDA_STALE_S)
    if feed == "settlement":
        return t >= SETTLEMENT_LATE and not yesterday_settled
    if feed == "live":
        return _older(now, st.last_ok, LIVE_STALE_S)
    if feed == "irradiance":
        return _older(now, st.last_ok, IRRADIANCE_STALE_S)
    return not pack_installed or st.failures >= PACK_MAX_FAILURES
