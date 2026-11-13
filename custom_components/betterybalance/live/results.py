"""The shadow's result from the newest bookings: totals and the cumulative sum per hour.

The result of a quarter is ``core.ledger.quarter_result(house, shadow)``, design 10.4:
``money(house) - money(shadow)``, positive is earned. The ``actual`` path is never part of
it. A quarter counts when both its house and shadow bookings exist; it is missing when its
result is NaN (a missing quarter or NaN money). Settled means both newest bookings carry
``price_source = "settled"``, so a TenneT correction (a newer settled rebooking) replaces the
quarter's result and is never counted twice.

- ``result_totals``: settled and provisional totals, and today's result (the Amsterdam day of
  ``now``). ``TotalsCache`` gives the same totals on every call and rereads only the Amsterdam
  days whose bookings changed since the last call (the minute loop calls it every minute).
- ``hourly_sums``: per UTC hour start, the cumulative settled result from the first booking to
  the hour's end, the ``sum`` of the external result statistic. Amsterdam is a whole-hour
  offset from UTC, so the hours tile every local day, the 23- and 25-hour days included.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

from ..core.ledger import quarter_result
from ..core.store import Store
from ..core.timegrid import ZONE

HOUR_S = 3600
_END = 2**62  # past every quarter


@dataclass
class Totals:
    settled_eur: float
    provisional_eur: float
    today_eur: float
    today_settled_share: float
    today_missing: int


@dataclass(frozen=True)
class _Day:
    settled_eur: float = 0.0
    provisional_eur: float = 0.0
    n: int = 0  # quarters with a finite result
    n_settled: int = 0
    n_missing: int = 0


def _local_day(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date()


def _day_bounds(day: dt.date) -> tuple[int, int]:
    lo = dt.datetime(day.year, day.month, day.day, tzinfo=ZONE)
    nxt = day + dt.timedelta(days=1)
    hi = dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=ZONE)
    return int(lo.timestamp()), int(hi.timestamp())


def _quarters(newest: dict) -> dict[int, tuple[float, bool]]:
    """``{quarter_ts: (result, settled)}`` of the quarters with a house and a shadow row."""
    out = {}
    for q in sorted({q for q, _ in newest}):
        house, shadow = newest.get((q, "house")), newest.get((q, "shadow"))
        if house is None or shadow is None:
            continue
        settled = house["price_source"] == "settled" and shadow["price_source"] == "settled"
        out[q] = (quarter_result(house, shadow), settled)
    return out


def _days(newest: dict) -> dict[dt.date, _Day]:
    acc: dict[dt.date, list] = {}
    for q, (r, settled) in _quarters(newest).items():
        a = acc.setdefault(_local_day(q), [0.0, 0.0, 0, 0, 0])
        if math.isnan(r):
            a[4] += 1
            continue
        a[1] += r
        a[2] += 1
        if settled:
            a[0] += r
            a[3] += 1
    return {day: _Day(*a) for day, a in acc.items()}


class TotalsCache:
    """``result_totals`` per call, rereading only the days with a booking after the last call.

    It follows the chain head: bookings are append-only and chained, so the bookings since
    the last call are the chain rows after its head. A head behind the cached one (another
    store) starts over.
    """

    def __init__(self) -> None:
        self._seq: int | None = None
        self._days: dict[dt.date, _Day] = {}

    def totals(self, store: Store, now: int) -> Totals:
        head = store.head()
        seq = 0 if head is None else int(head[0])
        if self._seq is None or seq < self._seq:
            self._days = _days(store.newest_bookings(0, _END))
        elif seq > self._seq:
            changed = {
                _local_day(int(row["quarter_ts"]))
                for kind, row in store.chain(self._seq + 1, seq)
                if kind == "booking"
            }
            for day in changed:
                lo, hi = _day_bounds(day)
                self._days.update(_days(store.newest_bookings(lo, hi)))
        self._seq = seq
        days = [self._days[d] for d in sorted(self._days)]
        today = self._days.get(_local_day(now), _Day())
        return Totals(
            settled_eur=float(sum(d.settled_eur for d in days)),
            provisional_eur=float(sum(d.provisional_eur for d in days)),
            today_eur=float(today.provisional_eur),
            today_settled_share=today.n_settled / today.n if today.n else 0.0,
            today_missing=int(today.n_missing),
        )


def result_totals(store: Store, now: int) -> Totals:
    """The totals from the newest bookings (module docstring).

    ``settled_eur`` sums the settled quarters, ``provisional_eur`` every quarter that is not
    missing (settled or not). Today, the Amsterdam day of ``now``: ``today_eur`` its result
    as booked, ``today_settled_share`` its settled share of the quarters that are not
    missing (0 without one), ``today_missing`` its missing quarters.
    """
    return TotalsCache().totals(store, now)


def hourly_sums(store: Store, from_ts: int, to_ts: int) -> list[tuple[int, float]]:
    """``(hour start, cumulative settled result to the hour's end)`` per UTC hour in
    ``[from_ts, to_ts)`` (``from_ts`` floored to its hour), summed from the first booking."""
    lo = int(from_ts) - int(from_ts) % HOUR_S
    per_hour: dict[int, float] = {}
    base = 0.0
    for q, (r, settled) in _quarters(store.newest_bookings(0, int(to_ts))).items():
        if not settled or math.isnan(r):
            continue
        if q < lo:
            base += r
        else:
            h = q - q % HOUR_S
            per_hour[h] = per_hour.get(h, 0.0) + r
    out, total = [], base
    for h in range(lo, int(to_ts), HOUR_S):
        total += per_hour.get(h, 0.0)
        out.append((h, float(total)))
    return out


def last_settled_quarter(store: Store) -> int | None:
    """The newest quarter whose house and shadow bookings are both settled; None without one."""
    settled = [q for q, (_, s) in _quarters(store.newest_bookings(0, _END)).items() if s]
    return settled[-1] if settled else None
