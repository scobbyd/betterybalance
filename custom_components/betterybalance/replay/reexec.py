"""Re-execution of a live day from the inputs the live box stored (design 9).

``reexecute`` runs ``live.site.LiveSite`` again over a live store and writes the chain into a
second store. The site code is the live one; only its inputs change:

- The window of each quarter is ``live.site.stored_inputs`` of the quarter's ``windows`` row
  (the stored bands as of the row's ``now``, the stored ``healthy`` tiers, ``da_known`` mask,
  load, PV, EV, plan mask and widening), ranked through ``LiveSite._rank`` as live. A row
  without arrays is a failed setup: the quarter runs without a ranking, as live.
- The nowcast of each minute is the run's stored ``nowcasts`` row (every field NaN is no
  nowcast).
- The flows of each minute are the stored ``flows_minute`` row, which the live minute also
  decides on (no row is a missing minute); bookings read the stored live minutes; settled
  rebookings the stored settlements.
- The ``events`` log drives the rest at its position: ``restart`` (a fresh ``LiveSite`` on
  the event's ``model_version``, the version the live site restored on, and its ``knobs``,
  the knobs of the run, and ``restore``),
  ``refit`` (the ``learning`` and ``drift`` flags), ``settle`` (the settled
  rebooking of the quarters in ``detail["quarters"]`` at the settlement row each one names in
  ``detail["published_at"]``, ``live.site.settlement_row``, so a row stored later never
  reaches it; an event without the list reads the newest settlement published by its ``ts``;
  through ``LiveSite._rebook_settled``) and ``model_switch`` (the site's model and the decisions'
  ``model_version`` become ``detail["version"]`` from ``detail["effective_ts"]``, else from
  the quarter boundary after the event; the model comes from ``models``, else ``lc.model``).
  Other kinds change no decision.

Runs: every ``restart`` event starts a run whose epoch is the event's seq. The live site
keys every write a re-run can repeat by that epoch (nowcasts, bands, windows, the writer of
a minute's flows), so each run is re-executed from its own rows: a run's minutes are its
nowcast rows (every ``on_minute`` writes one), its windows are its ``windows`` rows, its
bands are those the run saw (``Store.band`` as of a serve time and an epoch). The next
``restart`` event's ``last_minute`` names the run's last minute; a mismatch is a difference.

Wall-clock order inside a run: the live clock calls ``on_minute(ts)`` and the minute-1 setup
at ``ts``, so an event or a window built at ``t`` applies before every minute ``ts >= t``; at
one instant events come before the setup.

The walk starts at a ``restart`` event, the only place the live site re-anchors its state:
the newest at or before the span's start, else the first. The output chain continues from
the source chain at that event's ``head_seq``: rows up to it are read from the source, rows
after it are written into ``out``. ``compare`` covers exactly the span.
A restart inside a quarter re-decides the minutes from the quarter's start (R30); both runs
re-execute from their own rows.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.ledger import BOOKING_FIELDS, DECISION_FIELDS
from ..core.pricemodel import PriceModel
from ..core.rule.window import Nowcast
from ..core.site.contract import with_knobs
from ..core.store import NOWCAST_FIELDS, Store
from ..core.timegrid import ISP_S, isp_start
from ..live.site import (
    LIVE_STALE_S,
    LiveConfig,
    LiveSite,
    daily_result,
    settlement_at,
    settlement_row,
    stored_inputs,
)
from .engine import ReplayResult

MINUTE_S = 60
_CHAIN_FIELDS = ("seq", "prev_hash", "hash")
_TABLES = (("decisions", DECISION_FIELDS), ("bookings", BOOKING_FIELDS))


@dataclass
class ReexecResult:
    n_decisions: int
    n_bookings: int
    differences: list[str]


# -- the store the re-executed site sees ------------------------------------------------


class _SplitStore:
    """The live store's inputs and chain prefix, with the chain continued in ``out``.

    Reads of inputs go to ``src``; chain rows with ``seq <= prefix`` come from ``src``, later
    ones from ``out``; chain writes and events go to ``out``; input writes are dropped (the
    inputs are the stored ones). ``put_event`` writes nothing and returns the seq of the
    source ``restart`` event being replayed (``event``), so the site runs in the live epoch
    (``append_chained_with_event`` chains its rows into ``out`` and answers the same);
    ``last_event`` answers as of that event. The registry reads as empty: re-execution takes
    the site state from the ``refit`` events. Any other store call raises ``AttributeError``.
    """

    def __init__(self, src: Store, out: Store, prefix: int, events: list[dict]):
        self.src, self.out, self.prefix = src, out, int(prefix)
        self.events = events
        self.event: dict | None = None

    # inputs, from the live store
    def minutes(self, lo, hi, max_epoch=None):
        return self.src.minutes(lo, hi, max_epoch=max_epoch)

    def minute_count(self, lo, hi):  # LiveSite._minute_quarters checks its cache with it
        return self.src.minute_count(lo, hi)

    def hours(self, lo, hi):
        return self.src.hours(lo, hi)

    def live_minutes(self, lo, hi):
        return self.src.live_minutes(lo, hi)

    def band(self, *args, **kwargs):
        return self.src.band(*args, **kwargs)

    def window(self, qts, epoch=None):
        return self.src.window(qts, epoch=epoch)

    def last_nowcast_ts(self, epoch):
        return self.src.last_nowcast_ts(epoch)

    def last_event(self, kind):
        seq = self.event["seq"]
        found = [e for e in self.events if e["kind"] == kind and e["seq"] < seq]
        return found[-1] if found else None

    def registry(self, key):
        return None

    # input writes, dropped
    def put_minute(self, row):
        pass

    def put_nowcast(self, row):
        pass

    def put_window(self, row):
        pass

    def put_greed(self, row):
        pass

    def upsert_features(self, ts, **cols):
        pass

    # events and the chain
    def put_event(self, ts, kind, detail):
        return int(self.event["seq"])

    def append_chained(self, kind, row):
        return self.out.append_chained(kind, row)

    def append_chained_many(self, items):
        return self.out.append_chained_many(items)

    def append_chained_with_event(self, items, ts, kind, detail):
        return self.out.append_chained_many(items), self.put_event(ts, kind, detail)

    def head(self):
        return self.out.head()

    def chain(self, lo_seq, hi_seq):
        p = self.prefix
        old = self.src.chain(lo_seq, min(hi_seq, p)) if lo_seq <= p else []
        return old + self.out.chain(max(lo_seq, p + 1), hi_seq)

    def bookings(self, lo, hi):
        old = [r for r in self.src.bookings(lo, hi) if r["seq"] <= self.prefix]
        return old + self.out.bookings(lo, hi)

    def decisions(self, lo, hi):
        old = [r for r in self.src.decisions(lo, hi) if r["seq"] <= self.prefix]
        return old + self.out.decisions(lo, hi)


class _StoredSite(LiveSite):
    """``LiveSite`` with the stored nowcast of each minute instead of the live re-read."""

    def __init__(self, store: _SplitStore, lc: LiveConfig, nowcasts: dict[tuple, dict]):
        super().__init__(store, lc)
        self._stored_nc = nowcasts  # by (ts, epoch)

    def _nowcast_at(self, ts: int, qts: int, e: int) -> tuple[Nowcast | None, int | None]:
        row = self._stored_nc[(ts, self.epoch)]
        values = [float(row[f]) for f in NOWCAST_FIELDS]
        nc = None if all(math.isnan(v) for v in values) else Nowcast(*values)
        age = row["live_age_s"]
        self._live_stale = age is None or age > LIVE_STALE_S
        return nc, age


# -- the walk ---------------------------------------------------------------------------


def _seed(out: Store, src: Store, install_id: str, head_seq: int) -> None:
    """``out`` continues the source chain after ``head_seq`` (genesis for 0)."""
    out.set_meta("install_id", install_id)
    if head_seq > 0:
        (_, row), = src.chain(head_seq, head_seq)
        out.set_meta("chain_seq", head_seq)
        out.set_meta("chain_head", row["hash"])


class _Walk:
    def __init__(self, src: Store, lc: LiveConfig, models: dict[str, PriceModel]):
        self.src, self.lc, self.models = src, lc, models
        self.learning: bool | None = None  # None: the initial state's
        self.drift: bool | None = None
        self.version = lc.model_version
        self.switches: list[tuple[int, str]] = []  # (effective ts, version), pending
        self.site: _StoredSite | None = None
        self.differences: list[str] = []

    # events
    def apply(self, ev: dict) -> None:
        kind, detail, site = ev["kind"], ev["detail"], self.site
        if kind == "refit":
            self.learning, self.drift = bool(detail["learning"]), bool(detail["drift"])
            if site is not None:
                site.state = replace(site.state, learning=self.learning, drift=self.drift)
        elif kind == "model_switch":
            at = detail.get("effective_ts")
            at = int(isp_start(int(ev["ts"]))) + ISP_S if at is None else int(at)
            self.switches.append((at, str(detail["version"])))
        elif kind == "settle" and site is not None:
            quarters = detail.get("quarters", [])
            published = detail.get("published_at")
            for k, q in enumerate(quarters):
                if published is None:  # an event written before it named its rows
                    row = settlement_at(self.src, int(q), int(ev["ts"]))
                else:
                    row = settlement_row(self.src, int(q), int(published[k]))
                if row is None or not (math.isfinite(row["take"]) and math.isfinite(row["feed"])):
                    self.differences.append(
                        f"settle event seq {ev['seq']}: no stored settlement of quarter {int(q)}"
                    )
                    continue
                state = 0 if row["regulation_state"] is None else row["regulation_state"]
                site._rebook_settled(int(q), row["take"], row["feed"], state)

    def _lc(self, lc: LiveConfig) -> LiveConfig:
        """``lc`` on the model version in effect (its model from ``models``)."""
        model = self.models.get(self.version, self.lc.model)
        return replace(lc, model=model, model_version=self.version)

    def _switch_until(self, ts: int) -> None:
        due = [s for s in self.switches if s[0] <= ts]
        if due:
            self.version = due[-1][1]
            self.switches = [s for s in self.switches if s[0] > ts]
            if self.site is not None:
                self.site.lc = self._lc(self.site.lc)

    def restart(self, store: _SplitStore, ev: dict, nowcasts: dict[tuple, dict]) -> None:
        now = int(ev["ts"])
        self._switch_until(now)
        version = ev["detail"].get("model_version")
        if version is not None:  # the version the live site restored on
            self.version = str(version)
        knobs = ev["detail"].get("knobs")
        if knobs is not None:  # the run's own knobs (an options change reloads at a restart)
            self.lc = replace(self.lc, cfg=with_knobs(self.lc.cfg, knobs))
        self.site = _StoredSite(store, self._lc(self.lc), nowcasts)
        store.event = ev
        self.site.restore(now)
        st = self.site.state
        self.site.state = replace(
            st,
            learning=st.learning if self.learning is None else self.learning,
            drift=st.drift if self.drift is None else self.drift,
        )

    def setup(self, w: dict) -> None:
        site, qts = self.site, int(w["ts"])
        model = self.models.get(w["model_version"], self.lc.model)
        qi = None
        if w["load_kw"] is not None:
            qi = stored_inputs(self.src, w, model.tables, self.lc.cfg.knobs)
            if qi is None:
                self.differences.append(
                    f"windows {qts}: the stored bands do not rebuild the window"
                )
            soc = float(site.stepper.battery.soc_kwh)
            if soc != w["soc_kwh"]:
                self.differences.append(f"windows {qts}: soc_kwh {soc!r} != {w['soc_kwh']!r}")
        site._rank(qi, qts)

    def minutes(self, events: list[dict], end: int) -> tuple[int | None, bool]:
        """The minutes of the run from ``site._next`` while it has a nowcast row, before
        ``end``; events at their position. Returns the last minute (None without one) and
        whether ``end`` stopped the run."""
        site, src, nowcasts = self.site, self.src, self.site._stored_nc
        epoch = site.epoch
        first = ts = site._next
        set_q, j = None, 0
        while ts < end and (ts, epoch) in nowcasts:
            while j < len(events) and int(events[j]["ts"]) <= ts:
                self.apply(events[j])
                j += 1
            self._switch_until(ts)
            qts = int(isp_start(ts))
            if set_q != qts:
                w = src.window(qts, epoch=epoch)
                if w is not None and int(w["epoch"]) == epoch and int(w["now"]) <= ts:
                    self.setup(w)
                    set_q = qts
            site.on_minute(ts, None)  # the minute decides on its stored flows row, as live
            ts += MINUTE_S
        for ev in events[j:]:
            self.apply(ev)
        return (None if ts == first else ts - MINUTE_S), ts >= end


def _version_at(events: list[dict], k: int, default: str) -> str:
    """The model version in effect at the ``restart`` event ``events[k]``: its
    ``model_version``; for an event written before it carried one, the ``from_version`` of
    the first ``model_switch`` after it; else ``default``."""
    version = events[k]["detail"].get("model_version")
    if version is not None:
        return str(version)
    for ev in events[k + 1:]:
        if ev["kind"] == "model_switch" and ev["detail"].get("from_version") is not None:
            return str(ev["detail"]["from_version"])
    return default


def reexecute(
    src: Store,
    lc: LiveConfig,
    start: int,
    end: int,
    out: Store,
    models: dict[str, PriceModel] | None = None,
) -> ReexecResult:
    """Re-execute the live site over ``[start, end)`` from ``src`` into ``out`` (module rules).

    Only a ``restart`` re-anchors the live site, so the walk starts at the newest ``restart``
    event at or before ``start``, else at the first one (the install: the first restore
    holds the first minute and SoC the site began with), and runs forward to ``end``.
    ``out`` must hold no chain. ``lc`` gives the site, the install id (the genesis of a chain
    from seq 1), the model whose dual tables rebuild the windows (``models`` maps other model
    versions to theirs) and the model version when the starting ``restart`` event names none
    (``_version_at``; every restart names the version it restored on). ``differences`` lists the
    walk's own findings (a window that does not rebuild, a SoC that differs at a setup) and
    then ``compare(src, out, start, end)``; the counts are the re-executed rows in
    ``[start, end)`` (decisions by ``ts``, bookings by ``quarter_ts``).
    """
    start, end = int(start), int(end)
    events = src.events()
    restarts = [i for i, e in enumerate(events) if e["kind"] == "restart"]
    if not restarts:
        return ReexecResult(0, 0, ["the store holds no restart event: nothing ran live"])
    before = [i for i in restarts if int(events[i]["ts"]) <= start]
    k0 = before[-1] if before else restarts[0]
    if out.head() is not None:
        raise ValueError("out already holds a chain")
    if "head_seq" not in events[k0]["detail"]:
        return ReexecResult(0, 0, [f"restart event seq {events[k0]['seq']} has no head_seq"])
    prefix = int(events[k0]["detail"]["head_seq"])
    _seed(out, src, lc.install_id, prefix)
    store = _SplitStore(src, out, prefix, events)
    walk = _Walk(src, lc, models or {})
    walk.version = _version_at(events, k0, lc.model_version)
    for ev in events[:k0]:
        if ev["kind"] in ("refit", "model_switch"):
            walk.apply(ev)
    t0 = int(events[k0]["ts"])
    nowcasts = src.nowcasts(int(isp_start(t0)), end)
    runs = [i for i in range(k0, len(events)) if events[i]["kind"] == "restart"]
    runs = [i for i in runs if int(events[i]["ts"]) < end]
    for n, i in enumerate(runs):
        nxt = runs[n + 1] if n + 1 < len(runs) else None
        within = [
            e for e in events[i + 1 : nxt if nxt is not None else len(events)]
            if int(e["ts"]) < end
        ]
        walk.restart(store, events[i], nowcasts)
        last, stopped = walk.minutes(within, end)
        if nxt is not None and not stopped and "last_minute" in events[nxt]["detail"]:
            want = events[nxt]["detail"]["last_minute"]
            if last != want:
                walk.differences.append(
                    f"restart event seq {events[nxt]['seq']}: the run before it ends at "
                    f"{last}, the event names {want}"
                )
    differences = walk.differences + compare(src, out, start, end)
    n_dec, n_book = len(out.decisions(start, end)), len(out.bookings(start, end))
    return ReexecResult(n_dec, n_book, differences)


# -- comparison -------------------------------------------------------------------------


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare(a: Store, b: Store, start: int, end: int) -> list[str]:
    """The first difference per table, ``decisions`` (``ts`` in ``[start, end)``) then
    ``bookings`` (``quarter_ts`` in ``[start, end)``); empty when both are identical.

    Rows pair in seq order. A difference reads ``"<table> seq <n>: <field> <a> != <b>"``
    (the ledger fields first, then ``seq``, ``prev_hash``, ``hash``; NaN equals NaN), or
    ``"<table> seq <n>: only in a"`` (``b``) for a row without a partner.
    """
    out = []
    for table, fields in _TABLES:
        ra, rb = getattr(a, table)(start, end), getattr(b, table)(start, end)
        for x, y in zip(ra, rb, strict=False):
            bad = next(
                (f for f in (*fields, *_CHAIN_FIELDS) if not _same(x[f], y[f])), None
            )
            if bad is not None:
                out.append(f"{table} seq {x['seq']}: {bad} {x[bad]!r} != {y[bad]!r}")
                break
        else:
            if len(ra) != len(rb):
                extra, side = (ra, "a") if len(ra) > len(rb) else (rb, "b")
                out.append(f"{table} seq {extra[min(len(ra), len(rb))]['seq']}: only in {side}")
    return out


# -- the walk-forward replay against the live day ---------------------------------------


def daily_result_mad(live: Store, result: ReplayResult) -> float:
    """Mean absolute difference per Amsterdam day, EUR, between the live shadow result
    (``live.site.daily_result``, newest bookings) and a walk-forward replay's
    ``result_pairs``, over the replay days with a live booking; NaN without such a day."""
    pairs = result.result_pairs
    diffs = []
    for day, r in zip(pairs["day"], np.asarray(pairs["result_eur"], dtype=float), strict=True):
        got = daily_result(live, dt.date.fromisoformat(str(day)))
        if got["n_quarters"]:
            diffs.append(abs(got["result_eur"] - float(r)))
    return float(np.mean(diffs)) if diffs else math.nan
