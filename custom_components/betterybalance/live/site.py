"""The live site: the plan-2 trader on a live minute clock, every input and decision in the store.

``LiveSite`` drives the replay's quarter and minute steps (``replay.steps``) from a wall clock:

- ``setup_quarter`` at minute 1: the live master, the window from ``ModelBandSource`` (each
  served band recorded in ``bands``), the site view from the store's quarters, the PV
  expectation, the ranking; the ``windows`` row; at 18:00 Amsterdam the ``greed`` row.
- ``on_minute`` every minute: the flows row, the nowcast (re-read at elapsed 5 and 10, None
  while the live feed is stale), ``Stepper.minute``, the ``nowcasts`` row and the chained
  decision row (repeat holds are not written). At elapsed 14 the quarter is booked on the
  three paths (shadow, house, actual) at the provisional live end price and chained in that
  order, and the quarter's live values go to its feature row.
- ``restore`` after a start: site state, SoC anchor, the gap booked missing, a re-ranking from
  the stored window. ``refit``: the weekly site-model update.
- ``settle`` at a minute the loop processes next: the settled rebooking of the quarters whose
  newest settlement (published by then) is not the price of their newest booking, and the
  ``settle`` event re-execution replays at that minute.
- ``set_model`` at a minute the loop processes next: the ``model_switch`` event; the price
  model, its version and the band source change at the next quarter boundary.

Runs: ``restore`` starts a run whose ``epoch`` is the seq of its ``restart`` event. Every
write a re-run can repeat carries it: nowcasts, bands, windows and the writer of a minute's
flows. A minute decides on its stored flows row (the first writer's), so a re-run after a
restart inside a quarter (R30) and its re-execution see the same minute.

Every write goes through the store before the method returns. In memory the site keeps only
the ``Stepper``, the current ``Ranked``, the ``ModelBandSource``, the previous decision,
the flags of the last setup and minute and a cache of the view's minute quarters (a copy of
stored rows, checked against the store on every read); a restart rebuilds them from the store.

Minute convention (plan 2): minute ``ts`` holds ``[ts, ts + 60)``. The decision of minute
``ts`` reads live minutes labelled before ``ts`` only, so the stored inputs are the same
whatever the wall time of the call.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.battery import load_profile
from ..core.feeds.live import minute_to_isp
from ..core.feeds.openmeteo import stale_fallback
from ..core.ledger import (
    PATHS,
    booking_row,
    decision_row,
    is_repeat_hold,
    quarter_result,
    resettle,
)
from ..core.pricemodel import QUANTILES, PriceModel
from ..core.rule.trader import Decision
from ..core.rule.window import Nowcast, band_for_slot, build_window, window_slots
from ..core.site.contract import Knobs, SiteConfig, knobs_dict
from ..core.site.flows import residual_kw, to_quarters
from ..core.site.load import local_fields
from ..core.site.registry import (
    SiteModelState,
    drift_alarm,
    initial_state,
    state_from_json,
    state_to_json,
    weekly_update,
)
from ..core.site.separate import predict_pv_kw
from ..core.site.solar import clear_sky_ghi, daylight, solar_position
from ..core.site.strings import scale_check, string_labels
from ..core.store import Store
from ..core.timegrid import ISP_S, Table, align, isp_start
from ..replay.disaggregate import hourly_to_quarters
from ..replay.engine import GREED_HOUR
from ..replay.phases import (
    DAY_S,
    EV_DAYS,
    IRR_COLS,
    Ranked,
    greed_row,
    local_midnight,
    rows_before,
    site_view,
)
from ..replay.steps import (
    FLOW_REQUIRED,
    NOWCAST_REREAD_MIN,
    QuarterInputs,
    Stepper,
    da_known_mask,
    ev_passthrough_kwh,
    minute_flows,
    quarter_inputs,
    rank_quarter,
    reread_nowcast,
)
from .bands import ModelBandSource, StoredBandSource
from .copy import FLAGS
from .master import LIVE_COLS as _LIVE_COLS
from .master import live_master
from .pack_update import cov_source
from .views import pv_now

_LOGGER = logging.getLogger(__name__)

MINUTE_S = 60
LIVE_STALE_S = 180  # the nowcast is None while the newest live minute is older than this
LIVE_LOOKBACK_S = 3600
# quarter q is booked in its minute q + 840 (at wall q + 900 or later) from the live rows
# labelled before q + 840, known by q + 840: a row stored after the booking never reaches it
PROVISIONAL_CUT_S = ISP_S - MINUTE_S
# the site's live master: every feature reads at most 40 quarters back within the master
MASTER_DAYS_BACK = 2
FORECAST_MAX_AGE_S = 86400  # an older irradiance forecast falls back to climatology
CLIMATOLOGY_DAYS = 3
DEFAULT_CLEARNESS = 0.5
DEFAULT_TEMP_C = 10.0
SITE_STATE_KEY = "site_state"
REFIT_DAYS = 90
SOC_LOOKBACK_S = 7 * DAY_S
_CHAIN_PAGE = 512
_REQUIRED = FLOW_REQUIRED
_NAN_BAND = {"take": {q: math.nan for q in QUANTILES}, "feed": {q: math.nan for q in QUANTILES}}


@dataclass
class LiveConfig:
    cfg: SiteConfig
    install_id: str
    model: PriceModel
    model_version: str
    pack_version: str | None = None
    skip_tiers: tuple[str, ...] = ()  # tiers the window never uses ("B" without the IDA feed)


@dataclass
class MinuteResult:
    decision: Decision
    written: bool  # False for a repeat hold (no decision row)
    booked: list[dict]  # the chained booking rows when this minute closed a quarter


def daily_result(store: Store, day: dt.date) -> dict:
    """The shadow's result of one Amsterdam day from the newest bookings per quarter and path.

    ``result_eur`` sums ``quarter_result(house, shadow)`` (positive is earned) over the
    quarters where both are priced and not missing; the others count in ``n_missing``.
    """
    lo, hi = local_midnight(day), local_midnight(day + dt.timedelta(days=1))
    new = store.newest_bookings(lo, hi)
    total, n_missing, n = 0.0, 0, 0
    for q in sorted({q for q, _ in new}):
        house, shadow = new.get((q, "house")), new.get((q, "shadow"))
        if house is None or shadow is None:
            continue
        n += 1
        r = quarter_result(house, shadow)
        if math.isnan(r):
            n_missing += 1
        else:
            total += r
    return {"result_eur": total, "n_missing": n_missing, "n_quarters": n}


def settlement_at(store, qts: int, now: int) -> dict | None:
    """The newest settlement of quarter ``qts`` published by ``now`` (None without one).

    ``settle`` books this row; its event names the row's ``published_at`` (``settlement_row``).
    """
    rows = [
        r for r in store.settlements(qts, qts + ISP_S, newest=False)
        if r["published_at"] is not None and int(r["published_at"]) <= now
    ]
    return rows[-1] if rows else None


def settlement_row(store, qts: int, published_at: int) -> dict | None:
    """The stored settlement of quarter ``qts`` published at ``published_at`` (None without)."""
    rows = store.settlements(qts, qts + ISP_S, newest=False)
    return next((r for r in rows if r["published_at"] == int(published_at)), None)


def _concat(tables: list[Table]) -> Table:
    """Row-wise union of tables; a column a table lacks is NaN (float) or False (bool)."""
    tables = [t for t in tables if len(t["ts"])]
    if not tables:
        return {"ts": np.zeros(0, dtype=np.int64), "missing": np.zeros(0, dtype=bool)}
    keys = list(dict.fromkeys(k for t in tables for k in t))
    out: Table = {}
    for k in keys:
        ref = next(np.asarray(t[k]) for t in tables if k in t)
        parts = []
        for t in tables:
            n = len(t["ts"])
            if k in t:
                parts.append(np.asarray(t[k]))
            elif ref.dtype == bool:
                parts.append(np.zeros(n, dtype=bool))
            else:
                parts.append(np.full(n, np.nan))
        out[k] = np.concatenate(parts)
    return out


def stored_inputs(store, w: dict, tables: dict, knobs: Knobs) -> QuarterInputs | None:
    """The ``QuarterInputs`` of a stored ``windows`` row, the window rebuilt from the bands.

    The window comes from ``StoredBandSource`` as of the row's ``now`` with the row's
    ``healthy`` tiers and ``da_known`` mask; load, PV, EV, plan mask and widening are the
    stored arrays. None for the row of a failed setup (no arrays) and when the stored bands
    lack slot 0 (``build_window`` raises ``ValueError``).
    """
    if w["load_kw"] is None or w["healthy"] is None or w["da_known"] is None:
        return None
    src = StoredBandSource(store, w["model_version"], w["healthy"], tables)
    src.set_now(w["now"], w["epoch"])
    try:
        window = build_window(w["now"], src, w["healthy"], w["da_known"], tables, knobs)
    except ValueError:
        return None
    return QuarterInputs(
        now=int(w["now"]), window=window, load_kw=w["load_kw"], pv_kw=w["pv_kw"],
        ev_kw=w["ev_kw"], plan_mask=w["plan_mask"], widening=w["widening"],
        healthy=tuple(w["healthy"]), da_known=w["da_known"],
    )


# -- the site ---------------------------------------------------------------------------


class LiveSite:
    def __init__(self, store: Store, lc: LiveConfig):
        self.store = store
        self.lc = lc
        cfg = lc.cfg
        self.loss = load_profile(cfg.loss_profile, cfg.pack)
        self.stepper = Stepper(cfg, self.loss, cfg.pack.soc_floor * cfg.pack.capacity_kwh)
        self.state: SiteModelState = initial_state(cfg)
        self.ranked: Ranked | None = None
        self.source: ModelBandSource | None = None
        self.prev: Decision | None = None
        self.ev_passthrough_kwh = 0.0  # of the last booked quarter (no ledger column)
        self._ranked_qts: int | None = None
        self._nc = None
        self._next: int | None = None
        self._live_stale = True
        self._climatology = False
        self._da_missing = False
        self.epoch = 0  # the seq of the restart event of this run (restore sets it)
        self._seen_epoch: int | None = None  # restore: the runs whose minutes it reads
        # shown by the entities only, never read by a decision or a booking
        self.last_trade: dict | None = None  # the newest committed charge or discharge
        self.pv: dict | None = None  # the running quarter's PV expectation (views.pv_now)
        self._pv_failed = False  # pv_now raised once already (logged once)
        self._pending: tuple[int, PriceModel, str] | None = None  # (effective ts, model, version)
        self._qcache: dict | None = None  # the view's minute quarters (_minute_quarters)
        self._reset_quarter()

    # -- helpers ------------------------------------------------------------------------

    def _reset_quarter(self) -> None:
        self._outs: list = []
        self._grid: list[float] = []
        self._batt: list[float] = []
        self._soc_rec = math.nan
        self._missing = False
        self._q_ev = 0.0

    def _soc_pct(self, kwh: float) -> float:
        return float(kwh) / self.lc.cfg.pack.capacity_kwh * 100.0

    def _minute_quarters(self, lo: int, hi: int) -> Table:
        """``to_quarters`` of the stored minute rows in ``[lo, hi)`` (quarter boundaries), the
        quarters before the cached end from the cache.

        The cache holds the quarters of ``[c.lo, c.hi)`` and the minute rows behind each one.
        A call whose span starts at or after ``c.lo`` and ends at or after ``c.hi`` keeps the
        cached quarters from ``lo`` on and aggregates only the minutes of ``[c.hi, hi)``;
        ``to_quarters`` works per quarter, so the result is the one of a full read. Any other
        span, or a stored row count in ``[lo, c.hi)`` that differs from the cached one (a row
        written or pruned behind the cache), rebuilds it from a full read. ``restore`` and
        ``adopt_refit`` drop it. The caller gets copies of the cached arrays. A span off the
        quarter grid is a full read that leaves the cache as it is.
        """
        lo, hi = int(lo), int(hi)
        if lo % ISP_S or hi % ISP_S:
            return self._aggregate(lo, hi)[0]
        c = self._qcache
        if c is not None and c["lo"] <= lo <= c["hi"] <= hi:
            keep = c["q"]["ts"] >= lo
            n = int(c["n"][keep].sum())
            if self.store.minute_count(lo, c["hi"]) == n:
                old = {k: v[keep] for k, v in c["q"].items()}
                new, new_n = self._aggregate(c["hi"], hi)
                q = _concat([old, new]) if len(new["ts"]) else old
                n_all = np.concatenate([c["n"][keep], new_n])
                self._qcache = {"lo": lo, "hi": hi, "q": q, "n": n_all}
                return {k: v.copy() for k, v in q.items()}
        q, n = self._aggregate(lo, hi)
        self._qcache = {"lo": lo, "hi": hi, "q": q, "n": n}
        return {k: v.copy() for k, v in q.items()}

    def _aggregate(self, lo: int, hi: int) -> tuple[Table, np.ndarray]:
        """``to_quarters`` of the stored minute rows in ``[lo, hi)`` and the rows per quarter."""
        m = self.store.minutes(lo, hi)
        if not len(m["ts"]):
            return {"ts": np.zeros(0, dtype=np.int64)}, np.zeros(0, dtype=np.int64)
        q = to_quarters(m)
        quarter = isp_start(np.asarray(m["ts"], dtype=np.int64))
        _, n = np.unique(quarter, return_counts=True)
        return q, n.astype(np.int64)

    def _quarters(
        self, lo: int, hi: int, measured_gen: bool = False, cached: bool = False
    ) -> Table:
        """Quarter flows in ``[lo, hi)``: minute rows through ``to_quarters``, and before the
        first minute quarter the hourly backfill through ``hourly_to_quarters``. The string
        columns ``pv1_kwh`` .. ``pv8_kwh`` come along where the rows carry them.

        The site models learn all PV, so ``gen_kwh`` is 0 (as ``replay.phases.SiteModels``),
        unless ``measured_gen``: the refit's per-string fit reads the GEN port as a string.
        ``cached``: the minute quarters through ``_minute_quarters`` (the same table).
        """
        if cached:
            mq = self._minute_quarters(lo, hi)
        else:
            m = self.store.minutes(lo, hi)
            mq = to_quarters(m) if len(m["ts"]) else {"ts": np.zeros(0, dtype=np.int64)}
        if len(mq["ts"]):
            mq["hourly_origin"] = np.zeros(len(mq["ts"]), dtype=bool)
        first = int(mq["ts"][0]) if len(mq["ts"]) else int(hi)
        h = self.store.hours(lo, first)
        hq = hourly_to_quarters(h) if len(h["ts"]) else {"ts": np.zeros(0, dtype=np.int64)}
        if len(hq["ts"]):
            keep = (hq["ts"] >= lo) & (hq["ts"] < first)
            hq = {k: np.asarray(v)[keep] for k, v in hq.items()}
        q = _concat([hq, mq])
        if not measured_gen or "gen_kwh" not in q:
            q["gen_kwh"] = np.zeros(len(q["ts"]))
        return q

    def _serve_at(self, now: int) -> Table:
        """The live master at ``now``, served by the band source (recording at ``now``)."""
        master = live_master(self.store, now, MASTER_DAYS_BACK)
        if self.source is None:
            self.source = ModelBandSource(
                self.lc.model, self.lc.model_version, master, record=self.store.put_band,
                skip=self.lc.skip_tiers,
            )
        else:
            self.source.refresh(master)
        self.source.set_now(now, self.epoch)
        return master

    def _view(self, qts: int):
        """The load median and EV expectation at ``qts`` from the store's quarters before it."""
        cfg = self.lc.cfg
        q = self._quarters(qts - EV_DAYS * DAY_S, qts, cached=True)
        for c in ("grid_import_kwh", "grid_export_kwh", "batt_charge_kwh", "batt_discharge_kwh"):
            q.setdefault(c, np.full(len(q["ts"]), np.nan))
        residual = residual_kw(q, cfg.pack)
        pv_est = self._pv_est(q["ts"], qts)
        return site_view(q, residual, q["missing"], pv_est, self.state, qts, cfg)

    def _irr_history(self, now: int) -> Table | None:
        """Irradiance known at ``now``: the archive stored by ``refit``, then the newest
        forecast's rows the archive lacks."""
        tables = []
        for source in ("archive", "forecast"):
            got = self.store.fetch_at(now, source)
            if got is not None:
                tables.append({k: got[1][k] for k in ("ts", *IRR_COLS) if k in got[1]})
        if not tables:
            return None
        if len(tables) == 2:
            seen = np.isin(tables[1]["ts"], tables[0]["ts"])
            tables[1] = {k: np.asarray(v)[~seen] for k, v in tables[1].items()}
        t = _concat(tables)
        order = np.argsort(t["ts"], kind="stable")
        return {k: np.asarray(v)[order] for k, v in t.items()}

    def _pv_est(self, ts: np.ndarray, now: int) -> np.ndarray:
        champ = self.state.champion
        hist = self._irr_history(now)
        if champ is None or not champ.planes or hist is None:
            return np.zeros(len(ts))
        cfg = self.lc.cfg
        return predict_pv_kw(champ, align(hist, ts, IRR_COLS), cfg.latitude, cfg.longitude)

    def _window_irr(self, now: int, slots: np.ndarray) -> tuple[Table, int | None, bool]:
        """``(irradiance, forecast issue ts, climatology)`` for the window's PV expectation.

        A forecast older than ``FORECAST_MAX_AGE_S`` (or none) falls back to climatology:
        ``stale_fallback`` of the clear-sky GHI and the trailing 3-day clearness of the
        stored irradiance (``DEFAULT_CLEARNESS`` without one), all of it diffuse.
        """
        got = self.store.fetch_at(now, "forecast")
        if got is not None and now - got[0] <= FORECAST_MAX_AGE_S:
            return got[1], got[0], False
        cfg = self.lc.cfg
        zenith, _ = solar_position(slots, cfg.latitude, cfg.longitude)
        kt, temp = DEFAULT_CLEARNESS, DEFAULT_TEMP_C
        hist = self._irr_history(now)
        if hist is not None:
            recent = (hist["ts"] >= now - CLIMATOLOGY_DAYS * DAY_S) & (hist["ts"] < now)
            hz, _ = solar_position(hist["ts"][recent], cfg.latitude, cfg.longitude)
            ghi = np.asarray(hist["ghi_w_m2"], dtype=float)[recent]
            cs = clear_sky_ghi(hz)
            ok = daylight(hz) & np.isfinite(ghi)
            if ok.any() and cs[ok].sum() > 0:
                kt = float(ghi[ok].sum() / cs[ok].sum())
            t = np.asarray(hist["temp_c"], dtype=float)[recent]
            if np.isfinite(t).any():
                temp = float(np.nanmean(t))
        ghi = stale_fallback(clear_sky_ghi(zenith), kt)
        n = len(slots)
        irr = {
            "ts": slots, "ghi_w_m2": ghi, "dni_w_m2": np.zeros(n), "dhi_w_m2": ghi.copy(),
            "temp_c": np.full(n, temp),
        }
        return irr, None, True

    def _flags(self) -> list[str]:
        on = {
            "learning": bool(self.state.learning),
            "live_stale": self._live_stale,
            "irradiance_climatology": self._climatology,
            "pv_drift": bool(self.state.drift),
            "no_pack": self.lc.pack_version is None,
            "day_ahead_missing": self._da_missing,
        }
        return [f for f in FLAGS if on[f]]

    def _band(self, qts: int) -> dict:
        """The band of quarter ``qts`` in the current ranking's window (NaN without one)."""
        r = self.ranked
        if r is None:
            return _NAN_BAND
        k = int(np.searchsorted(r.window.ts, qts))
        if k >= len(r.window.ts) or int(r.window.ts[k]) != qts:
            return _NAN_BAND
        return band_for_slot(r.window, k)

    def _live_age(self, ts: int) -> int | None:
        """Seconds since the newest priced live minute labelled before ``ts`` became known."""
        lm = self.store.live_minutes(ts - LIVE_LOOKBACK_S, ts)
        ok = np.isfinite(lm["buy_live"]) & np.isfinite(lm["sell_live"])
        if not ok.any():
            return None
        return int(ts - int(lm["ts"][ok][-1]) - MINUTE_S)

    def _provisional(self, qts: int) -> tuple[dict, dict]:
        """Booking prices of a quarter from the live minutes labelled before
        ``qts + PROVISIONAL_CUT_S`` (state of the last priced minute, 0 without one) and the
        quarter's live feature values."""
        lm = self.store.live_minutes(qts - 1800 - MINUTE_S, qts + PROVISIONAL_CUT_S)
        isp = minute_to_isp(lm, np.array([qts], dtype=np.int64))
        values = {c: float(isp[c][0]) for c in _LIVE_COLS}
        inq = (lm["ts"] >= qts) & np.isfinite(lm["state"])
        inq &= np.isfinite(lm["buy_live"]) & np.isfinite(lm["sell_live"])
        state = int(lm["state"][inq][-1]) if inq.any() else 0
        prices = {"buy": values["buy_live_end"], "sell": values["sell_live_end"], "state": state}
        return prices, values

    def _book(self, qts: int, missing: bool) -> list[dict]:
        """Shadow, house and actual rows of a quarter, chained in that order in one
        transaction (all of them are stored or none), and the quarter's live feature values."""
        items, values = self._book_items(qts, missing)
        out = self.store.append_chained_many(items)
        if self._outs:
            self.store.upsert_features(qts, **values)
        return out

    def _book_items(
        self, qts: int, missing: bool, paths: tuple[str, ...] = PATHS
    ) -> tuple[list[tuple[str, dict]], dict]:
        """The booking items of a quarter (the paths of ``paths``, in ``PATHS`` order) and
        its live feature values; nothing is written."""
        cfg = self.lc.cfg
        prices, values = self._provisional(qts)
        nan = np.full(ISP_S // MINUTE_S, np.nan)
        if self._outs:
            net = np.array([o.net_kw for o in self._outs], dtype=float)
            house = np.array([o.house_kw for o in self._outs], dtype=float)
            ac = np.array([o.batt_ac_kw for o in self._outs], dtype=float)
            grid, batt = np.array(self._grid), np.array(self._batt)
        else:
            net = house = ac = grid = batt = nan
        soc_shadow = self._soc_pct(self.stepper.battery.soc_kwh)
        rows = {
            "shadow": booking_row(qts, "shadow", net, ac, soc_shadow, prices, cfg.terms, missing),
            "house": booking_row(
                qts, "house", house, np.zeros(len(house)), math.nan, prices, cfg.terms, missing
            ),
            "actual": booking_row(
                qts, "actual", grid, batt, self._soc_rec, prices, cfg.terms, missing
            ),
        }
        return [("booking", rows[p]) for p in PATHS if p in paths], values

    def _last_booked(self) -> tuple[int | None, set[str], dict | None]:
        """The newest quarter with a booking that is not a settled rebooking: its
        ``quarter_ts``, the paths booked for it and its shadow row (None without one)."""
        head = self.store.head()
        hi = 0 if head is None else head[0]
        while hi >= 1:
            lo = max(1, hi - _CHAIN_PAGE + 1)
            for kind, row in reversed(self.store.chain(lo, hi)):
                if kind == "booking" and row["price_source"] != "settled":
                    q = int(row["quarter_ts"])
                    rows = [
                        r for r in self.store.bookings(q, q + ISP_S)
                        if r["price_source"] != "settled"
                    ]
                    shadow = next((r for r in rows if r["path"] == "shadow"), None)
                    return q, {r["path"] for r in rows}, shadow
            hi = lo - 1
        return None, set(), None

    def _first_quarter(self, now: int) -> int | None:
        """The quarter of the oldest stored decision, else of the oldest minute row within
        ``SOC_LOOKBACK_S``; None without either (nothing ran yet)."""
        first = self.store.chain(1, 1)
        if first and first[0][0] == "decision":
            return int(isp_start(int(first[0][1]["ts"])))
        m = self.store.minutes(now - SOC_LOOKBACK_S, now, max_epoch=self._seen_epoch)
        return int(isp_start(int(m["ts"][0]))) if len(m["ts"]) else None

    def _recorded_soc_kwh(self, now: int) -> float:
        m = self.store.minutes(now - SOC_LOOKBACK_S, now, max_epoch=self._seen_epoch)
        soc = np.asarray(m["soc_pct"], dtype=float)
        ok = np.isfinite(soc)
        if not ok.any():
            return math.nan
        return float(soc[ok][-1]) / 100.0 * self.lc.cfg.pack.capacity_kwh

    def _rank(self, qi: QuarterInputs | None, qts: int) -> Ranked | None:
        """The ranking of quarter ``qts`` from ``qi`` at the stepper's SoC: the one path from
        inputs to a ranking for a fresh setup, a restart and re-execution alike.

        Sets the stepper's plan quarters, ``ranked`` and the nowcast; ``qi`` None (a failed
        setup) leaves no ranking and no nowcast for the quarter.
        """
        self._ranked_qts = qts
        if qi is None:
            self.ranked, self._nc = None, None
            return None
        cfg, soc = self.lc.cfg, self.stepper.battery.soc_kwh
        self.ranked = rank_quarter(qi, soc, cfg, self.loss, self.stepper.trader)
        self.stepper.set_plan(self.ranked)
        self._nc = self.ranked.nowcast
        return self.ranked

    def _rerank_stored(self, qts: int) -> None:
        """Rank the quarter ``qts`` again from its ``windows`` row and the stored bands."""
        w = self.store.window(qts, epoch=self.epoch - 1)  # set up by an earlier run
        if w is None:
            return
        qi = stored_inputs(self.store, w, self.lc.model.tables, self.lc.cfg.knobs)
        if qi is not None:
            self._rank(qi, qts)

    # -- the model ----------------------------------------------------------------------

    def set_model(self, model: PriceModel, version: str, now: int) -> None:
        """Switch the price model at the quarter boundary after ``now``.

        ``now`` is the next minute the live loop processes (under the minute lock), so the
        ``model_switch`` event applies before that minute in re-execution as live. The event
        names the new ``version``, ``effective_ts`` (``isp_start(now) + ISP_S``), the
        ``from_version`` in effect and the ``cov_source`` of the new model
        (``live.pack_update.cov_source``). From ``effective_ts`` on, every minute, window and
        band carries the new version, and the band source serves the new model. A second
        call before the boundary replaces the pending switch (the last event wins, as in
        re-execution). The pending switch is kept in memory: a new site object starts on the
        model in effect and its caller puts a pending switch again.
        """
        now = int(now)
        at = int(isp_start(now)) + ISP_S
        detail = {
            "version": version, "effective_ts": at, "from_version": self.lc.model_version,
            "cov_source": cov_source(model),
        }
        self.store.put_event(now, "model_switch", detail)
        self._pending = (at, model, version)

    @property
    def pending_model(self) -> tuple[int, str] | None:
        """The pending switch ``(effective_ts, version)``; None without one."""
        return None if self._pending is None else (self._pending[0], self._pending[2])

    def _apply_switch(self, ts: int) -> None:
        """Make the pending model the live one when its boundary is at or before ``ts``."""
        if self._pending is None or self._pending[0] > int(ts):
            return
        _, model, version = self._pending
        self._pending = None
        self.lc = replace(self.lc, model=model, model_version=version)
        self.source = None  # the next serve builds the band source on the new model

    # -- the clock ----------------------------------------------------------------------

    @property
    def next_minute(self) -> int | None:
        """The minute ``on_minute`` accepts next; None before ``restore``."""
        return self._next

    def restore(self, now: int) -> None:
        """Continue from the store at ``now``: the next minute is ``isp_start(now)``.

        Loads ``registry["site_state"]``, anchors the shadow at the last booked shadow SoC
        (else the newest recorded SoC, else the floor; clamped to the pack's band), books every
        whole quarter after the last booked one up to ``isp_start(now)`` missing on the three
        paths, puts a ``restart`` event (the anchor SoC, ``head_seq``, the chain head seq
        before the gap bookings, 0 for none: re-execution starts there, the ``model_version``
        in effect: re-execution starts its walk on it, and the ``knobs`` of the run: the
        export and the verifier read them per run) in one transaction
        with the gap bookings, re-anchors the stepper and ranks the running quarter again
        from its stored window when one exists. Without any booking the gap starts at
        the quarter of the oldest decision (else the oldest minute row), so a partial first
        quarter is booked too. A last quarter booked on fewer than three paths (a store
        written before the triple was atomic) gets its other paths booked missing. A ``now``
        before the end of the last booked quarter (a clock jump back) resumes after that
        quarter, so no quarter is booked twice. A pending model switch whose boundary is at or
        before ``now`` applies first (re-execution applies it at the ``restart`` event).
        """
        now = int(now)
        self._apply_switch(now)
        self._qcache = None
        cfg = self.lc.cfg
        pack = cfg.pack
        qts = int(isp_start(now))
        raw = self.store.registry(SITE_STATE_KEY)
        self.state = state_from_json(raw) if raw is not None else initial_state(cfg)
        head = self.store.head()
        prev = self.store.last_event("restart")
        # the minute rows of earlier runs only (re-execution holds this run's rows already)
        self._seen_epoch = 0 if prev is None else int(prev["seq"])
        last_minute = None if prev is None else self.store.last_nowcast_ts(prev["seq"])
        q_last, have, shadow = self._last_booked()
        soc = math.nan
        if shadow is not None and math.isfinite(shadow["soc_pct"]):
            soc = shadow["soc_pct"] / 100.0 * pack.capacity_kwh
        if not math.isfinite(soc):
            soc = self._recorded_soc_kwh(now)
        floor, ceiling = pack.soc_floor * pack.capacity_kwh, pack.soc_ceiling * pack.capacity_kwh
        soc = floor if not math.isfinite(soc) else min(max(soc, floor), ceiling)
        self.stepper.reanchor(soc)
        self._reset_quarter()
        if q_last is not None:
            first = q_last if have != set(PATHS) else q_last + ISP_S
            nxt = max(qts, q_last + ISP_S)
        else:
            first = self._first_quarter(now)
            first, nxt = (qts if first is None else first), qts
        items = []
        for q in range(first, nxt, ISP_S):
            todo = tuple(p for p in PATHS if not (q == q_last and p in have))
            items += self._book_items(q, missing=True, paths=todo)[0]
        detail = {
            "soc_kwh": soc, "head_seq": 0 if head is None else head[0], "last_minute": last_minute,
            "model_version": self.lc.model_version, "knobs": knobs_dict(cfg.knobs),
        }
        # the gap bookings and the restart event in one transaction: both or neither
        _, self.epoch = self.store.append_chained_with_event(items, now, "restart", detail)
        self.prev, self.ranked, self._ranked_qts, self._nc = None, None, None, None
        self._next = nxt
        self.last_trade = self._stored_last_trade(now)
        self.pv = None
        self._rerank_stored(nxt)

    def setup_quarter(self, now: int) -> bool:
        """Minute 1: window, ranking and the ``windows`` row; False without a slot-0 band.

        The live master spans ``MASTER_DAYS_BACK`` days; ``da_known`` is
        ``da_known_mask & isfinite(da)`` (holes included). At 18:00 Amsterdam also the
        ``greed`` row. A failed setup also writes its ``windows`` row, without the arrays,
        so re-execution knows the quarter ran without a ranking.
        """
        now = int(now)
        self._apply_switch(now)
        cfg, store = self.lc.cfg, self.store
        qts = int(isp_start(now))
        master = self._serve_at(now)
        slots = window_slots(now)
        da = align(master, slots, ["da_eur_mwh"])["da_eur_mwh"]
        published = da_known_mask(now, slots)
        da_known = published & np.isfinite(da)
        self._da_missing = bool((published & ~np.isfinite(da)).any())
        view = self._view(qts)
        irr, issue, climatology = self._window_irr(now, slots)
        self._climatology = climatology and not cfg.pv.none
        qi = quarter_inputs(now, self.source, cfg, view, irr, da_known=da_known)
        soc = self.stepper.battery.soc_kwh
        row = {
            "ts": qts, "epoch": self.epoch, "now": now, "model_version": self.lc.model_version,
            "pack_version": self.lc.pack_version, "soc_kwh": soc, "irr_issue_ts": issue,
            "flags": self._flags(),
        }
        if self._rank(qi, qts) is None:
            self.pv = None
            store.put_window(row)
            return False
        store.put_window(
            dict(
                row, healthy=qi.healthy, da_known=qi.da_known, widening=qi.widening,
                load_kw=qi.load_kw, pv_kw=qi.pv_kw, ev_kw=qi.ev_kw, plan_mask=qi.plan_mask,
            )
        )
        wall, _, _ = local_fields(np.array([qts], dtype=np.int64))
        if int(wall[0]) == GREED_HOUR * 4:
            store.put_greed(greed_row(now, self.ranked, soc, cfg))
        self._set_pv(irr, int(slots[0]), float(qi.pv_kw[0]))
        return True

    def _set_pv(self, irr: Table, slot: int, total_kw: float) -> None:
        """The display-only PV expectation, after every write of the setup: a failure logs
        once and leaves ``pv`` None, never touching the ranking or the stored inputs."""
        cfg = self.lc.cfg
        try:
            self.pv = pv_now(
                self.state.champion, irr, slot, cfg.latitude, cfg.longitude, total_kw
            )
        except Exception:
            self.pv = None
            if not self._pv_failed:
                _LOGGER.exception("PV expectation for display failed at %s", slot)
            self._pv_failed = True

    def on_minute(self, ts: int, flows: dict | None) -> MinuteResult:
        """One minute (rules in the module docstring); ``flows`` None is a missing minute.

        Raises ``ValueError`` before ``restore`` and when ``ts`` is not the next minute.
        """
        ts = int(ts)
        if self._next is None:
            raise ValueError("restore(now) must run before the first minute")
        if ts != self._next:
            raise ValueError(f"minute {ts} is not the next minute {self._next}")
        self._apply_switch(ts)
        cfg, store = self.lc.cfg, self.store
        qts = int(isp_start(ts))
        e = (ts - qts) // MINUTE_S
        if flows is not None:
            store.put_minute(dict(flows, ts=ts, epoch=self.epoch))
        grid, batt, soc_rec, missing = minute_flows(self._stored_flows(ts), cfg.pack)
        nc, age = self._nowcast_at(ts, qts, e)
        learning = bool(self.state.learning)
        soc_before = self.stepper.battery.soc_kwh
        out = self.stepper.minute(ts, nc, grid, batt, missing, learning)
        if not missing and qts in self.stepper.plan_quarters:
            self._q_ev += ev_passthrough_kwh(out.house_kw, out.batt_ac_kw, cfg.ev.charger_kw)
        nan = math.nan
        store.put_nowcast(
            {
                "ts": ts, "epoch": self.epoch, "quarter_ts": qts,
                "take_point": nan if nc is None else nc.take_point,
                "feed_point": nan if nc is None else nc.feed_point,
                "w_take": nan if nc is None else nc.w_take,
                "w_feed": nan if nc is None else nc.w_feed,
                "p_dual": nan if nc is None else nc.p_dual,
                "take_q50": nan if nc is None else nc.take_q50,
                "feed_q50": nan if nc is None else nc.feed_q50,
                "live_age_s": age,
            }
        )
        d = out.decision
        written = not is_repeat_hold(self.prev, d)
        if written:
            row = decision_row(
                d, self._band(qts), nc, self._soc_pct(soc_before), self.lc.model_version
            )
            store.append_chained("decision", row)
        if d.committed and d.action in ("charge", "discharge"):
            self.last_trade = {"ts": int(d.ts), "action": d.action, "kw": float(d.kw)}
        self.prev = d
        self._outs.append(out)
        self._grid.append(grid)
        self._batt.append(batt)
        self._soc_rec = soc_rec
        self._missing = self._missing or missing
        booked: list[dict] = []
        if ts + MINUTE_S == qts + ISP_S:
            booked = self._book(qts, self._missing)
            self.ev_passthrough_kwh = self._q_ev
            self._reset_quarter()
        self._next = ts + MINUTE_S
        return MinuteResult(d, written, booked)

    def _stored_flows(self, ts: int) -> dict | None:
        """The stored flows of minute ``ts`` (the required columns), None without a row.

        The minute decides on the stored row, not on the flows handed in: a re-run after a
        restart keeps the first run's row (insert-or-ignore), so both runs and re-execution
        see the same minute. Rows of this run or earlier ones only (``max_epoch``): a later
        run's row for a minute this run had none for never reaches its re-execution.
        """
        m = self.store.minutes(ts, ts + MINUTE_S, max_epoch=self.epoch)
        if not len(m["ts"]):
            return None
        return {c: float(m[c][0]) for c in _REQUIRED}

    def _nowcast_at(self, ts: int, qts: int, e: int) -> tuple[Nowcast | None, int | None]:
        """The nowcast the trader sees at minute ``ts`` (elapsed ``e``) and the live age.

        Reset at elapsed 0, re-read at ``NOWCAST_REREAD_MIN`` from bands served at ``ts``,
        None while the live feed is stale. Re-execution replaces this with the stored row.
        """
        if e == 0:
            self._nc = None
        elif e in NOWCAST_REREAD_MIN and self.ranked is not None and self._ranked_qts == qts:
            self._serve_at(ts)
            self._nc = reread_nowcast(self.source, self.ranked, qts, e, self.lc.cfg.knobs)
            self.ranked.nowcast = self._nc
        age = self._live_age(ts)
        self._live_stale = age is None or age > LIVE_STALE_S
        return (None if self._live_stale else self._nc), age

    def _settled_items(
        self, qts: int, take: float, feed: float, state: int
    ) -> list[tuple[str, dict]]:
        """The settled rebooking of quarter ``qts``: ``resettle`` of the newest booking per
        path at ``(take, feed, state)``, in ``PATHS`` order; nothing is written.

        ``settle`` and re-execution of its ``settle`` event both book these items, so the
        rebooked rows are identical by construction.
        """
        newest = {r["path"]: r for r in self.store.bookings(qts, qts + ISP_S)}  # seq order
        terms = self.lc.cfg.terms
        return [
            ("booking", resettle(newest[p], take, feed, int(state), terms))
            for p in PATHS
            if p in newest
        ]

    def _rebook_settled(self, qts: int, take: float, feed: float, state: int) -> list[dict]:
        """``_settled_items`` of quarter ``qts`` chained in one transaction."""
        return self.store.append_chained_many(self._settled_items(qts, take, feed, state))

    def settle(self, rows: list[dict], now: int) -> list[int]:
        """Store settlement rows and rebook their quarters settled at ``now``; the quarters.

        ``now`` is the next minute the live loop processes, so the ``settle`` event applies
        before that minute in re-execution as live. Each row is stored (``put_settlement``,
        a stored row is kept); a row with a NaN take or feed is skipped, so its quarter stays
        as booked. A quarter is rebooked (``_settled_items``) when it has bookings and its
        newest settlement published by ``now`` (``settlement_at``) is not the price of every
        path's newest booking: a provisional or ``none`` booking, or one settled at an older
        publication. A row published again with the same prices books nothing. The
        rebookings of every quarter and one ``settle`` event go to the store in one
        transaction (both or neither); the event's ``detail`` lists the rebooked
        ``quarters`` and, per quarter, the ``published_at`` of the settlement it booked, so
        re-execution reads that row and no other (none rebooked: no event). Raises
        ``ValueError`` for a row without ``published_at`` or published after ``now``.
        """
        now = int(now)
        ok = []
        for r in rows:
            if not (math.isfinite(float(r["take"])) and math.isfinite(float(r["feed"]))):
                continue
            if r.get("published_at") is None or int(r["published_at"]) > now:
                raise ValueError(
                    f"settlement of {int(r['quarter_ts'])} published at "
                    f"{r.get('published_at')}, after the settle minute {now}"
                )
            ok.append(r)
        for r in ok:
            self.store.put_settlement(dict(r))
        done, published, items = [], [], []
        for q in sorted({int(r["quarter_ts"]) for r in ok}):
            s = settlement_at(self.store, q, now)
            if s is None or not (math.isfinite(s["take"]) and math.isfinite(s["feed"])):
                continue
            state = 0 if s["regulation_state"] is None else int(s["regulation_state"])
            price = (float(s["take"]), float(s["feed"]), state)
            newest = self.store.newest_bookings(q, q + ISP_S)
            if not newest or all(
                b["price_source"] == "settled" and (b["buy"], b["sell"], b["state"]) == price
                for b in newest.values()
            ):
                continue
            items += self._settled_items(q, *price)
            done.append(q)
            published.append(int(s["published_at"]))
        if done:
            detail = {"quarters": done, "published_at": published}
            self.store.append_chained_with_event(items, now, "settle", detail)
        return done

    def _stored_last_trade(self, now: int) -> dict | None:
        """The newest committed charge or discharge decision of the last week (None without)."""
        rows = self.store.decisions(int(now) - SOC_LOOKBACK_S, int(now))
        for r in reversed(rows):
            if r["committed"] and r["action"] in ("charge", "discharge"):
                return {"ts": int(r["ts"]), "action": r["action"], "kw": float(r["kw"])}
        return None

    def status(self) -> dict:
        return {
            "decision": self.prev,
            "ranked": self.ranked,
            "soc_kwh": float(self.stepper.battery.soc_kwh),
            "head": self.store.head(),
            "flags": self._flags(),
            "learning": bool(self.state.learning),
            "drift": bool(self.state.drift),
            "last_trade": self.last_trade,
            "pv": self.pv,
        }

    def fit(self, now: int, irr_archive: Table) -> SiteModelState:
        """``weekly_update`` and the drift check on the store's quarters before ``now``.

        Reads the store and the current state and writes nothing, so it may run beside the
        live minutes; ``adopt_refit`` makes the result the site's state.
        """
        now = int(now)
        cfg = self.lc.cfg
        measured = self._quarters(now - REFIT_DAYS * DAY_S, int(isp_start(now)), True)
        q = dict(measured, gen_kwh=np.zeros(len(measured["ts"])))
        irr_before = rows_before(irr_archive, now)
        state = weekly_update(self.state, measured, irr_before, cfg, now)
        champ = state.champion
        if champ is None or not champ.planes:
            pv = np.zeros(len(q["ts"]))
        else:
            pv = predict_pv_kw(
                champ, align(irr_archive, q["ts"], IRR_COLS), cfg.latitude, cfg.longitude
            )
        drift = drift_alarm(
            q, pv, irr_before, cfg.latitude, cfg.longitude, now,
            limit=cfg.knobs.drift_alarm, pack=cfg.pack,
        )
        return replace(state, drift=bool(drift))

    def scale_off(self, state: SiteModelState) -> str | None:
        """The entered strings whose fitted kWp in ``state``'s latest fit (the challenger) lies
        more than ``strings.SCALE_TOLERANCE`` from the entered kWp, as the repair-issue
        placeholder ("1, 2"), "" when all checked strings agree.

        None, no verdict, without entered planes, without a fit, or when the fit took the
        blind path (``model.supervised`` False): a blind scale carries the level multiplier
        and says nothing about the entered figures, so it neither raises nor clears the issue.
        """
        planes = self.lc.cfg.pv.planes
        model = state.challenger
        if not planes or model is None or not model.supervised:
            return None
        return string_labels(planes, scale_check(model, planes))

    def adopt_refit(self, state: SiteModelState, at: int, irr_archive: Table) -> None:
        """Make a ``fit`` result the site's state from ``at`` on.

        ``at`` is the next minute the live loop processes, so the ``refit`` event applies
        before that minute in re-execution as live. Stores the state as
        ``registry["site_state"]``, the archive as the forecast fetch ``archive`` issued at
        ``at`` (the PV estimate of past quarters reads it) and the event ``refit``.
        """
        at = int(at)
        self.state = state
        self._qcache = None
        self.store.set_registry(SITE_STATE_KEY, state_to_json(state), at)
        self.store.put_fetch(at, "archive", at, {k: irr_archive[k] for k in ("ts", *IRR_COLS)})
        self.store.put_event(
            at,
            "refit",
            {
                "learning": bool(state.learning),
                "drift": bool(state.drift),
                "champion": state.champion is not None,
                "learned_days": int(state.learned_days),
            },
        )

    def refit(self, now: int, irr_archive: Table) -> SiteModelState:
        """``fit`` and ``adopt_refit`` at ``now`` in one step (the weekly site-model update)."""
        self.adopt_refit(self.fit(now, irr_archive), now, irr_archive)
        return self.state
