"""Phases 1 and 4 to 6 of the replay engine (``replay.engine``), and the shared helpers.

Phase 1 (``SiteModels``, ``site_view``) walks the site models forward, phase 4
(``Prices``, ``book_quarter``) books the ledgers, phase 5 (``greed_row``) checks the rule's
appetite at 18:00 and phase 6 (``drop_empty_days`` to ``actual_soc_kwh``, ``result_pairs``)
closes the replay. Phases 2 and 3 (the ranking and the minute steps) and ``book_paths``
(which books the virtual and actual rows through ``book_quarter`` and adds the house row)
live in ``replay.steps``.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from ..core.ledger import LedgerRow, money_eur, rows_for_quarter
from ..core.pricemodel import QUANTILES
from ..core.rule.thresholds import Thresholds
from ..core.rule.trader import Decision
from ..core.rule.window import Nowcast, Window
from ..core.site.contract import SiteConfig, tax_mode_at
from ..core.site.ev import EvStats, detect_sessions, ev_stats, session_mask
from ..core.site.flows import residual_kw
from ..core.site.load import LoadModel, load_median
from ..core.site.registry import SiteModelState, drift_alarm, initial_state, weekly_update
from ..core.site.separate import predict_pv_kw
from ..core.site.shading import share_by_string
from ..core.site.socpath import SocPath
from ..core.timegrid import ZONE, Table, align, isp_start

DAY_S = 86400
IRR_COLS = ["ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c"]
LOAD_DAYS = 7
EV_DAYS = 28
# ``site_report["irradiance"]`` without ``ReplayInputs.irr_forecast``: the open-meteo archive is
# the day's own archived model output (ERA5 where missing), not the multi-day forecast a live box
# would have for the slots 1 to 3 days ahead, so the replay's PV expectation is optimistic and
# the bias favours the virtual path. With ``irr_forecast`` it is ``IrrByLead.note()``.
IRRADIANCE_SOURCE = "archive (day-0 model output; optimistic)"
NIGHT_FROM_HOUR, NIGHT_TO_HOUR = 22, 6
GREED_COLUMNS = (
    "day",
    "soc_kwh",
    "charge_threshold",
    "sale_threshold",
    "night_median_take",
    "night_median_feed",
    "surplus_tomorrow_kwh",
    "low_soc_applies",
    "low_soc_ok",
    "high_soc_applies",
    "high_soc_ok",
)


def local_midnight(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=ZONE).timestamp())


def local_at(d: dt.date, hour: int) -> int:
    return int(dt.datetime(d.year, d.month, d.day, hour, tzinfo=ZONE).timestamp())


def local_date(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date()


def rows_before(table: Table, now: int) -> Table:
    """The rows of ``table`` with ``ts < now`` (every column of the row length)."""
    ts = np.asarray(table["ts"], dtype=np.int64)
    keep = ts < int(now)
    return {k: np.asarray(v)[keep] for k, v in table.items()}


def row_index(ts_sorted: np.ndarray, t: int) -> int | None:
    k = int(np.searchsorted(ts_sorted, int(t)))
    return k if k < len(ts_sorted) and int(ts_sorted[k]) == int(t) else None


def update_instants(start: int, end: int) -> list[int]:
    """``start`` and every local Monday 00:00 in ``(start, end)``."""
    out = [int(start)]
    d = local_date(start) + dt.timedelta(days=1)
    while True:
        t = local_midnight(d)
        if t >= end:
            return out
        if d.weekday() == 0 and t > start:
            out.append(t)
        d += dt.timedelta(days=1)


@dataclass
class Ranked:
    window: Window
    path: SocPath
    th: Thresholds
    nowcast: Nowcast
    plan_mask: np.ndarray


@dataclass
class QuarterRun:
    net_kw: np.ndarray  # the virtual grid net per minute
    grid_kw: np.ndarray  # the recorded grid net per minute (flat quarter values without minutes)
    ev_passthrough_kwh: float
    decisions: list[Decision] = field(default_factory=list)
    house_kw: np.ndarray | None = None  # the house without a battery, grid - batt_actual
    batt_ac_kw: np.ndarray | None = None  # the virtual battery's AC flow per minute


# -- phase 1: site models ---------------------------------------------------------------


@dataclass
class SiteView:
    """What the rule may know about the site at one quarter."""

    state: SiteModelState
    load: LoadModel
    ev_stats: EvStats | None
    n_sessions: int


class SiteModels:
    """The walk-forward site models: PV registry weekly, load median and EV per quarter."""

    def __init__(self, q: Table, irr: Table, start: int, end: int, cfg: SiteConfig):
        self.cfg = cfg
        # the replay learns all PV: GEN-port production is validation data only, so the
        # site models (residual, blind PV fit, drift) see the flows with gen_kwh at 0; the
        # weekly refit reads the measured flows, where the per-string fit counts the GEN port
        # as one more string (registry.weekly_update zeroes it for the blind fit)
        self.q_measured = q
        self.q = dict(q, gen_kwh=np.zeros(len(q["ts"])))
        self.irr = irr
        self.ts = np.asarray(self.q["ts"], dtype=np.int64)
        self.residual = residual_kw(self.q, cfg.pack)
        self.missing = np.asarray(self.q["missing"], dtype=bool)
        self.irr_q = align(self.irr, self.ts, IRR_COLS)
        self.instants = update_instants(start, end)
        self.state = initial_state(cfg)
        self.pv_est = np.zeros(len(self.ts))
        self.drift_seen: list[bool] = []

    def _weekly(self, instant: int) -> None:
        """``weekly_update`` and the drift check on rows strictly before ``instant``."""
        cfg = self.cfg
        q_before = rows_before(self.q, instant)
        irr_before = rows_before(self.irr, instant)
        measured = rows_before(self.q_measured, instant)
        state = weekly_update(self.state, measured, irr_before, cfg, instant)
        champ = state.champion
        if champ is None:
            self.pv_est = np.zeros(len(self.ts))
        else:
            self.pv_est = predict_pv_kw(champ, self.irr_q, cfg.latitude, cfg.longitude)
        before = self.ts < instant
        drift = drift_alarm(
            q_before,
            self.pv_est[before],
            irr_before,
            cfg.latitude,
            cfg.longitude,
            instant,
            limit=cfg.knobs.drift_alarm,
            pack=cfg.pack,
        )
        self.drift_seen.append(bool(drift))
        self.state = replace(state, drift=bool(drift))

    def at(self, qts: int) -> SiteView:
        """The site models for the quarter starting at ``qts`` (rows before ``qts`` only)."""
        while self.instants and self.instants[0] <= qts:
            self._weekly(self.instants.pop(0))
        return site_view(
            self.q, self.residual, self.missing, self.pv_est, self.state, qts, self.cfg
        )


def site_view(
    q: Table,
    residual: np.ndarray,
    missing: np.ndarray,
    pv_est: np.ndarray,
    state: SiteModelState,
    qts: int,
    cfg: SiteConfig,
) -> SiteView:
    """The load median and the EV expectation at ``qts`` on the quarter rows before it.

    ``residual``, ``missing`` and ``pv_est`` run along ``q["ts"]``; ``state`` is the PV
    registry as of its last weekly update. The load estimate is ``residual + pv_est``.
    """
    all_ts = np.asarray(q["ts"], dtype=np.int64)
    hi = int(np.searchsorted(all_ts, qts))  # rows [.., hi) lie before the quarter
    lo = int(np.searchsorted(all_ts, qts - EV_DAYS * DAY_S))
    ts = all_ts[lo:hi]
    load_est = residual[lo:hi] + pv_est[lo:hi]
    mask = missing[lo:hi].copy()
    stats, n_sessions = None, 0
    if cfg.ev.present or cfg.ev.plan is not None:
        profile = load_median(load_est, ts, qts, mask=mask, days=EV_DAYS)
        sessions = detect_sessions(load_est, ts, profile, cfg.ev.charger_kw, cfg.ev.min_minutes)
        first = int(ts[0]) if len(ts) else qts - EV_DAYS * DAY_S
        stats = ev_stats(sessions, first, qts - 1)
        n_sessions = len(sessions)
        mask |= session_mask(sessions, ts)
    load = load_median(
        load_est,
        ts,
        qts,
        mask=mask,
        days=LOAD_DAYS,
        weekday_split=cfg.knobs.weekday_split,
    )
    return SiteView(state, load, stats, n_sessions)


# -- phase 4: ledgers -------------------------------------------------------------------


class Prices:
    """Settled prices of a quarter from the master, the live end values while unsettled."""

    def __init__(self, master: Table):
        self.ts = np.asarray(master["ts"], dtype=np.int64)
        self.buy = np.asarray(master["buy_eur_mwh"], dtype=float)
        self.sell = np.asarray(master["sell_eur_mwh"], dtype=float)
        self.state = np.asarray(master["regulation_state"], dtype=float)
        self.buy_live = np.asarray(master["buy_live_end"], dtype=float)
        self.sell_live = np.asarray(master["sell_live_end"], dtype=float)

    def settled(self, qts: int) -> tuple[float, float]:
        """Settled buy and sell (NaN when unsettled or off the master grid)."""
        k = row_index(self.ts, qts)
        if k is None:
            return float("nan"), float("nan")
        return float(self.buy[k]), float(self.sell[k])

    def at(self, qts: int) -> dict:
        """``buy``, ``sell``, ``state``, ``settled`` for ``rows_for_quarter``.

        A NaN settled buy or sell takes the live end value and ``settled = False``. A NaN
        regulation state is booked as 0 (it labels the row and does not change the money).
        """
        k = row_index(self.ts, qts)
        if k is None:
            return {"buy": np.nan, "sell": np.nan, "state": 0, "settled": False}
        buy, sell = float(self.buy[k]), float(self.sell[k])
        settled = bool(np.isfinite(buy) and np.isfinite(sell))
        if not settled:
            buy = buy if np.isfinite(buy) else float(self.buy_live[k])
            sell = sell if np.isfinite(sell) else float(self.sell_live[k])
        state = float(self.state[k])
        return {
            "buy": buy,
            "sell": sell,
            "state": int(state) if np.isfinite(state) else 0,
            "settled": settled,
        }


def unpriced(prices: dict) -> bool:
    """True when the quarter has no finite buy and sell price (it is booked ``missing``)."""
    return not (np.isfinite(prices["buy"]) and np.isfinite(prices["sell"]))


def book_quarter(
    qts: int,
    run: QuarterRun,
    prices: dict,
    cfg: SiteConfig,
    learning: bool,
    missing: bool,
) -> tuple[LedgerRow, LedgerRow]:
    """Virtual and actual rows of one quarter, both the ``gross_split`` of minute grid nets.

    The actual path splits the recorded grid net ``run.grid_kw``, the virtual path
    ``run.net_kw``, at the same granularity, so an identical battery books identical rows.
    A quarter without a finite buy and sell price is booked ``missing`` on both paths.
    """
    missing = missing or unpriced(prices)
    terms = cfg.terms
    virtual = rows_for_quarter(
        qts, "virtual", run.net_kw, prices, terms, learning, run.ev_passthrough_kwh, missing
    )
    actual = rows_for_quarter(qts, "actual", run.grid_kw, prices, terms, learning, 0.0, missing)
    return virtual, actual


def gross_split_bias_eur(recorded: tuple[float, float], actual: LedgerRow, terms) -> float:
    """Money of the recorded quarter gross minus the booked (split) actual money, in EUR.

    It shows the within-interval two-way flow the split cannot see. 0 for a missing row or
    a quarter without recorded energies.
    """
    imp, exp = recorded
    if actual.missing or not (np.isfinite(imp) and np.isfinite(exp)):
        return 0.0
    total, _ = money_eur(
        imp,
        exp,
        actual.buy_eur_mwh,
        actual.sell_eur_mwh,
        actual.state,
        terms,
        tax_mode_at(terms, actual.ts),
    )
    return float(total - actual.money_eur)


# -- phase 5: greed rows ----------------------------------------------------------------


def at_quantile(band: np.ndarray, q: float) -> np.ndarray:
    """Per slot, the band (quantiles x slots) at quantile ``q``, linear between rows."""
    return np.array([np.interp(q, QUANTILES, band[:, j]) for j in range(band.shape[1])])


def greed_row(now: int, ranked: Ranked, soc_kwh: float, cfg: SiteConfig) -> dict:
    """The greed check at 18:00: is the rule's appetite right for the SoC and what is coming?

    The thresholds are the medians over the night slots (22:00 to 06:00), which for the
    ranking is the single value it gives every slot; slot 0's without night slots.
    """
    pack, knobs = cfg.pack, cfg.knobs
    w, th, path = ranked.window, ranked.th, ranked.path
    today = local_date(now)
    tomorrow = today + dt.timedelta(days=1)
    night = (w.ts >= local_at(today, NIGHT_FROM_HOUR)) & (w.ts < local_at(tomorrow, NIGHT_TO_HOUR))
    nan = float("nan")
    take = at_quantile(w.take[:, night], knobs.q_take) if night.any() else np.zeros(0)
    feed = at_quantile(w.feed[:, night], knobs.q_feed) if night.any() else np.zeros(0)
    night_take = float(np.median(take)) if len(take) else nan
    night_feed = float(np.median(feed)) if len(feed) else nan
    pick = night if night.any() else np.arange(len(w.ts)) == 0
    charge_th = float(np.median(th.charge[pick]))
    sale_th = float(np.median(th.discharge[pick]))
    nxt = (w.ts >= local_midnight(tomorrow)) & (
        w.ts < local_midnight(tomorrow + dt.timedelta(days=1))
    )
    surplus = float(np.maximum(-path.net_kw[nxt], 0.0).sum() * 0.25)
    cap = pack.capacity_kwh
    low = soc_kwh <= pack.soc_floor * cap + 0.2 * cap
    high = soc_kwh >= 0.8 * cap and surplus >= 0.5 * cap
    return {
        "day": today.isoformat(),
        "soc_kwh": float(soc_kwh),
        "charge_threshold": charge_th,
        "sale_threshold": sale_th,
        "night_median_take": night_take,
        "night_median_feed": night_feed,
        "surplus_tomorrow_kwh": surplus,
        "low_soc_applies": bool(low),
        "low_soc_ok": not (low and charge_th < night_take),
        "high_soc_applies": bool(high),
        "high_soc_ok": not (high and sale_th > night_feed),
    }


# -- phase 6: closing -------------------------------------------------------------------


def drop_empty_days(pairs: Table, virtual: list[LedgerRow]) -> Table:
    """Drop the days whose every quarter is missing (no included quarter to compare)."""
    per_day = Counter(local_date(r.ts).isoformat() for r in virtual)
    keep = np.array(
        [per_day[d] > int(n) for d, n in zip(pairs["day"], pairs["n_missing"], strict=True)],
        dtype=bool,
    )
    return {k: np.asarray(v)[keep] for k, v in pairs.items()}


def result_pairs(house: list[LedgerRow], virtual: list[LedgerRow]) -> Table:
    """Per local day: ``result_eur`` = house money minus virtual money, and ``n_missing``.

    Positive means the shadow battery earned. Rows pair by ``ts``; a quarter missing on
    either path is excluded and counted in ``n_missing``. No Balans revenue is netted.
    Days whose every quarter is missing are dropped (``drop_empty_days``).
    """
    by_ts = {r.ts: r for r in virtual}
    sums: dict[dt.date, list[float]] = {}
    for h in house:
        v = by_ts.get(h.ts)
        if v is None:
            continue
        s = sums.setdefault(local_date(h.ts), [0.0, 0.0])
        if h.missing or v.missing:
            s[1] += 1
        else:
            s[0] += h.money_eur - v.money_eur
    days = sorted(sums)
    pairs = {
        "day": np.array([d.isoformat() for d in days], dtype=object),
        "result_eur": np.array([sums[d][0] for d in days], dtype=float),
        "n_missing": np.array([int(sums[d][1]) for d in days], dtype=np.int64),
    }
    return drop_empty_days(pairs, virtual)


def modal_tiers(tiers: list[tuple[int, str]]) -> dict[str, str]:
    """The most frequent slot-0 tier per local month (ties: the first in tier order)."""
    by_month: dict[str, Counter] = {}
    for qts, tier in tiers:
        by_month.setdefault(local_date(qts).strftime("%Y-%m"), Counter())[tier] += 1
    return {
        m: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for m, c in sorted(by_month.items())
    }


def wrong_side_count(decisions: list[Decision], prices: Prices, cfg: SiteConfig) -> int:
    """Committed quarters whose settled price lies on the wrong side, each counted once.

    A quarter counts when one of its committed decisions (``commit``, ``precommit``,
    ``negative_take``) has its quarter's settled price on the wrong side of the threshold
    that decision used: a charge when the settled buy lies above its charge threshold, a
    discharge when the settled sell lies below its discharge threshold, a ``negative_take``
    when ``buy / 1000 + fee_import >= 0``. Unsettled quarters do not count. One count per
    quarter keeps the number a per-ISP rate, comparable across rungs.
    """
    fee = cfg.terms.fee_import_eur_kwh
    wrong: set[int] = set()
    for d in decisions:
        if not d.committed:
            continue
        quarter = int(isp_start(d.ts))
        if quarter in wrong:
            continue
        buy, sell = prices.settled(quarter)
        if d.reason == "negative_take":
            bad = np.isfinite(buy) and buy / 1000.0 + fee >= 0.0
        elif d.action == "charge":
            bad = np.isfinite(buy) and buy > d.charge_threshold
        elif d.action == "discharge":
            bad = np.isfinite(sell) and sell < d.discharge_threshold
        else:
            bad = False
        if bad:
            wrong.add(quarter)
    return len(wrong)


def site_summary(
    cfg: SiteConfig,
    site: SiteModels,
    view: SiteView | None,
    learning_days: int,
    no_band_quarters: int,
    gross_bias_eur: float = 0.0,
    soc_end_anchor: dict | None = None,
    irradiance: str = IRRADIANCE_SOURCE,
) -> dict:
    """The site-model summary of the replay (the last state and the last EV statistics),
    where the terminal valuation read the recorded SoC (``end_soc_anchor``) and the source of
    the window's PV regressors (``irradiance``). With a shading profile on the champion,
    ``shading_share`` holds the share of the modelled energy it removes per string over the
    window's irradiance (shading.share_by_string); without one the key is absent, so plan-2
    summaries stay as they were."""
    state = site.state
    champ = state.champion
    pv = "none" if cfg.pv.none else ("planes_given" if cfg.pv.planes else "learned")
    stats = view.ev_stats if view is not None else None

    def _list(a) -> list[float]:
        return [float(x) for x in np.asarray(a, dtype=float)]

    out = {
        "pv": pv,
        "planes": [
            {"tilt_deg": float(p.tilt_deg), "azimuth_deg": float(p.azimuth_deg)}
            for p in (champ.planes if champ is not None else ())
        ],
        "scales_kwp": _list(champ.scales_kwp) if champ is not None else [],
        "holdout_error": None if champ is None else champ.holdout_error,
        "null_error": None if champ is None else champ.null_error,
        "stability": None if champ is None else tuple(float(x) for x in champ.stability),
        "learning_days": int(learning_days),
        "learned_days": int(state.learned_days),
        "drift": bool(any(site.drift_seen)),
        "ev": {
            "sessions": 0 if view is None else int(view.n_sessions),
            "p_session": [] if stats is None else _list(stats.p_session),
            "median_start_slot": [] if stats is None else _list(stats.median_start_slot),
            "median_energy_kwh": [] if stats is None else _list(stats.median_energy_kwh),
        },
        "irradiance": irradiance,
        "gen_handling": "learned",
        "no_band_quarters": int(no_band_quarters),
        "gross_split_bias_eur": float(gross_bias_eur),
        "soc_end_anchor": soc_end_anchor,
    }
    if champ is not None and champ.shading is not None:
        out["shading_share"] = share_by_string(champ, site.irr_q, cfg.latitude, cfg.longitude)
    return out


def actual_soc_kwh(
    q: Table, minute: Table | None, instant: int, capacity: float, at_end: bool
) -> float:
    """The recorded SoC (``soc_pct``) at ``instant`` in cells kWh, NaN without one.

    With minute flows: the minute row at ``instant`` for the start anchor, the minute row
    at ``instant - 60`` (the last minute) for the end anchor. Without: the quarter that
    ends at ``instant`` (``instant - 900``), so the value never averages a minute after the
    anchor except the start minute itself.
    """
    if minute is not None:
        table, at = minute, int(instant) - (60 if at_end else 0)
    else:
        table, at = q, int(instant) - 900
    k = row_index(np.asarray(table["ts"], dtype=np.int64), at)
    if k is None:
        return float("nan")
    return float(np.asarray(table["soc_pct"], dtype=float)[k]) / 100.0 * capacity


END_ANCHOR_REACH_S = 3600  # the end anchor falls back to the last finite SoC this far back


def _local_minute(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), ZONE).strftime("%Y-%m-%d %H:%M")


def end_soc_anchor(q: Table, minute: Table | None, end: int, capacity: float) -> tuple[float, dict]:
    """The recorded SoC (cells kWh) for the terminal valuation and where it was read.

    The anchor is ``actual_soc_kwh(..., end, at_end=True)``: the last minute, else the last
    quarter. When it is NaN, the last finite SoC of a row that ends within the final
    ``END_ANCHOR_REACH_S`` before ``end`` stands in (minute rows when given, else quarter
    rows). Returns ``(kwh, {"ts": row label, "minutes_before_end": end minus the row's
    end, in minutes})``. Raises ``ValueError`` naming the last finite SoC before ``end``
    (local time) when no row in that reach has one.
    """
    value = actual_soc_kwh(q, minute, end, capacity, at_end=True)
    table, step = (minute, 60) if minute is not None else (q, 900)
    ts = np.asarray(table["ts"], dtype=np.int64)
    soc = np.asarray(table["soc_pct"], dtype=float)
    if np.isfinite(value):
        return value, {"ts": int(end) - step, "minutes_before_end": 0}
    usable = np.isfinite(soc) & (ts + step <= int(end))
    near = usable & (ts + step >= int(end) - END_ANCHOR_REACH_S)
    if near.any():
        k = int(np.flatnonzero(near)[-1])
        lag = (int(end) - int(ts[k]) - step) // 60
        return float(soc[k]) / 100.0 * capacity, {"ts": int(ts[k]), "minutes_before_end": lag}
    last = (
        f"the last finite SoC is at {_local_minute(int(ts[np.flatnonzero(usable)[-1]]))}"
        if usable.any()
        else "the flows have no finite SoC before it"
    )
    raise ValueError(
        f"no recorded SoC at the end anchor ({_local_minute(end)}) or within the "
        f"{END_ANCHOR_REACH_S // 60} minutes before it; {last}"
    )


def anchored_first_day(
    q: Table,
    minute: Table | None,
    first: dt.date,
    last: dt.date,
    capacity: float,
    max_days: int | None = None,
) -> dt.date:
    """The first local day from ``first`` with a recorded SoC at its start anchor.

    The engine reads the recorded SoC at the start (``actual_soc_kwh``: the minute row at
    the start, else the quarter that ends there). An export that starts at the replay's
    first midnight has no quarter before it, so the start moves one day on per missing
    anchor, at most ``max_days`` times (None: up to ``last``). The end anchor is checked
    first (``end_soc_anchor`` raises). Raises ``ValueError`` when no day has a start anchor.
    """
    end = local_midnight(last + dt.timedelta(days=1))
    end_soc_anchor(q, minute, end, capacity)
    day, tries = first, 0
    while day <= last and (max_days is None or tries < max_days):
        if np.isfinite(actual_soc_kwh(q, minute, local_midnight(day), capacity, at_end=False)):
            return day
        day, tries = day + dt.timedelta(days=1), tries + 1
    reach = f"within {max_days} days of" if max_days is not None else "from"
    raise ValueError(f"no recorded SoC at the start anchor {reach} {first.isoformat()}")


def greed_table(rows: list[dict]) -> Table:
    out: Table = {}
    for c in GREED_COLUMNS:
        vals = [r[c] for r in rows]
        if c == "day":
            out[c] = np.array(vals, dtype=object)
        elif c.endswith(("_applies", "_ok")):
            out[c] = np.array(vals, dtype=bool)
        else:
            out[c] = np.array(vals, dtype=float)
    return out
