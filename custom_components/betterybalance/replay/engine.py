"""Replay engine: the site models, the window, the trader and the virtual battery, minute by minute.

``replay`` walks an archive quarter by quarter in six phases, each a function below:

1. ``SiteModels``: the PV registry walks forward (``weekly_update`` at the start and at every
   local Monday 00:00, on rows strictly before that instant), the load median and the EV
   expectation are rebuilt at every quarter on rows strictly before the quarter.
2. ``steps.quarter_inputs`` and ``steps.rank_quarter``: at minute 1 the window, the load,
   PV and EV expectations, the SoC path, the need and the ranking.
3. ``steps.Stepper.minute``: the nowcast (re-read at elapsed 5 and 10 by
   ``steps.reread_nowcast``), ``Trader.minute``, the executor and the virtual battery; the
   virtual net per minute.
4. ``steps.book_paths``: one ledger row per path from the settled (else live) prices. All
   paths are the same ``gross_split`` of minute grid nets: the actual one of the recorded
   grid net, the virtual one of ``grid + (batt_virtual - batt_actual)``, the house one of
   ``grid - batt_actual`` (the same house without a battery).
5. ``greed_row``: at 18:00 local, the thresholds against the coming night's prices.
6. ``_close``: terminal valuation, daily pairs, the shadow result per day (house minus
   virtual), the bootstrap interval and the summaries.

Phases 2 to 4 are the live-callable steps of ``replay.steps``; phases 1, 5 and 6 keep their
helpers in ``replay.phases``.

Honesty: every model a quarter sees was fitted on rows before that quarter, every band comes
from the walk-forward cache (``replay.bands``) at an honest horizon, and the ledger reads the
settled prices of the quarter itself only to book money, never to decide. The PV
expectation reads ``ReplayInputs.irr_forecast`` when given (``replay.irradiance.IrrByLead``:
per slot the open-meteo Previous Runs lead a live box had at the decision time, the archive
before 2024, beyond lead 3 and where a lead has no value; ``site_report["irradiance"] =
irr_forecast.note()``). Without it the irradiance archive (the open-meteo historical-forecast
API's archived model output for the day itself, ERA5 where missing) stands in for the
forecast, also for the slots 1 to 3 days ahead where a live box has only a multi-day
forecast; that PV expectation is optimistic and the bias favours the virtual path
(``site_report["irradiance"] = phases.IRRADIANCE_SOURCE``). The site-model fits read the
archive in both cases.
The replay learns all PV; GEN-port production is validation data only: the engine never
reads ``gen_kwh`` (the site models see it as 0, so their residual is ``grid - batt`` like
the trader's), and ``gen_kwh`` stays in the flows for the validation tasks
(``site_report["gen_handling"] = "learned"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

from ..core.battery import load_profile
from ..core.ledger import (
    LedgerRow,
    bootstrap_interval,
    daily_pairs,
    terminal_valuation,
    trailing_mean_buy,
)
from ..core.rule.trader import Decision
from ..core.rule.window import band_for_slot
from ..core.site.contract import Knobs, SiteConfig
from ..core.site.flows import ac_legs
from ..core.site.load import local_fields
from ..core.timegrid import ISP_S, Table, isp_start
from .bands import BandCache, CachedBandSource
from .irradiance import IrrByLead
from .phases import (
    DAY_S,
    Prices,
    Ranked,
    SiteModels,
    SiteView,
    actual_soc_kwh,
    drop_empty_days,
    end_soc_anchor,
    greed_row,
    greed_table,
    gross_split_bias_eur,
    local_date,
    modal_tiers,
    result_pairs,
    row_index,
    site_summary,
    wrong_side_count,
)
from .steps import (
    DA_PUBLISHED_HOUR,  # noqa: F401 (re-exported: the engine's plan-2 name)
    NOWCAST_REREAD_MIN,
    MinuteOut,
    Stepper,
    _ev_passthrough_kwh,
    _grid_batt_kw,
    book_paths,
    da_known_mask,  # noqa: F401 (re-exported: the engine's plan-2 name)
    quarter_inputs,
    rank_quarter,
    reread_nowcast,
)

_MIN_S = 60
_MINUTES = ISP_S // _MIN_S
# the columns a flows row cannot lack (core.site.flows._REQUIRED)
_REQUIRED_FLOWS = (
    "grid_import_kwh",
    "grid_export_kwh",
    "batt_charge_kwh",
    "batt_discharge_kwh",
    "soc_pct",
)
GREED_HOUR = 18
TRAILING_BUY_DAYS = 7


@dataclass
class ReplayInputs:
    cfg: SiteConfig
    flows_quarter: Table
    master: Table
    cache: BandCache
    irr: Table
    start: int
    end: int
    flows_minute: Table | None = None
    seed: int = 0
    soc0_kwh: float | None = None  # None: the actual SoC at start
    irr_forecast: IrrByLead | None = None  # None: the window's PV regressors read ``irr``


@dataclass
class ReplayResult:
    virtual: list[LedgerRow]
    actual: list[LedgerRow]
    house: list[LedgerRow]  # the same house without a battery (grid - batt_actual)
    decisions: list[Decision]
    soc_virtual_kwh: np.ndarray  # per quarter, the virtual SoC at the quarter's end
    greed: Table
    site_report: dict
    tiers_by_month: dict[str, str]
    learning_days: int
    terminal_eur: float
    pairs: Table
    result_pairs: Table  # per local day: house minus virtual money, positive is earned
    interval: tuple[float, float, float]
    cycles_per_day: float
    wrong_side: int
    knobs: Knobs
    # per quarter with a ranking: slot 0's take and feed quantiles (``window.band_for_slot``)
    quarter_bands: dict[int, dict] = field(default_factory=dict)
    # per decision (rows of ``decisions``): the virtual SoC before the minute (kWh) and the
    # nowcast take and feed points the trader saw (NaN without a nowcast)
    minute_marks: np.ndarray | None = None


# -- the replay -------------------------------------------------------------------------


def _start_check(cache: BandCache, start: int) -> None:
    k = row_index(np.asarray(cache.ts, dtype=np.int64), start)
    if k is None or int(cache.fold_of[k]) < 0:
        raise ValueError(
            f"replay start {start} has no priced band row (fold_of -1 or off the cache grid); "
            "start the band cache earlier"
        )


def _minute_flows(inputs: ReplayInputs, cfg: SiteConfig):
    """``(ts, grid kW, batt kW, bad)`` per minute row, or None without minute flows."""
    m = inputs.flows_minute
    if m is None:
        return None
    bad = np.zeros(len(m["ts"]), dtype=bool)
    for c in _REQUIRED_FLOWS:
        bad |= np.isnan(np.asarray(m[c], dtype=float))
    grid, batt = _grid_batt_kw(m, cfg.pack, 3600.0 / _MIN_S)
    return np.asarray(m["ts"], dtype=np.int64), grid, batt, bad


def _quarter_flows(
    qts: int, k: int | None, site: SiteModels, minute, q_grid: np.ndarray, q_batt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The 15 minute grid and battery flows (kW) of the quarter at ``qts`` and ``missing``.

    Missing: no quarter row, the quarter row is flagged, its recorded grid gross (or,
    without minute flows, its battery flow) is NaN, or (with minute flows) a minute row is
    absent or has a NaN in a required column. Without minute flows the quarter values
    stand flat for every minute, on both paths. A missing quarter has NaN flows.
    """
    nan = np.full(_MINUTES, np.nan)
    if k is None or bool(site.missing[k]) or not np.isfinite(q_grid[k]):
        return nan, nan, True
    if minute is None:
        if not np.isfinite(q_batt[k]):
            return nan, nan, True
        return np.full(_MINUTES, float(q_grid[k])), np.full(_MINUTES, float(q_batt[k])), False
    m_ts, m_grid, m_batt, m_bad = minute
    j = row_index(m_ts, qts)
    if j is None or j + _MINUTES > len(m_ts) or int(m_ts[j + _MINUTES - 1]) != qts + ISP_S - _MIN_S:
        return nan, nan, True
    if bool(m_bad[j : j + _MINUTES].any()):
        return nan, nan, True
    return m_grid[j : j + _MINUTES], m_batt[j : j + _MINUTES], False


def _close(
    inputs: ReplayInputs,
    cfg: SiteConfig,
    start: int,
    end: int,
    stepper: Stepper,
    prices: Prices,
    site: SiteModels,
    trace: dict,
) -> ReplayResult:
    """Phase 6: terminal valuation, pairs without empty days, the daily shadow result
    (``phases.result_pairs``), the interval and the summaries."""
    pack, virtual = cfg.pack, trace["virtual"]
    _, ed = ac_legs(pack)
    soc_actual_end, end_anchor = end_soc_anchor(
        inputs.flows_quarter, inputs.flows_minute, end, pack.capacity_kwh
    )
    trailing = trailing_mean_buy(prices.buy, prices.ts, end, TRAILING_BUY_DAYS)
    terminal = terminal_valuation(stepper.battery.soc_kwh, soc_actual_end, trailing, ed)
    pairs = drop_empty_days(daily_pairs(virtual, trace["actual"], cfg.terms), virtual)
    interval = bootstrap_interval(pairs["diff_eur"], seed=inputs.seed)
    days = (end - start) / DAY_S
    cycles = stepper.battery.throughput_kwh / (2.0 * pack.capacity_kwh) / days if days > 0 else 0.0
    learning_days = len(trace["learning_days"])
    return ReplayResult(
        virtual=virtual,
        actual=trace["actual"],
        house=trace["house"],
        decisions=trace["decisions"],
        soc_virtual_kwh=np.asarray(trace["soc"], dtype=float),
        greed=greed_table(trace["greed"]),
        site_report=site_summary(
            cfg,
            site,
            trace["view"],
            learning_days,
            trace["no_band"],
            trace["gross_bias"],
            end_anchor,
            **({} if inputs.irr_forecast is None else {"irradiance": inputs.irr_forecast.note()}),
        ),
        tiers_by_month=modal_tiers(trace["tiers"]),
        learning_days=learning_days,
        terminal_eur=float(terminal),
        pairs=pairs,
        result_pairs=result_pairs(trace["house"], virtual),
        interval=tuple(float(x) for x in interval),
        cycles_per_day=float(cycles),
        wrong_side=wrong_side_count(trace["decisions"], prices, cfg),
        knobs=cfg.knobs,
        quarter_bands=trace["bands"],
        minute_marks=np.asarray(trace["marks"], dtype=float).reshape(-1, 3),
    )


def _rank_at_minute_1(
    qts: int,
    source: CachedBandSource,
    cfg: SiteConfig,
    loss,
    stepper: Stepper,
    view: SiteView,
    irr: Table,
    greed_due: bool,
    trace: dict,
    irr_forecast: IrrByLead | None = None,
) -> Ranked | None:
    """Phases 2 and 5 at minute 1: ``quarter_inputs``, ``rank_quarter`` on the virtual SoC
    and, at 18:00, the greed row. None when slot 0 has no band (counted in ``no_band``)."""
    now = qts + _MIN_S
    qi = quarter_inputs(now, source, cfg, view, irr, irr_forecast=irr_forecast)
    if qi is None:
        trace["no_band"] += 1
        return None
    soc_kwh = stepper.battery.soc_kwh
    ranked = rank_quarter(qi, soc_kwh, cfg, loss, stepper.trader)
    trace["tiers"].append((qts, ranked.window.tier[0]))
    trace["bands"][qts] = band_for_slot(ranked.window, 0)
    if greed_due:
        trace["greed"].append(greed_row(now, ranked, soc_kwh, cfg))
    return ranked


def _run_quarter(
    qts: int,
    grid_min: np.ndarray,
    batt_min: np.ndarray,
    missing: bool,
    learning: bool,
    stepper: Stepper,
    cfg: SiteConfig,
    source: CachedBandSource,
    rank,
    marks: list | None = None,
) -> tuple[list[MinuteOut], float]:
    """Phase 3: the fifteen minutes of the quarter at ``qts`` and its EV passthrough (kWh).

    ``rank()`` runs at minute 1 and returns the ranking (or None, which keeps the previous
    EV plan quarters); the nowcast is re-read at elapsed 5 and 10. Minute 0 has no nowcast
    of this quarter (the trader falls back to the stored ranking). A missing quarter holds
    (``Stepper.minute``), but the ranking at minute 1 still runs. ``marks`` collects per
    minute the SoC before it and the nowcast points (``ReplayResult.minute_marks``).
    """
    outs: list[MinuteOut] = []
    ranked, nc, ev_kwh = None, None, 0.0
    for e in range(_MINUTES):
        ts = qts + e * _MIN_S
        if e == 1:
            ranked = rank()
            stepper.set_plan(ranked)
            if ranked is not None:
                nc = ranked.nowcast
        elif e in NOWCAST_REREAD_MIN and ranked is not None:
            nc = reread_nowcast(source, ranked, qts, e, cfg.knobs)
            ranked.nowcast = nc
        if marks is not None:
            nan = float("nan")
            take, feed = (nan, nan) if nc is None else (nc.take_point, nc.feed_point)
            marks.append((stepper.battery.soc_kwh, take, feed))
        out = stepper.minute(ts, nc, grid_min[e], batt_min[e], missing, learning)
        if not missing and qts in stepper.plan_quarters:
            ev_kwh += _ev_passthrough_kwh(out.house_kw, out.batt_ac_kw, cfg.ev.charger_kw)
        outs.append(out)
    return outs, ev_kwh


def replay(inputs: ReplayInputs, knobs: Knobs | None = None, progress=None) -> ReplayResult:
    """Replay ``[start, end)`` quarter by quarter (phases in the module docstring).

    ``knobs`` overrides ``cfg.knobs`` for this run. ``progress(done, total)`` is called after
    every quarter. Raises ``ValueError`` when the start row has no band in the cache.
    ``soc0_kwh`` defaults to the recorded SoC at ``start`` (``phases.actual_soc_kwh``: the
    minute row at ``start``, else the quarter that ends at ``start``), the floor without
    one, clamped to the band (the CLI and the acceptance tool move the start to a day with a
    recorded SoC, ``phases.anchored_first_day``). The terminal valuation reads the recorded
    SoC at ``end`` the same way (the last minute, else the last quarter), else the last
    finite SoC within the final 60 minutes (``phases.end_soc_anchor``, recorded in
    ``site_report["soc_end_anchor"]``); without one it raises ``ValueError`` before the
    first quarter.
    Deterministic: the only random draw is the bootstrap, seeded with ``inputs.seed``.
    """
    cfg = inputs.cfg if knobs is None else replace(inputs.cfg, knobs=knobs)
    start, end = int(isp_start(inputs.start)), int(inputs.end)
    _start_check(inputs.cache, start)
    pack = cfg.pack
    end_soc_anchor(inputs.flows_quarter, inputs.flows_minute, end, pack.capacity_kwh)  # fail early
    loss = load_profile(cfg.loss_profile, pack)
    source = CachedBandSource(inputs.cache)
    prices = Prices(inputs.master)
    site = SiteModels(inputs.flows_quarter, inputs.irr, start, end, cfg)
    q = inputs.flows_quarter
    q_imp = np.asarray(q["grid_import_kwh"], dtype=float)
    q_exp = np.asarray(q["grid_export_kwh"], dtype=float)
    minute = _minute_flows(inputs, cfg)
    q_grid, q_batt = _grid_batt_kw(q, pack, 3600.0 / ISP_S)
    floor, ceiling = pack.soc_floor * pack.capacity_kwh, pack.soc_ceiling * pack.capacity_kwh
    soc0 = inputs.soc0_kwh
    if soc0 is None:
        soc0 = actual_soc_kwh(q, inputs.flows_minute, start, pack.capacity_kwh, at_end=False)
        soc0 = floor if not np.isfinite(soc0) else soc0
    stepper = Stepper(cfg, loss, min(max(float(soc0), floor), ceiling))

    quarters = np.arange(start, end, ISP_S, dtype=np.int64)
    walls, _, _ = local_fields(quarters)
    trace = {
        "virtual": [],
        "actual": [],
        "house": [],
        "decisions": [],
        "greed": [],
        "soc": [],
        "tiers": [],
        "learning_days": set(),
        "bands": {},
        "marks": [],
        "no_band": 0,
        "gross_bias": 0.0,
        "view": None,
    }
    for i, qts in enumerate(quarters.tolist()):
        view = trace["view"] = site.at(qts)  # phase 1
        learning = bool(view.state.learning)
        k = row_index(site.ts, qts)
        grid_min, batt_min, missing = _quarter_flows(qts, k, site, minute, q_grid, q_batt)
        greed_due = int(walls[i]) == GREED_HOUR * 4

        def rank(qts=qts, view=view, greed_due=greed_due) -> Ranked | None:
            return _rank_at_minute_1(  # phases 2 and 5
                qts,
                source,
                cfg,
                loss,
                stepper,
                view,
                inputs.irr,
                greed_due,
                trace,
                inputs.irr_forecast,
            )

        outs, ev_kwh = _run_quarter(  # phase 3
            qts, grid_min, batt_min, missing, learning, stepper, cfg, source, rank, trace["marks"]
        )
        trace["decisions"].extend(o.decision for o in outs)
        trace["soc"].append(stepper.battery.soc_kwh)
        v_row, a_row, h_row = book_paths(  # phase 4
            qts, outs, grid_min, prices.at(qts), cfg, learning, missing, ev_kwh
        )
        if k is not None:
            recorded = (float(q_imp[k]), float(q_exp[k]))
            trace["gross_bias"] += gross_split_bias_eur(recorded, a_row, cfg.terms)
        trace["virtual"].append(v_row)
        trace["actual"].append(a_row)
        trace["house"].append(h_row)
        if learning:
            trace["learning_days"].add(local_date(qts).isoformat())
        if progress is not None:
            progress(i + 1, len(quarters))
    return _close(inputs, cfg, start, end, stepper, prices, site, trace)  # phase 6


# -- tables -----------------------------------------------------------------------------


def _table(rows: list, cls) -> Table:
    """Columns of a dataclass list: int64, float64, bool or object by the field type."""
    out: Table = {}
    for f in fields(cls):
        vals = [getattr(r, f.name) for r in rows]
        kind = {"int": np.int64, "float": np.float64, "bool": bool}.get(str(f.type), object)
        out[f.name] = np.array(vals, dtype=kind)
    return out


def decisions_table(decisions: list[Decision]) -> Table:
    """One column per ``Decision`` field. Non-finite margins and thresholds stay as they are."""
    return _table(decisions, Decision)


def ledger_table(rows: list[LedgerRow]) -> Table:
    """One column per ``LedgerRow`` field. NaN money of missing quarters stays NaN."""
    return _table(rows, LedgerRow)
