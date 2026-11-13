"""The replay engine's quarter and minute steps, callable from a replay or a live clock.

``replay.engine`` drives these steps over an archive; the live loop drives the same steps
from the wall clock, and re-execution rebuilds a live day from the stored inputs.

1. ``quarter_inputs``: at minute 1 the window, the load, PV and EV expectations and the EV
   plan mask (the model-heavy half; no SoC, no trader). ``QuarterInputs.da_known`` keeps
   the day-ahead mask the window was built with, so the window can be rebuilt from it.
2. ``rank_quarter``: the SoC path from the SoC at minute 1, the need, ``Trader.rank`` and
   the nowcast.
3. ``reread_nowcast``: the nowcast at elapsed 5 and 10.
4. ``Stepper.minute``: one minute of the trader, the executor and the virtual battery.
5. ``book_paths``: the virtual, actual and house ledger rows of one quarter.

This module imports nothing from ``replay.engine`` (the engine imports it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..core.battery import Executor, VirtualBattery, execution_profile
from ..core.features import ELAPSED_MIN
from ..core.ledger import LedgerRow, rows_for_quarter
from ..core.pricemodel import QUANTILES
from ..core.rule.need import need_kwh
from ..core.rule.trader import COMMITTED_REASONS, Decision, Trader
from ..core.rule.window import Nowcast, Window, _widen, build_window, nowcast, window_slots
from ..core.site.contract import Knobs, Pack, SiteConfig
from ..core.site.ev import expected_ev_kw, plan_deadline, plan_slots
from ..core.site.flows import ac_legs
from ..core.site.load import forecast_load_kw, local_fields
from ..core.site.registry import need_widening
from ..core.site.separate import predict_pv_kw
from ..core.site.socpath import soc_path
from ..core.timegrid import Table, align, isp_start
from .phases import IRR_COLS, QuarterRun, Ranked, SiteView, book_quarter, unpriced

if TYPE_CHECKING:
    from .irradiance import IrrByLead

_MIN_S = 60
DA_PUBLISHED_HOUR = 13  # the day-ahead auction result is public from about 13:00 local
NOWCAST_REREAD_MIN = tuple(e for e in ELAPSED_MIN if e > 1)  # (5, 10)


def da_known_mask(now: int, slots_ts: np.ndarray) -> np.ndarray:
    """True for slots on the local day of ``now`` or earlier, and for tomorrow's slots from
    13:00 local on. Never for the day after tomorrow."""
    wall, _, day = local_fields(np.array([int(now)], dtype=np.int64))
    last = int(day[0]) + (1 if int(wall[0]) >= DA_PUBLISHED_HOUR * 4 else 0)
    _, _, slot_day = local_fields(np.asarray(slots_ts, dtype=np.int64))
    return slot_day <= last


def _grid_batt_kw(table: Table, pack: Pack, per_hour: float) -> tuple[np.ndarray, np.ndarray]:
    """``(grid net kW, battery AC kW)`` per flows row, import and charge positive.

    ``grid = (import - export) * per_hour`` and ``batt = (charge_ac - discharge_ac) *
    per_hour``, through ``ac_legs`` for a dc-side pack (as ``residual_kw``). ``per_hour`` is
    60 for minute rows and 4 for quarter rows. Both ledger paths are built from these.
    """
    charge = np.asarray(table["batt_charge_kwh"], dtype=float)
    discharge = np.asarray(table["batt_discharge_kwh"], dtype=float)
    if pack.side == "dc":
        ec, ed = ac_legs(pack)
        charge, discharge = charge / ec, discharge * ed
    imp = np.asarray(table["grid_import_kwh"], dtype=float)
    exp = np.asarray(table["grid_export_kwh"], dtype=float)
    return (imp - exp) * per_hour, (charge - discharge) * per_hour


def _ev_passthrough_kwh(residual: float, batt_ac: float, charger_kw: float) -> float:
    """EV energy of one plan-slot minute that the pack did not serve.

    The EV share of the residual is ``min(max(residual, 0), charger_kw)``; the pack serves
    the EV only with what it delivers beyond the rest of the house load.
    """
    ev_load = min(max(residual, 0.0), charger_kw)
    rest = max(residual - charger_kw, 0.0)
    served = min(max(max(-batt_ac, 0.0) - rest, 0.0), ev_load)
    return (ev_load - served) / 60.0


# public names for the live loop (``live.site``); the engine keeps the plan-2 private names
grid_batt_kw = _grid_batt_kw
ev_passthrough_kwh = _ev_passthrough_kwh

# the flows columns a minute needs; a stored row without one of them is a missing minute
FLOW_REQUIRED = (
    "grid_import_kwh", "grid_export_kwh", "batt_charge_kwh", "batt_discharge_kwh", "soc_pct",
)


def minute_flows(flows: dict | None, pack: Pack) -> tuple[float, float, float, bool]:
    """``(grid kW, battery AC kW, recorded SoC %, missing)`` of one stored minute row.

    No row, or a non-finite ``FLOW_REQUIRED`` column, is a missing minute with NaN values.
    The live minute (``live.site``) and the verifier (``verify.check``) both read a minute
    through this function.
    """
    nan = float("nan")
    if flows is None or any(not np.isfinite(float(flows[c])) for c in FLOW_REQUIRED):
        return nan, nan, nan, True
    one = {c: np.array([float(flows[c])]) for c in FLOW_REQUIRED}
    g, b = _grid_batt_kw(one, pack, 3600.0 / _MIN_S)
    return float(g[0]), float(b[0]), float(flows["soc_pct"]), False


# -- quarter steps ----------------------------------------------------------------------


@dataclass
class QuarterInputs:
    """What the ranking of one quarter reads besides the SoC, at ``now`` (minute 1)."""

    now: int
    window: Window
    load_kw: np.ndarray
    pv_kw: np.ndarray
    ev_kw: np.ndarray
    plan_mask: np.ndarray
    widening: float
    healthy: tuple[str, ...]
    da_known: np.ndarray  # bool per window slot, as the window was built (holes included)


def quarter_inputs(
    now: int,
    source,
    cfg: SiteConfig,
    view: SiteView,
    irr: Table,
    da_known: np.ndarray | None = None,
    irr_forecast: IrrByLead | None = None,
) -> QuarterInputs | None:
    """Window, load, PV and EV expectations and the EV plan mask at ``now``.

    ``source`` provides ``bands``, ``tables_at`` and ``healthy_tiers``. ``da_known``
    defaults to ``da_known_mask(now, window_slots(now))``; a stored mask rebuilds the same
    window. None when slot 0 has no band (``build_window`` raises ``ValueError``).
    The PV regressors come from ``irr_forecast.at(now, window.ts)`` when given (the
    Previous Runs lead a live box had at ``now``), else from ``irr`` on the window slots.
    """
    qts = int(isp_start(now))
    knobs = cfg.knobs
    if da_known is None:
        da_known = da_known_mask(now, window_slots(now))
    da_known = np.asarray(da_known, dtype=bool)
    healthy = tuple(source.healthy_tiers(qts))
    try:
        window = build_window(now, source, healthy, da_known, source.tables_at(qts), knobs)
    except ValueError:
        return None
    load = forecast_load_kw(view.load, window.ts)
    if view.state.champion is not None:
        regressors = (
            irr_forecast.at(now, window.ts)
            if irr_forecast is not None
            else align(irr, window.ts, IRR_COLS)
        )
        pv = predict_pv_kw(view.state.champion, regressors, cfg.latitude, cfg.longitude)
    else:
        pv = np.zeros(len(window.ts))
    ev = np.zeros(len(window.ts))
    mask = np.zeros(len(window.ts), dtype=bool)
    if cfg.ev.plan is not None:
        deadline = plan_deadline(cfg.ev.plan, now)
        if deadline is not None:
            take_mid = window.take[QUANTILES.index(0.5)]
            mask = plan_slots(
                take_mid, window.ts, qts, deadline, cfg.ev.plan.energy_kwh, cfg.ev.charger_kw
            )
    elif view.ev_stats is not None:
        ev = expected_ev_kw(view.ev_stats, window.ts, cfg.ev.charger_kw)
    return QuarterInputs(
        now=int(now),
        window=window,
        load_kw=load,
        pv_kw=pv,
        ev_kw=ev,
        plan_mask=mask,
        widening=need_widening(view.state, knobs),
        healthy=healthy,
        da_known=da_known,
    )


def rank_quarter(
    qi: QuarterInputs, soc_kwh: float, cfg: SiteConfig, loss, trader: Trader
) -> Ranked:
    """SoC path from ``soc_kwh``, the need, ``trader.rank`` (stores the ranking) and the nowcast."""
    w = qi.window
    path = soc_path(soc_kwh, cfg.pack, loss, w.ts, qi.load_kw, qi.pv_kw, qi.ev_kw, qi.plan_mask)
    need = need_kwh(path, soc_kwh, cfg.pack, widening=qi.widening)
    th = trader.rank(w, path, need, qi.now)
    return Ranked(w, path, th, nowcast(w, cfg.knobs), qi.plan_mask)


def reread_nowcast(source, ranked: Ranked, qts: int, elapsed: int, knobs: Knobs) -> Nowcast:
    """The h0 cell ``(tier, 0, elapsed)`` of the slot-0 tier as a nowcast, else the previous.

    Raw quantiles widened about q50 by the cell's ``cov``, as ``build_window`` does.
    """
    got = source.bands(ranked.window.tier[0], 0, elapsed, qts)
    if got is None:
        return ranked.nowcast
    cov = float(got["cov"])
    take = np.array([got["take"][q] for q in QUANTILES], float)
    feed = np.array([got["feed"][q] for q in QUANTILES], float)
    p_dual = float(got["p_dual"])
    if not (np.isfinite(cov) and np.isfinite(take).all() and np.isfinite(feed).all()):
        return ranked.nowcast
    one = Window(
        ts=np.array([qts], dtype=np.int64),
        take=_widen(take, cov)[:, None],
        feed=_widen(feed, cov)[:, None],
        p_dual=np.array([p_dual]),
        dual_spread=np.array([np.nan]),
        predicted=np.zeros(1, dtype=bool),
        tier=[ranked.window.tier[0]],
        elapsed_min=elapsed,
    )
    return nowcast(one, knobs)


# -- minute step ------------------------------------------------------------------------


@dataclass
class MinuteOut:
    """One minute: the decision, the virtual grid net, the house net without a battery
    (``grid - batt_actual``) and the virtual battery's AC flow. NaN kW on a missing minute."""

    decision: Decision
    net_kw: float
    house_kw: float
    batt_ac_kw: float


class Stepper:
    """The minute-level state that runs across quarters: trader, executor and battery."""

    def __init__(self, cfg: SiteConfig, loss, soc0_kwh: float):
        self.cfg = cfg
        self.loss = loss
        self.reanchor(soc0_kwh)

    def reanchor(self, soc_kwh: float) -> None:
        """A fresh trader, executor and battery at ``soc_kwh``, and no EV plan quarters."""
        cfg = self.cfg
        self.trader = Trader(cfg, self.loss)
        self.executor = Executor(execution_profile(cfg.execution_profile, cfg.knobs))
        self.battery = VirtualBattery(cfg.pack, self.loss, soc_kwh, standby=cfg.knobs.standby)
        self.plan_quarters: set[int] = set()

    def set_soc(self, soc_kwh: float) -> None:
        """A fresh battery at ``soc_kwh``; the trader, the executor and the plan stay."""
        cfg = self.cfg
        self.battery = VirtualBattery(cfg.pack, self.loss, soc_kwh, standby=cfg.knobs.standby)

    def set_plan(self, ranked: Ranked | None) -> None:
        """The EV plan quarters of a new ranking. None (no ranking) keeps the previous."""
        if ranked is not None:
            planned = ranked.window.ts[ranked.plan_mask]
            self.plan_quarters = {int(t) for t in planned}

    def minute(
        self,
        ts: int,
        nowcast: Nowcast | None,
        grid_kw: float,
        batt_kw: float,
        missing: bool,
        learning: bool,
    ) -> MinuteOut:
        """The minute at ``ts``: ``grid_kw`` and ``batt_kw`` are the recorded grid net and
        battery AC flow (kW, import and charge positive).

        A missing minute holds: the executor gets 0 kW, the battery is not stepped (its SoC
        stays, no standby drain) and the decision is ``hold`` with reason ``missing``.
        The trader sees the bus net without the battery, ``grid - batt_actual``. The virtual
        grid net is ``grid + (batt_virtual - batt_actual)``, so a virtual battery that moves
        as the recorded one gives the recorded grid net bit for bit.
        """
        nan = float("nan")
        if missing:
            self.executor.request(ts, 0.0, grid_trade=False)
            d = Decision(ts, "hold", 0.0, False, "missing", nan, nan, nan, nan, learning)
            return MinuteOut(d, nan, nan, nan)
        grid, batt = float(grid_kw), float(batt_kw)
        res = grid - batt
        plan_slot = int(isp_start(ts)) in self.plan_quarters
        d = self.trader.minute(ts, nowcast, res, self.battery.soc_kwh, plan_slot, learning)
        grid_trade = d.reason in COMMITTED_REASONS
        delivered = self.executor.request(ts, d.kw, grid_trade)
        ac = self.battery.step(delivered, _MIN_S)
        return MinuteOut(d, grid + (ac - batt), res, ac)


# -- booking ----------------------------------------------------------------------------


def book_paths(
    qts: int,
    outs: list[MinuteOut],
    grid_kw: np.ndarray,
    prices: dict,
    cfg: SiteConfig,
    learning: bool,
    missing: bool,
    ev_passthrough_kwh: float,
) -> tuple[LedgerRow, LedgerRow, LedgerRow]:
    """Virtual, actual and house rows of one quarter, each the ``gross_split`` of minute nets.

    Virtual and actual are ``phases.book_quarter``. The house row books the same house
    without a battery, ``grid - batt_actual`` per minute, with the same money path and the
    same ``missing`` rule (a quarter without a finite buy and sell price is missing).
    The house row has no EV passthrough (0 kWh): it has no pack to serve the EV.
    """
    run = QuarterRun(
        net_kw=np.array([o.net_kw for o in outs], dtype=float),
        grid_kw=np.asarray(grid_kw, dtype=float).copy(),
        ev_passthrough_kwh=ev_passthrough_kwh,
        decisions=[o.decision for o in outs],
        house_kw=np.array([o.house_kw for o in outs], dtype=float),
        batt_ac_kw=np.array([o.batt_ac_kw for o in outs], dtype=float),
    )
    virtual, actual = book_quarter(qts, run, prices, cfg, learning, missing)
    house = rows_for_quarter(
        qts,
        "house",
        run.house_kw,
        prices,
        cfg.terms,
        learning,
        0.0,
        missing or unpriced(prices),
    )
    return virtual, actual, house
