"""EV charging: session detection, weekday statistics, expected shape and plan placement."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

from ..timegrid import ISP_S, ZONE
from .contract import EvPlan
from .load import LoadModel, forecast_load_kw, local_fields

_DETECT_FRACTION = 0.9


@dataclass(frozen=True)
class Session:
    start_ts: int
    end_ts: int
    energy_kwh: float
    mean_kw: float


def detect_sessions(
    load_kw: np.ndarray,
    ts: np.ndarray,
    profile: LoadModel,
    charger_kw: float = 11.0,
    min_minutes: int = 45,
) -> list[Session]:
    """Runs of consecutive quarters whose excess load is at least 90 % of the charger."""
    load = np.asarray(load_kw, dtype=float)
    ts = np.asarray(ts, dtype=np.int64)
    if len(ts) == 0:
        return []
    excess = load - forecast_load_kw(profile, ts)
    hot = np.nan_to_num(excess, nan=-np.inf) >= _DETECT_FRACTION * charger_kw
    # a run breaks on a drop or on a gap in the time axis
    brk = np.concatenate(([True], (~hot[1:]) | (~hot[:-1]) | (np.diff(ts) != ISP_S)))
    starts = np.flatnonzero(brk & hot)
    min_q = max(min_minutes // 15, 1)
    out: list[Session] = []
    for i in starts:
        j = i + 1
        while j < len(ts) and hot[j] and ts[j] - ts[j - 1] == ISP_S:
            j += 1
        if j - i < min_q:
            continue
        energy = float(excess[i:j].sum() * 0.25)
        hours = (j - i) * 0.25
        out.append(Session(int(ts[i]), int(ts[i]) + (j - i) * ISP_S, energy, energy / hours))
    return out


def session_mask(sessions: list[Session], ts: np.ndarray) -> np.ndarray:
    """True where a timestamp lies inside a session."""
    ts = np.asarray(ts, dtype=np.int64)
    out = np.zeros(len(ts), dtype=bool)
    for s in sessions:
        out |= (ts >= s.start_ts) & (ts < s.end_ts)
    return out


@dataclass
class EvStats:
    p_session: np.ndarray
    median_start_slot: np.ndarray
    median_energy_kwh: np.ndarray


def ev_stats(sessions: list[Session], first_ts: int, last_ts: int) -> EvStats:
    """Per weekday: share of days with a session (at most 1), median start slot, median energy."""
    p = np.full(7, np.nan)
    start = np.full(7, np.nan)
    energy = np.full(7, np.nan)
    _, _, days = local_fields(np.arange(first_ts, last_ts + 1, 3600, dtype=np.int64))
    day_set = np.unique(days)
    n_days = np.bincount((day_set + 3) % 7, minlength=7)
    if sessions:
        s_ts = np.array([s.start_ts for s in sessions], dtype=np.int64)
        s_slot, s_dow, s_day = local_fields(s_ts)
        s_kwh = np.array([s.energy_kwh for s in sessions])
        for d in range(7):
            pick = s_dow == d
            if pick.any() and n_days[d] > 0:
                p[d] = len(np.unique(s_day[pick])) / n_days[d]
                start[d] = float(np.median(s_slot[pick]))
                energy[d] = float(np.median(s_kwh[pick]))
    return EvStats(p, start, energy)


def expected_ev_kw(stats: EvStats, slots_ts: np.ndarray, charger_kw: float) -> np.ndarray:
    """Expected EV load: p * charger_kw from the median start for the median energy.

    A session that runs past midnight spills into the early slots of the next day. Where the
    same-day rule and the spill-over rule both apply, the larger value wins.
    """
    slot, dow, _ = local_fields(slots_ts)

    def shape(d: np.ndarray, at: np.ndarray) -> np.ndarray:
        p = stats.p_session[d]
        start = stats.median_start_slot[d]
        quarters = np.ceil(stats.median_energy_kwh[d] / (charger_kw * 0.25) - 1e-9)
        ok = ~(np.isnan(p) | np.isnan(start) | np.isnan(quarters))
        s0 = np.where(ok, start, 0.0)
        q = np.where(ok, quarters, 0.0)
        inside = ok & (at >= s0) & (at < s0 + q)
        return np.where(inside, np.nan_to_num(p) * charger_kw, 0.0)

    return np.maximum(shape(dow, slot), shape((dow - 1) % 7, slot + 96))


def plan_deadline(plan: EvPlan, now: int) -> int | None:
    """Epoch s of the next by_local on an allowed weekday, strictly after now."""
    if not plan.days:
        return None
    hh, mm = (int(x) for x in plan.by_local.split(":")[:2])
    today = dt.datetime.fromtimestamp(now, ZONE).date()
    for k in range(9):
        day = today + dt.timedelta(days=k)
        if day.weekday() not in plan.days:
            continue
        cand = int(dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=ZONE).timestamp())
        if cand > now:
            return cand
    return None


def plan_slots(
    take_cost_eur_mwh: np.ndarray,
    slots_ts: np.ndarray,
    now: int,
    deadline: int,
    energy_kwh: float,
    charger_kw: float,
) -> np.ndarray:
    """Boolean mask of the cheapest slots in [now, deadline) that hold the planned energy."""
    cost = np.asarray(take_cost_eur_mwh, dtype=float)
    ts = np.asarray(slots_ts, dtype=np.int64)
    out = np.zeros(len(ts), dtype=bool)
    need = math.ceil(energy_kwh / (charger_kw * 0.25) - 1e-9)
    idx = np.flatnonzero((ts >= now) & (ts < deadline))
    if need <= 0 or len(idx) == 0:
        return out
    order = np.argsort(cost[idx], kind="stable")
    out[idx[order[:need]]] = True
    return out
