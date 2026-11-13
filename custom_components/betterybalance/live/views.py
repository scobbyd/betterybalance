"""What the entities show, computed from ``LiveSite.status()`` and the current ranking.

Pure and free of Home Assistant. Prices are EUR/MWh in the core and ct/kWh here, at the
sensor boundary (``ct_kwh``). Attributes are machine values (plain decimals); the sentences
come from ``live.copy``. The UI shows the shadow's own result only: nothing here reads the
``actual`` path (design 10.4).

- ``plan_path``: the planned battery over the ranked window, through the pack's limits and
  loss profile: under ``dp`` the trader's own policy walked forward at the planned SoC, under
  ``ranking`` an approximation from the charge and sale slots.
- ``plan_value``: that plan valued at one quantile of the take and feed bands, house money
  minus shadow money (positive is earned), plus the energy the plan leaves in the pack at the
  rule's terminal value (``dp_terminal_value``). The plan sensor's state is the value at q50 over
  the first ``n`` slots, its range the values at q10 and q90 with the plan held fixed
  (design 13.2), widened to hold the q50 value (``plan_view``).
- ``state_view``: the state sensor (the decision, its verdict, the next trade window, the
  last trade, the ledger head and the status flags).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core.battery import LossProfile, VirtualBattery, default_policy_kw
from ..core.ledger import money_eur
from ..core.pricemodel import QUANTILES
from ..core.rule.thresholds import DP_ACTIONS, dp_policy, dp_terminal_value, soc_grid
from ..core.site.contract import Knobs, Pack, Terms, tax_mode_at
from ..core.site.plant import plane_shape
from ..core.site.separate import PvModel, _local_month
from ..core.site.shading import apply_shading
from ..core.timegrid import ISP_S, Table, align, isp_start
from ..replay.phases import IRR_COLS, Ranked, at_quantile
from ..replay.report import ATTRIBUTION
from . import copy as CP

PLAN_SLOTS = 144  # 36 hours
SLOT_H = ISP_S / 3600.0
ACTIONS = ("hold", "charge", "discharge")


def ct_kwh(eur_mwh: float) -> float:
    """EUR/MWh to ct/kWh."""
    return float(eur_mwh) / 10.0


def _ct(x: float, dec: int = 2) -> float | None:
    """ct/kWh of an EUR/MWh value, rounded; None for a non-finite one."""
    x = float(x)
    return round(ct_kwh(x), dec) if math.isfinite(x) else None


@dataclass
class PlannedPath:
    action: np.ndarray  # per slot: "charge", "discharge" (a grid trade) or "hold"
    kw: np.ndarray  # per slot: the pack's AC kW, charge positive
    soc_kwh: np.ndarray  # slots + 1: cells kWh at each slot start and after the last
    capacity_kwh: float = math.nan  # the pack's, for the SoC lane in %


def plan_path(
    ranked: Ranked,
    pack: Pack,
    loss: LossProfile,
    terms: Terms | None = None,
    knobs: Knobs | None = None,
    now: int | None = None,
) -> PlannedPath:
    """The planned battery over the window from the path's start SoC, through the pack's
    limits and loss profile (``VirtualBattery``, one slot per step).

    A ``dp`` ranking (``ranked.th.impl``) walks the trader's own policy forward: at each slot
    the q50-scenario action of ``dp_policy`` at the grid state nearest the planned SoC (the
    rounding ``dp`` uses for ``value_stored``): charge at ``charge_kw``, discharge at
    ``discharge_kw`` or the default policy on the window's net load. It needs ``terms`` and
    ``knobs``; ``now`` defaults to ``ranking_now``.

    A ``ranking`` ranking has no policy per SoC, so it keeps an approximation: full charge
    power in every ``th.charge_slots`` slot, full discharge power in every ``th.sale_slots``
    slot (a slot in both counts as charge), else the default policy. It charges every marked
    slot whatever the SoC the plan reached, so it can over-state the trading.
    """
    th, path, w = ranked.th, ranked.path, ranked.window
    n = len(w.ts)
    net = np.nan_to_num(np.asarray(path.net_kw, dtype=float))
    policy = grid = None
    if th.impl == "dp":
        if terms is None or knobs is None:
            raise ValueError("a dp ranking needs terms and knobs to walk its policy")
        now = ranking_now(ranked) if now is None else int(now)
        policy = dp_policy(w, path, pack, terms, knobs, 0.0, now, loss)
        grid = soc_grid(pack)
    else:
        charge = np.asarray(th.charge_slots, dtype=bool)
        sale = np.asarray(th.sale_slots, dtype=bool) & ~charge
    action = np.full(n, "hold", dtype=object)
    bat = VirtualBattery(pack, loss, float(path.soc_kwh[0]), standby=False)
    kw = np.zeros(n)
    soc = np.empty(n + 1)
    soc[0] = bat.soc_kwh
    for j in range(n):
        if policy is not None:
            code = int(policy[j, int(np.abs(grid - bat.soc_kwh).argmin())])
            action[j] = DP_ACTIONS[code]
        elif charge[j]:
            action[j] = "charge"
        elif sale[j]:
            action[j] = "discharge"
        if action[j] == "charge":
            want = pack.charge_kw
        elif action[j] == "discharge":
            want = -pack.discharge_kw
        else:
            want = default_policy_kw(float(net[j]), pack)
        kw[j] = bat.step(want, ISP_S)
        soc[j + 1] = bat.soc_kwh
    return PlannedPath(action.astype(str), kw, soc, float(pack.capacity_kwh))


def _slot_money(net_kw: float, take: float, feed: float, terms: Terms, ts: int) -> float:
    imp, exp = max(net_kw, 0.0) * SLOT_H, max(-net_kw, 0.0) * SLOT_H
    total, _ = money_eur(imp, exp, take, feed, 0, terms, tax_mode_at(terms, ts))
    return total


def ranking_now(ranked: Ranked) -> int:
    """The instant the ranking was made: its quarter start plus the window's elapsed minutes
    (minute 1 for a setup), the ``now`` the rule passed to its thresholds."""
    w = ranked.window
    return int(w.ts[0]) + int(w.elapsed_min) * 60


def plan_value(
    planned: PlannedPath,
    ranked: Ranked,
    q: float,
    terms: Terms,
    pack: Pack,
    knobs: Knobs,
    now: int | None = None,
    n: int | None = None,
) -> float:
    """Sum over the first ``n`` slots (all when None) of house money minus shadow money, both
    legs priced at quantile ``q`` of the take and feed bands, plus what the plan leaves in the
    pack valued as the rule values it: ``(soc_kwh[n] - soc_kwh[0]) *
    dp_terminal_value(window, pack, terms, knobs, now) * (1 + btw)`` (``now`` defaults to
    ``ranking_now``). The rule's EUR per cells kWh exclude BTW (``slot_costs``), the money of
    ``money_eur`` includes it, so the term takes the BTW factor to be in the same EUR. The
    terminal term is the same at every quantile. A slot with a non-finite price adds
    nothing."""
    w = ranked.window
    m = len(w.ts) if n is None else min(int(n), len(w.ts))
    take = at_quantile(w.take[:, :m], q)
    feed = at_quantile(w.feed[:, :m], q)
    house = np.nan_to_num(np.asarray(ranked.path.net_kw, dtype=float))
    total = 0.0
    for j in range(m):
        if not (math.isfinite(take[j]) and math.isfinite(feed[j])):
            continue
        ts = int(w.ts[j])
        h = _slot_money(float(house[j]), take[j], feed[j], terms, ts)
        s = _slot_money(float(house[j] + planned.kw[j]), take[j], feed[j], terms, ts)
        total += h - s
    left = float(planned.soc_kwh[m]) - float(planned.soc_kwh[0])
    if left != 0.0:
        now = ranking_now(ranked) if now is None else int(now)
        per_kwh = dp_terminal_value(w, pack, terms, knobs, now) * (1.0 + terms.btw)
        total += left * per_kwh
    return float(total)


def plan_view(
    ranked: Ranked,
    planned: PlannedPath,
    terms: Terms,
    pack: Pack,
    knobs: Knobs,
    now: int | None = None,
    n: int = PLAN_SLOTS,
) -> tuple[float, dict]:
    """The plan sensor: state the value at q50 (EUR, 2 decimals, ``plan_value`` with its
    terminal term) over the first ``min(n, slots)`` slots; attributes the per-slot arrays
    (ct/kWh, SoC % at each slot start, action), the range and the open-meteo attribution.

    ``result_low`` and ``result_high`` are the smallest and largest of the q10, q50 and q90
    values: the plan is held fixed, so the value is affine in the band rows but not monotone
    in the quantile (the take and feed legs pull opposite ways), and the range must hold the
    state it is printed around.
    """
    w = ranked.window
    m = min(int(n), len(w.ts))
    values = [plan_value(planned, ranked, q, terms, pack, knobs, now, m) for q in (0.1, 0.5, 0.9)]
    attrs: dict = {"ts": [int(t) for t in w.ts[:m]]}
    for side, band in (("take", w.take), ("feed", w.feed)):
        for q in (0.1, 0.5, 0.9):
            row = band[QUANTILES.index(q), :m]
            attrs[f"{side}_q{round(q * 100)}"] = [_ct(v) for v in row]
    cap = float(planned.capacity_kwh)
    attrs["soc_pct"] = [
        round(float(s) / cap * 100.0, 1) if cap > 0 else None for s in planned.soc_kwh[:m]
    ]
    attrs["action"] = [str(a) for a in planned.action[:m]]
    attrs["result_low"] = round(min(values), 2)
    attrs["result_high"] = round(max(values), 2)
    attrs["attribution"] = ATTRIBUTION
    return round(values[1], 2), attrs


def next_window(ranked: Ranked, now: int, planned: PlannedPath | None = None) -> dict | None:
    """The first contiguous run of charge or sale slots from the quarter of ``now`` on:
    ``{"start", "end", "side"}`` (epoch s, side ``charge`` or ``discharge``); None without
    one.

    Under ``dp`` with a ``planned`` path the slots are the plan's own actions (the policy
    walked at the planned SoC, what the plan card shows); otherwise the ranking's q50 masks,
    a slot in both counting as a charge slot."""
    ts = np.asarray(ranked.window.ts)
    if planned is not None and ranked.th.impl == "dp":
        action = np.asarray(planned.action)[: len(ts)]
        charge, sale = action == "charge", action == "discharge"
    else:
        charge = np.asarray(ranked.th.charge_slots, dtype=bool)
        sale = np.asarray(ranked.th.sale_slots, dtype=bool) & ~charge
    k = int(np.searchsorted(ts, int(isp_start(int(now)))))
    trade = np.flatnonzero((charge | sale)[k:])
    if not len(trade):
        return None
    j = k + int(trade[0])
    mask, side = (charge, "charge") if charge[j] else (sale, "discharge")
    e = j
    while e + 1 < len(ts) and mask[e + 1]:
        e += 1
    return {"start": int(ts[j]), "end": int(ts[e]) + ISP_S, "side": side}


def plan_for(ranked: Ranked, pack: Pack, loss: LossProfile, terms: Terms, knobs: Knobs) -> dict:
    """The plan of one ranking: ``{"planned": PlannedPath, "state", "attrs"}`` (``plan_path``
    and ``plan_view`` at ``ranking_now``). Blocking under ``dp`` (a backward induction)."""
    now = ranking_now(ranked)
    planned = plan_path(ranked, pack, loss, terms, knobs, now)
    state, attrs = plan_view(ranked, planned, terms, pack, knobs, now, PLAN_SLOTS)
    return {"planned": planned, "state": state, "attrs": attrs}


def pv_now(
    model: PvModel | None, irr: Table, slot: int, lat: float, lon: float, total_kw: float
) -> dict:
    """The PV expectation of the quarter ``slot``: ``kw`` (the window's, ``total_kw``) and
    ``per_string`` (kW per ``Plane.string``, ``planeN`` for a plane without one; planes of
    one string summed, each plane through the model's shading profile), the strings scaled
    to the total when the AC limit clips it, and
    ``model_version`` (``pv-<fitted_at>`` of the champion, None without one)."""
    out = {"kw": float(total_kw), "per_string": {}, "model_version": None}
    if model is None:
        return out
    out["model_version"] = f"pv-{int(model.fitted_at)}"
    if not model.planes:
        return out
    ts = np.array([int(slot)], dtype=np.int64)
    one = align(irr, ts, IRR_COLS)
    factor = float(model.month_factor[int(_local_month(ts)[0]) - 1])
    parts: dict[str, float] = {}
    shade = (
        np.ones(len(model.planes))
        if model.shading is None
        else apply_shading(model.shading, np.ones((1, len(model.planes))), ts, lat, lon)[0]
    )
    for i, (plane, scale) in enumerate(zip(model.planes, model.scales_kwp)):
        kw = float(plane_shape(one, plane, lat, lon)[0]) * float(shade[i]) * float(scale) * factor
        key = plane.string or f"plane{i + 1}"
        parts[key] = parts.get(key, 0.0) + (kw if math.isfinite(kw) else 0.0)
    raw, total = sum(parts.values()), float(total_kw)
    if total <= 0:
        k = 0.0
    elif raw > total:
        k = total / raw  # the AC limit clipped the total
    else:
        k = 1.0
    out["per_string"] = {key: round(v * k, 3) for key, v in parts.items()}
    return out


def _threshold_ct(d) -> float:
    """The threshold of the decision's side in ct/kWh (the charge side for a hold)."""
    th = d.discharge_threshold if d.action == "discharge" else d.charge_threshold
    return ct_kwh(th) if math.isfinite(float(th)) else math.nan


def state_view(
    status: dict, ranked: Ranked | None, lang: str, planned: PlannedPath | None = None
) -> tuple[str, dict]:
    """The state sensor: the action of the newest decision (``hold`` before the first) and
    its attributes (module docstring). ``kw`` is the pack's AC kW, charge positive.
    ``planned`` is the ranking's plan (``plan_for``): the next window follows it."""
    d = status.get("decision")
    now = int(d.ts) if d is not None else (int(ranked.window.ts[0]) if ranked else None)
    window = (
        next_window(ranked, now, planned) if ranked is not None and now is not None else None
    )
    head = status.get("head")
    last = status.get("last_trade")
    flags = list(status.get("flags") or [])
    attrs = {
        "kw": round(float(d.kw), 2) if d is not None else 0.0,
        "committed": bool(d.committed) if d is not None else False,
        "reason": d.reason if d is not None else None,
        "verdict": (
            CP.verdict(d.reason, d.action, d.kw, _threshold_ct(d), window, lang)
            if d is not None
            else None
        ),
        "next_window": window,
        "last_trade": (
            CP.trade_line(last["action"], last["kw"], last["ts"], lang) if last else None
        ),
        "ledger_head": None if head is None else {"seq": int(head[0]), "hash": str(head[1])},
        "status": flags,
        "status_text": [CP.status_line(f, lang) for f in flags],
    }
    action = d.action if d is not None and d.action in ACTIONS else "hold"
    return action, attrs
