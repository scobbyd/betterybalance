"""Marginal value of stored energy: per-slot charge and discharge thresholds.

Greed is a state, not a knob. The value of a stored kWh follows from the SoC, the energy
the house still needs before the window ends and the expected prices. The ``ranking``
implementation buys the need in the cheapest slots, then scans the sale slots from the
best down and accepts a sale when the sun refills it, when a grid charge slot pays for it
with a margin above the cycle hurdle, or when the pack holds energy above the need. The
``dp`` implementation solves the same question by backward induction on a SoC grid over
three price scenarios per slot and reads the value as the slope of the value function.

The rule works excl. BTW: BTW is a factor on the whole bill and cancels in every
comparison. Prices in a ``Window`` are EUR/MWh; costs and values here are EUR per kWh
of cells energy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..battery import LossProfile, default_policy_kw
from ..pricemodel import QUANTILES
from ..site.contract import TAX_MODES, Knobs, Pack, Terms, tax_mode_at
from ..site.flows import ac_legs
from ..site.socpath import SLOT_H, SocPath
from .need import charge_slots_needed
from .window import Window

DP_WEIGHTS = (0.3, 0.4, 0.3)  # q10, q50, q90 price scenarios
DP_GRID_KWH = 1.0  # cells kWh between SoC grid states
DP_SCENARIO_ROWS = tuple(QUANTILES.index(q) for q in (0.1, 0.5, 0.9))
DP_TERMINAL_SLOTS = 96  # the terminal value reads the window's last day
_Q50_SCENARIO = 1
_HOLD, _CHARGE, _DISCHARGE = 0, 1, 2  # action order; a tie goes to the lower index
DP_ACTIONS = ("hold", "charge", "discharge")  # dp_policy's codes
_TIE_EUR = 1e-9


@dataclass
class Thresholds:
    """Per-slot thresholds of one window.

    ``charge`` and ``discharge`` are grid prices in EUR/MWh (charge at or below, sell at
    or above). ``value_stored`` is EUR per stored kWh. ``charge_slots`` and ``sale_slots``
    are boolean masks over the window's slots.
    """

    charge: np.ndarray
    discharge: np.ndarray
    value_stored: np.ndarray
    n_charge: int
    charge_slots: np.ndarray
    sale_slots: np.ndarray
    impl: str


def wear_eur_kwh(pack: Pack, knobs: Knobs) -> float:
    """Wear per cells kWh discharged: ``wear_eur_cycle / capacity_kwh``."""
    return knobs.wear_eur_cycle / pack.capacity_kwh


def _quantile_row(q: float, knob: str) -> int:
    """Row index of quantile ``q`` in ``QUANTILES``, or ValueError naming the knob."""
    if q not in QUANTILES:
        raise ValueError(f"{knob} must be one of {QUANTILES}, got {q!r}")
    return QUANTILES.index(q)


def tax_legs_for_mode(terms: Terms, mode: str) -> tuple[float, float]:
    """Energy tax in EUR/kWh on the import and on the export leg for a tax mode.

    ``"none"`` taxes the import only (no netting), ``"net_importer"`` taxes both (an export
    offsets taxed import), ``"net_exporter"`` taxes neither (the export already exceeds the
    import over the year). An unknown mode raises ValueError.
    """
    tax = terms.tax_eur_kwh
    if mode == "none":
        return tax, 0.0
    if mode == "net_importer":
        return tax, tax
    if mode == "net_exporter":
        return 0.0, 0.0
    raise ValueError(f"tax_mode must be one of {TAX_MODES}, got {mode!r}")


def _tax_legs(terms: Terms, now: int) -> tuple[float, float]:
    """Tax legs for the tax mode at ``now`` (``tax_legs_for_mode``)."""
    return tax_legs_for_mode(terms, tax_mode_at(terms, now))


def slot_costs(
    window: Window, pack: Pack, terms: Terms, knobs: Knobs, now: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per slot ``buy_cost`` and ``sale_value`` in EUR per cells kWh.

    ``buy_cost = (take[q_take] / 1000 + fee_import + tax_in) / ec``: one stored kWh costs
    ``1 / ec`` grid kWh. ``sale_value = (feed[q_feed] / 1000 - fee_export + tax_out) * ed
    - wear``: one stored kWh delivers ``ed`` grid kWh and wears the pack. The tax mode at
    ``now`` sets the energy tax legs (``_tax_legs``).
    """
    ec, ed = ac_legs(pack)
    take = window.take[_quantile_row(knobs.q_take, "q_take")]
    feed = window.feed[_quantile_row(knobs.q_feed, "q_feed")]
    tax_in, tax_out = _tax_legs(terms, now)
    buy_cost = (take / 1000.0 + terms.fee_import_eur_kwh + tax_in) / ec
    sale_value = (feed / 1000.0 - terms.fee_export_eur_kwh + tax_out) * ed - wear_eur_kwh(
        pack, knobs
    )
    return buy_cost, sale_value


def dual_penalty_eur_kwh(window: Window, c: int, s: int, pack: Pack) -> float:
    """Expected dual-pricing loss of a pair, charge slot ``c`` and sale slot ``s``.

    ``p_dual[s] * dual_spread[s] / 1000 * ed + p_dual[c] * dual_spread[c] / 1000 / ec``:
    both legs are at risk, each weighted by its own probability of a dual quarter.
    """
    ec, ed = ac_legs(pack)
    sale_leg = window.p_dual[s] * window.dual_spread[s] / 1000.0 * ed
    buy_leg = window.p_dual[c] * window.dual_spread[c] / 1000.0 / ec
    return float(sale_leg + buy_leg)


def _surplus_per_slot(soc_path: SocPath) -> np.ndarray:
    """Per-slot AC surplus kWh from ``surplus_to_come_kwh`` by differencing (a fresh copy)."""
    s2c = np.asarray(soc_path.surplus_to_come_kwh, dtype=float)
    return s2c - np.append(s2c[1:], 0.0)


def _consume(surplus: np.ndarray, start: int, kwh: float) -> None:
    """Take ``kwh`` from ``surplus`` greedily from slot ``start`` on, in place."""
    left = kwh
    for j in range(start, len(surplus)):
        if left <= 0.0:
            break
        take = min(surplus[j], left)
        surplus[j] -= take
        left -= take


def _select_sales(
    window: Window,
    soc_path: SocPath,
    pack: Pack,
    knobs: Knobs,
    buy_cost: np.ndarray,
    sale_value: np.ndarray,
    need: float,
    n_need: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The charge and sale slot masks of the ``ranking`` rule.

    The need slots are the ``n_need`` cheapest buy slots. Then every slot, best sale value
    first, becomes a sale by the first branch that holds:

    1. Free refill: the pack holds a sale's energy at ``s``, the sale clears the hurdle, and
       AC surplus strictly after ``s`` covers ``sale_kwh / ec`` (the charge leg loses the
       rest). The surplus is consumed from the earliest slot on. The refill counts at cost
       zero and ignores the export value the surplus would have earned (the ``dp`` rung
       carries that).
    2. Pair: the cheapest free charge slot ``c`` with room for a charge slot's energy,
       before ``s`` or at any time when the pack already holds the energy at ``s``, pays
       for the sale with a margin above the hurdle plus the dual penalty (if enabled).
    3. Pack energy: the pack holds a sale's energy at ``s``, the sale clears the hurdle,
       and the energy above the need still covers it.

    A slot that fails every branch is skipped, never a stop: a lower sale may still find
    a cheap charge slot before it. Free refills come before pairs so the pack's own energy
    stays in reserve. A full pack has no room, so it forms no pairs.
    """
    n = len(buy_cost)
    hurdle = knobs.hurdle_ct_kwh / 100.0
    ec, ed = ac_legs(pack)
    slot_kwh = pack.charge_kw * SLOT_H * ec  # cells kWh gained per charge slot
    sale_kwh = pack.discharge_kw * SLOT_H / ed  # cells kWh consumed per sale slot
    refill_ac_kwh = sale_kwh / ec  # AC surplus kWh that puts sale_kwh back into the cells
    floor_kwh = pack.soc_floor * pack.capacity_kwh
    order_buy = np.argsort(buy_cost, kind="stable")
    charge = np.zeros(n, dtype=bool)
    charge[order_buy[:n_need]] = True
    sale = np.zeros(n, dtype=bool)
    pack_unpromised = max(0.0, float(soc_path.soc_kwh[0]) - floor_kwh - need)
    surplus = _surplus_per_slot(soc_path)
    room_ok = np.asarray(soc_path.room_charge_kwh) >= slot_kwh
    min_buy = float(buy_cost.min())
    idx = np.arange(n)
    for s in np.argsort(-sale_value, kind="stable").tolist():
        if charge[s]:
            continue
        value = float(sale_value[s])
        if value <= hurdle and value - min_buy <= hurdle:
            continue  # neither a sale on its own nor any pair clears the hurdle
        has_energy = float(soc_path.room_discharge_kwh[s]) >= sale_kwh
        # 1. the sun refills this sale: absorbed AC surplus becomes surplus * ec cells kWh,
        # so refilling sale_kwh cells needs sale_kwh / ec kWh of AC surplus
        if has_energy and value > hurdle and float(surplus[s + 1 :].sum()) >= refill_ac_kwh:
            _consume(surplus, s + 1, refill_ac_kwh)
            sale[s] = True
            continue
        # 2. a grid charge slot pays for this sale
        free = ~charge & ~sale & (idx != s) & room_ok & ((idx < s) | has_energy)
        if free.any():
            cands = np.flatnonzero(free)
            c = int(cands[np.argmin(buy_cost[cands])])
            penalty = dual_penalty_eur_kwh(window, c, s, pack) if knobs.dual_penalty else 0.0
            if value - float(buy_cost[c]) > hurdle + penalty:
                sale[s] = True
                charge[c] = True
                continue
        # 3. energy already in the pack above the need
        if has_energy and value > hurdle and pack_unpromised >= sale_kwh:
            sale[s] = True
            pack_unpromised -= sale_kwh
    return charge, sale


def _value_stored(
    soc_path: SocPath,
    pack: Pack,
    knobs: Knobs,
    buy_cost: np.ndarray,
    sale_value: np.ndarray,
    sale: np.ndarray,
    need: float,
    n_need: int,
) -> np.ndarray:
    """EUR per stored kWh per slot.

    Below the need, a stored kWh saves the dearest need purchase: the buy cost of the
    ``n_need``-th cheapest slot. Otherwise it is worth the worst accepted sale before
    wear, or with no sale the cheapest buy cost (the price of putting it back).
    """
    n = len(buy_cost)
    floor_kwh = pack.soc_floor * pack.capacity_kwh
    below_need = np.asarray(soc_path.soc_kwh[:n]) - floor_kwh < need
    if sale.any():
        rest = float(sale_value[sale].min()) + wear_eur_kwh(pack, knobs)
    else:
        rest = float(buy_cost.min())
    out = np.full(n, rest)
    if n_need >= 1:
        order_buy = np.argsort(buy_cost, kind="stable")
        nth = float(buy_cost[order_buy[min(n_need, n) - 1]])
        out[below_need] = nth
    return out


def ranking(
    window: Window,
    soc_path: SocPath,
    pack: Pack,
    terms: Terms,
    knobs: Knobs,
    need: float,
    now: int,
) -> Thresholds:
    """Thresholds from a ranking of the window's slots (``_select_sales``, ``_value_stored``).

    ``charge`` is the dearest ``q_take`` take over the charge slots (-inf with none) and
    ``discharge`` the cheapest ``q_feed`` feed over the sale slots (+inf with none), the
    same value in every slot.
    """
    buy_cost, sale_value = slot_costs(window, pack, terms, knobs, now)
    n_need = charge_slots_needed(need, pack)
    charge, sale = _select_sales(window, soc_path, pack, knobs, buy_cost, sale_value, need, n_need)
    n = len(buy_cost)
    take = window.take[_quantile_row(knobs.q_take, "q_take")]
    feed = window.feed[_quantile_row(knobs.q_feed, "q_feed")]
    charge_th = float(take[charge].max()) if charge.any() else -np.inf
    discharge_th = float(feed[sale].min()) if sale.any() else np.inf
    value = _value_stored(soc_path, pack, knobs, buy_cost, sale_value, sale, need, n_need)
    return Thresholds(
        charge=np.full(n, charge_th),
        discharge=np.full(n, discharge_th),
        value_stored=value,
        n_charge=int(charge.sum()),
        charge_slots=charge,
        sale_slots=sale,
        impl="ranking",
    )


def soc_grid(pack: Pack) -> np.ndarray:
    """Cells kWh states from ``floor_kwh`` in ``DP_GRID_KWH`` steps, ``ceiling_kwh`` last.

    The last step is the remainder of the band and may be shorter than ``DP_GRID_KWH``.
    """
    floor_kwh = pack.soc_floor * pack.capacity_kwh
    ceiling_kwh = pack.soc_ceiling * pack.capacity_kwh
    n_full = math.ceil((ceiling_kwh - floor_kwh) / DP_GRID_KWH - 1e-9)
    return np.append(floor_kwh + DP_GRID_KWH * np.arange(n_full), ceiling_kwh)


@dataclass
class _Move:
    """One action over one slot at every grid state.

    ``cells_kwh`` is the SoC change (charge positive), ``ac_kwh`` the AC energy the pack
    draws (positive) or delivers (negative).
    """

    cells_kwh: np.ndarray
    ac_kwh: np.ndarray


class _Limits:
    """The band limits of one slot at every grid state, through the loss profile.

    A request that does not fit the room to the ceiling (or the energy above the floor)
    stores (draws) exactly the room, and its AC energy is recomputed for that cells
    energy, as ``VirtualBattery.step`` does. The clipped AC energies depend on the state
    only, so they are computed once per state here, not per slot.
    """

    def __init__(self, grid: np.ndarray, pack: Pack, loss: LossProfile):
        self._loss = loss
        self.room = pack.soc_ceiling * pack.capacity_kwh - grid
        self.avail = grid - pack.soc_floor * pack.capacity_kwh
        self._ac_room = np.array([loss.ac_for_cells_charge(r / SLOT_H) * SLOT_H for r in self.room])
        self._ac_avail = np.array([loss.ac_from_cells(a / SLOT_H) * SLOT_H for a in self.avail])

    def charge(self, ac_kw: float) -> _Move:
        """Charge at ``ac_kw`` AC for one slot, bounded by the room."""
        cells_req = self._loss.cells_from_ac(ac_kw) * SLOT_H
        clip = cells_req > self.room
        cells = np.where(clip, self.room, cells_req)
        ac = np.where(clip, self._ac_room, ac_kw * SLOT_H)
        return _Move(cells, ac)

    def discharge(self, ac_kw: float) -> _Move:
        """Deliver ``ac_kw`` AC for one slot, bounded by the energy above the floor."""
        cells_req = self._loss.cells_for_ac_discharge(ac_kw) * SLOT_H
        clip = cells_req > self.avail
        cells = np.where(clip, self.avail, cells_req)
        ac = np.where(clip, self._ac_avail, ac_kw * SLOT_H)
        return _Move(-cells, -ac)

    def hold(self, net_kw: float, pack: Pack) -> _Move:
        """The default policy: absorb the surplus or cover the load within the limits."""
        ac_kw = default_policy_kw(net_kw, pack)
        if ac_kw > 0:
            return self.charge(ac_kw)
        if ac_kw < 0:
            return self.discharge(-ac_kw)
        zero = np.zeros_like(self.room)
        return _Move(zero, zero.copy())


def dp_terminal_value(window: Window, pack: Pack, terms: Terms, knobs: Knobs, now: int) -> float:
    """EUR per cells kWh left above the floor at the window end.

    The mean of the ``k`` cheapest ``buy_cost`` values among the window's last
    ``DP_TERMINAL_SLOTS`` slots, ``k = charge_slots_needed(ceiling_kwh - floor_kwh, pack)``:
    the price of refilling the band from empty on the last day. A window with fewer than
    ``k`` slots there takes all of them.
    """
    buy_cost, _ = slot_costs(window, pack, terms, knobs, now)
    last = np.sort(buy_cost[-DP_TERMINAL_SLOTS:])
    span = (pack.soc_ceiling - pack.soc_floor) * pack.capacity_kwh
    k = min(charge_slots_needed(span, pack), len(last))
    return float(last[: max(k, 1)].mean())


def _slope(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """``-dV/ds`` per row: central difference inside the grid, one-sided at the band edges.

    The difference divides by the true grid spacing, so the shorter last step still gives
    EUR per kWh (with the uniform 1 kWh step it is ``(V[s - 1] - V[s + 1]) / 2``).
    """
    out = np.empty_like(values)
    out[:, 1:-1] = (values[:, :-2] - values[:, 2:]) / (grid[2:] - grid[:-2])
    out[:, 0] = (values[:, 0] - values[:, 1]) / (grid[1] - grid[0])
    out[:, -1] = (values[:, -2] - values[:, -1]) / (grid[-1] - grid[-2])
    return out


@dataclass
class _DpRun:
    """The backward induction of ``dp``: the value table and the q50-scenario policy."""

    n: int
    grid: np.ndarray
    nearest: np.ndarray  # per slot, the grid state nearest the default SoC path
    values: np.ndarray  # (n + 1, states)
    policy: np.ndarray  # (n, states): the q50-scenario optimal action code
    tax_in: float
    tax_out: float
    wear: float


def _dp_backward(
    window: Window,
    soc_path: SocPath,
    pack: Pack,
    terms: Terms,
    knobs: Knobs,
    now: int,
    loss: LossProfile,
) -> _DpRun:
    """The one copy of the ``dp`` recursion (``dp`` docstring). ``policy[q, s]`` is the
    q50-scenario optimal action (``_HOLD``, ``_CHARGE``, ``_DISCHARGE``; a tie goes to the
    lower code) at slot ``q`` and grid state ``s``; ``dp`` reads it at ``nearest[q]``."""
    if not (np.isfinite(window.take).all() and np.isfinite(window.feed).all()):
        raise ValueError("window has non-finite prices")
    n = window.take.shape[1]
    grid = soc_grid(pack)
    limits = _Limits(grid, pack, loss)
    charge_move = limits.charge(pack.charge_kw)
    discharge_move = limits.discharge(pack.discharge_kw)
    tax_in, tax_out = _tax_legs(terms, now)
    wear = wear_eur_kwh(pack, knobs)
    weights = np.asarray(DP_WEIGHTS)
    rows = list(DP_SCENARIO_ROWS)
    take_eur = window.take[rows] / 1000.0 + terms.fee_import_eur_kwh + tax_in  # (3, n)
    feed_eur = window.feed[rows] / 1000.0 - terms.fee_export_eur_kwh + tax_out
    net_kw = np.nan_to_num(
        np.asarray(soc_path.net_kw, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    soc = np.asarray(soc_path.soc_kwh[:n], dtype=float)
    nearest = np.abs(soc[:, None] - grid[None, :]).argmin(axis=1)
    floor_kwh = float(grid[0])
    values = np.empty((n + 1, len(grid)))
    values[n] = -(grid - floor_kwh) * dp_terminal_value(window, pack, terms, knobs, now)
    policy = np.empty((n, len(grid)), dtype=int)
    for q in range(n - 1, -1, -1):
        net_kwh = float(net_kw[q]) * SLOT_H
        moves = (limits.hold(float(net_kw[q]), pack), charge_move, discharge_move)
        totals = np.empty((len(moves), len(weights), len(grid)))  # (action, scenario, state)
        for a, move in enumerate(moves):
            grid_kwh = net_kwh + move.ac_kwh
            import_kwh = np.maximum(grid_kwh, 0.0)
            export_kwh = np.maximum(-grid_kwh, 0.0)
            wear_eur = wear * np.maximum(-move.cells_kwh, 0.0)
            future = np.interp(grid + move.cells_kwh, grid, values[q + 1])
            totals[a] = (
                import_kwh * take_eur[:, q, None]
                - export_kwh * feed_eur[:, q, None]
                + wear_eur
                + future
            )
        best = totals.min(axis=0)  # (scenario, state)
        values[q] = weights @ best
        at_best = totals[:, _Q50_SCENARIO, :] <= best[_Q50_SCENARIO, :] + _TIE_EUR
        policy[q] = np.argmax(at_best, axis=0)
    return _DpRun(n, grid, nearest, values, policy, tax_in, tax_out, wear)


def dp_policy(
    window: Window,
    soc_path: SocPath,
    pack: Pack,
    terms: Terms,
    knobs: Knobs,
    need: float,
    now: int,
    loss: LossProfile,
) -> np.ndarray:
    """Read-only: the q50-scenario optimal action of ``dp`` per slot and per grid state of
    ``soc_grid(pack)``, shape ``(slots, states)``, codes ``DP_ACTIONS`` index (0 hold the
    default policy, 1 charge at ``charge_kw``, 2 discharge at ``discharge_kw``). The same
    backward induction as ``dp`` (``_dp_backward``); ``need`` is unused, as in ``dp``."""
    return _dp_backward(window, soc_path, pack, terms, knobs, now, loss).policy


def dp(
    window: Window,
    soc_path: SocPath,
    pack: Pack,
    terms: Terms,
    knobs: Knobs,
    need: float,
    now: int,
    loss: LossProfile,
) -> Thresholds:
    """Thresholds from a stochastic dynamic programme by backward induction.

    States ``s`` are the cells kWh of ``soc_grid(pack)``. Per slot ``q`` from the last to
    the first and per price scenario ``i`` (q10, q50, q90 with ``DP_WEIGHTS``), the pack
    sees the scenario price and then picks hold (the default policy on ``net_kw[q]``),
    charge at ``charge_kw`` or discharge at ``discharge_kw`` (``_Limits``)::

        V[N, s] = -(s - floor_kwh) * dp_terminal_value(...)
        V[q, s] = sum_i w_i * min_a (cost_i(q, s, a) + interp(V[q + 1], s + cells(a)))

    with ``grid = net_kwh + ac_kwh`` (import positive) and ``cost_i = import * (take_i /
    1000 + fee_import + tax_in) - export * (feed_i / 1000 - fee_export + tax_out) + wear
    * discharged cells``. Wear falls on the discharged cells only (the brief's ``|cells
    moved|`` read that way): one equivalent full cycle costs exactly ``wear_eur_cycle``,
    charge and the default-policy absorb carry none, and the value function agrees with
    the discharge threshold, which adds wear on the sale leg only (as ``slot_costs``).
    Losses enter the dynamics only through ``loss``; ``ec, ed = ac_legs(pack)`` appear
    only in the threshold conversion below. The tax legs are those of ``slot_costs`` at
    ``now``. ``value_stored[q] = -dV/ds`` (``_slope``) at the grid state
    nearest ``soc_path.soc_kwh[q]``, and::

        charge[q] = (value_stored[q] * ec - fee_import - tax_in - hurdle / 2) * 1000
        discharge[q] = ((value_stored[q] + wear) / ed + fee_export - tax_out + hurdle / 2) * 1000

    No dual penalty: the three scenarios carry the price tail. ``charge_slots`` and
    ``sale_slots`` mark the slots where the q50-scenario optimal action at that nearest
    state is charge or discharge (a tie goes to hold). ``need`` is unused: the value
    function carries the need through the net load and the terminal value. So the
    learning-state widening of the need (``Knobs.learning_widen``, ``need_widening``) does
    not change the ``dp`` dispatch; a ``dp``-side equivalent is an open item.

    A non-finite ``net_kw[q]`` counts as 0 for that slot (the ``soc_path`` convention for
    unknown inputs), so one gap cannot blank the window. A non-finite take or feed price
    in any quantile row (the scenario rows and the ``q_take`` row of the terminal value)
    raises ``ValueError``: ``build_window`` guarantees finite bands, so this only makes an
    upstream bug loud.
    """
    run = _dp_backward(window, soc_path, pack, terms, knobs, now, loss)
    n, grid, nearest, values = run.n, run.grid, run.nearest, run.values
    tax_in, tax_out, wear = run.tax_in, run.tax_out, run.wear
    action_q50 = run.policy[np.arange(n), nearest]
    value = _slope(values[:n], grid)[np.arange(n), nearest]
    ec, ed = ac_legs(pack)
    hurdle = knobs.hurdle_ct_kwh / 100.0
    charge_th = (value * ec - terms.fee_import_eur_kwh - tax_in - hurdle / 2) * 1000.0
    discharge_th = ((value + wear) / ed + terms.fee_export_eur_kwh - tax_out + hurdle / 2) * 1000.0
    charge_slots = action_q50 == _CHARGE
    return Thresholds(
        charge=charge_th,
        discharge=discharge_th,
        value_stored=value,
        n_charge=int(charge_slots.sum()),
        charge_slots=charge_slots,
        sale_slots=action_q50 == _DISCHARGE,
        impl="dp",
    )


def thresholds(
    window: Window,
    soc_path: SocPath,
    pack: Pack,
    terms: Terms,
    knobs: Knobs,
    need: float,
    now: int,
    loss: LossProfile | None = None,
) -> Thresholds:
    """Dispatch on ``knobs.impl``: ``ranking`` or ``dp`` (``dp`` needs ``loss``)."""
    if knobs.impl == "ranking":
        return ranking(window, soc_path, pack, terms, knobs, need, now)
    if knobs.impl == "dp":
        if loss is None:
            raise ValueError("impl 'dp' needs a loss profile")
        return dp(window, soc_path, pack, terms, knobs, need, now, loss)
    raise ValueError(f"impl must be 'ranking' or 'dp', got {knobs.impl!r}")
