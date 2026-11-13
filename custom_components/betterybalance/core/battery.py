"""Virtual battery, loss profiles and execution profiles for the replay.

Sign convention: AC power is positive while the battery draws from the AC side (charging)
and negative while it delivers (discharging). All inverses are closed-form quadratics.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .site.contract import Knobs, Pack

# Same resolution as pricemodel.shipped_model: from this file, not importlib.resources.
PROFILES_DIR = Path(__file__).resolve().parent.parent / "models" / "profiles"


class LossProfile(Protocol):
    standby_w: float

    def cells_from_ac(self, ac_kw: float) -> float:
        """Cells kW stored while charging at ac_kw from the AC side."""

    def ac_from_cells(self, cells_kw: float) -> float:
        """AC kW delivered while discharging cells_kw from the cells."""

    def cells_for_ac_discharge(self, ac_kw: float) -> float:
        """Cells kW drawn to deliver ac_kw (inverse of ac_from_cells)."""

    def ac_for_cells_charge(self, cells_kw: float) -> float:
        """AC kW drawn to store cells_kw (inverse of cells_from_ac)."""


class _Flat:
    standby_w = 0.0

    def __init__(self, eta_rt: float):
        self._leg = math.sqrt(eta_rt)

    def cells_from_ac(self, ac_kw: float) -> float:
        return ac_kw * self._leg

    def ac_from_cells(self, cells_kw: float) -> float:
        return cells_kw * self._leg

    def cells_for_ac_discharge(self, ac_kw: float) -> float:
        return ac_kw / self._leg

    def ac_for_cells_charge(self, cells_kw: float) -> float:
        return cells_kw / self._leg


def flat_profile(eta_rt: float) -> LossProfile:
    """Each leg sqrt(eta_rt), no standby."""
    return _Flat(eta_rt)


def _loss_kw(stage: tuple[float, float], p_kw: float) -> float:
    return (stage[0] * p_kw + stage[1] * p_kw * p_kw) / 1000.0


def _grow_root(stage: tuple[float, float], total_kw: float) -> float:
    """P with P + loss(P) = total (positive root)."""
    a, b = stage[0] / 1000.0, stage[1] / 1000.0
    if b == 0:
        return total_kw / (1.0 + a)
    return (-(1.0 + a) + math.sqrt((1.0 + a) ** 2 + 4.0 * b * total_kw)) / (2.0 * b)


def _shrink_root(stage: tuple[float, float], net_kw: float) -> float:
    """P with P - loss(P) = net (smaller root)."""
    a, b = stage[0] / 1000.0, stage[1] / 1000.0
    if b == 0:
        return net_kw / (1.0 - a)
    return ((1.0 - a) - math.sqrt(max((1.0 - a) ** 2 - 4.0 * b * net_kw, 0.0))) / (2.0 * b)


class _Deye:
    def __init__(self, data: dict):
        self._to_port = (data["ac_to_port_w"]["a"], data["ac_to_port_w"]["b"])
        self._to_ac = (data["port_to_ac_w"]["a"], data["port_to_ac_w"]["b"])
        self._side = (data["side_each_way_w"]["a"], data["side_each_way_w"]["b"])
        self.standby_w = float(data["standby_w"])

    def cells_from_ac(self, ac_kw: float) -> float:
        port = _grow_root(self._to_port, ac_kw)
        return port - _loss_kw(self._side, port)

    def ac_for_cells_charge(self, cells_kw: float) -> float:
        port = _shrink_root(self._side, cells_kw)
        return port + _loss_kw(self._to_port, port)

    def ac_from_cells(self, cells_kw: float) -> float:
        port = _grow_root(self._side, cells_kw)
        return port - _loss_kw(self._to_ac, port)

    def cells_for_ac_discharge(self, ac_kw: float) -> float:
        port = _shrink_root(self._to_ac, ac_kw)
        return port + _loss_kw(self._side, port)


def deye_profile(data: dict) -> LossProfile:
    """Stage losses a*P + b*P^2 watts on the port power P in kW, from the profile JSON."""
    return _Deye(data)


def load_profile(name: str, pack: Pack) -> LossProfile:
    """The only loader: "flat" or "deye"."""
    if name == "flat":
        return flat_profile(pack.eta_rt)
    if name == "deye":
        return deye_profile(json.loads((PROFILES_DIR / "deye_sun12k.json").read_text()))
    raise ValueError(f"unknown loss profile {name!r}")


# the flat profile names no inverter: string power counts as AC at factor 1, as the plant
# model's per-kWp shape does on the blind path (no DC-to-AC loss)
FLAT_DC_TO_AC = (0.0, 0.0)


def pv_dc_to_ac_stage(name: str) -> tuple[float, float]:
    """The DC-to-AC stage loss (a, b), a * P + b * P^2 W at P kW, that DC string power passes.

    "deye": the profile's ``port_to_ac_w`` stage (the inverter's DC bus to AC; the profile has
    no separate PV leg, and MPPT losses sit inside the measured string power). "flat":
    FLAT_DC_TO_AC.
    """
    if name == "flat":
        return FLAT_DC_TO_AC
    if name == "deye":
        data = json.loads((PROFILES_DIR / "deye_sun12k.json").read_text())
        return (float(data["port_to_ac_w"]["a"]), float(data["port_to_ac_w"]["b"]))
    raise ValueError(f"unknown loss profile {name!r}")


@dataclass(frozen=True)
class ExecutionProfile:
    latency_s: int = 60
    dwell_s: int = 300
    trade_floor_w: float = 3000.0
    step_w: float = 0.0
    deadband_w: float = 0.0
    charge_cap_w: float | None = None
    discharge_cap_w: float | None = None


IDEAL = ExecutionProfile()
DEYE_AMPS = ExecutionProfile(step_w=51.2, trade_floor_w=3072.0)  # one ampere at 51,2 V
DEYE_TOU = ExecutionProfile(deadband_w=1000.0, charge_cap_w=11300.0)
_TEMPLATES = {"ideal": IDEAL, "deye_amps": DEYE_AMPS, "deye_tou": DEYE_TOU}


def execution_profile(name: str, knobs: Knobs) -> ExecutionProfile:
    """Template shape (step, deadband, caps) plus the knob timing and trade floor."""
    if name not in _TEMPLATES:
        raise ValueError(f"unknown execution profile {name!r}")
    t = _TEMPLATES[name]
    floor = knobs.trade_floor_w
    if t.step_w > 0:
        # The template floor is a hardware minimum (60 A); the knob can only raise it.
        floor = max(floor, t.trade_floor_w)
        floor = math.ceil(floor / t.step_w - 1e-9) * t.step_w
    return ExecutionProfile(
        latency_s=knobs.latency_s,
        dwell_s=knobs.dwell_s,
        trade_floor_w=floor,
        step_w=t.step_w,
        deadband_w=t.deadband_w,
        charge_cap_w=t.charge_cap_w,
        discharge_cap_w=t.discharge_cap_w,
    )


_LONG_AGO = -(2**62)  # an action start the dwell can no longer reach


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


class Executor:
    """Applies a requested setpoint the way a real writer would (latency, dwell, floor, step)."""

    def __init__(self, profile: ExecutionProfile):
        self._p = profile
        self._current = 0.0  # delivered setpoint now
        self._queue: deque[tuple[int, float]] = deque()  # (effective_ts, setpoint), in order
        self._last_ts: int | None = None
        # Dwell guards charge/discharge reversals only: a hold neither counts nor resets it.
        self._action_sign = 0  # sign of the last non-zero action
        self._action_since: int | None = None  # when that action took effect

    @classmethod
    def unknown(cls, profile: ExecutionProfile, sign: int, since: int | None) -> Executor:
        """An executor whose delivered setpoint and target are unknown (NaN, nothing queued).

        Its last action had ``sign`` (-1, 0 or 1) and took effect at ``since``; None (or
        ``sign`` 0) means long enough ago that the dwell no longer applies. A NaN output means
        the setpoint is not known. The verifier runs one per state that a run started before
        its range can be in (``verify.check``).
        """
        e = cls(profile)
        e._current = math.nan
        e._action_sign = int(sign)
        if e._action_sign != 0:
            e._action_since = _LONG_AGO if since is None else int(since)
        return e

    def state(self, now: int) -> tuple:
        """What decides the outputs after ``now``: ``(delivered, queue, sign of the last
        action, its start)``, the start None once the dwell can no longer apply to it."""
        since = self._action_since
        if since is not None and now - since >= self._p.dwell_s:
            since = None
        return (self._current, tuple(self._queue), self._action_sign, since)

    def _shape(self, kw: float, grid_trade: bool) -> float:
        p = self._p
        w = abs(kw) * 1000.0
        if grid_trade and w < p.trade_floor_w:
            return 0.0
        if p.step_w > 0:
            w = math.floor(w / p.step_w + 1e-9) * p.step_w
        out = math.copysign(w / 1000.0, kw) if w > 0 else 0.0
        cap = p.charge_cap_w if out > 0 else p.discharge_cap_w
        if cap is not None and w > cap:
            out = math.copysign(cap / 1000.0, out)
        return out

    def _apply_due(self, ts: int) -> None:
        while self._queue and self._queue[0][0] <= ts:
            self._current = self._queue.popleft()[1]

    def request(self, ts: int, kw: float, grid_trade: bool = True) -> float:
        """Delivered AC setpoint for the minute starting at ts."""
        p = self._p
        if self._last_ts is not None and ts < self._last_ts:
            raise ValueError(f"ts {ts} is earlier than the previous request {self._last_ts}")
        self._last_ts = ts
        self._apply_due(ts)
        target = self._queue[-1][1] if self._queue else self._current
        want = self._shape(kw, grid_trade)
        reversal = want != 0 and self._action_sign != 0 and _sign(want) != self._action_sign
        if reversal and ts - self._action_since < p.dwell_s:
            want = target
        elif (
            p.deadband_w > 0
            and _sign(want) == _sign(target)
            and abs(abs(want) - abs(target)) * 1000.0 < p.deadband_w
        ):
            want = target
        # an unknown (NaN) setpoint kept as an unknown target is no change (``unknown``)
        if want != target and not (math.isnan(want) and math.isnan(target)):
            if want != 0 and _sign(want) != self._action_sign:
                self._action_sign, self._action_since = _sign(want), ts + p.latency_s
            self._queue.append((ts + p.latency_s, want))
            self._apply_due(ts)
        return self._current


class VirtualBattery:
    """Minute-stepped pack with a loss profile, power limits and a SoC band in cells kWh."""

    def __init__(self, pack: Pack, loss: LossProfile, soc_kwh: float, standby: bool = False):
        self._pack, self._loss, self._standby = pack, loss, standby
        self._soc = float(soc_kwh)
        self._through = 0.0

    @property
    def soc_kwh(self) -> float:
        return self._soc

    @property
    def throughput_kwh(self) -> float:
        return self._through

    def step(self, ac_kw: float, dt_s: int) -> float:
        """Apply ac_kw (charge positive) for dt_s; return the AC kW actually exchanged."""
        if dt_s < 0:
            raise ValueError("dt_s must be >= 0")
        pk, h = self._pack, dt_s / 3600.0
        if self._standby:
            self._soc = max(self._soc - self._loss.standby_w / 1000.0 * h, 0.0)
        if ac_kw > 0:
            ac = min(ac_kw, pk.charge_kw)
            cells = self._loss.cells_from_ac(ac)
            room = max(pk.soc_ceiling * pk.capacity_kwh - self._soc, 0.0)
            if cells * h > room:
                cells = room / h
                ac = self._loss.ac_for_cells_charge(cells)
            self._soc += cells * h
            self._through += cells * h
            return ac
        if ac_kw < 0:
            ac = min(-ac_kw, pk.discharge_kw)
            cells = self._loss.cells_for_ac_discharge(ac)
            avail = max(self._soc - pk.soc_floor * pk.capacity_kwh, 0.0)
            if cells * h > avail:
                cells = avail / h
                ac = self._loss.ac_from_cells(cells)
            self._soc -= cells * h
            self._through += cells * h
            return -ac
        return 0.0


def default_policy_kw(residual_kw: float, pack: Pack) -> float:
    """Absorb surplus, cover load: clip(-residual, -discharge_kw, charge_kw)."""
    return max(-pack.discharge_kw, min(pack.charge_kw, -residual_kw))
