"""Expected SoC path over the decision window under the default policy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..battery import LossProfile, VirtualBattery, default_policy_kw
from .contract import Pack

SLOT_S = 900
SLOT_H = 0.25


@dataclass
class SocPath:
    """Slot arrays have n entries; soc_kwh has n + 1 (soc_kwh[0] is the start state).

    room_* are cells kWh at the start of slot q. net_kw is the pack-seen net load
    (planned EV slots exclude the EV). surplus_to_come_kwh[q] sums slots q.. (q included).
    """

    ts: np.ndarray
    soc_kwh: np.ndarray
    room_charge_kwh: np.ndarray
    room_discharge_kwh: np.ndarray
    surplus_to_come_kwh: np.ndarray
    net_kw: np.ndarray


def soc_path(
    soc0_kwh: float,
    pack: Pack,
    loss: LossProfile,
    slots_ts: np.ndarray,
    load_kw: np.ndarray,
    pv_kw: np.ndarray,
    ev_kw: np.ndarray,
    ev_plan_mask: np.ndarray,
) -> SocPath:
    """Simulate the default policy; planned EV slots are pass-through (never served by the pack).

    NaN inputs (unknown expectations) count as 0.
    """
    load, pv, ev = (np.nan_to_num(np.asarray(a, dtype=float)) for a in (load_kw, pv_kw, ev_kw))
    mask = np.asarray(ev_plan_mask, dtype=bool)
    n = len(slots_ts)
    net = load + np.where(mask, 0.0, ev) - pv
    floor, ceiling = pack.soc_floor * pack.capacity_kwh, pack.soc_ceiling * pack.capacity_kwh
    soc = np.empty(n + 1)
    soc[0] = min(max(float(soc0_kwh), floor), ceiling)
    bat = VirtualBattery(pack, loss, soc[0], standby=False)
    for q in range(n):
        bat.step(default_policy_kw(float(net[q]), pack), SLOT_S)
        soc[q + 1] = bat.soc_kwh
    surplus = np.maximum(-net, 0.0) * SLOT_H
    return SocPath(
        ts=np.asarray(slots_ts),
        soc_kwh=soc,
        room_charge_kwh=ceiling - soc[:n],
        room_discharge_kwh=soc[:n] - floor,
        surplus_to_come_kwh=np.cumsum(surplus[::-1])[::-1],
        net_kw=net,
    )
