"""Energy need: the cells energy the pack must still gain before the window ends.

The need is the cells energy the pack must still buy to carry the expected net load to
the terminal target. Plan 2 knows one target, the floor, so the terminal energy above
the floor is 0. The net load is AC-side, so it converts to cells through the legs.
"""

from __future__ import annotations

import math

import numpy as np

from ..site.contract import Pack
from ..site.flows import ac_legs
from ..site.socpath import SLOT_H, SocPath

TERMINAL_FLOOR_KWH = 0.0


def need_kwh(soc_path: SocPath, soc_kwh: float, pack: Pack, widening: float = 1.0) -> float:
    """Cells kWh the pack must still gain to serve the window's net load.

    ``max(0, (demand_cells - surplus_cells + terminal - (soc_kwh - floor_kwh)) * widening)``
    with ``demand_cells = sum(max(net, 0)) * 0.25 / ed`` (an AC kWh served draws ``1 / ed``
    cells kWh), ``surplus_cells = sum(max(-net, 0)) * 0.25 * ec`` (an AC kWh absorbed
    stores ``ec`` cells kWh) and ``terminal = 0`` for the floor target. ``net_kw`` already
    holds the EV expectation and leaves out planned EV slots (``soc_path``). ``widening``
    scales the need up for a learning or uncertain site.
    """
    ec, ed = ac_legs(pack)
    net = np.asarray(soc_path.net_kw, dtype=float)
    demand_cells = float(np.maximum(net, 0.0).sum()) * SLOT_H / ed
    surplus_cells = float(np.maximum(-net, 0.0).sum()) * SLOT_H * ec
    floor_kwh = pack.soc_floor * pack.capacity_kwh
    above_floor = float(soc_kwh) - floor_kwh
    return max(0.0, (demand_cells - surplus_cells + TERMINAL_FLOOR_KWH - above_floor) * widening)


def charge_slots_needed(need: float, pack: Pack) -> int:
    """Charge slots that cover ``need`` cells kWh: ``ceil(need / (charge_kw * 0.25 * ec))``."""
    ec, _ = ac_legs(pack)
    return math.ceil(need / (pack.charge_kw * SLOT_H * ec))
