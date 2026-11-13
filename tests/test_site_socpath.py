import numpy as np
import pytest
from betterybalance.core import battery as B
from betterybalance.core.site import socpath as SP
from betterybalance.core.site.contract import Pack

PACK = Pack(48, 12, 12, 0.1, 1.0, eta_rt=1.0)
TS = np.arange(0, 96 * 900, 900, dtype=np.int64)
NO = np.zeros(96, bool)


def _path(soc0, load, pv, ev, mask):
    return SP.soc_path(soc0, PACK, B.flat_profile(1.0), TS, load, pv, ev, mask)


def test_sunny_day_fills_to_the_ceiling_and_surplus_counts_down():
    pv = np.zeros(96)
    pv[32:64] = 8.0  # 8 kW for 8 h = 64 kWh of surplus
    p = _path(10.0, np.zeros(96), pv, np.zeros(96), NO)
    assert p.soc_kwh[-1] == pytest.approx(48.0) and p.room_charge_kwh[0] == pytest.approx(38.0)
    assert p.surplus_to_come_kwh[0] == pytest.approx(64.0) and p.surplus_to_come_kwh[64] == 0.0
    assert p.soc_kwh[32 + 19] == pytest.approx(48.0)  # 38 kWh at 8 kW takes 19 quarters


def test_night_load_drains_to_the_floor_and_stops():
    p = _path(8.0, np.full(96, 2.0), np.zeros(96), np.zeros(96), NO)
    assert p.soc_kwh[-1] == pytest.approx(4.8) and p.room_discharge_kwh[0] == pytest.approx(3.2)
    assert p.net_kw[0] == 2.0


def test_planned_ev_slots_are_pass_through():
    ev = np.zeros(96)
    ev[8:16] = 11.0
    mask = np.zeros(96, bool)
    mask[8:16] = True
    p = _path(40.0, np.zeros(96), np.zeros(96), ev, mask)
    assert p.soc_kwh[16] == pytest.approx(40.0)
    assert p.net_kw[8] == 0.0
    unplanned = _path(40.0, np.zeros(96), np.zeros(96), ev, NO)
    assert unplanned.soc_kwh[16] == pytest.approx(40.0 - 8 * 11.0 * 0.25)
