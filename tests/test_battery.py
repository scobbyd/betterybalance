import json

import pytest
from betterybalance.core import battery as B
from betterybalance.core.site.contract import Knobs, Pack

PACK = Pack(48, 12, 12, 0.1, 1.0, eta_rt=0.81, side="dc")
DEYE = json.loads((B.PROFILES_DIR / "deye_sun12k.json").read_text())


def test_flat_profile_splits_round_trip_evenly():
    p = B.flat_profile(0.81)
    assert p.cells_from_ac(10.0) == pytest.approx(9.0)
    assert p.ac_from_cells(10.0) == pytest.approx(9.0)
    assert p.cells_for_ac_discharge(9.0) == pytest.approx(10.0)


def test_deye_profile_matches_loss_map_headline_table():
    p = B.deye_profile(DEYE)
    # port 4 kW discharge: loss 37,9*4 + 2,6*16 = 193 W; side 3,7*4+2,1*16 = 48 W
    cells_kw = 4.0 + 0.048
    assert p.ac_from_cells(cells_kw) == pytest.approx(4.0 - 0.193, abs=0.005)
    # AC -> cells -> AC round trip at 4 kW port lands at 90,7 % within 0,5 points
    ac_in = 4.0 + (12.1 * 4 + 3.4 * 16) / 1000
    assert p.ac_from_cells(p.cells_from_ac(ac_in)) / ac_in == pytest.approx(0.907, abs=0.005)


def test_deye_inverses_are_consistent():
    p = B.deye_profile(DEYE)
    for ac in (0.5, 4.0, 11.0):
        assert p.ac_from_cells(p.cells_for_ac_discharge(ac)) == pytest.approx(ac)
        assert p.ac_for_cells_charge(p.cells_from_ac(ac)) == pytest.approx(ac)
    assert p.standby_w == 97.0


def test_load_profile_names():
    assert B.load_profile("flat", PACK).cells_from_ac(10.0) == pytest.approx(9.0)
    assert B.load_profile("deye", PACK).standby_w == 97.0
    with pytest.raises(ValueError, match="nope"):
        B.load_profile("nope", PACK)


def test_executor_latency_dwell_floor_and_step():
    ex = B.Executor(B.ExecutionProfile(latency_s=60, dwell_s=300, trade_floor_w=3000, step_w=500))
    assert ex.request(0, 5.0) == 0.0  # latency: nothing in the first minute
    assert ex.request(60, 5.0) == 5.0
    assert ex.request(120, 5.3) == 5.0  # step 500 W rounds down
    assert ex.request(180, -6.0) == 5.0  # dwell: a sign change within 5 min is ignored
    assert ex.request(420, -6.0) == 5.0 and ex.request(480, -6.0) == -6.0
    ex.request(900, -2.0)
    assert ex.request(960, -2.0) == 0.0  # under the 3 kW trade floor is a hold


def test_executor_floor_skipped_without_grid_trade():
    ex = B.Executor(B.IDEAL)
    ex.request(0, 1.0, grid_trade=False)
    assert ex.request(60, 1.0, grid_trade=False) == 1.0


def test_executor_deadband_keeps_previous_magnitude():
    ex = B.Executor(B.DEYE_TOU)
    ex.request(0, 8.0)
    assert ex.request(60, 8.0) == 8.0
    assert ex.request(120, 8.6) == 8.0  # 600 W change stays inside the 1 000 W deadband
    assert ex.request(180, 9.2) == 8.0 and ex.request(240, 9.2) == 9.2


def test_executor_caps_apply_last():
    ex = B.Executor(B.ExecutionProfile(charge_cap_w=5000.0, discharge_cap_w=4000.0, dwell_s=0))
    ex.request(0, 9.0)
    assert ex.request(60, 9.0) == 5.0
    ex.request(120, -9.0)
    assert ex.request(180, -9.0) == -4.0


def test_execution_profile_templates_and_knobs():
    k = Knobs(latency_s=30, dwell_s=120, trade_floor_w=2000.0)
    assert B.execution_profile("ideal", k) == B.ExecutionProfile(30, 120, 2000.0)
    amps = B.execution_profile("deye_amps", Knobs())
    assert amps.step_w == 51.2 and amps.trade_floor_w == pytest.approx(3072.0)
    high = B.execution_profile("deye_amps", Knobs(trade_floor_w=5000.0))
    assert high.trade_floor_w == pytest.approx(98 * 51.2)  # 5.017,6 W
    assert B.execution_profile("deye_tou", k).trade_floor_w == 2000.0
    tou = B.execution_profile("deye_tou", Knobs())
    assert tou.deadband_w == 1000.0 and tou.charge_cap_w == 11300.0
    with pytest.raises(ValueError, match="bogus"):
        B.execution_profile("bogus", Knobs())


def test_virtual_battery_respects_band_and_losses():
    vb = B.VirtualBattery(PACK, B.flat_profile(0.81), soc_kwh=47.5)
    # only 0,5 kWh of room, 0,9 per leg
    assert vb.step(12.0, 900) == pytest.approx(0.5 / 0.9 / 0.25)
    assert vb.soc_kwh == pytest.approx(48.0)
    vb2 = B.VirtualBattery(PACK, B.flat_profile(0.81), soc_kwh=5.0)
    assert vb2.step(-12.0, 900) == pytest.approx(-(0.2 * 0.9) / 0.25)  # floor 4,8 kWh
    assert vb2.soc_kwh == pytest.approx(4.8)


def test_virtual_battery_throughput_power_limit_and_standby():
    vb = B.VirtualBattery(PACK, B.flat_profile(1.0), soc_kwh=24.0)
    assert vb.step(20.0, 3600) == pytest.approx(12.0)  # pack charge limit
    assert vb.step(-6.0, 3600) == pytest.approx(-6.0)
    assert vb.throughput_kwh == pytest.approx(18.0)
    vs = B.VirtualBattery(PACK, B.deye_profile(DEYE), soc_kwh=24.0, standby=True)
    vs.step(0.0, 3600)
    assert vs.soc_kwh == pytest.approx(24.0 - 0.097) and vs.throughput_kwh == 0.0


def test_default_policy_absorbs_surplus_and_covers_load():
    assert B.default_policy_kw(-3.0, PACK) == 3.0 and B.default_policy_kw(20.0, PACK) == -12.0


def test_executor_dwell_ignores_holds_and_blocks_reversals():
    ex = B.Executor(B.IDEAL)
    ex.request(0, 5.0)
    assert ex.request(60, 5.0) == 5.0  # charge takes effect at 60
    ex.request(120, 0.0)
    assert ex.request(180, 0.0) == 0.0  # hold takes effect at 180
    ex.request(240, 5.0)
    assert ex.request(300, 5.0) == 5.0  # same side again is not blocked

    ex2 = B.Executor(B.IDEAL)
    ex2.request(0, 5.0)
    ex2.request(60, 5.0)
    ex2.request(120, 0.0)
    ex2.request(180, 0.0)
    ex2.request(240, -6.0)
    assert ex2.request(300, -6.0) == 0.0  # reversal within 300 s of the charge at 60


def test_executor_latency_longer_than_request_cadence_queues_changes():
    ex = B.Executor(B.ExecutionProfile(latency_s=120, dwell_s=0, trade_floor_w=0))
    reqs = [(0, 5.0), (60, 6.0), (120, 7.0), (180, 7.0), (240, 7.0)]
    got = [ex.request(ts, kw) for ts, kw in reqs]
    assert got == [0.0, 0.0, 5.0, 6.0, 7.0]


def test_executor_rejects_backwards_ts_and_battery_negative_dt():
    ex = B.Executor(B.IDEAL)
    ex.request(100, 4.0)
    ex.request(100, 4.0)
    with pytest.raises(ValueError, match="ts"):
        ex.request(99, 4.0)
    with pytest.raises(ValueError, match="dt_s"):
        B.VirtualBattery(PACK, B.flat_profile(0.81), soc_kwh=24.0).step(1.0, -60)
