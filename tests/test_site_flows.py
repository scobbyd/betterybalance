import pathlib

import numpy as np
import pytest
from betterybalance.core.site import contract as C
from betterybalance.core.site import flows as F

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_site_flows_minute.csv"


def test_read_write_roundtrip(tmp_path):
    t = F.read_flows(FIX)
    assert set(t) >= set(F.COLUMNS) and t["ts"].dtype == np.int64 and np.isnan(t["pv_kwh"]).all()
    F.write_flows(t, tmp_path / "f.csv")
    t2 = F.read_flows(tmp_path / "f.csv")
    assert np.array_equal(t["grid_import_kwh"], t2["grid_import_kwh"])
    assert np.array_equal(t["load_kwh"], t2["load_kwh"], equal_nan=True)


def test_to_quarters_sums_energy_and_flags_gaps():
    t = F.read_flows(FIX)  # 2 full quarters of minute rows, the third quarter misses minute 7
    q = F.to_quarters(t)
    assert len(q["ts"]) == 3 and q["step_s"][0] == 900
    assert q["grid_import_kwh"][0] == pytest.approx(t["grid_import_kwh"][:15].sum())
    assert q["soc_min_pct"][1] <= q["soc_pct"][1] <= q["soc_max_pct"][1]
    assert q["missing"][2] and np.isnan(q["grid_import_kwh"][2]) and not q["missing"][0]


def test_to_quarters_refuses_hourly_rows():
    t = F.read_flows(FIX)
    t["step_s"][:] = 3600
    with pytest.raises(ValueError, match="disaggregation"):
        F.to_quarters(t)


def test_residual_is_ac_side_and_adds_gen():
    pack = C.Pack(48, 12, 12, 0.1, 1.0, eta_rt=0.81, side="dc")  # ec = ed = 0,9
    q = {
        "missing": np.array([False]),
        "grid_import_kwh": np.array([1.0]),
        "grid_export_kwh": np.array([0.0]),
        "batt_charge_kwh": np.array([0.9]),
        "batt_discharge_kwh": np.array([0.0]),
        "gen_kwh": np.array([0.5]),
    }
    # charge 0,9 kWh DC = 1,0 kWh AC; residual = (1 - 0 - 1 + 0 + 0,5) * 4 = 2 kW
    assert F.residual_kw(q, pack)[0] == pytest.approx(2.0)
    assert F.ac_legs(pack) == pytest.approx((0.9, 0.9))


def test_residual_ac_side_pack_uses_meter_energy_and_nan_when_missing():
    pack = C.Pack(48, 12, 12, 0.1, 1.0, side="ac")
    q = {
        "missing": np.array([False, True]),
        "grid_import_kwh": np.array([1.0, np.nan]),
        "grid_export_kwh": np.array([0.0, np.nan]),
        "batt_charge_kwh": np.array([1.0, np.nan]),
        "batt_discharge_kwh": np.array([0.0, np.nan]),
        "gen_kwh": np.array([np.nan, np.nan]),
    }
    r = F.residual_kw(q, pack)
    assert r[0] == pytest.approx(0.0) and np.isnan(r[1])


def test_string_columns_roundtrip_and_sum(tmp_path):
    t = F.read_flows(FIX)
    t = {k: v[:15] for k, v in t.items()}
    t["pv1_kwh"] = np.full(15, 0.01)
    F.write_flows(t, tmp_path / "f.csv")
    t2 = F.read_flows(tmp_path / "f.csv")
    assert np.array_equal(t["pv1_kwh"], t2["pv1_kwh"]) and "pv2_kwh" not in t2
    assert F.to_quarters(t2)["pv1_kwh"][0] == pytest.approx(0.15)
    assert "pv1_kwh" not in F.to_quarters(F.read_flows(FIX))
