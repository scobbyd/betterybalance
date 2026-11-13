import pathlib

import numpy as np
import pytest
from betterybalance.core import ledger as LG
from betterybalance.core.site import flows as F
from betterybalance.core.site.contract import Terms
from betterybalance.replay import disaggregate as D

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_site_flows_hourly.csv"


def test_hourly_rows_become_four_quarters_with_the_same_energy():
    h = F.read_flows(FIX)
    q = D.hourly_to_quarters(h)
    assert len(q["ts"]) == 12 and (q["step_s"] == 900).all() and q["ts"][1] - q["ts"][0] == 900
    assert q["grid_import_kwh"][:4].sum() == pytest.approx(h["grid_import_kwh"][0])
    assert q["soc_pct"][3] == h["soc_pct"][0]
    assert not q["missing"][0]


def test_pv_follows_the_plant_shape_inside_the_hour():
    h = F.read_flows(FIX)
    shape_ts = np.arange(h["ts"][0], h["ts"][0] + 3 * 3600, 900)
    shape = np.tile(np.array([0.0, 1.0, 2.0, 1.0]), 3)
    q = D.hourly_to_quarters(h, shape, shape_ts)
    # the fixture's first hour has 1,0 kWh of PV
    assert q["pv_kwh"][:4].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.25])
    assert q["pv_kwh"][:4].sum() == pytest.approx(h["pv_kwh"][0])


def test_refuses_non_hourly_rows_and_flattens_back():
    h = F.read_flows(FIX)
    q = D.hourly_to_quarters(h)
    q["grid_import_kwh"][:4] = [1.0, 0.0, 0.0, 0.0]
    flat = D.flatten_to_hours(q)
    assert flat["grid_import_kwh"][:4].tolist() == [0.25] * 4
    with pytest.raises(ValueError):
        D.hourly_to_quarters(q)


def test_hourly_bias_is_zero_on_flat_prices_and_flagged_when_large():
    terms = Terms()

    def rows(imports, prices):
        return [
            LG.rows_for_quarter(
                i * 900,
                "actual",
                np.full(15, imp * 4),
                {"buy": p, "sell": p, "state": 1, "settled": True},
                terms,
            )
            for i, (imp, p) in enumerate(zip(imports, prices, strict=True))
        ]

    quarters = rows([1.0, 0.0, 0.0, 0.0], [100, 100, 100, 100])
    flat = rows([0.25] * 4, [100, 100, 100, 100])
    assert D.hourly_bias(quarters, flat, 10.0)["bias_eur"] == pytest.approx(0.0)
    quarters = rows([1.0, 0.0, 0.0, 0.0], [400, 0, 0, 0])
    flat = rows([0.25] * 4, [400, 0, 0, 0])
    b = D.hourly_bias(quarters, flat, 1.0)
    assert b["bias_eur"] == pytest.approx(-(0.75 * 0.4) * 1.21)
    assert b["indicative_only"]


def test_missing_required_value_marks_the_hour_and_optional_nan_stays_nan():
    h = F.read_flows(FIX)
    h["grid_import_kwh"][1] = np.nan
    h["load_kwh"][2] = np.nan
    q = D.hourly_to_quarters(h)
    assert q["missing"].tolist() == [False] * 4 + [True] * 4 + [False] * 4
    assert np.isnan(q["soc_pct"][4:8]).all()
    assert np.isnan(q["load_kwh"][8:]).all() and not q["missing"][8]


def test_shape_falls_back_to_flat_when_incomplete_or_zero():
    h = F.read_flows(FIX)
    shape_ts = np.array([h["ts"][0] + 900 * k for k in range(4)])
    zero = D.hourly_to_quarters(h, np.zeros(4), shape_ts)
    assert zero["pv_kwh"][:4].tolist() == pytest.approx([0.25] * 4)
    partial = D.hourly_to_quarters(h, np.ones(4), shape_ts)   # hours 2 and 3 are not in shape_ts
    assert partial["pv_kwh"][4:8].tolist() == pytest.approx([0.1] * 4)


def test_flatten_propagates_missing_and_nan():
    h = F.read_flows(FIX)
    h["grid_import_kwh"][0] = np.nan
    flat = D.flatten_to_hours(D.hourly_to_quarters(h))
    assert flat["missing"][:4].all() and np.isnan(flat["grid_import_kwh"][:4]).all()


def test_string_columns_split_like_pv():
    h = F.read_flows(FIX)
    h["pv1_kwh"] = np.array([1.0, 0.0, np.nan])  # the first hour carries the pv_kwh of 1,0
    h["pv3_kwh"] = h["pv_kwh"].copy()
    flat = D.hourly_to_quarters(h)
    assert flat["pv1_kwh"][:4].tolist() == pytest.approx([0.25] * 4)
    assert flat["pv1_kwh"][:4].sum() == pytest.approx(1.0)
    assert np.isnan(flat["pv1_kwh"][8:]).all() and not flat["missing"][8]
    assert flat["pv3_kwh"].tolist() == pytest.approx(flat["pv_kwh"].tolist())
    assert "pv2_kwh" not in flat
    shape_ts = np.arange(h["ts"][0], h["ts"][0] + 3 * 3600, 900)
    shape = np.tile(np.array([0.0, 1.0, 2.0, 1.0]), 3)
    shaped = D.hourly_to_quarters(h, shape, shape_ts)
    assert shaped["pv1_kwh"][:4].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.25])
    assert shaped["pv1_kwh"][:4].tolist() == pytest.approx(shaped["pv_kwh"][:4].tolist())
    assert shaped["pv3_kwh"].tolist() == pytest.approx(shaped["pv_kwh"].tolist())


def test_string_columns_go_nan_in_a_missing_hour_and_flatten_back():
    h = F.read_flows(FIX)
    h["pv1_kwh"] = np.array([1.0, 0.4, 0.8])
    h["grid_import_kwh"][1] = np.nan
    q = D.hourly_to_quarters(h)
    assert np.isnan(q["pv1_kwh"][4:8]).all()
    q["pv1_kwh"][:4] = [1.0, 0.0, 0.0, 0.0]
    assert D.flatten_to_hours(q)["pv1_kwh"][:4].tolist() == [0.25] * 4
