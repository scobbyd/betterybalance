import json

import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core import pricemodel as PM

from tests.test_features import _synthetic_master


def test_five_quantiles_sorted_and_tables_present():
    m = _synthetic_master(days=10)
    model = PM.fit_model(m, tiers=("A",), horizons=(1,))
    assert PM.QUANTILES == (0.1, 0.25, 0.5, 0.75, 0.9) and model.meta["format"] == 2
    mx = F.build_matrix(m, "A", h=1)
    p = PM.predict_rows(model, "A", 1, mx.X, mx.names)
    assert np.all(p["buy_q10"] <= p["buy_q25"]) and np.all(p["buy_q25"] <= p["buy_q50"])
    assert np.all(p["sell_q75"] <= p["sell_q90"])
    assert np.asarray(model.tables["dual_rate"]).shape == (4, 12)
    assert len(model.tables["dual_spread"]) == 4


def test_format1_file_loads_with_interpolated_quantiles_and_defaults():
    m = _synthetic_master(days=10)
    model = PM.fit_model(m, tiers=("A",), horizons=(1,))
    raw = json.loads(PM.to_json(model))
    cell = raw["models"]["A:1:-"]
    for side in ("w_buy", "w_sell"):
        cell[side].pop("0.25")
        cell[side].pop("0.75")
    cell.pop("cov_factor")
    raw.pop("tables")
    raw["meta"].pop("format")
    old = PM.from_json(json.dumps(raw))
    hm = old.models[("A", 1, None)]
    assert np.allclose(hm.w_buy[0.25], (hm.w_buy[0.1] + hm.w_buy[0.5]) / 2) and hm.cov_factor == 1.0
    assert old.meta["tables_fallback"]
    assert PM.dual_rate_at(old.tables, 12, 6) == PM.DUAL_RATE_FALLBACK
    assert PM.dual_spread_at(old.tables, 12) == 0.0


def test_coverage_factor_hits_the_target_and_is_clipped():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 2.0, 5000)
    mid = np.zeros(5000)
    lo = np.full(5000, -1.0)
    hi = np.full(5000, 1.0)
    f = PM.coverage_factor(y, lo, mid, hi, target=0.8)
    assert f == pytest.approx(2.0 * 1.2816, rel=0.05)
    assert PM.coverage_factor(y * 100, lo, mid, hi) == 5.0
    assert PM.coverage_factor(y[:10], lo[:10], mid[:10], hi[:10]) == 1.0
    p = {
        "buy_q10": np.array([-1.0]),
        "buy_q25": np.array([-0.5]),
        "buy_q50": np.array([0.0]),
        "buy_q75": np.array([0.5]),
        "buy_q90": np.array([1.0]),
    }
    PM.apply_coverage(p, "buy", 2.0)
    assert p["buy_q10"][0] == -2.0 and p["buy_q75"][0] == 1.0 and p["buy_q50"][0] == 0.0


def test_dual_tables_from_history():
    m = _synthetic_master(days=10)
    t = PM.dual_tables(m, np.ones(len(m["ts"]), bool))
    rate = np.asarray(t["dual_rate"])
    assert rate.shape == (4, 12) and np.all((rate >= 0) & (rate <= 1))
    assert len(t["dual_spread"]) == 4


def test_format2_roundtrip_and_raw_predictions():
    m = _synthetic_master(days=10)
    model = PM.fit_model(m, tiers=("A",), horizons=(1,))
    model.models[("A", 1, None)].cov_factor = 1.7
    back = PM.from_json(PM.to_json(model))
    assert back.models[("A", 1, None)].cov_factor == 1.7
    assert back.cov_factor("A", 1, 0.1) == 1.7 and back.cov_factor("B", 1, 0.5) == 1.0
    assert back.tables == model.tables and not back.meta.get("tables_fallback")
    mx = F.build_matrix(m, "A", h=1)
    a = PM.predict_rows(model, "A", 1, mx.X, mx.names)
    b = PM.predict_rows(back, "A", 1, mx.X, mx.names)
    assert np.array_equal(a["buy_q25"], b["buy_q25"])
    row = dict(zip(mx.names, mx.X[0], strict=True))
    pr = PM.predict_one(back, "A", 1, row)
    assert pr.cov_factor == 1.7 and pr.buy[0.25] == pytest.approx(a["buy_q25"][0])
