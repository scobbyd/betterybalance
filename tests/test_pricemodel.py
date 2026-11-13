import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core import pricemodel as PM

from tests.test_features import _synthetic_master


def test_fit_predict_roundtrip_json():
    m = _synthetic_master(days=10)
    model = PM.fit_model(m, tiers=("A", "C"), horizons=(0, 1, 4))
    assert ("C", 1, None) in model.models and ("A", 0, 5) in model.models
    mx = F.build_matrix(m, "C", h=1)
    p = PM.predict_rows(model, "C", 1, mx.X, mx.names)
    assert np.all(p["buy_q10"] <= p["buy_q50"]) and np.all(p["buy_q50"] <= p["buy_q90"])
    assert np.all((p["p_dual"] >= 0) & (p["p_dual"] <= 1))
    model2 = PM.from_json(PM.to_json(model))
    p2 = PM.predict_rows(model2, "C", 1, mx.X, mx.names)
    assert np.array_equal(p["buy_q50"], p2["buy_q50"])


def test_predict_rejects_wrong_feature_names():
    m = _synthetic_master(days=5)
    model = PM.fit_model(m, tiers=("A",), horizons=(1,))
    mx = F.build_matrix(m, "A", h=1)
    with pytest.raises(ValueError):
        PM.predict_rows(model, "A", 1, mx.X, list(reversed(mx.names)))


def test_table_baseline_fallback_and_shape():
    m = _synthetic_master(days=5)
    n = len(m["ts"])
    train = np.arange(n) < n // 2
    tb = PM.TableBaseline().fit(m, train)
    p = tb.predict_rows(m, np.where(~train)[0])
    assert len(p["buy_q50"]) == (~train).sum() and np.isfinite(p["sell_q90"]).all()


def test_train_mask_limits_rows_and_changes_weights():
    m = _synthetic_master(days=10)
    n = len(m["ts"])
    mask = np.arange(n) < n // 2
    full = PM.fit_model(m, tiers=("A",), horizons=(1,))
    part = PM.fit_model(m, tiers=("A",), horizons=(1,), train_mask=mask)
    mx = F.build_matrix(m, "A", h=1)
    k = np.searchsorted(m["ts"], mx.ts_target)
    assert part.models[("A", 1, None)].n_train == int(mask[k].sum())
    assert not np.array_equal(
        part.models[("A", 1, None)].w_buy[0.5], full.models[("A", 1, None)].w_buy[0.5]
    )
    assert part.meta["cells"] == 1 and part.meta["min_rows"] == PM.MIN_ROWS
    assert part.meta["train_start"] <= part.meta["train_end"]


def test_small_cell_is_omitted_and_raises_missing_cell_error():
    m = _synthetic_master(days=5)
    mask = np.arange(len(m["ts"])) < 80
    model = PM.fit_model(m, tiers=("C",), horizons=(1, 24), train_mask=mask)
    assert PM.has_cell(model, "C", 1) and not PM.has_cell(model, "C", 24)
    mx = F.build_matrix(m, "C", h=24)
    with pytest.raises(PM.MissingCellError, match="fewer than 50"):
        PM.predict_rows(model, "C", 24, mx.X, mx.names)
    with pytest.raises(KeyError):
        PM.predict_one(model, "C", 24, {})


def test_one_class_label_gives_none_weight_and_half_probability():
    m = _synthetic_master(days=5)
    m["regulation_state"] = np.ones(len(m["ts"]))
    model = PM.fit_model(m, tiers=("A",), horizons=(1,))
    assert model.models[("A", 1, None)].w_dual is None
    mx = F.build_matrix(m, "A", h=1)
    model2 = PM.from_json(PM.to_json(model))
    assert model2.models[("A", 1, None)].w_dual is None
    assert np.all(PM.predict_rows(model2, "A", 1, mx.X, mx.names)["p_dual"] == 0.5)


def test_predict_one_matches_predict_rows():
    m = _synthetic_master(days=5)
    model = PM.fit_model(m, tiers=("C",), horizons=(1,))
    mx = F.build_matrix(m, "C", h=1)
    rows = PM.predict_rows(model, "C", 1, mx.X, mx.names)
    one = PM.predict_one(model, "C", 1, dict(zip(mx.names, mx.X[0], strict=True)))
    for q in PM.QUANTILES:
        assert one.buy[q] == pytest.approx(rows["buy_q%d" % round(q * 100)][0], rel=1e-12)
        assert one.sell[q] == pytest.approx(rows["sell_q%d" % round(q * 100)][0], rel=1e-12)
    assert one.p_dual == pytest.approx(rows["p_dual"][0], rel=1e-12)
    assert one.p_negsell == pytest.approx(rows["p_negsell"][0], rel=1e-12)
    assert one.p_spike == pytest.approx(rows["p_spike"][0], rel=1e-12)


def test_table_baseline_fallback_rows_get_training_wide_quantiles():
    m = _synthetic_master(days=5)
    n = len(m["ts"])
    train = np.arange(n) < n // 2
    tb = PM.TableBaseline().fit(m, train)
    tb.cells["buy"].clear()  # every cell unseen
    m["da_eur_mwh"][n - 1] = np.nan
    p = tb.predict_rows(m, np.array([n - 2, n - 1]))
    want = np.quantile(np.asarray(m["buy_eur_mwh"])[train], 0.5)
    assert np.array_equal(p["buy_q50"], [want, want])


def test_meta_dates_are_local_calendar_dates():
    # 2025-10-28 23:30 UTC is 2025-10-29 00:30 CET; 2026-03-28 23:30 UTC is 2026-03-29 local
    # (before the DST change that day, UTC+1), and 2026-10-24 22:30 UTC is 2026-10-25 local (CEST).
    import datetime as dt

    def utc(*a):
        return int(dt.datetime(*a, tzinfo=dt.UTC).timestamp())

    assert PM._iso_date(utc(2025, 10, 28, 23, 30)) == "2025-10-29"
    assert PM._iso_date(utc(2025, 10, 28, 22, 30)) == "2025-10-28"
    assert PM._iso_date(utc(2026, 10, 24, 22, 30)) == "2026-10-25"
    assert PM._iso_date(utc(2026, 10, 24, 21, 30)) == "2026-10-24"
    assert PM._iso_date(utc(2026, 10, 25, 22, 30)) == "2026-10-25"
    assert PM._iso_date(utc(2026, 10, 25, 23, 30)) == "2026-10-26"
