import datetime as dt
import zoneinfo

import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core.timegrid import isp_grid

AMS = zoneinfo.ZoneInfo("Europe/Amsterdam")


def _e(*a):
    return int(dt.datetime(*a, tzinfo=AMS).timestamp())


def _synthetic_master(days=3):
    start = _e(2026, 10, 14, 0)
    g = isp_grid(start, start + days * 86400)  # whole 96-slot days, also across the DST change
    n = len(g)
    rng = np.random.default_rng(1)
    da = 80 + 30 * np.sin(np.arange(n) / 96 * 2 * np.pi)
    settled = {
        "ts": g,
        "buy_eur_mwh": da + rng.normal(0, 20, n),
        "sell_eur_mwh": da + rng.normal(0, 20, n),
        "regulation_state": rng.choice([-1, 0, 1, 2], n).astype(float),
    }
    ida = {
        a: {"ts": g, "ida_eur_mwh": da + 5.0, "ida_volume_mwh": np.full(n, np.nan)}
        for a in ("IDA1", "IDA2", "IDA3")
    }
    live = {
        "ts": g,
        **{
            c: da + 1.0
            for c in (
                "buy_live_end",
                "sell_live_end",
                "buy_live_e1",
                "sell_live_e1",
                "buy_live_e5",
                "sell_live_e5",
                "buy_live_e10",
                "sell_live_e10",
            )
        },
    }
    return F.build_master({"ts": g, "da_eur_mwh": da}, settled, ida, live, g)


def test_ida3_is_unknown_for_morning_targets_and_before_clearing():
    m = _synthetic_master()
    mx = F.build_matrix(m, "B", h=1)
    i3 = mx.names.index("ida3_known")
    hours = np.array([dt.datetime.fromtimestamp(t, AMS).hour for t in mx.ts_target])
    assert not mx.X[hours < 12, i3].any()  # morning: never
    assert mx.X[(hours >= 12), i3].all()  # afternoon at h=1: known (forecast after 10:10)
    mx24 = F.build_matrix(m, "B", h=24)  # 6 h ahead: a 15:00 target forecast at 09:01
    hours24 = np.array([dt.datetime.fromtimestamp(t, AMS).hour for t in mx24.ts_target])
    assert not mx24.X[(hours24 >= 12) & (hours24 < 16), i3].any()


def test_dual_label_comes_from_state_code_and_lag1_is_live_end():
    m = _synthetic_master()
    mx = F.build_matrix(m, "C", h=1)
    k = np.searchsorted(m["ts"], mx.ts_target)
    assert np.array_equal(mx.y_dual, m["regulation_state"][k] == 2)
    assert np.allclose(mx.X[:, mx.names.index("lag1_buy")], m["buy_live_end"][k - 2])


def test_tier_d_h0_uses_elapsed_columns():
    m = _synthetic_master()
    mx = F.build_matrix(m, "D", h=0, elapsed_min=5)
    assert "buy_live_e5" in mx.names and "elapsed_min" in mx.names
    assert mx.X[0, mx.names.index("elapsed_min")] == 5


def test_build_master_rejects_gapped_grid():
    g = isp_grid(_e(2026, 10, 14, 0), _e(2026, 10, 15, 0))
    g = np.delete(g, 10)
    n = len(g)
    settled = {"ts": g, **{c: np.zeros(n) for c in ("buy_eur_mwh", "sell_eur_mwh")}}
    settled["regulation_state"] = np.zeros(n)
    with pytest.raises(ValueError, match="contiguous"):
        F.build_master({"ts": g, "da_eur_mwh": np.zeros(n)}, settled, {}, None, g)


def test_index_arithmetic_with_distinct_ramps():
    g = isp_grid(_e(2026, 10, 14, 0), _e(2026, 10, 17, 0))
    n = len(g)
    i = np.arange(n, dtype=float)
    settled = {
        "ts": g,
        "buy_eur_mwh": 2000 + i,
        "sell_eur_mwh": 3000 + i,
        "regulation_state": np.zeros(n),
    }
    live = {"ts": g}
    for j, c in enumerate(
        ("buy_live_end", "sell_live_end", "buy_live_e1", "sell_live_e1",
         "buy_live_e5", "sell_live_e5", "buy_live_e10", "sell_live_e10")
    ):
        live[c] = (4000 + 1000 * j) + i
    m = F.build_master({"ts": g, "da_eur_mwh": 1000 + i}, settled, {}, live, g)
    mx = F.build_matrix(m, "D", h=2)
    k = 150
    row = int(np.flatnonzero(mx.ts_target == g[k])[0])

    def col(name):
        return mx.X[row, mx.names.index(name)]

    assert col("lag1_buy") == 4000 + (k - 3)
    assert col("lag2_buy") == 4000 + (k - 4)
    assert col("lag4_buy") == 4000 + (k - 6)
    assert np.isclose(col("roll1h_buy_mean"), 4000 + np.mean(np.arange(k - 7, k - 3)))
    assert np.isclose(col("roll3h_buy_mean"), 4000 + np.mean(np.arange(k - 15, k - 3)))
    assert col("buy_live_e1") == 6000 + (k - 2)
    # settled columns are labels only: no feature value comes from the settled buy ramp
    settled_vals = {2000.0 + j for j in range(k - 20, k + 1)}
    assert not settled_vals & set(mx.X[row].tolist())


def test_label_free_matrix_keeps_unsettled_rows_with_the_same_features():
    m = _synthetic_master(days=3)
    full = F.build_matrix(m, "C", 1)
    for c in ("buy_eur_mwh", "sell_eur_mwh", "regulation_state"):
        m[c] = m[c].copy()
        m[c][96 * 2 :] = np.nan
    assert len(F.build_matrix(m, "C", 1).ts_target) < len(full.ts_target)
    free = F.build_matrix(m, "C", 1, labels=False)
    assert free.ts_target.tolist() == full.ts_target.tolist()
    assert np.array_equal(free.X, full.X) and free.names == full.names
    assert np.isnan(free.y_buy).all() and np.isnan(free.y_dual).all()
