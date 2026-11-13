import numpy as np
import pytest
from betterybalance.core import evaluate as E

from tests.test_features import _synthetic_master


def test_auc_mae_and_wrong_side():
    y = np.array([0, 0, 1, 1], bool)
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert E.auc(y, s) == 0.75 and E.mae(np.array([1.0, 3.0]), np.array([2.0, 2.0])) == 1.0
    assert np.isnan(E.auc(np.zeros(3, bool), s[:3]))
    da = np.array([100.0, 100.0, 100.0, 100.0])
    assert (
        E.wrong_side_rate(
            np.array([120.0, 80.0, 100.0, 130.0]), np.array([110.0, 90.0, 50.0, 90.0]), da
        )
        == 1 / 3
    )


def test_walk_forward_blocks_do_not_overlap():
    m = _synthetic_master(days=30)
    ts = m["ts"]
    start = ts[96 * 20]
    end = ts[-1] + 900
    folds = E.walk_forward(ts, start, end, refit_every_days=5, train_days=15)
    assert (
        len(folds) == 2
        and folds[0].test.max() < folds[1].test.min()
        and folds[1].train.max() < folds[1].test.min()
    )


def test_evaluate_and_decay_shape():
    m = _synthetic_master(days=30)
    ts = m["ts"]
    folds = E.walk_forward(ts, ts[96 * 22], ts[-1] + 900, refit_every_days=4, train_days=20)
    r = E.evaluate(m, folds, tiers=("A", "C"), horizons=(1, 4))
    assert "mae_q50" in r["C"][1]["buy"] and "auc_dual" in r["C"][1] and "baseline" in r
    assert (
        "mae" in r["naive_da"][1]["buy"]
        and "mae" in r["naive_persist"][4]["sell"]
        and "wrong_side" in r["C"][1]["buy"]
    )
    assert r["A"][1]["buy"]["n"] == r["C"][1]["buy"]["n"] == r["naive_da"][1]["buy"]["n"]
    d = E.decay_curve(r)
    assert d["A"][1] == 0.0 and set(d["C"]) == {1, 4}
    md = E.render_markdown(r, d, E.provisional_gap(m), {"window": "test"})
    assert "| h |" in md and "," in md and "| n |" in md


def _local(day: str) -> int:
    import datetime as dt

    from betterybalance.core.timegrid import ZONE

    return int(dt.datetime.fromisoformat(day).replace(tzinfo=ZONE).timestamp())


def test_walk_forward_steps_local_days_across_dst():
    import datetime as dt

    from betterybalance.core.timegrid import ZONE, isp_grid

    ts = isp_grid(_local("2026-10-10"), _local("2026-11-01"))
    folds = E.walk_forward(ts, _local("2026-10-22"), _local("2026-10-29"), 1, train_days=7)
    assert len(folds) == 7
    sizes = [len(f.test) for f in folds]
    assert sizes == [96, 96, 96, 100, 96, 96, 96]
    for f in folds:
        first = dt.datetime.fromtimestamp(int(ts[f.test[0]]), ZONE)
        train0 = dt.datetime.fromtimestamp(int(ts[f.train[0]]), ZONE)
        assert (first.hour, first.minute) == (0, 0) and (train0.hour, train0.minute) == (0, 0)
        assert (first.date() - train0.date()).days == 7
        assert f.train.max() < f.test.min() and not set(f.train) & set(f.test)
    week = E.walk_forward(ts, _local("2026-10-22"), _local("2026-10-29"), 7, train_days=7)
    assert len(week) == 1 and len(week[0].test) == 7 * 96 + 4


def test_decay_curve_keys_h0_per_elapsed_minute():
    def cell(v):
        return {"buy": {"mae_q50": v}}

    r = {
        "A": {0: {1: cell(50.0), 5: cell(50.0), 10: cell(40.0)}, 1: cell(50.0)},
        "D": {0: {1: cell(45.0), 5: cell(40.0), 10: cell(20.0)}, 1: cell(40.0)},
    }
    d = E.decay_curve(r)
    assert d["D"][1] == pytest.approx(0.2)
    assert d["D"][0] == pytest.approx({1: 0.1, 5: 0.2, 10: 0.5})
    assert d["A"][0] == {1: 0.0, 5: 0.0, 10: 0.0}
