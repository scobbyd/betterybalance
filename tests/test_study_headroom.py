import math

import numpy as np
import sklearn.ensemble
from betterybalance.core import evaluate as E
from betterybalance.core.features import build_matrix

from tests.test_features import _synthetic_master
from tools.study.october_2026 import (
    adopt_boosting,
    boosted_headroom,
    clip_to_first,
    relative_gain,
)


def test_adopt_boosting_rule():
    lin = {0: 20.0, 1: 22.0, 2: 24.0, 3: 25.0, 4: 26.0}
    assert adopt_boosting(lin, {h: v * 0.94 for h, v in lin.items()}) is True
    assert adopt_boosting(lin, {h: v * 0.97 for h, v in lin.items()}) is False


class _Recorder:
    calls: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        _Recorder.calls.append(("fit", X.copy()))
        return self

    def predict(self, X):
        _Recorder.calls.append(("predict", X.copy()))
        return np.zeros(len(X))


def test_boosted_headroom_fits_on_train_rows_and_scores_test_rows_only(monkeypatch):
    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingRegressor", _Recorder)
    _Recorder.calls = []
    m = _synthetic_master(days=20)
    ts = m["ts"]
    folds = E.walk_forward(ts, ts[96 * 14], ts[-1] + 900, refit_every_days=3, train_days=10)
    out = boosted_headroom(m, folds, tier="A", horizons=(1,))
    mx = build_matrix(m, "A", 1)
    k = np.searchsorted(ts, mx.ts_target)
    fits = [x for kind, x in _Recorder.calls if kind == "fit"]
    preds = [x for kind, x in _Recorder.calls if kind == "predict"]
    assert len(fits) == len(preds) == len(folds) and math.isfinite(out[1])
    for f, fx, px in zip(folds, fits, preds, strict=True):
        train = np.isin(k, f.train)
        test = np.isin(k, f.test)
        assert np.array_equal(fx, mx.X[train]) and np.array_equal(px, mx.X[test])
        assert mx.ts_target[train].max() < mx.ts_target[test].min()


def _res(values: dict) -> dict:
    return {t: {h: {"buy": {"mae_q50": v}} for h, v in per.items()} for t, per in values.items()}


def test_relative_gain_incomplete_horizons_give_no_verdict():
    full = _res({"A": {1: 50.0, 2: 50.0, 3: 50.0, 4: 50.0}, "B": {1: 45, 2: 45, 3: 45, 4: 45}})
    r = relative_gain(full, "B", "A")
    assert abs(r["mean"] - 0.1) < 1e-12 and r["load_bearing"] is True
    part = _res({"A": {1: 50.0, 2: 50.0}, "B": {1: 45.0, 2: 45.0}})
    r = relative_gain(part, "B", "A")
    assert math.isnan(r["mean"]) and r["load_bearing"] is None


def test_clip_to_first_drops_early_train_rows_only():
    ts = np.arange(0, 100 * 900, 900, dtype=np.int64)
    folds = [E.Fold(train=np.arange(0, 50), test=np.arange(50, 60))]
    out = clip_to_first(folds, ts, int(ts[20]))
    assert out[0].train.min() == 20 and np.array_equal(out[0].test, folds[0].test)


def test_ida_window_clips_only_when_a_whole_day_is_missing():
    from tools.study.october_2026 import ida_window

    m = _synthetic_master(days=6)
    ts = m["ts"]
    for col in ("ida1_eur_mwh", "ida2_eur_mwh", "ida3_eur_mwh"):
        m[col] = np.full(len(ts), 50.0)
    m["ida3_eur_mwh"][:48] = np.nan  # afternoon-only auction: first value at 12:00 on day one
    info, clip = ida_window(m, int(ts[0]))
    assert clip is None and info["clipped"] is False
    m["ida1_eur_mwh"][: 96 * 2] = np.nan  # IDA1 history starts on day three
    info, clip = ida_window(m, int(ts[0]))
    assert clip == int(ts[96 * 2]) and info["clipped"] is True
