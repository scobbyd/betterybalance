# tests/test_replay_bands.py
import numpy as np
from betterybalance.replay import bands as BC

from tests.test_features import _synthetic_master


def test_bands_exist_only_in_test_rows_and_never_from_own_training(tmp_path):
    m = _synthetic_master(days=40)
    ts = m["ts"]
    start, end = int(ts[96 * 25]), int(ts[-1] + 900)
    c = BC.build_band_cache(
        m, start, end, tiers=("A", "C"), horizons=(0, 1, 4), refit_every_days=5, train_days=20
    )
    k = c.take[("A", 1, None)]
    assert k.shape == (5, len(ts)) and np.isnan(k[2, : 96 * 25]).all()
    assert np.isfinite(k[2, 96 * 30])
    rows = np.flatnonzero(c.fold_of >= 0)
    assert (ts[rows] >= c.fold_train_end[c.fold_of[rows]]).all()
    assert np.all(np.diff(c.fold_of[rows]) >= 0)
    first = rows[c.fold_of[rows] == 0]
    later = rows[c.fold_of[rows] == 2]
    assert np.all(c.cov[("A", 1, None)][first] == 1.0)
    assert np.any(c.cov[("A", 1, None)][later] != 1.0)
    assert np.all(np.diff(k[:, 96 * 30]) >= 0) and len(c.tables) == len(c.fold_train_end)
    BC.save_cache(c, tmp_path / "b.npz")
    back = BC.load_cache(tmp_path / "b.npz")
    assert np.array_equal(back.take[("C", 0, 5)], c.take[("C", 0, 5)], equal_nan=True)
    assert back.meta["key"] == c.meta["key"]
    only_a = BC.build_band_cache(
        m, start, end, tiers=("A",), horizons=(0, 1, 4), refit_every_days=5, train_days=20
    )
    assert only_a.meta["key"] != c.meta["key"]
    corrected = dict(m, buy_eur_mwh=m["buy_eur_mwh"] + 0.01)
    args = (start, end, 5, 20, ("A", "C"), (0, 1, 4))
    assert BC.cache_key(m, *args) == c.meta["key"] != BC.cache_key(corrected, *args)


def test_cached_source_answers_or_returns_none():
    m = _synthetic_master(days=40)
    ts = m["ts"]
    c = BC.build_band_cache(
        m,
        int(ts[96 * 25]),
        int(ts[-1] + 900),
        tiers=("A",),
        horizons=(0, 1),
        refit_every_days=5,
        train_days=20,
    )
    src = BC.CachedBandSource(c)
    b = src.bands("A", 1, None, int(ts[96 * 30]))
    assert set(b["take"]) == {0.1, 0.25, 0.5, 0.75, 0.9} and 0.0 <= b["p_dual"] <= 1.0
    assert b["cov"] > 0
    assert src.bands("A", 1, None, int(ts[10])) is None
    assert src.bands("D", 1, None, int(ts[96 * 30])) is None
    assert src.healthy_tiers(int(ts[96 * 30])) == ("A",)
    assert "dual_rate" in src.tables_at(int(ts[96 * 30]))


def test_embargo_keeps_every_priced_row_far_from_its_training_end():
    m = _synthetic_master(days=40)
    ts = m["ts"]
    start = int(ts[96 * 25])
    c = BC.build_band_cache(
        m,
        start,
        int(ts[-1] + 900),
        tiers=("A",),
        horizons=(1, 24),
        refit_every_days=5,
        train_days=20,
    )
    embargo = 4 * 86400  # the ruled 96 h, a literal so a weaker EMBARGO_S fails here
    assert BC.EMBARGO_S >= embargo
    lag = 24 * 900 + embargo
    rows = np.flatnonzero(c.fold_of >= 0)
    f = c.fold_of[rows]
    for h in c.meta["horizons"]:  # every cell of a priced row was forecast 72 h after training
        assert (ts[rows] - h * 900 + 60 - embargo >= c.fold_train_end[f]).all()
    assert ts[rows[0]] - start >= lag
    later = f + 1 < len(c.fold_train_end)  # the latest fold that satisfies the embargo prices
    assert (ts[rows][later] - lag < c.fold_train_end[f[later] + 1]).all()
    assert np.isnan(c.take[("A", 24, None)][2, ts < ts[rows[0]]]).all()
    src = BC.CachedBandSource(c)
    assert src.tables_at(int(c.fold_train_end[1])) is c.tables[0]  # follows fold_of
