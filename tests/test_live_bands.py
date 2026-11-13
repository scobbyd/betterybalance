import hashlib

import numpy as np
import pytest
from betterybalance.core import pricemodel as PM
from betterybalance.core.store import Store
from betterybalance.live import bands as LB
from betterybalance.replay import bands as RB

from tests.test_features import _synthetic_master

# sha256 of _cache_digest on the 41-day build_band_cache below, recorded on d821a9e
# (before cache_from_models was factored out)
CACHE_GOLDEN = "85fa16699729d32bed50510dec69a37c57e6f1e23551dad07a648145c1ef3194"


def _key_text(key) -> str:
    tier, h, e = key
    return "%s:%d:%s" % (tier, h, "-" if e is None else e)


def _cache_digest(c) -> str:
    d = hashlib.sha256()
    for a in (c.ts, c.fold_of, c.fold_train_end):
        d.update(np.ascontiguousarray(a, dtype=np.int64).tobytes())
    for group in ("take", "feed", "p_dual", "cov"):
        arrays = getattr(c, group)
        for text, key in sorted((_key_text(k), k) for k in arrays):
            d.update(text.encode())
            d.update(np.ascontiguousarray(arrays[key], dtype=np.float64).tobytes())
    return d.hexdigest()


def _model_and_master():
    m = _synthetic_master(days=30)
    train = np.zeros(len(m["ts"]), bool)
    train[: 96 * 20] = True
    return PM.fit_model(m, ("A", "C"), (0, 1, 4), train_mask=train), m


def test_model_source_matches_the_cache_without_embargo():
    model, m = _model_and_master()
    t = int(m["ts"][96 * 25])
    cache = RB.cache_from_models(
        m,
        [model],
        np.array([int(m["ts"][96 * 20])]),
        int(m["ts"][96 * 21]),
        int(m["ts"][-1]) + 900,
        ("A", "C"),
        (0, 1, 4),
        lag=0,
    )
    got = LB.ModelBandSource(model, "v1", m).bands("A", 4, None, t)
    k = int(np.searchsorted(cache.ts, t))
    assert [got["take"][q] for q in PM.QUANTILES] == pytest.approx(
        cache.take[("A", 4, None)][:, k].tolist()
    )
    assert got["cov"] == model.cov_factor("A", 4, 0.5)


def test_stored_source_returns_exactly_what_was_served():
    model, m = _model_and_master()
    s = Store(":memory:")
    s.migrate()
    t = int(m["ts"][96 * 25])
    live = LB.ModelBandSource(model, "v1", m, record=s.put_band)
    live.set_now(t - 840)
    a = live.bands("C", 1, None, t)
    stored = LB.StoredBandSource(s, "v1", ("A", "C"), model.tables)
    stored.set_now(t - 840)
    b = stored.bands("C", 1, None, t)
    other = LB.StoredBandSource(s, "v2", ("A",), {})
    other.set_now(t - 840)
    assert a == b and other.bands("C", 1, None, t) is None
    stored.set_now(t - 900)  # before the serve: nothing was recorded yet
    assert stored.bands("C", 1, None, t) is None


def test_stored_source_serves_each_band_as_of_its_serve_time():
    model, m = _model_and_master()
    s = Store(":memory:")
    s.migrate()
    t = int(m["ts"][96 * 25])
    t1, t2 = t - 840, t - 780
    live = LB.ModelBandSource(model, "v1", m, record=s.put_band)
    live.set_now(t1)
    a = live.bands("A", 4, None, t)
    live.set_now(t1 + 30)
    assert live.bands("A", 4, None, t) == a  # an identical re-serve adds no row
    late = dict(m, da_eur_mwh=m["da_eur_mwh"] + 50.0)  # a late feed changes the cell
    live.refresh(late)
    live.set_now(t2)
    b = live.bands("A", 4, None, t)
    assert a != b
    rows = s._db.execute("select recorded_at from bands order by recorded_at").fetchall()
    assert rows == [(t1,), (t2,)]
    stored = LB.StoredBandSource(s, "v1", ("A",), model.tables)
    for now, want in ((t1, a), (t1 + 30, a), (t2, b), (t2 + 600, b)):
        stored.set_now(now)
        assert stored.bands("A", 4, None, t) == want


def test_stored_source_needs_a_serve_time():
    model, m = _model_and_master()
    s = Store(":memory:")
    s.migrate()
    with pytest.raises(ValueError, match="set_now"):
        LB.StoredBandSource(s, "v1", ("A",), model.tables).bands(
            "A", 4, None, int(m["ts"][96 * 25])
        )


def test_recording_needs_a_serve_time():
    model, m = _model_and_master()
    s = Store(":memory:")
    s.migrate()
    with pytest.raises(ValueError, match="set_now"):
        LB.ModelBandSource(model, "v1", m, record=s.put_band).bands(
            "A", 4, None, int(m["ts"][96 * 25])
        )


def test_healthy_tiers_drop_c_without_live_values():
    model, m = _model_and_master()
    t = int(m["ts"][96 * 25])
    for c in ("buy_live_end", "sell_live_end"):
        m[c] = m[c].copy()
        m[c][96 * 24 :] = np.nan
    assert LB.ModelBandSource(model, "v1", m).healthy_tiers(t) == ("A",)


def test_build_band_cache_is_unchanged():
    m = _synthetic_master(days=41)
    c = RB.build_band_cache(
        m,
        int(m["ts"][96 * 30]),
        int(m["ts"][-1]) + 900,
        tiers=("A",),
        horizons=(0, 1, 4),
        refit_every_days=7,
        train_days=20,
    )
    assert _cache_digest(c) == CACHE_GOLDEN  # recorded on the unchanged code


def test_model_source_serves_targets_whose_price_is_not_settled_yet():
    model, m = _model_and_master()
    t = int(m["ts"][96 * 25])
    want = LB.ModelBandSource(model, "v1", m).bands("C", 1, None, t)
    for c in ("buy_eur_mwh", "sell_eur_mwh", "regulation_state"):
        m[c] = m[c].copy()
        m[c][96 * 24 :] = np.nan  # a live master: the newest settlements end a day back
    assert want is not None
    src = LB.ModelBandSource(model, "v1", m)
    assert src.bands("C", 1, None, t) == want and src.healthy_tiers(t) == ("A", "C")


def test_a_skipped_tier_is_never_healthy():
    from betterybalance.live.bands import ModelBandSource

    src = object.__new__(ModelBandSource)
    src.skip = ("B",)
    src._band = lambda tier, h, e, ts: {"ok": True}
    assert src.healthy_tiers(0) == ("A", "C", "D")
    src.skip = ()
    assert src.healthy_tiers(0) == ("A", "B", "C", "D")
