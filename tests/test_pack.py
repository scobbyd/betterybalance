import json

import numpy as np
import pytest
from betterybalance.core import pack as PK
from betterybalance.core import pricemodel as PM
from betterybalance.replay import bands as RB

from tests.test_features import _synthetic_master


def _series(m):
    out = []
    for d in (10, 17, 24):
        train = np.zeros(len(m["ts"]), bool)
        train[: 96 * d] = True
        out.append((int(m["ts"][96 * d]), PM.fit_model(m, ("A",), (0, 1, 4), train_mask=train)))
    return out


def _quarters(m):
    q = {c: np.asarray(m[c]) for c in PK.QUARTER_COLUMNS if c in m}
    for c in ("ida1_eur_mwh", "ida2_eur_mwh", "ida3_eur_mwh"):
        q[c] = np.full(len(m["ts"]), np.nan)
    return q


@pytest.fixture(scope="module")
def ms():
    # fit once: fit_model stamps meta["fitted_at"] with the wall clock
    m = _synthetic_master(days=41)
    return m, _series(m)


def test_write_and_load_roundtrip(tmp_path, ms):
    m, s = ms
    man = PK.write_pack(tmp_path, "2026-11", _quarters(m), s, built="2026-11-01")
    p = PK.load_pack(tmp_path)
    assert man["n_models"] == 3 and p.version == "2026-11"
    assert p.train_ends.tolist() == [t for t, _ in s]
    assert PM.to_json(p.model(1)) == PM.to_json(s[1][1])
    assert np.isnan(p.quarters["ida1_eur_mwh"]).all()
    assert man["ida_included"] is False and "da_eur_mwh" in man["licences"]


def test_sha_mismatch_is_refused(tmp_path, ms):
    m, s = ms
    PK.write_pack(tmp_path, "2026-11", _quarters(m), s, built="2026-11-01")
    man = json.loads((tmp_path / "manifest.json").read_text())
    man["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(PK.PackError, match="sha256"):
        PK.load_pack(tmp_path)


def test_series_bands_respect_the_embargo(tmp_path, ms):
    m, s = ms
    PK.write_pack(tmp_path, "2026-11", _quarters(m), s, built="2026-11-01")
    p = PK.load_pack(tmp_path)
    c = RB.cache_from_series(p, int(m["ts"][96 * 10]), int(m["ts"][-1]) + 900, ("A",), (0, 1, 4))
    first = int(np.searchsorted(c.ts, p.train_ends[0] + RB.embargo_lag((0, 1, 4))))
    assert (c.fold_of[:first] == -1).all() and c.fold_of[first] == 0


def test_pack_bytes_are_deterministic_and_master_roundtrips(tmp_path, ms):
    m, s = ms
    a = PK.write_pack(tmp_path / "a", "2026-11", _quarters(m), s, built="2026-11-01")
    b = PK.write_pack(tmp_path / "b", "2026-11", _quarters(m), s, built="2026-11-01")
    name = PK.npz_name("2026-11")
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert a == b and a["coverage"] == {"from": "2026-10-14", "to": "2026-11-23"}
    p = PK.load_pack(tmp_path / "a")
    mm = p.master()
    for key in m:
        if key.startswith("ida"):
            continue
        assert np.array_equal(np.asarray(mm[key]), np.asarray(m[key]), equal_nan=True), key
    model, version, end = p.newest()
    # 96 * 24 slots after 2026-10-14 00:00 cross the DST change: the end is 2026-11-06 23:00
    # local, so the last trained quarter (22:45) is on 2026-11-06
    assert end == s[-1][0] and version == "pack-2026-11-2026-11-06"
    # the end 2026-10-24 00:00 is exclusive: the last trained day is 2026-10-23
    assert p.model_version(0) == "pack-2026-11-2026-10-23"
    assert PM.to_json(model) == PM.to_json(s[-1][1])


def test_serves_format_and_contents_are_checked(tmp_path, ms):
    m, s = ms
    s = s[:1]
    PK.write_pack(tmp_path, "2026-11", _quarters(m), s, built="2026-11-01", serves=">=9.0")
    with pytest.raises(PK.PackError, match="pack serves >=9.0"):
        PK.load_pack(tmp_path)
    assert PK.serves_version(">=0.3.0,<0.4", "0.3.0") and not PK.serves_version(">0.3", "0.3.0")
    man = PK.read_manifest(tmp_path)
    man["format"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(PK.PackError, match="format"):
        PK.load_pack(tmp_path)
    q = _quarters(m)
    del q["buy_live_end"]
    with pytest.raises(PK.PackError, match="buy_live_end"):
        PK.write_pack(tmp_path, "2026-11", q, s, built="2026-11-01")
    with pytest.raises(PK.PackError, match="ascending"):
        PK.write_pack(tmp_path, "2026-11", _quarters(m), s + s, built="2026-11-01")


def test_builder_series_ends_at_the_data_end_and_never_trains_past_its_end():
    from tools import build_data_pack as B

    m = _synthetic_master(days=41)
    ts = m["ts"]
    start, end = int(ts[0]), int(ts[-1]) + 900
    folds = B.series_folds(ts, start, end, refit_days=7, train_days=14)
    ends = [e for e, _ in folds]
    assert ends[-1] == end and len(ends) == 4
    assert all(B.plus_local_days(a, 7) == b for a, b in zip(ends, ends[1:], strict=False))
    assert B.plus_local_days(ends[0], -14) >= start > B.plus_local_days(ends[0], -21)
    for e, rows in folds:
        assert ts[rows].max() + 900 == e and ts[rows].min() == B.plus_local_days(e, -14)


def test_builder_fit_on_the_cut_master_equals_the_full_fit():
    from tools import build_data_pack as B

    m = _synthetic_master(days=20)
    rows = np.arange(96 * 9, 96 * 16)
    mask = np.zeros(len(m["ts"]), bool)
    mask[rows] = True
    kw = {"tiers": ("C", "D"), "horizons": (0, 24)}  # the longest lookback: tier C at h24
    full = PM.fit_model(m, train_mask=mask, meta={"pack": "x"}, **kw)
    part = B._fit(m, rows, {"pack": "x"}, **kw)
    full.meta.pop("fitted_at")
    part.meta.pop("fitted_at")
    assert len(full.models) == 8 and PM.to_json(part) == PM.to_json(full)


def _rewrite_manifest(path, **changes):
    man = json.loads((path / "manifest.json").read_text())
    man.update(changes)
    (path / "manifest.json").write_text(json.dumps(man))


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"version": "2026-13"}, "YYYY-MM"),
        ({"version": "../x"}, "YYYY-MM"),
        ({"n_models": "1"}, "n_models"),
        ({"n_models": 1.0}, "n_models"),
        ({"train_ends": [1.5]}, "train_ends"),
        ({"train_ends": "x"}, "train_ends"),
        ({"serves": ">=0.3.0b1"}, "serves"),
    ],
)
def test_malformed_manifests_raise_pack_error(tmp_path, ms, changes, match):
    m, s = ms
    PK.write_pack(tmp_path, "2026-11", _quarters(m), s[:1], built="2026-11-01")
    _rewrite_manifest(tmp_path, **changes)
    with pytest.raises(PK.PackError, match=match):
        PK.load_pack(tmp_path)


def test_prerelease_version_and_gapped_grid_raise_pack_error(tmp_path, ms):
    m, s = ms
    with pytest.raises(PK.PackError, match="0.3.0b1"):
        PK.serves_version(">=0.3.0", "0.3.0b1")
    q = {c: np.delete(np.asarray(v), 5) for c, v in _quarters(m).items()}
    with pytest.raises(PK.PackError, match="contiguous"):
        PK.write_pack(tmp_path, "2026-11", q, s[:1], built="2026-11-01")


def test_models_are_read_by_index_not_by_sorted_name(tmp_path, ms):
    m, s = ms
    PK.write_pack(tmp_path, "2026-11", _quarters(m), s, built="2026-11-01")
    _rewrite_manifest(tmp_path, n_models=4, train_ends=[t for t, _ in s] + [s[-1][0] + 900])
    with pytest.raises(PK.PackError, match="model_003"):
        PK.load_pack(tmp_path)


def test_cut_series_cache_equals_the_uncut_cache(tmp_path, ms):
    from betterybalance.core import features as F

    m, s = ms
    PK.write_pack(tmp_path, "2026-11", _quarters(m), s, built="2026-11-01")
    p = PK.load_pack(tmp_path)
    tiers, horizons = ("A", "C", "D"), (0, 1, 24)
    start, end = int(m["ts"][96 * 20]), int(m["ts"][96 * 35])
    cut = RB.cache_from_series(p, start, end, tiers, horizons)
    full = RB.cache_from_models(
        p.master(), [p.model(i) for i in range(3)], p.train_ends, start, end, tiers, horizons
    )
    assert F.LOOKBACK_ISP == 24 + 2 + 12 - 1
    assert len(cut.ts) == 96 * 15 + F.LOOKBACK_ISP < len(full.ts)
    k = np.searchsorted(full.ts, cut.ts)
    assert np.array_equal(cut.fold_of, full.fold_of[k]) and (cut.fold_of >= 0).any()
    for group in ("take", "feed", "p_dual", "cov"):
        a, b = getattr(cut, group), getattr(full, group)
        assert a.keys() == b.keys()
        for key in a:
            assert np.array_equal(a[key], b[key][..., k], equal_nan=True), (group, key)


def test_builder_refuses_a_span_shorter_than_one_window_with_a_usage_error(tmp_path, ms, capsys):
    from tools import build_data_pack as B

    m, _ = ms
    cols = ("da_eur_mwh", "buy_eur_mwh", "sell_eur_mwh", "regulation_state", *PK.IDA_COLUMNS)
    np.savez(tmp_path / "master_inputs.npz", ts=m["ts"], **{c: np.asarray(m[c]) for c in cols})
    argv = ["--data", str(tmp_path), "--from", "2026-10-14", "--to", "2026-11-22"]
    with pytest.raises(SystemExit) as exc:
        B.main([*argv, "--month", "2026-11", "--out", str(tmp_path / "out")])
    assert exc.value.code == 2 and "less than one training window" in capsys.readouterr().err
