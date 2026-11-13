# tests/test_pack_update.py
import datetime as dt
import shutil

import numpy as np
import pytest
from betterybalance.core import pack as PK
from betterybalance.core import pricemodel as PM
from betterybalance.core.pack import PackError, load_pack
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import pack_update as PU

from tests.test_features import _synthetic_master
from tests.test_live_site import _flows, _model, _run, _site
from tests.test_pack import _quarters, _series

_SERIES: dict = {}


def _fitted():
    # fit once: fit_model stamps meta["fitted_at"] with the wall clock
    if not _SERIES:
        m = _synthetic_master(days=41)
        _SERIES["m"], _SERIES["s"] = m, _series(m)
    return _SERIES["m"], _SERIES["s"]


def _release(tmp_path, month="2026-11"):
    m, s = _fitted()
    src = tmp_path / "rel" / month
    PK.write_pack(src, month, _quarters(m), s, built="2026-11-01")
    return src


def test_install_newer_and_ignore_same(tmp_path):
    src = _release(tmp_path)
    rel = PU.check_release(str(src), None)
    assert rel.version == "2026-11" and rel.source == str(src)
    path = PU.install(rel, tmp_path / "pack")
    assert path == tmp_path / "pack" / "2026-11"
    assert load_pack(path).version == "2026-11" and PU.check_release(str(src), "2026-11") is None
    assert PU.check_release(str(src), "2026-12") is None  # older than the installed one
    # a file:// URL names the same folder
    assert PU.check_release(src.as_uri(), None).version == "2026-11"
    assert sorted(p.name for p in (tmp_path / "pack").iterdir()) == ["2026-11"]


def test_install_keeps_the_previous_version_and_drops_older_ones(tmp_path):
    dest = tmp_path / "pack"
    for month in ("2026-11", "2026-12", "2027-01"):
        PU.install(PU.check_release(str(_release(tmp_path, month)), None), dest)
    assert sorted(p.name for p in dest.iterdir()) == ["2026-12", "2027-01"]


def test_corrupt_release_leaves_the_installed_pack(tmp_path):
    good = PU.install(PU.check_release(str(_release(tmp_path)), None), tmp_path / "pack")
    bad = _release(tmp_path, "2026-12")
    (bad / "betterybalance-pack-2026-12.npz").write_bytes(b"broken")
    with pytest.raises(PackError):
        PU.install(PU.check_release(str(bad), "2026-11"), tmp_path / "pack")
    assert load_pack(good).version == "2026-11" and not list((tmp_path / "pack").glob(".tmp-*"))
    assert sorted(p.name for p in (tmp_path / "pack").iterdir()) == ["2026-11"]


def test_a_release_without_its_npz_fails_without_a_trace(tmp_path):
    src = _release(tmp_path)
    (src / PK.npz_name("2026-11")).unlink()
    with pytest.raises(PackError):
        PU.install(PU.check_release(str(src), None), tmp_path / "pack")
    assert not (tmp_path / "pack").exists() or not list((tmp_path / "pack").iterdir())


def test_vanished_source_raises_and_never_downgrades(tmp_path):
    with pytest.raises(PackError):
        PU.check_release(str(tmp_path / "missing"), "2026-11")
    with pytest.raises(PackError):
        PU.check_release((tmp_path / "missing").as_uri(), None)
    pack = load_pack(_release(tmp_path))
    shipped = PM.shipped_model()
    newest = int(pack.train_ends[-1])
    # the pack's newest training end is before the shipped one: the shipped model stays
    model, version = PU.choose_model((shipped, "shipped-x", newest + 900), pack)
    assert model is shipped and version == "shipped-x"
    # an equal training end is no upgrade either
    assert PU.choose_model((shipped, "shipped-x", newest), pack)[1] == "shipped-x"
    # later than the shipped one: the pack's newest
    assert PU.choose_model((shipped, "shipped-x", newest - 900), pack)[1] == pack.newest()[1]
    # never older than the model in use: a pack model in use that ends later stays
    cur = (pack.model(-1), "pack-2027-01-2027-01-04", newest + 86400)
    got = PU.choose_model((shipped, "shipped-x", newest - 900), pack, current=cur)
    assert got[1] == "pack-2027-01-2027-01-04" and got[0] is cur[0]
    # no pack: the shipped model, or the one in use when it is newer
    assert PU.choose_model((shipped, "shipped-x", 0), None)[1] == "shipped-x"
    assert PU.choose_model((shipped, "shipped-x", 0), None, current=cur)[1] == cur[1]


def test_a_pack_model_without_coverage_inherits_the_shipped_factors(tmp_path):
    pack = load_pack(_release(tmp_path))
    raw = pack.model(-1)
    assert {hm.cov_factor for hm in raw.models.values()} == {1.0}  # the ruling's premise
    shipped = PM.shipped_model()
    shared = [k for k in raw.models if k in shipped.models]
    assert shared and any(shipped.models[k].cov_factor != 1.0 for k in shared)
    model, version = PU.choose_model((shipped, "shipped-x", 0), pack)
    assert version == pack.newest()[1] and model.meta["cov_source"] == "shipped"
    for k in raw.models:
        want = shipped.models[k].cov_factor if k in shipped.models else 1.0
        assert model.models[k].cov_factor == want
    assert model.tables is raw.tables
    assert {hm.cov_factor for hm in pack.model(-1).models.values()} == {1.0}  # not mutated
    # a pack model with its own coverage keeps it
    own = PM.from_json(PM.to_json(raw))
    k0 = next(iter(own.models))
    own.models[k0].cov_factor = 1.25
    got, src = PU.with_coverage(own, shipped)
    assert src == "own" and got is own and got.models[k0].cov_factor == 1.25
    # the shipped model itself is "own"
    assert PU.cov_source(PU.choose_model((shipped, "shipped-x", 2**40), pack)[0]) == "own"


def test_shipped_entry_names_the_exclusive_training_end():
    shipped = PM.shipped_model()
    model, version, end = PU.shipped_entry(shipped)
    day = shipped.meta["train_end"]
    assert model is shipped and version == f"shipped-{day}"
    nxt = dt.date.fromisoformat(day) + dt.timedelta(days=1)
    assert end == int(dt.datetime(nxt.year, nxt.month, nxt.day, tzinfo=ZONE).timestamp())


def test_model_switch_waits_for_the_quarter_boundary():
    site, store, start = _site()
    old = site.lc.model_version
    new_model = _model(30, 14)
    _run(site, start, 7)  # minutes 0 .. 6 of the first quarter
    site.set_model(new_model, "v-new", site.next_minute)  # at minute 7
    ev = store.last_event("model_switch")
    assert ev["ts"] == start + 7 * 60
    assert ev["detail"] == {
        "version": "v-new", "effective_ts": start + 900, "from_version": old,
        "cov_source": "own",
    }
    versions = {}
    for ts in range(start + 7 * 60, start + 900 + 3 * 60, 60):
        if (ts - start) % 900 == 60:
            site.setup_quarter(ts)
        site.on_minute(ts, dict(_flows(ts), ts=ts))
        versions[(ts - start) // 60] = site.lc.model_version
    assert all(versions[k] == old for k in range(7, 15))
    assert all(versions[k] == "v-new" for k in (15, 16, 17))
    assert site.lc.model is new_model and site.source.version == "v-new"
    rows = store.decisions(start, start + 900 + 3 * 60)
    assert all(r["model_version"] == old for r in rows if r["ts"] < start + 900)
    assert all(r["model_version"] == "v-new" for r in rows if r["ts"] >= start + 900)
    assert store.window(start)["model_version"] == old
    assert store.window(start + 900)["model_version"] == "v-new"
    bands = store._db.execute(
        "SELECT DISTINCT model_version FROM bands WHERE recorded_at >= ?", (start + 900,)
    ).fetchall()
    assert bands == [("v-new",)]


def test_a_restart_before_the_boundary_keeps_the_switch_pending():
    site, store, start = _site()
    _run(site, start, 7)
    site.set_model(_model(30, 14), "v-new", site.next_minute)
    site.restore(start + 10 * 60)  # the same site object (an in-process restore)
    assert site.lc.model_version == "v-test"
    _run(site, start, 16)  # restore resumes at the quarter start
    assert site.lc.model_version == "v-new"


def test_set_model_records_the_shipped_coverage_source(tmp_path):
    site, store, _ = _site()
    pack = load_pack(_release(tmp_path))
    model, version = PU.choose_model((PM.shipped_model(), "shipped-x", 0), pack)
    site.set_model(model, version, site.next_minute)
    assert store.last_event("model_switch")["detail"]["cov_source"] == "shipped"


def test_model_state_from_the_events():
    ev = [
        {"seq": 1, "ts": 0, "kind": "restart", "detail": {}},
        {"seq": 2, "ts": 420, "kind": "model_switch",
         "detail": {"version": "a", "effective_ts": 900}},
        {"seq": 3, "ts": 2000, "kind": "model_switch", "detail": {"version": "b"}},
    ]
    assert PU.model_state(ev, 600, "s") == ("s", (900, "a"))
    assert PU.model_state(ev, 900, "s") == ("a", None)
    assert PU.model_state(ev, 2600, "s") == ("a", (2700, "b"))  # effective_ts defaults
    assert PU.model_state(ev, 2700, "s") == ("b", None)
    assert PU.model_state([], 2700, "s") == ("s", None)


def test_resolve_finds_a_version_in_the_pack_or_the_shipped_model(tmp_path):
    pack = load_pack(_release(tmp_path))
    shipped = PU.shipped_entry(PM.shipped_model())
    got = PU.resolve(pack.model_version(1), shipped, pack)
    assert got is not None and got[1] == int(pack.train_ends[1])
    assert got[0].meta["cov_source"] == "shipped"
    assert PU.resolve(shipped[1], shipped, pack) == (shipped[0], shipped[2])
    assert PU.resolve("pack-2020-01-2020-01-01", shipped, pack) is None
    assert PU.resolve(pack.model_version(1), shipped, None) is None


def test_store_records_the_installed_pack(tmp_path):
    store = Store(":memory:")
    store.migrate()
    assert store.active_pack() is None
    store.put_pack("2026-11", 100, {"version": "2026-11"}, "/x/2026-11")
    store.put_pack("2026-12", 200, {"version": "2026-12"}, "/x/2026-12")
    got = store.active_pack()
    assert got == {
        "version": "2026-12", "installed_at": 200, "manifest": {"version": "2026-12"},
        "path": "/x/2026-12", "active": 1,
    }
    rows = store._db.execute("SELECT version, active FROM pack ORDER BY version").fetchall()
    assert rows == [("2026-11", 0), ("2026-12", 1)]


def test_install_copies_only_the_pack_files(tmp_path):
    src = _release(tmp_path)
    (src / "notes.txt").write_text("not part of the pack", encoding="utf-8")
    path = PU.install(PU.check_release(str(src), None), tmp_path / "pack")
    assert sorted(p.name for p in path.iterdir()) == sorted([PK.MANIFEST, PK.npz_name("2026-11")])
    shutil.rmtree(src)  # the installed copy stands on its own
    assert np.asarray(load_pack(path).train_ends).size == 3


def test_a_crash_between_the_renames_restores_the_old_folder(tmp_path):
    dest = tmp_path / "pack"
    good = PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    # the state a crash leaves between ``final.rename(old)`` and ``tmp.rename(final)``
    good.rename(dest / ".old-2026-11")
    (dest / ".tmp-2026-11").mkdir()
    PU.install(PU.check_release(str(_release(tmp_path, "2026-12")), None), dest)
    assert sorted(p.name for p in dest.iterdir()) == ["2026-11", "2026-12"]
    assert load_pack(dest / "2026-11").version == "2026-11"


def test_recover_drops_an_old_folder_only_beside_a_loading_version(tmp_path):
    dest = tmp_path / "pack"
    good = PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    shutil.copytree(good, dest / ".old-2026-11")
    PU.recover(dest)  # the version loads: the old copy goes
    assert sorted(p.name for p in dest.iterdir()) == ["2026-11"]
    shutil.copytree(good, dest / ".old-2026-11")
    (good / PK.npz_name("2026-11")).write_bytes(b"broken")
    PU.recover(dest)  # the version does not load: the old copy replaces it
    assert sorted(p.name for p in dest.iterdir()) == ["2026-11"]
    assert load_pack(dest / "2026-11").version == "2026-11"


def test_a_partly_inherited_coverage_is_mixed(tmp_path):
    pack = load_pack(_release(tmp_path))
    shipped = PM.from_json(PM.to_json(PM.shipped_model()))
    gone = next(k for k in pack.model(-1).models if k in shipped.models)
    del shipped.models[gone]
    model, src = PU.with_coverage(pack.model(-1), shipped)
    assert src == "mixed" and model.meta["cov_source"] == "mixed"
    assert model.models[gone].cov_factor == 1.0


def test_resolve_falls_back_to_the_previous_pack(tmp_path):
    dest = tmp_path / "pack"
    PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    PU.install(PU.check_release(str(_release(tmp_path, "2026-12")), None), dest)
    assert PU.previous_folder(dest, "2026-12") == dest / "2026-11"
    assert PU.previous_folder(dest, "2026-11") is None
    active, previous = load_pack(dest / "2026-12"), load_pack(dest / "2026-11")
    shipped = PU.shipped_entry(PM.shipped_model())
    old = previous.model_version(2)
    assert old != active.model_version(2)  # the version names the pack's month
    assert PU.resolve(old, shipped, active) is None
    got = PU.resolve(old, shipped, active, previous)
    assert got is not None and got[1] == int(previous.train_ends[2])


def test_models_from_dir_maps_every_installed_model(tmp_path):
    dest = tmp_path / "pack"
    PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    PU.install(PU.check_release(str(_release(tmp_path, "2026-12")), None), dest)
    shipped = PM.shipped_model()
    models = PU.models_from_dir(dest, shipped)
    want = {PU.shipped_entry(shipped)[1]}
    for month in ("2026-11", "2026-12"):
        p = load_pack(dest / month)
        want |= {p.model_version(i) for i in range(len(p))}
    assert set(models) == want and len(models) == 7
    assert models[PU.shipped_entry(shipped)[1]] is shipped
    assert all(m.meta.get("cov_source") == "shipped" for k, m in models.items() if k[:5] == "pack-")
    assert PU.models_from_dir(tmp_path / "none", shipped) == {PU.shipped_entry(shipped)[1]: shipped}


def test_installs_never_prune_the_pack_of_the_model_in_use(tmp_path):
    dest = tmp_path / "pack"
    PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    in_use = load_pack(dest / "2026-11").model_version(2)
    keep = {PU.version_month(in_use)}
    assert keep == {"2026-11"}
    for month in ("2026-12", "2027-01"):  # two releases whose newest end does not advance
        PU.install(PU.check_release(str(_release(tmp_path, month)), None), dest, keep=keep)
    assert sorted(p.name for p in dest.iterdir()) == ["2026-11", "2026-12", "2027-01"]
    assert PU.month_folder(dest, in_use) == dest / "2026-11"
    assert in_use in PU.models_from_dir(dest, PM.shipped_model())
    PU.install(PU.check_release(str(_release(tmp_path, "2027-02")), None), dest)  # no keep
    assert sorted(p.name for p in dest.iterdir()) == ["2027-01", "2027-02"]
    assert PU.version_month("shipped-2026-10-30") is None and PU.month_folder(dest, None) is None
