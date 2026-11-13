# tests/test_reexec.py
import datetime as dt
import math
from types import SimpleNamespace

import numpy as np
import pytest
from betterybalance.core import ledger as LG
from betterybalance.core import pricemodel as PM
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import site as LS
from betterybalance.replay import __main__ as CLI
from betterybalance.replay import reexec as RX

from tests.test_live_site import _flows, _run, _site


def test_reexecution_is_byte_identical_across_a_restart():
    site, store, start = _site()
    _run(site, start, 1440)
    # a settlement pull applied at the next minute: the first hour rebooked settled
    rows = [
        {
            "quarter_ts": start + 900 * k, "take": 180.0 + k, "feed": 20.0 - k,
            "regulation_state": k % 3 - 1, "published_at": start + 86000, "source": "tennet",
        }
        for k in range(4)
    ]
    assert site.settle(rows, site.next_minute) == [start + 900 * k for k in range(4)]
    ev = store.last_event("settle")
    assert ev["detail"]["published_at"] == [start + 86000] * 4
    # a row stored after the event but published before it: the event names its own rows
    store.put_settlement(dict(rows[1], take=999.0, published_at=start + 86200))
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 1440 * 60 + 1800)
    _run(site2, start + 1440 * 60 + 1800, 600)
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 1440 * 60 + 1800 + 600 * 60, out)
    assert r.differences == [] and RX.compare(store, out, start, start + 2 * 86400) == []
    assert out.head() == store.head()
    assert r.n_decisions == len(store.decisions(start, start + 2 * 86400)) > 0
    assert r.n_bookings == len(store.bookings(start, start + 2 * 86400)) == (96 + 2 + 40 + 4) * 3
    settled = [b for b in out.bookings(start, start + 3600) if b["price_source"] == "settled"]
    assert len(settled) == 12 and {b["buy"] for b in settled} == {180.0, 181.0, 182.0, 183.0}


def test_reexecution_reads_the_stored_nowcasts():
    site, store, start = _site()
    _run(site, start, 120)
    nc = store.nowcasts(start, start + 120 * 60, epoch=site.epoch)
    # a written decision row in a commit window (elapsed 1 to 10) with a fresh nowcast
    row = next(
        d for d in store.decisions(start, start + 120 * 60)
        if 1 <= (d["ts"] - d["quarter_ts"]) // 60 <= 10 and math.isfinite(d["nowcast_take"])
    )
    stored = nc[row["ts"]]
    store.put_nowcast(dict(stored, ts=row["ts"], take_point=stored["take_point"] + 500.0))
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 120 * 60, out)
    diffs = RX.compare(store, out, start, start + 120 * 60)
    assert diffs and diffs[0].startswith("decisions") and f"seq {row['seq']}:" in diffs[0]
    assert r.differences == diffs
    got = next(d for d in out.decisions(start, start + 120 * 60) if d["seq"] == row["seq"])
    assert got["nowcast_take"] == row["nowcast_take"] + 500.0


def test_a_failed_setup_is_stored_and_reexecuted(monkeypatch):
    site, store, start = _site()
    real = LS.quarter_inputs
    failed = start + 2 * 900  # the third quarter has no slot-0 band

    def qi(now, *args, **kwargs):
        return None if int(now) - int(now) % 900 == failed else real(now, *args, **kwargs)

    monkeypatch.setattr(LS, "quarter_inputs", qi)
    _run(site, start, 4 * 15)
    w = store.window(failed)
    assert w is not None and w["now"] == failed + 60 and w["load_kw"] is None
    assert site.status()["ranked"] is not None  # the fourth quarter ranked again
    monkeypatch.setattr(LS, "quarter_inputs", real)
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 4 * 900, out)
    assert r.differences == [] and out.head() == store.head()
    rows = store.decisions(failed + 60, failed + 900)
    assert rows and all(math.isnan(d["band_take_q50"]) for d in rows)


def test_a_settle_event_is_rebooked_in_place():
    site, store, start = _site()
    _run(site, start, 2 * 15)
    # what a settlement pull does (plan 3 task 11): the settlement, the rebooking, the event
    store.put_settlement(
        {
            "quarter_ts": start, "take": 210.0, "feed": 190.0, "regulation_state": 1,
            "published_at": start + 1800, "source": "synthetic",
        }
    )
    booked = site._rebook_settled(start, 210.0, 190.0, 1)
    assert [r["path"] for r in booked] == ["shadow", "house", "actual"]
    assert all(r["price_source"] == "settled" for r in booked)
    store.put_event(start + 1800, "settle", {"quarters": [start]})
    _run(site, start + 1800, 15)
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 3 * 900, out)
    assert r.differences == [] and out.head() == store.head()


def test_a_span_without_a_restart_is_reexecuted_from_the_newest_restart_before_it():
    site, store, start = _site()
    _run(site, start, 2 * 1440)
    day2 = start + 86400
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, day2, day2 + 86400, out)
    assert [e["kind"] for e in store.events()] == ["restart"]  # only the start of the run
    assert r.differences == [] and out.head() == store.head()
    assert r.n_decisions == len(store.decisions(day2, day2 + 86400)) > 0
    assert r.n_bookings == len(store.bookings(day2, day2 + 86400)) == 96 * 3


def test_a_span_before_a_restart_is_compared_from_its_start():
    site, store, start = _site()
    _run(site, start, 300)
    t0 = start + 300 * 60 + 1800
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(t0)
    _run(site2, t0, 120)
    end = t0 + 120 * 60
    # a span from before the first restart (the install) and one from inside the first run
    for lo in (start - 3600, start + 3600):
        out = Store(":memory:")
        out.migrate()
        r = RX.reexecute(store, site.lc, lo, end, out)
        assert r.differences == [] and r.n_decisions == len(store.decisions(lo, end)) > 0
    # a tampered decision in [start, t0) is reported
    row = store.decisions(start + 3600, t0)[0]
    store._db.execute("UPDATE decisions SET kw = kw + 1.0 WHERE seq = ?", (row["seq"],))
    store._db.commit()
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start + 3600, end, out)
    assert r.differences[0] == (
        f"decisions seq {row['seq']}: kw {row['kw'] + 1.0!r} != {row['kw']!r}"
    )


def test_daily_result_mad_against_a_walk_forward_result():
    site, store, start = _site()
    _run(site, start, 1440)
    day = dt.datetime.fromtimestamp(start, ZONE).date()
    live = LS.daily_result(store, day)["result_eur"]
    pairs = {
        "day": np.array([day.isoformat(), (day + dt.timedelta(days=1)).isoformat()], object),
        "result_eur": np.array([live + 2.5, 7.0]),
        "n_missing": np.zeros(2, dtype=np.int64),
    }
    # the second day has no live booking, so only the first day counts
    assert RX.daily_result_mad(store, SimpleNamespace(result_pairs=pairs)) == pytest.approx(2.5)
    empty = {"day": np.array([], object), "result_eur": np.zeros(0), "n_missing": np.zeros(0)}
    assert math.isnan(RX.daily_result_mad(store, SimpleNamespace(result_pairs=empty)))


def test_report_states_the_difference_from_the_live_day():
    from dataclasses import replace

    from betterybalance.replay import engine as E
    from betterybalance.replay import report as R

    from tests.test_replay_engine import _inputs

    result = E.replay(_inputs())
    site, store, start = _site()
    _run(site, start, 1440)
    day = dt.datetime.fromtimestamp(start, ZONE).date()
    live = LS.daily_result(store, day)["result_eur"]
    pairs = {
        "day": np.array([day.isoformat()], object),
        "result_eur": np.array([live - 1.25]),
        "n_missing": np.zeros(1, dtype=np.int64),
    }
    text = R.render_report(replace(result, result_pairs=pairs), {}, live_store=store)
    assert "Differs from the live day by a mean absolute 1,25 EUR per day." in text
    # the synthetic replay's days are not live days of this store
    assert R.NO_LIVE_DAY_SENTENCE in R.render_report(result, {}, live_store=store)
    assert "live day" not in R.render_report(result, {})


def test_cli_reexecute_prints_identical(tmp_path, capsys):
    path = tmp_path / "store.sqlite3"
    site, store, start = _site(store=Store(path))
    _run(site, start, 90, flows=lambda ts: None if ts == start + 600 else _flows(ts))
    n_dec = len(store.decisions(start, start + 86400))
    n_book = len(store.bookings(start, start + 86400))
    store.close()
    (tmp_path / "install_id").write_text("install-1\n", encoding="utf-8")
    (tmp_path / "model.json").write_text(PM.to_json(site.lc.model), encoding="utf-8")
    (tmp_path / "site.json").write_text(
        '{"latitude": 52.1, "longitude": 5.2, "pack": {"capacity_kwh": 48, "charge_kw": 12, '
        '"discharge_kw": 12, "soc_floor": 0.1, "soc_ceiling": 1.0}, "pv": {"none": true}}',
        encoding="utf-8",
    )
    day = dt.datetime.fromtimestamp(start, ZONE).date().isoformat()
    args = [
        "--reexecute", str(path), "--install-id-file", str(tmp_path / "install_id"),
        "--site", str(tmp_path / "site.json"), "--model", str(tmp_path / "model.json"),
        "--from", day, "--to", day, "--out", str(tmp_path / "out"),
    ]
    assert CLI.main(args) == 0
    assert capsys.readouterr().out.strip() == (
        f"identical: {R_eu(n_dec)} decisions, {R_eu(n_book)} bookings"
    )
    assert (tmp_path / "out" / "reexec.sqlite3").exists()
    (tmp_path / "install_id").write_text("install-2\n", encoding="utf-8")
    assert CLI.main(args) == 1
    assert capsys.readouterr().out.startswith("first difference: decisions seq 1:")


def R_eu(n: int) -> str:
    from betterybalance.core.fmt import eu

    return eu(n, 0)


def test_restart_event_names_the_chain_head():
    site, store, start = _site()
    _run(site, start, 20)
    head = store.head()[0]
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 3 * 900)
    ev = store.events()[-1]
    assert ev["kind"] == "restart" and ev["detail"]["head_seq"] == head
    assert store.events()[0]["detail"]["head_seq"] == 0
    assert LG.verify_chain(store.chain(1, store.head()[0]), LG.genesis_hash("install-1")).ok


def test_a_sensor_anchored_restart_is_reexecuted_byte_identical():
    site, store, start = _site()
    _run(site, start, 20, flows=lambda ts: None)  # nothing valid yet: the shadow never stepped
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 1800, 90.0)
    _run(site2, start + 1800, 60)
    assert store.last_event("restart")["detail"]["soc_sensor_pct"] == 90.0
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 1800 + 3600, out)
    assert r.differences == [] and out.head() == store.head()


@pytest.mark.parametrize(
    "crash, restart, hole", [(22, 22, None), (19, 24, None), (7, 9, None), (22, 22, 18)]
)
def test_a_mid_quarter_restart_is_reexecuted_byte_identical(crash, restart, hole):
    # R30: the second run re-decides the running quarter from its start; each run reads its
    # own nowcasts, bands, windows and the flows rows of its own or an earlier run, so both
    # re-execute exactly. ``hole``: a minute without flows in the first run that the second
    # run's flows fill (the second run's row never reaches the first run's re-execution).
    site, store, start = _site()

    def flows(ts):
        return None if hole is not None and ts == start + hole * 60 else _flows(ts)

    _run(site, start, crash, flows=flows)  # minutes 0 .. crash - 1
    site2 = LS.LiveSite(store, site.lc)
    t = start + restart * 60
    site2.restore(t)
    ev = store.events()[-1]
    assert ev["detail"]["last_minute"] == start + (crash - 1) * 60 and site2.epoch == ev["seq"]
    q = t - (t - start) % 900
    for ts in range(q, q + 900, 60):  # the running quarter, as the restore ranked it
        site2.on_minute(ts, dict(_flows(ts), ts=ts))
    _run(site2, q + 900, 30)
    end = q + 900 + 30 * 60
    head = ev["detail"]["head_seq"]
    assert any(d["seq"] <= head for d in store.decisions(q, q + 900))  # decided twice
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, end, out)
    assert r.differences == [] and RX.compare(store, out, start, end) == []
    assert out.head() == store.head()
    assert r.n_decisions == len(store.decisions(start, end))
    if hole is not None:  # the second run stored the hole minute, with its epoch
        assert store.minutes(start + hole * 60, start + hole * 60 + 60)["ts"].size == 1
        first = store.minutes(start + hole * 60, start + hole * 60 + 60, max_epoch=site.epoch)
        assert first["ts"].size == 0


def test_a_failing_display_pv_leaves_the_setup_and_reexecution_untouched(monkeypatch, caplog):
    site, store, start = _site()

    def boom(*args, **kwargs):
        raise RuntimeError("display only")

    monkeypatch.setattr(LS, "pv_now", boom)
    _run(site, start, 2 * 15)
    assert site.status()["pv"] is None and site.status()["ranked"] is not None
    for q in (start, start + 900):
        w = store.window(q)
        assert w is not None and w["load_kw"] is not None  # the full windows row
    assert sum("PV expectation for display failed" in r.message for r in caplog.records) == 1
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 2 * 900, out)
    assert r.differences == [] and RX.compare(store, out, start, start + 2 * 900) == []
    assert out.head() == store.head()


def _minutes(site, lo, hi):
    """The live clock from minute ``lo`` to ``hi`` (exclusive), minute 1 set up first."""
    for ts in range(lo, hi, 60):
        if ts % 900 == 60:
            site.setup_quarter(ts)
        site.on_minute(ts, dict(_flows(ts), ts=ts))


def test_reexecution_is_byte_identical_across_a_model_switch():
    from dataclasses import replace

    from tests.test_live_site import _model

    site, store, start = _site()
    lc0, old, new = site.lc, site.lc.model, _model(30, 14)
    q = 900
    _minutes(site, start, start + 7 * 60)
    site.set_model(new, "v-new", site.next_minute)  # minute 7: from the second quarter on
    _minutes(site, start + 7 * 60, start + 2 * q + 5 * 60)
    # a switch back, pending across an in-process restore inside its quarter
    site.set_model(old, "v-test", site.next_minute)
    t1 = start + 2 * q + 9 * 60
    site.restore(t1)
    _minutes(site, start + 2 * q, start + 4 * q + 3 * 60)
    # an HA restart: a fresh site on the model in effect, the pending switch put again
    _minutes(site, start + 4 * q + 3 * 60, start + 4 * q + 6 * 60)
    site.set_model(new, "v-new", site.next_minute)
    t2 = start + 4 * q + 8 * 60
    site2 = LS.LiveSite(store, replace(lc0, model=old, model_version="v-test"))
    site2.restore(t2)
    site2.set_model(new, "v-new", site2.next_minute)
    _minutes(site2, start + 4 * q, start + 6 * q)
    end = start + 6 * q
    switches = [e for e in store.events() if e["kind"] == "model_switch"]
    assert [e["detail"]["effective_ts"] for e in switches] == [
        start + q, start + 3 * q, start + 5 * q, start + 5 * q,
    ]
    versions = {r["quarter_ts"]: r["model_version"] for r in store.decisions(start, end)}
    assert versions[start] == versions[start + 4 * q] == "v-test"
    assert versions[start + q] == versions[start + 2 * q] == versions[start + 5 * q] == "v-new"
    assert store.window(start + q)["model_version"] == "v-new"
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, lc0, start, end, out, models={"v-new": new, "v-test": old})
    assert r.differences == [] and RX.compare(store, out, start, end) == []
    assert out.head() == store.head() and r.n_decisions == len(store.decisions(start, end))
    # without the switch events the decisions differ in their model_version
    store._db.execute("DELETE FROM events WHERE kind = 'model_switch'")
    store._db.commit()
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, lc0, start, end, out, models={"v-new": new, "v-test": old})
    assert r.differences


def test_a_start_time_switch_to_the_shipped_model_reexecutes_byte_identical():
    # an HA start that finds the model in effect gone: a model_switch at the restart's wall
    # time, effective at the quarter start the restored run decides from, before the restart
    from dataclasses import replace

    from tests.test_live_site import _model

    site0, store, start = _site()
    new = _model(30, 14)
    lc_new = replace(site0.lc, model=new, model_version="v-new")
    site = LS.LiveSite(store, lc_new)
    site.restore(start)
    _minutes(site, start, start + 20 * 60)  # minutes 0 .. 19
    t = start + 22 * 60 + 17
    q = start + 900
    store.put_event(
        t, "model_switch",
        {"version": "v-test", "effective_ts": q, "from_version": "v-new",
         "cov_source": "own", "reason": "model not available"},
    )
    site2 = LS.LiveSite(store, site0.lc)  # the shipped stand-in "v-test"
    site2.restore(t)
    _minutes(site2, q, q + 2 * 900)
    end = q + 2 * 900
    runs = {}
    for d in store.decisions(q, q + 900):
        runs.setdefault(d["model_version"], []).append(d["ts"])
    assert set(runs) == {"v-new", "v-test"}  # the first run's and the re-run's decisions
    out = Store(":memory:")
    out.migrate()
    models = {"v-new": new, "v-test": site0.lc.model}
    r = RX.reexecute(store, lc_new, start, end, out, models=models)
    assert r.differences == [] and out.head() == store.head()


def test_cli_reexecute_reads_the_pack_models(tmp_path, capsys, monkeypatch):
    from betterybalance.core.pack import load_pack
    from betterybalance.live import pack_update as PU

    from tests.test_pack_update import _release

    dest = tmp_path / "pack"
    PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    pack = load_pack(dest / "2026-11")
    model, version = PU.choose_model((PM.shipped_model(), "shipped-x", 0), pack)
    path = tmp_path / "store.sqlite3"
    site, store, start = _site(store=Store(path))
    base = site.lc.model
    _minutes(site, start, start + 7 * 60)
    site.set_model(model, version, site.next_minute)
    _minutes(site, start + 7 * 60, start + 3 * 900)
    assert store.window(start + 900)["model_version"] == version
    store.close()
    (tmp_path / "install_id").write_text("install-1\n", encoding="utf-8")
    (tmp_path / "base.json").write_text(PM.to_json(base), encoding="utf-8")
    (tmp_path / "site.json").write_text(
        '{"latitude": 52.1, "longitude": 5.2, "pack": {"capacity_kwh": 48, "charge_kw": 12, '
        '"discharge_kw": 12, "soc_floor": 0.1, "soc_ceiling": 1.0}, "pv": {"none": true}}',
        encoding="utf-8",
    )
    day = dt.datetime.fromtimestamp(start, ZONE).date().isoformat()
    args = [
        "--reexecute", str(path), "--install-id-file", str(tmp_path / "install_id"),
        "--site", str(tmp_path / "site.json"), "--model", str(tmp_path / "base.json"),
        "--from", day, "--to", day, "--out", str(tmp_path / "out"),
    ]
    seen = []
    real = CLI.reexecute

    def spy(*a, **kw):
        seen.append(kw.get("models"))
        return real(*a, **kw)

    monkeypatch.setattr(CLI, "reexecute", spy)
    assert CLI.main([*args, "--pack-dir", str(dest)]) == 0
    assert capsys.readouterr().out.startswith("identical: ")
    models = seen[-1]
    assert version in models and models[version].meta["cov_source"] == "shipped"
    assert PM.to_json(models[version]) == PM.to_json(model)  # the model the live site ran
    assert CLI.main(args) == 0 and seen[-1] is None  # without --pack-dir: no map
    # two releases whose newest end does not advance: the in-use pack stays installed
    keep = {PU.version_month(version)}
    for month in ("2026-12", "2027-01"):
        PU.install(PU.check_release(str(_release(tmp_path, month)), None), dest, keep=keep)
    assert (dest / "2026-11").is_dir()
    assert CLI.main([*args, "--pack-dir", str(dest)]) == 0
    assert capsys.readouterr().out.startswith("identical: ")
    assert PM.to_json(seen[-1][version]) == PM.to_json(model)


def test_cli_reexecutes_the_day_after_a_model_switch_identically(tmp_path, capsys):
    """A switch on day D with no restart after it: the walk of day D + 1 starts at D's
    restart, on the version that restart names, not on the first window of D + 1."""
    from betterybalance.live import pack_update as PU

    from tests.test_pack_update import _release

    dest = tmp_path / "pack"
    PU.install(PU.check_release(str(_release(tmp_path)), None), dest)
    from betterybalance.core.pack import load_pack

    pack = load_pack(dest / "2026-11")
    model, version = PU.choose_model((PM.shipped_model(), "shipped-x", 0), pack)
    path = tmp_path / "store.sqlite3"
    site, store, start = _site(store=Store(path))
    base = site.lc.model
    restart = store.last_event("restart")
    assert restart["detail"]["model_version"] == site.lc.model_version == "v-test"
    _minutes(site, start, start + 7 * 60)
    site.set_model(model, version, site.next_minute)
    day2 = start + 86400
    _minutes(site, start + 7 * 60, day2 + 2 * 900)
    assert [e["kind"] for e in store.events()] == ["restart", "model_switch"]
    assert store.window(day2 + 900)["model_version"] == version
    store.close()
    (tmp_path / "install_id").write_text("install-1\n", encoding="utf-8")
    (tmp_path / "base.json").write_text(PM.to_json(base), encoding="utf-8")
    (tmp_path / "site.json").write_text(
        '{"latitude": 52.1, "longitude": 5.2, "pack": {"capacity_kwh": 48, "charge_kw": 12, '
        '"discharge_kw": 12, "soc_floor": 0.1, "soc_ceiling": 1.0}, "pv": {"none": true}}',
        encoding="utf-8",
    )
    day = dt.datetime.fromtimestamp(day2, ZONE).date().isoformat()
    args = [
        "--reexecute", str(path), "--install-id-file", str(tmp_path / "install_id"),
        "--site", str(tmp_path / "site.json"), "--model", str(tmp_path / "base.json"),
        "--from", day, "--to", day, "--out", str(tmp_path / "out"), "--pack-dir", str(dest),
    ]
    assert CLI.main(args) == 0
    assert capsys.readouterr().out.startswith("identical: ")


def test_an_old_restart_event_without_a_version_reads_the_next_switch():
    events = [
        {"kind": "restart", "detail": {"head_seq": 0}},
        {"kind": "settle", "detail": {}},
        {"kind": "model_switch", "detail": {"version": "b", "from_version": "a"}},
    ]
    assert RX._version_at(events, 0, "z") == "a"
    assert RX._version_at(events[:2], 0, "z") == "z"
    events[0]["detail"]["model_version"] = "r"
    assert RX._version_at(events, 0, "z") == "r"


def test_reexecution_runs_each_run_on_the_knobs_its_restart_names():
    from dataclasses import replace

    from betterybalance.core.site import contract as C

    site, store, start = _site()
    _run(site, start, 4 * 15)
    knobs = {**C.knobs_dict(site.lc.cfg.knobs), "latency_s": 180, "hurdle_ct_kwh": 1.0}
    lc2 = replace(site.lc, cfg=C.with_knobs(site.lc.cfg, knobs))
    site2 = LS.LiveSite(store, lc2)
    site2.restore(start + 3600)
    _run(site2, start + 3600, 4 * 15)
    assert store.last_event("restart")["detail"]["knobs"] == knobs
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, lc2, start, start + 7200, out)  # the caller holds the new knobs
    assert r.differences == [] and out.head() == store.head()
