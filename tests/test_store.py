# tests/test_store.py
import math

import numpy as np
import pytest
from betterybalance.core.store import RETENTION_DAYS, Store, blob, unblob


def _store(tmp_path=None):
    s = Store(":memory:" if tmp_path is None else tmp_path / "s.sqlite3")
    s.migrate()
    return s


def test_file_store_uses_wal_and_records_the_schema(tmp_path):
    s = _store(tmp_path)
    assert s.meta("schema_version") == "1"
    assert s._db.execute("pragma journal_mode").fetchone()[0] == "wal"


def test_minutes_roundtrip_nan_as_null():
    s = _store()
    s.put_minute(
        {
            "ts": 1788300000,
            "grid_import_kwh": 0.02,
            "grid_export_kwh": 0.0,
            "batt_charge_kwh": 0.05,
            "batt_discharge_kwh": 0.0,
            "soc_pct": 50.0,
            "load_kwh": float("nan"),
            "n_samples": 12,
        }
    )
    t = s.minutes(1788300000, 1788300060)
    assert (
        t["step_s"][0] == 60
        and t["grid_import_kwh"][0] == 0.02
        and np.isnan(t["load_kwh"][0])
        and np.isnan(t["pv3_kwh"][0])
    )


def test_blob_is_exact():
    a = np.array([0.1, 1 / 3, np.nan, -2.5e-17])
    b = unblob(blob(a))
    assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]) and np.isnan(b[2])


def test_settlement_correction_appends_and_newest_wins():
    s = _store()
    row = {
        "quarter_ts": 1788300000,
        "take": 100.0,
        "feed": 90.0,
        "regulation_state": 1,
        "published_at": 1788390000,
        "source": "tennet",
    }
    assert s.put_settlement(row) and not s.put_settlement(row)
    assert s.put_settlement(dict(row, take=120.0, published_at=1788400000))
    assert [r["take"] for r in s.settlements(1788300000, 1788300900)] == [120.0]
    assert len(s.settlements(1788300000, 1788300900, newest=False)) == 2


def test_prune_keeps_bookings_and_drops_old_minutes():
    s = _store()
    s.set_meta("install_id", "synthetic-install")  # controller ruling: the chain needs an anchor
    now = 1830000000
    day = 86400
    for ts in (now - 401 * day, now - 399 * day):
        s.put_minute(
            {
                "ts": ts,
                "grid_import_kwh": 0.0,
                "grid_export_kwh": 0.0,
                "batt_charge_kwh": 0.0,
                "batt_discharge_kwh": 0.0,
                "soc_pct": 50.0,
                "n_samples": 12,
            }
        )
    s.append_chained("decision", _decision(ts=now - 500 * day, quarter_ts=now - 500 * day))
    s.append_chained(
        "booking",
        {
            "quarter_ts": now - 500 * day,
            "path": "shadow",
            "import_kwh": 0.0,
            "export_kwh": 0.0,
            "batt_charge_kwh": 0.0,
            "batt_discharge_kwh": 0.0,
            "soc_pct": 50.0,
            "missing": False,
            "price_source": "none",
            "buy": float("nan"),
            "sell": float("nan"),
            "state": 0,
            "money_eur": float("nan"),
            "tax_eur": float("nan"),
        },
    )
    assert s.prune(now)["flows_minute"] == 1 and len(s.minutes(0, now)["ts"]) == 1
    assert len(s.bookings(0, now)) == 1 and RETENTION_DAYS == 400
    assert "decisions" not in ST.PRUNED and len(s.decisions(0, now)) == 1
    genesis = LG.genesis_hash("synthetic-install")
    assert LG.verify_chain(s.chain(1, s.head()[0]), genesis).ok


def test_newer_schema_is_refused(tmp_path):
    s = _store(tmp_path)
    s.set_meta("schema_version", "2")
    s.close()
    with pytest.raises(ValueError, match="newer than this integration"):
        Store(tmp_path / "s.sqlite3").migrate()


# --- Beyond the brief's stubs: every method of the store ----------------------------------

from betterybalance.core import ledger as LG  # noqa: E402
from betterybalance.core import store as ST  # noqa: E402
from betterybalance.core.site import flows as FL  # noqa: E402

Q = 1788300000
NAN = float("nan")


def _decision(ts=Q + 60, **kw):
    row = {
        "ts": ts,
        "quarter_ts": Q,
        "action": "charge",
        "kw": 3,
        "committed": True,
        "reason": "below",
        "margin_w": 0.0,
        "value_stored": -0.0,
        "charge_threshold": 80.5,
        "discharge_threshold": NAN,
        "band_take_q10": 1.0,
        "band_take_q50": 2.0,
        "band_take_q90": 3.0,
        "band_feed_q10": 0.5,
        "band_feed_q50": 1.5,
        "band_feed_q90": 2.5,
        "nowcast_take": NAN,
        "nowcast_feed": NAN,
        "soc_pct": 49.5,
        "learning": False,
        "model_version": "m1",
    }
    row.update(kw)
    return row


def _booking(qts=Q, path="shadow", seq_tag=0.0, **kw):
    row = {
        "quarter_ts": qts,
        "path": path,
        "import_kwh": 0,
        "export_kwh": 0.25 + seq_tag,
        "batt_charge_kwh": 0.0,
        "batt_discharge_kwh": 0.25,
        "soc_pct": 50,
        "missing": False,
        "price_source": "provisional",
        "buy": 100.0,
        "sell": 50.0,
        "state": 1,
        "money_eur": -0.01,
        "tax_eur": 0.0,
    }
    row.update(kw)
    return row


def _chained_store(tmp_path=None):
    s = _store(tmp_path)
    s.set_meta("install_id", "synthetic-install")
    return s


def test_chain_round_trip_through_sqlite_verifies(tmp_path):
    s = _chained_store(tmp_path)
    r1 = s.append_chained("decision", _decision())
    r2 = s.append_chained("booking", _booking())
    r3 = s.append_chained(
        "booking", _booking(path="house", missing=True, money_eur=NAN, tax_eur=NAN)
    )
    assert (r1["seq"], r2["seq"], r3["seq"]) == (1, 2, 3)
    assert r1["prev_hash"] == LG.genesis_hash("synthetic-install") and r2["prev_hash"] == r1["hash"]
    assert s.head() == (3, r3["hash"]) and s.meta("chain_seq") == "3"
    s.close()
    back = Store(tmp_path / "s.sqlite3")
    back.migrate()
    items = back.chain(1, back.head()[0])
    assert [k for k, _ in items] == ["decision", "booking", "booking"]
    assert LG.verify_chain(items, LG.genesis_hash("synthetic-install")).ok
    d = items[0][1]
    assert (
        d["kw"] == 3.0 and isinstance(d["committed"], bool) and np.isnan(d["discharge_threshold"])
    )
    assert items[2][1]["missing"] is True and np.isnan(items[2][1]["money_eur"])
    assert back.chain(2, 2) == items[1:2]


def test_append_chained_needs_install_id_and_a_known_kind():
    s = _store()
    with pytest.raises(ValueError, match="install_id"):
        s.append_chained("decision", _decision())
    s.set_meta("install_id", "x")
    with pytest.raises(ValueError, match="kind"):
        s.append_chained("trade", _decision())
    assert s.head() is None


def test_append_chained_many_chains_in_one_transaction(tmp_path):
    s = _chained_store(tmp_path)
    first = s.append_chained("decision", _decision())
    rows = s.append_chained_many(
        [("booking", _booking(path=p)) for p in ("shadow", "house", "actual")]
    )
    assert [r["seq"] for r in rows] == [2, 3, 4] and rows[0]["prev_hash"] == first["hash"]
    assert s.head() == (4, rows[-1]["hash"])
    assert LG.verify_chain(s.chain(1, 4), LG.genesis_hash("synthetic-install")).ok
    real, calls = s._insert, []

    def failing(table, row, columns, verb="INSERT OR REPLACE"):
        calls.append(table)
        if len(calls) == 2:
            raise RuntimeError("disk gone")
        return real(table, row, columns, verb)

    s._insert = failing
    with pytest.raises(RuntimeError):
        s.append_chained_many(
            [("booking", _booking(qts=Q + 900, path=p)) for p in ("shadow", "house", "actual")]
        )
    s._insert = real
    assert s.head() == (4, rows[-1]["hash"]) and s.meta("chain_seq") == "4"
    assert s.bookings(Q + 900, Q + 1800) == []
    assert s.append_chained_many([]) == []


def test_tampered_row_fails_verification():
    s = _chained_store()
    s.append_chained("decision", _decision())
    s.append_chained("booking", _booking())
    with s._db:
        s._db.execute("update bookings set export_kwh = 9.0 where seq = 2")
    check = LG.verify_chain(s.chain(1, 2), LG.genesis_hash("synthetic-install"))
    assert not check.ok and check.first_bad_seq == 2


def test_decisions_bookings_and_newest_bookings():
    s = _chained_store()
    s.append_chained("decision", _decision(ts=Q + 60))
    s.append_chained("decision", _decision(ts=Q + 900))
    s.append_chained("booking", _booking(path="shadow"))
    s.append_chained("booking", _booking(path="house"))
    s.append_chained("booking", _booking(path="shadow", price_source="settled", seq_tag=1.0))
    s.append_chained("booking", _booking(qts=Q + 900, path="shadow"))
    assert [r["ts"] for r in s.decisions(Q, Q + 900)] == [Q + 60]
    assert [r["seq"] for r in s.bookings(Q, Q + 900)] == [3, 4, 5]
    newest = s.newest_bookings(Q, Q + 1800)
    assert set(newest) == {(Q, "shadow"), (Q, "house"), (Q + 900, "shadow")}
    assert newest[(Q, "shadow")]["seq"] == 5 and newest[(Q, "shadow")]["price_source"] == "settled"
    assert newest[(Q, "shadow")]["export_kwh"] == 1.25


def test_minutes_feed_to_quarters_and_reject_unknown_columns():
    s = _store()
    for i in range(15):
        s.put_minute(
            {
                "ts": Q + 60 * i,
                "step_s": 60,
                "grid_import_kwh": 0.01,
                "grid_export_kwh": 0.0,
                "batt_charge_kwh": 0.0,
                "batt_discharge_kwh": 0.0,
                "soc_pct": 40.0 + i,
                "soc_min_pct": 40.0 + i,
                "soc_max_pct": 41.0 + i,
                "pv2_kwh": 0.002,
                "n_samples": 4,
            }
        )
    s.put_minute(
        {
            "ts": Q,
            "grid_import_kwh": 9.0,
            "grid_export_kwh": 0.0,
            "batt_charge_kwh": 0.0,
            "batt_discharge_kwh": 0.0,
            "soc_pct": 1.0,
        }
    )  # insert-or-ignore
    t = s.minutes(Q, Q + 900)
    assert set(FL.COLUMNS) | set(FL.STRING_COLUMNS) <= set(t) and t["ts"].dtype == np.int64
    assert t["grid_import_kwh"][0] == 0.01 and t["n_samples"][0] == 4
    q = FL.to_quarters(t)
    assert q["grid_import_kwh"][0] == pytest.approx(0.15) and q["pv2_kwh"][0] == pytest.approx(0.03)
    assert q["soc_min_pct"][0] == 40.0 and q["soc_max_pct"][0] == 55.0
    with pytest.raises(ValueError, match="pv9_kwh"):
        s.put_minute({"ts": Q + 900, "pv9_kwh": 0.1})


def test_hours_upsert_and_step():
    s = _store()
    s.put_hours(
        {
            "ts": np.array([Q, Q + 3600]),
            "step_s": np.array([3600, 3600]),
            "grid_import_kwh": np.array([1.0, np.nan]),
            "pv1_kwh": np.array([0.5, 0.25]),
        }
    )
    s.put_hours({"ts": np.array([Q]), "grid_import_kwh": np.array([2.0])})
    h = s.hours(Q, Q + 7200)
    assert h["step_s"].tolist() == [3600, 3600] and h["grid_import_kwh"][0] == 2.0
    assert np.isnan(h["grid_import_kwh"][1]) and h["pv1_kwh"].tolist() == [0.5, 0.25]
    assert "n_samples" not in h and set(FL.COLUMNS) <= set(h)


def test_features_merge_non_nan_values():
    s = _store()
    assert len(ST.FEATURE_COLUMNS) == 12 and ST.FEATURE_COLUMNS[0] == "da_eur_mwh"
    s.upsert_features(Q, da_eur_mwh=80.0, ida1_eur_mwh=NAN)
    s.upsert_features(Q, da_eur_mwh=NAN, buy_live_end=120.0)
    s.upsert_features(Q + 900)
    f = s.features(Q, Q + 1800)
    assert f["ts"].tolist() == [Q, Q + 900] and f["da_eur_mwh"][0] == 80.0
    assert (
        f["buy_live_end"][0] == 120.0
        and np.isnan(f["ida1_eur_mwh"][0])
        and np.isnan(f["da_eur_mwh"][1])
    )
    with pytest.raises(ValueError, match="nope"):
        s.upsert_features(Q, nope=1.0)


def test_live_minutes_roundtrip():
    s = _store()
    s.put_live_minutes(
        [
            {"ts": Q, "buy_live": 100.0, "sell_live": 90.0, "state": 1, "source": "tennet"},
            {"ts": Q + 60, "buy_live": NAN, "sell_live": 80.0, "state": None, "source": "tenergy"},
        ]
    )
    t = s.live_minutes(Q, Q + 120)
    assert (
        t["ts"].tolist() == [Q, Q + 60] and t["buy_live"][0] == 100.0 and np.isnan(t["buy_live"][1])
    )
    assert (
        t["state"][0] == 1
        and np.isnan(t["state"][1])
        and t["source"].tolist() == ["tennet", "tenergy"]
    )


def _band_row(**kw):
    row = {
        "target_ts": Q,
        "tier": "A",
        "h": 4,
        "e": -1,
        "model_version": "v1",
        "p_dual": 0.2,
        "cov": 1.1,
    }
    row.update({f"take_q{q}": 10.0 * i for i, q in enumerate((10, 25, 50, 75, 90))})
    row.update({f"feed_q{q}": 5.0 * i for i, q in enumerate((10, 25, 50, 75, 90))})
    return dict(row, **kw)


def test_band_lookup_and_one_row_per_serve_time():
    s = _store()
    assert s.put_band(_band_row(), Q - 840)
    assert not s.put_band(_band_row(), Q - 840)  # the same band again: nothing to add
    with pytest.raises(ValueError, match="recorded_at"):  # a different band at one serve time
        s.put_band(_band_row(p_dual=0.9), Q - 840)
    got = s.band(Q, "A", 4, None, "v1")
    assert got["p_dual"] == 0.2 and got["take_q90"] == 40.0 and got["e"] == -1
    assert got["recorded_at"] == Q - 840
    assert s.band(Q, "A", 4, -1, "v2") is None and s.band(Q, "A", 4, 5, "v1") is None


def test_a_band_that_turns_unavailable_is_recorded_as_none():
    s = _store()
    nan = {k: float("nan") for k in _band_row() if k.startswith(("take_", "feed_"))}
    gone = _band_row(p_dual=float("nan"), cov=float("nan"), **nan)
    assert not s.put_band(gone, Q - 900)  # never served before: nothing to record
    assert s._db.execute("select count(*) from bands").fetchone()[0] == 0
    assert s.put_band(_band_row(), Q - 840)
    assert s.put_band(dict(gone), Q - 780)  # the cell turned unavailable
    assert not s.put_band(dict(gone), Q - 720)
    assert math.isnan(s.band(Q, "A", 4, None, "v1", as_of=Q - 780)["take_q50"])
    assert s.band(Q, "A", 4, None, "v1", as_of=Q - 781)["take_q50"] == 20.0


def test_band_is_as_of_and_an_identical_reserve_adds_no_row():
    s = _store()
    t1, t2 = Q - 840, Q - 780
    a, b = _band_row(), _band_row(take_q50=99.0)
    assert s.put_band(a, t1)
    assert not s.put_band(dict(a), t1 + 30)  # identical re-serve
    assert s.put_band(b, t2)
    assert not s.put_band(dict(b), t2 + 60)
    assert s._db.execute("select count(*) from bands").fetchone()[0] == 2
    assert s.band(Q, "A", 4, None, "v1", as_of=t1 - 1) is None
    assert s.band(Q, "A", 4, None, "v1", as_of=t1)["take_q50"] == 20.0
    assert s.band(Q, "A", 4, None, "v1", as_of=t2 - 1)["take_q50"] == 20.0
    assert s.band(Q, "A", 4, None, "v1", as_of=t2)["take_q50"] == 99.0
    assert s.band(Q, "A", 4, None, "v1")["take_q50"] == 99.0
    assert s.put_band(a, t2 + 120)  # back to A after B is a change again
    assert s.band(Q, "A", 4, None, "v1")["take_q50"] == 20.0


def test_window_blobs_decode_to_arrays():
    s = _store()
    row = {
        "ts": Q,
        "now": Q + 60,
        "model_version": "v1",
        "pack_version": "2026-11",
        "healthy": ("A", "C"),
        "da_known": np.array([True, False, True]),
        "soc_kwh": 12.5,
        "widening": 1.5,
        "irr_issue_ts": Q - 3600,
        "load_kw": np.array([0.4, NAN, 0.6]),
        "pv_kw": np.array([0.0, 1 / 3, 2.0]),
        "ev_kw": np.zeros(3),
        "plan_mask": np.array([False, True, False]),
        "flags": ["learning"],
    }
    s.put_window(row)
    w = s.window(Q)
    assert w["da_known"].dtype == bool and w["da_known"].tolist() == [True, False, True]
    assert w["plan_mask"].tolist() == [False, True, False] and w["pv_kw"][1] == 1 / 3
    assert np.isnan(w["load_kw"][1]) and w["healthy"] == ("A", "C") and w["flags"] == ["learning"]
    assert w["now"] == Q + 60 and w["soc_kwh"] == 12.5 and s.window(Q + 900) is None


def test_nowcasts_by_minute():
    s = _store()
    vals = dict(zip(ST.NOWCAST_FIELDS, (100.0, 90.0, 0.5, 0.4, 0.1, 95.0, NAN)))
    s.put_nowcast({"ts": Q + 300, "quarter_ts": Q, "live_age_s": 30, **vals})
    got = s.nowcasts(Q, Q + 900, epoch=0)
    assert list(got) == [Q + 300] and got[Q + 300]["take_point"] == 100.0
    assert (
        got[Q + 300]["quarter_ts"] == Q
        and got[Q + 300]["live_age_s"] == 30
        and np.isnan(got[Q + 300]["feed_q50"])
    )


def test_nowcasts_and_windows_are_kept_per_run_epoch():
    s = _store()
    vals = dict(zip(ST.NOWCAST_FIELDS, (100.0, 90.0, 0.5, 0.4, 0.1, 95.0, 85.0)))
    s.put_nowcast({"ts": Q + 60, "quarter_ts": Q, "epoch": 3, **vals})
    s.put_nowcast({"ts": Q + 60, "quarter_ts": Q, "epoch": 8, **dict(vals, take_point=7.0)})
    s.put_nowcast({"ts": Q + 120, "quarter_ts": Q, "epoch": 8, **vals})
    assert s.nowcasts(Q, Q + 900, epoch=3)[Q + 60]["take_point"] == 100.0
    assert s.nowcasts(Q, Q + 900, epoch=8)[Q + 60]["take_point"] == 7.0
    assert sorted(s.nowcasts(Q, Q + 900)) == [(Q + 60, 3), (Q + 60, 8), (Q + 120, 8)]
    assert s.last_nowcast_ts(8) == Q + 120 and s.last_nowcast_ts(3) == Q + 60
    assert s.last_nowcast_ts(4) is None
    s.put_window({"ts": Q, "now": Q + 60, "epoch": 3, "soc_kwh": 1.0})
    s.put_window({"ts": Q, "now": Q + 60, "epoch": 8, "soc_kwh": 2.0})
    assert s.window(Q, epoch=3)["soc_kwh"] == 1.0 and s.window(Q, epoch=7)["soc_kwh"] == 1.0
    assert s.window(Q)["soc_kwh"] == 2.0 and s.window(Q, epoch=8)["epoch"] == 8
    assert s.window(Q, epoch=2) is None


def test_last_event_of_a_kind():
    s = _store()
    assert s.last_event("restart") is None
    a = s.put_event(Q, "restart", {"n": 1})
    s.put_event(Q + 60, "refit", {})
    assert s.last_event("restart")["seq"] == a
    b = s.put_event(Q + 120, "restart", {"n": 2})
    want = {"seq": b, "ts": Q + 120, "kind": "restart", "detail": {"n": 2}}
    assert s.last_event("restart") == want


def test_recent_events_are_the_newest_in_seq_order():
    s = _store()
    assert s.recent_events(10) == []
    seqs = [s.put_event(Q + 60 * i, "settle", {"i": i}) for i in range(12)]
    got = s.recent_events(10)
    assert [e["seq"] for e in got] == seqs[2:] and got[-1]["detail"] == {"i": 11}
    assert s.recent_events(0) == []


def test_bands_are_read_as_of_a_serve_time_and_a_run_epoch():
    s = _store()
    t = Q - 840
    first, second = _band_row(), _band_row(take_q50=55.0)
    assert s.put_band(first, t, epoch=3)
    assert s.put_band(second, t, epoch=8)  # a re-run re-serves the same minute differently
    assert s.put_band(_band_row(take_q50=66.0), t - 60, epoch=8)
    assert s.band(Q, "A", 4, None, "v1", as_of=t, epoch=3)["take_q50"] == 20.0
    assert s.band(Q, "A", 4, None, "v1", as_of=t, epoch=8)["take_q50"] == 55.0
    # rows of a later run never reach an earlier run's read, at any serve time
    assert s.band(Q, "A", 4, None, "v1", as_of=t + 600, epoch=3)["take_q50"] == 20.0
    assert s.band(Q, "A", 4, None, "v1", as_of=t - 60, epoch=3) is None
    assert s.band(Q, "A", 4, None, "v1", as_of=t - 60, epoch=8)["take_q50"] == 66.0
    assert s.band(Q, "A", 4, None, "v1", as_of=t)["take_q50"] == 55.0  # newest of all epochs
    assert not s.put_band(dict(second), t, epoch=8)
    with pytest.raises(ValueError, match="recorded_at"):
        s.put_band(_band_row(take_q50=1.0), t, epoch=8)


def test_fetch_at_takes_the_newest_fetch_at_or_before_now():
    s = _store()
    t1 = {"ts": np.array([Q, Q + 3600], dtype=np.int64), "ghi": np.array([0.1, NAN])}
    t2 = {"ts": np.array([Q + 3600], dtype=np.int64), "ghi": np.array([1 / 3])}
    s.put_fetch(Q - 7200, "forecast", Q - 7100, t1)
    s.put_fetch(Q - 3600, "forecast", Q - 3500, t2)
    assert s.fetch_at(Q - 7200, "forecast") is None and s.fetch_at(Q, "archive") is None
    issue, t = s.fetch_at(Q - 3600, "forecast")
    assert issue == Q - 7200 and t["ts"].dtype == np.int64 and np.isnan(t["ghi"][1])
    issue, t = s.fetch_at(Q, "forecast")
    assert issue == Q - 3600 and t["ghi"].tolist() == [1 / 3]


def test_registry_meta_events_greed():
    s = _store()
    s.set_registry("site_models", '{"a": 1}', Q)
    s.set_registry("site_models", '{"a": 2}', Q + 1)
    assert s.registry("site_models") == '{"a": 2}' and s.registry("nope") is None
    s.set_meta("install_id", "abc")
    assert s.meta("install_id") == "abc" and s.meta("nope") is None
    a = s.put_event(Q, "restart", {"why": "boot"})
    b = s.put_event(Q + 60, "settle", {"n": 3})
    assert b == a + 1 and [e["kind"] for e in s.events()] == ["restart", "settle"]
    assert s.events(b) == [{"seq": b, "ts": Q + 60, "kind": "settle", "detail": {"n": 3}}]
    from betterybalance.replay.phases import GREED_COLUMNS

    assert ST.GREED_COLUMNS == GREED_COLUMNS
    g = {c: 1.0 for c in GREED_COLUMNS} | {"day": "2026-10-25", "low_soc_ok": True}
    s.put_greed(g)
    s.put_greed(dict(g, soc_kwh=2.0))
    assert s._db.execute("select count(*), max(soc_kwh) from greed").fetchone() == (1, 2.0)


def test_runs_lifecycle():
    s = _store()
    rid = s.new_run(Q, Q - 86400, Q)
    s.put_run_quarters(
        rid,
        [
            {
                "quarter_ts": Q,
                "path": p,
                "import_kwh": 0.1,
                "export_kwh": 0.0,
                "money_eur": NAN,
                "missing": True,
            }
            for p in LG.PATHS
        ],
    )
    s.put_run_decisions(rid, [_decision(), dict(_decision(ts=Q + 120), seq=7, hash="x")])
    s.finish_run(
        rid, Q + 10, "done", {"result_eur": 1.5, "lo_eur_day": NAN}, "runs/2026-10-31/report.md"
    )
    (run,) = s.runs()
    assert run["run_id"] == rid and run["status"] == "done" and run["headline"]["result_eur"] == 1.5
    assert np.isnan(run["headline"]["lo_eur_day"]) and run["report_path"].endswith("report.md")
    assert (
        s._db.execute("select count(*) from run_quarters where run_id = ?", (rid,)).fetchone()[0]
        == 3
    )
    assert s._db.execute("select count(*) from run_decisions").fetchone()[0] == 2
    assert s.new_run(Q, Q, Q) == rid + 1 and s.runs()[1]["status"] == "running"


def test_prune_counts_every_pruned_table_and_vacuum():
    s = _chained_store()
    now = 1830000000
    old = now - 401 * 86400
    s.put_minute({"ts": old, "soc_pct": 1.0})
    s.put_live_minutes([{"ts": old, "buy_live": 1.0, "sell_live": 1.0, "state": 0, "source": "x"}])
    s.upsert_features(old, da_eur_mwh=1.0)
    s.put_band(
        {"target_ts": old, "tier": "A", "h": 1, "e": -1, "model_version": "v", "p_dual": 0.5}, old
    )
    s.put_window({"ts": old, "now": old, "da_known": np.ones(2, bool)})
    s.put_nowcast({"ts": old, "quarter_ts": old})
    s.put_fetch(old, "forecast", old, {"ts": np.array([old])})
    s.append_chained("decision", _decision(ts=old))
    s.append_chained("booking", _booking(qts=old))
    s.put_settlement(
        {
            "quarter_ts": old,
            "take": 1.0,
            "feed": 1.0,
            "regulation_state": 0,
            "published_at": old,
            "source": "x",
        }
    )
    counts = s.prune(now)
    assert counts == {t: 1 for t in ST.PRUNED}
    assert len(s.bookings(0, now)) == 1 and len(s.settlements(0, now)) == 1 and s.head()[0] == 2
    s.vacuum()
    assert s.prune(now) == {t: 0 for t in ST.PRUNED}


def test_vacuum_and_checkpoint_leave_an_empty_wal(tmp_path):
    import os

    path = tmp_path / "store.sqlite3"
    s = Store(path)
    s.migrate()
    for k in range(2000):
        s.set_meta(f"k{k}", "x" * 200)
    assert os.path.getsize(str(path) + "-wal") > 0
    s.vacuum()
    assert os.path.getsize(str(path) + "-wal") == 0
    s.set_meta("after", "y")
    assert os.path.getsize(str(path) + "-wal") > 0
    assert s.checkpoint()[0] == 0 and os.path.getsize(str(path) + "-wal") == 0
    s.close()
