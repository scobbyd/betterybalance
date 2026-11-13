import csv
import datetime as dt
import io
import json
import math
import zipfile
from dataclasses import replace

import pytest
from betterybalance.core import ledger as LG
from betterybalance.core.site import contract as C
from betterybalance.core.timegrid import ZONE
from betterybalance.live import export as X

from tests.test_live_results import _settle_rows
from tests.test_live_site import _run, _site


def _ledger(files) -> list[dict]:
    return list(csv.DictReader(io.StringIO(files["ledger.csv"].decode())))


def test_ledger_columns_join_the_two_kinds_once():
    cols = LG.LEDGER_COLUMNS
    assert cols[:2] == ("seq", "kind") and cols[-2:] == ("prev_hash", "hash")
    assert len(cols) == len(set(cols)) and X.LEDGER_COLUMNS is cols
    assert set(LG.DECISION_FIELDS) | set(LG.BOOKING_FIELDS) <= set(cols)


def test_export_roundtrips_floats_and_carries_no_location():
    site, store, start = _site()
    _run(site, start, 60)
    files = X.export_ledger(store, site.lc, start, start + 3600, include_flows=False)
    assert set(files) == {"ledger.csv", "settlements.csv", "manifest.json"}
    rows = _ledger(files)
    assert tuple(rows[0]) == X.LEDGER_COLUMNS
    assert [int(r["seq"]) for r in rows] == sorted(int(r["seq"]) for r in rows)
    m = json.loads(files["manifest.json"])
    assert m["install_id_hash"] == LG.genesis_hash("install-1") and "latitude" not in json.dumps(m)
    assert "longitude" not in json.dumps(m)
    assert m["chain_head"] == store.head()[1] and m["flows_included"] is False
    # the rows read back from the CSV verify against the manifest's anchors
    items = [(r["kind"], r) for r in rows]
    assert m["first_prev_hash"] == m["install_id_hash"] == rows[0]["prev_hash"]
    check = LG.verify_chain(items, m["first_prev_hash"])
    assert check.ok and check.n == m["n_rows"] == len(rows)
    assert m["seq_span"] == [int(rows[0]["seq"]), int(rows[-1]["seq"])]
    assert m["last_hash"] == rows[-1]["hash"] == m["chain_head"]
    # bools as true/false, floats as repr, NaN as an empty cell, never "None"
    text = files["ledger.csv"].decode()
    assert "None" not in text and "nan" not in text.lower()
    dec = next(r for r in rows if r["kind"] == "decision")
    assert dec["committed"] in ("true", "false") and dec["import_kwh"] == ""
    house = next(r for r in rows if r["kind"] == "booking" and r["path"] == "house")
    assert house["soc_pct"] == "" and house["missing"] == "false"
    stored = store.chain(int(house["seq"]), int(house["seq"]))[0][1]
    assert house["import_kwh"] == repr(stored["import_kwh"])
    # the manifest: the site dict rebuilds the config with any location, terms included
    site_cfg = C.from_dict({**m["site"], "latitude": 0.0, "longitude": 0.0})
    assert site_cfg == replace(site.lc.cfg, latitude=0.0, longitude=0.0)
    assert m["format"] == 1 and m["range"] == [start, start + 3600]
    assert m["model_versions"] == ["v-test"] and m["pack_version"] == "p1"
    assert m["integration_version"] == "0.3.0" and m["n_quarters"] == 4
    assert [e["epoch"] for e in m["events"]] == [store.last_event("restart")["seq"]]
    assert math.isclose(m["events"][0]["soc_kwh"], m["soc0_kwh"])
    assert m["result_settled_eur"] == 0.0  # nothing settled yet


def test_export_after_a_later_settlement_is_seq_contiguous():
    site, store, start = _site()
    _run(site, start, 2 * 1440)
    day2 = start + 86400  # the synthetic days are 24 h (October, no DST change in range)
    rows_in = _settle_rows(store, start, range(96), published=start + 2 * 86400 - 600)
    assert len(site.settle(rows_in, start + 2 * 86400)) == 96
    files = X.export_ledger(store, site.lc, start, day2, include_flows=False)
    rows = _ledger(files)
    m = json.loads(files["manifest.json"])
    seqs = [int(r["seq"]) for r in rows]
    assert seqs == list(range(m["seq_span"][0], m["seq_span"][1] + 1))
    assert m["seq_span"][1] == store.head()[0]  # the settled day-1 bookings close the span
    # day-2 decision rows sit between day 1's provisional and settled bookings
    assert any(r["kind"] == "decision" and int(r["ts"]) >= day2 for r in rows)
    assert LG.verify_chain([(r["kind"], r) for r in rows], m["first_prev_hash"]).ok
    assert m["n_quarters"] == 96
    # the settled result of day 1 only: money(house) - money(shadow) of the newest bookings
    newest = store.newest_bookings(start, day2)
    want = sum(
        LG.quarter_result(newest[(q, "house")], newest[(q, "shadow")])
        for q in range(start, day2, 900)
    )
    assert math.isclose(m["result_settled_eur"], want, abs_tol=1e-9)
    sett = list(csv.DictReader(io.StringIO(files["settlements.csv"].decode())))
    assert {int(r["quarter_ts"]) for r in sett} == set(range(start, day2, 900))


def test_flows_only_on_request_and_zip_is_deterministic():
    site, store, start = _site()
    _run(site, start, 30)
    a = X.zip_bytes(X.export_ledger(store, site.lc, start, start + 1800, True))
    b = X.zip_bytes(X.export_ledger(store, site.lc, start, start + 1800, True))
    assert a == b and "flows_minute.csv" in zipfile.ZipFile(io.BytesIO(a)).namelist()
    flows = list(
        csv.DictReader(io.StringIO(zipfile.ZipFile(io.BytesIO(a)).read("flows_minute.csv").decode()))
    )
    assert [int(r["ts"]) for r in flows] == list(range(start, start + 1800, 60))
    assert json.loads(zipfile.ZipFile(io.BytesIO(a)).read("manifest.json"))["flows_included"]


def test_settlements_csv_keeps_every_version():
    site, store, start = _site()
    _run(site, start, 2 * 15)
    site.settle(_settle_rows(store, start, range(2)), start + 90000)
    corr = _settle_rows(store, start, [1], take=300.0, published=start + 99999)
    site.settle(corr, start + 100000)
    files = X.export_ledger(store, site.lc, start, start + 1800, include_flows=False)
    sett = list(csv.DictReader(io.StringIO(files["settlements.csv"].decode())))
    assert [(int(r["quarter_ts"]), float(r["take"])) for r in sett] == [
        (start, 100.0), (start + 900, 100.0), (start + 900, 300.0),
    ]
    m = json.loads(files["manifest.json"])
    newest = store.newest_bookings(start, start + 1800)
    want = sum(
        LG.quarter_result(newest[(q, "house")], newest[(q, "shadow")]) for q in (start, start + 900)
    )
    assert math.isclose(m["result_settled_eur"], want, abs_tol=1e-9)


def test_results_live_has_the_plan2_shape_and_daily_heads():
    site, store, start = _site()
    _run(site, start, 4 * 15)
    site.settle(_settle_rows(store, start, range(4)), start + 90000)
    d = X.export_results_live(store, site.lc, start, start + 86400, "0.3.0", heads=True)
    assert {
        "format", "version", "source", "window", "contract_mode", "site_class",
        "tiers_by_month", "learning_days", "terminal_eur", "interval", "daily",
    } <= set(d)
    assert d["source"] == "live" and d["terminal_eur"] == 0.0
    day = d["window"]["from"]
    newest = store.newest_bookings(start, start + 3600)
    want = sum(
        newest[(q, "actual")]["money_eur"] - newest[(q, "shadow")]["money_eur"]
        for q in range(start, start + 3600, 900)
    )
    assert d["daily"] == [{"day": day, "diff_eur": round(want, 2)}]
    assert set(d["tiers_by_month"]) == {day[:7]} and "latitude" not in json.dumps(d)
    # the day is not over in the store (no row of the next day): no head yet
    assert d["daily_heads"] == []
    plain = X.export_results_live(store, site.lc, start, start + 86400, "v", False)
    assert "daily_heads" not in plain


def test_daily_heads_name_the_head_before_the_first_row_of_the_next_day():
    site, store, start = _site()
    _run(site, start, 1440 + 30)
    d = X.export_results_live(store, site.lc, start, start + 86400, "v", heads=True)
    nxt = min(
        [r["seq"] for r in store.decisions(start + 86400, start + 2 * 86400)]
        + [r["seq"] for r in store.bookings(start + 86400, start + 2 * 86400)]
    )
    row = store.chain(nxt - 1, nxt - 1)[0][1]
    day = dt.datetime.fromtimestamp(start, ZONE).date().isoformat()
    assert d["daily_heads"] == [{"day": day, "seq": nxt - 1, "hash": row["hash"]}]
    assert d["daily"] == []  # nothing settled: no daily difference


def test_ledger_tail_joins_the_newest_settlement():
    site, store, start = _site()
    _run(site, start, 4 * 15)
    site.settle(_settle_rows(store, start, range(2)), start + 90000)
    corr = _settle_rows(store, start, [1], take=300.0, published=start + 99999)
    site.settle(corr, start + 100000)
    rows = X.ledger_tail(store, limit=500, now=start + 3600)
    dec = [
        r for r in store.decisions(start, start + 3600) if r["committed"] or r["action"] != "hold"
    ]
    assert dec and [r["seq"] for r in rows] == [r["seq"] for r in reversed(dec)]  # newest first
    for r in rows:
        q = (r["quarter_ts"] - start) // 900
        assert r["take"] == {0: 100.0, 1: 300.0}.get(q)
        assert r["feed"] == (50.0 if q in (0, 1) else None)
        assert r["regulation_state"] == (1 if q in (0, 1) else None)
        assert all(not (isinstance(v, float) and math.isnan(v)) for v in r.values())
    assert len(X.ledger_tail(store, limit=2, now=start + 3600)) == min(2, len(rows))


def test_streamed_zip_and_folder_equal_the_in_memory_export(tmp_path):
    site, store, start = _site()
    _run(site, start, 30)
    for flows in (False, True):
        files = X.export_ledger(store, site.lc, start, start + 1800, flows)
        path = tmp_path / f"x{flows}.zip"
        names = X.write_export_zip(store, site.lc, start, start + 1800, flows, path)
        assert path.read_bytes() == X.zip_bytes(files) and names == sorted(files)
        folder = tmp_path / "dir"
        assert X.write_export_dir(store, site.lc, start, start + 1800, flows, folder) == names
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == files
    # the ledger pages through the chain: a page of one row gives the same file
    small = X.export_ledger(store, site.lc, start, start + 1800, False)
    X.CHAIN_PAGE, page = 1, X.CHAIN_PAGE
    try:
        assert X.export_ledger(store, site.lc, start, start + 1800, False) == small
    finally:
        X.CHAIN_PAGE = page


def test_export_range_refuses_reversed_dates_long_flows_and_overflow():
    d = dt.date(2026, 11, 2)
    assert X.export_range(d, d, True) == X.day_range(d, d)
    assert X.export_range(d, d + dt.timedelta(days=30), True)[1] > 0  # 31 days with flows
    with pytest.raises(ValueError, match="at most 31 days"):
        X.export_range(d, d + dt.timedelta(days=31), True)
    assert X.export_range(d, d + dt.timedelta(days=400), False)  # no cap without flows
    with pytest.raises(ValueError, match="on or before"):
        X.export_range(d + dt.timedelta(days=1), d, False)
    with pytest.raises(ValueError, match="out of range"):
        X.export_range(d, dt.date(9999, 12, 31), False)


def test_export_range_refusals_carry_a_translation_key_and_placeholders():
    d = dt.date(2026, 11, 2)
    late, end, far = d + dt.timedelta(days=1), dt.date(9999, 12, 31), d + dt.timedelta(days=31)
    cases = (
        ((late, d, False), "export_order", {"from": "2026-11-03", "to": "2026-11-02"}),
        ((d, end, False), "export_dates", {"from": "2026-11-02", "to": "9999-12-31"}),
        ((d, far, True), "export_flow_days", {"max_days": "31", "days": "32"}),
    )
    for args, key, placeholders in cases:
        with pytest.raises(X.RangeRefusal) as err:
            X.export_range(*args)
        assert err.value.key == key and err.value.placeholders == placeholders
        assert key in X.REFUSAL_KEYS
    with pytest.raises(ValueError, match="unknown refusal key"):
        X.RangeRefusal("x", "export_typo")


def test_a_range_without_rows_exports_an_empty_span():
    site, store, start = _site()
    files = X.export_ledger(store, site.lc, start - 3600, start, include_flows=False)
    m = json.loads(files["manifest.json"])
    assert m["seq_span"] is None and m["n_rows"] == 0 and m["first_prev_hash"] is None
    assert store.chain_span(start - 3600, start) is None
    assert _ledger(files) == [] and files["ledger.csv"].decode().startswith("seq,kind,")
