import csv
import json
import math
import pathlib
import random
import subprocess
import sys
import tempfile
from dataclasses import replace

import pytest
from betterybalance.core import ledger as LG
from betterybalance.core.battery import DEYE_AMPS, DEYE_TOU, ExecutionProfile, Executor
from betterybalance.core.site import contract as C
from betterybalance.live import export as X
from betterybalance.live import site as LS
from betterybalance.verify import check as V

from tests.test_live_results import _settle_rows
from tests.test_live_site import _flows, _negative_take, _run, _site

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write(d: pathlib.Path, files: dict) -> pathlib.Path:
    for name, data in files.items():
        (d / name).write_bytes(data)
    return d


def _charging(ts):
    """0,5 kW house load while the real battery charges 2 kW (house and actual paths differ)."""
    return _flows(ts, imp=2.5 / 60, charge=2.0 / 60)


def _export(tmp_path, monkeypatch, flows=True):
    """8 quarters in which the shadow commits (every nowcast takes at -200 EUR/MWh), settled."""
    _negative_take(monkeypatch)
    site, store, start = _site()
    _run(site, start, 8 * 15, flows=_charging)
    site.settle(_settle_rows(store, start, range(8)), start + 90000)
    return _write(tmp_path, X.export_ledger(store, site.lc, start, start + 7200, flows))


def _read(d) -> list[dict]:
    with open(d / "ledger.csv", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(d, rows) -> None:
    with open(d / "ledger.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LG.LEDGER_COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _manifest(d) -> dict:
    return json.loads((d / "manifest.json").read_text())


def _set_manifest(d, m) -> None:
    (d / "manifest.json").write_text(json.dumps(m, sort_keys=True, indent=2) + "\n")


def _rewrite(d, edit, rechain=False):
    """Apply edit(rows) to ledger.csv and return the seq of the first changed row; with rechain,
    recompute every hash from that row on with ledger.chain_row and put the new last hash in the
    manifest (an attacker with root)."""
    rows = _read(d)
    before = [dict(r) for r in rows]
    edit(rows)
    k = next(i for i, (a, b) in enumerate(zip(before, rows, strict=True)) if a != b)
    if rechain:
        prev = rows[k]["prev_hash"]
        for r in rows[k:]:
            r.update(LG.chain_row(r["kind"], int(r["seq"]), r, prev))
            r["prev_hash"] = prev
            prev = r["hash"]
        m = _manifest(d)
        m["last_hash"] = m["chain_head"] = prev
        _set_manifest(d, m)
    _write_rows(d, rows)
    return int(rows[k]["seq"])


def _first(rows, kind, **filters) -> dict:
    return next(
        r for r in rows if r["kind"] == kind and all(r[k] == v for k, v in filters.items())
    )


def _terms(d):
    return C.from_dict({**_manifest(d)["site"], "latitude": 0.0, "longitude": 0.0}).terms


def _reprice(row: dict, terms) -> None:
    """``money_eur`` and ``tax_eur`` of a priced, not missing booking from its own cells."""
    q = int(row["quarter_ts"])
    money, tax = LG.money_eur(
        float(row["import_kwh"]), float(row["export_kwh"]), float(row["buy"]),
        float(row["sell"]), int(row["state"]), terms, C.tax_mode_at(terms, q),
    )
    row["money_eur"], row["tax_eur"] = repr(money), repr(tax)


def _shift_energy_and_money(d, rows, q: str | None = None) -> None:
    """+0,1 kWh shadow import in quarter ``q`` (default the first settled one), on every shadow
    booking of that quarter (the provisional one and its settled rebooking, as a careful
    attacker would), the money recomputed to match and the manifest's settled total moved by
    the same amount."""
    terms = _terms(d)
    if q is None:
        q = _first(rows, "booking", path="shadow", price_source="settled")["quarter_ts"]
    delta = 0.0
    for r in rows:
        if r["kind"] == "booking" and r["path"] == "shadow" and r["quarter_ts"] == q:
            old = float(r["money_eur"])
            r["import_kwh"] = repr(float(r["import_kwh"]) + 0.1)
            _reprice(r, terms)
            if r["price_source"] == "settled":
                delta = float(r["money_eur"]) - old
    m = _manifest(d)
    m["result_settled_eur"] -= delta  # result = house - shadow
    _set_manifest(d, m)


def _drop(d, pred) -> None:
    """Delete the ledger rows matching ``pred``, renumber and rechain every row after the
    first deleted one, and make the manifest agree (end anchors and settled total), as an
    attacker with root would. ``n_quarters`` stays."""
    rows = _read(d)
    k = next(i for i, r in enumerate(rows) if pred(r))
    seq, prev = int(rows[k]["seq"]), rows[k]["prev_hash"]
    keep = [r for r in rows if not pred(r)]
    for i, r in enumerate(keep[k:]):
        r.update(LG.chain_row(r["kind"], seq + i, r, prev))
        prev = r["hash"]
    _write_rows(d, keep)
    m = _manifest(d)
    last = int(keep[-1]["seq"])
    m.update(
        n_rows=len(keep), last_hash=prev, chain_head=prev, chain_head_seq=last,
        seq_span=[m["seq_span"][0], last],
    )
    _set_manifest(d, m)
    items, _, m, _ = V.read_export(d)
    start, end = m["range"]
    m["result_settled_eur"] = V._settled_total(
        [r for k, r in items if k == "booking" and start <= r["quarter_ts"] < end]
    )
    _set_manifest(d, m)


# -- an untouched export ------------------------------------------------------------------


def test_untouched_export_passes(monkeypatch):
    d = _export(pathlib.Path(tempfile.mkdtemp()), monkeypatch)
    r = V.verify_export(d)
    assert r.ok and r.quarters == 8
    assert r.lines[0] == "chain intact, 8 quarters recomputed, totals match, energies match"
    assert any("rewrite" in line and "ledger_head" in line for line in r.lines[1:])
    shadow = [x for x in _read(d) if x["kind"] == "booking" and x["path"] == "shadow"]
    assert any(float(x["batt_charge_kwh"] or 0) > 0 for x in shadow)  # the shadow traded
    # the inputs the chain does not anchor are named
    assert any("settlements.csv" in x and "TenneT" in x for x in r.lines)
    terms = next(x for x in r.lines if x.startswith("Terms used"))
    assert "import fee 0,0165 EUR/kWh" in terms and "BTW 21 %" in terms
    assert "tax mode net_exporter" in terms and "dual pricing on" in terms
    assert any("flows_minute.csv" in x and "not chained" in x for x in r.lines)


def test_read_export_types_the_rows(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    items, settlements, manifest, flows = V.read_export(d)
    assert len(items) == manifest["n_rows"] and flows is not None
    kind, row = items[0]
    assert kind == "decision" and isinstance(row["committed"], bool) and isinstance(row["ts"], int)
    house = next(r for k, r in items if k == "booking" and r["path"] == "house")
    assert math.isnan(house["soc_pct"]) and house["missing"] is False
    assert isinstance(house["state"], int) and "ts" not in house  # its own kind's fields only
    assert len(settlements) == 8 and isinstance(settlements[0]["published_at"], int)
    assert len(flows["ts"]) == 8 * 15 and flows["ts"].dtype.kind == "i"
    (d / "flows_minute.csv").unlink()
    assert V.read_export(d)[3] is None


def test_manifest_names_each_settle_event(tmp_path, monkeypatch):
    m = _manifest(_export(tmp_path, monkeypatch))
    assert len(m["settles"]) == 1
    ev = m["settles"][0]
    assert len(ev["quarters"]) == len(ev["published_at"]) == 8 and isinstance(ev["seq"], int)


# -- tampering ----------------------------------------------------------------------------


def test_tampered_decision_breaks_the_chain(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    seq = _rewrite(d, lambda rows: _first(rows, "decision").update(kw="9.9"))
    r = V.verify_export(d)
    assert not r.ok and r.lines[0] == f"chain broken at seq {seq}: hash mismatch"


def test_dropped_last_row_is_caught_by_the_manifest_anchor(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    rows = _read(d)
    last = int(rows[-1]["seq"])
    _write_rows(d, rows[:-1])  # the rows left still chain
    r = V.verify_export(d)
    assert not r.ok and r.lines[0] == f"chain broken at seq {last}: chain truncated"


def test_a_rechain_that_keeps_the_old_last_hash_is_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    m = _manifest(d)
    _rewrite(d, lambda rows: _first(rows, "decision").update(kw="9.9"), rechain=True)
    _set_manifest(d, m)  # the manifest someone else holds
    r = V.verify_export(d)
    last = m["seq_span"][1]
    assert r.lines[0] == f"chain broken at seq {last}: last hash differs from the manifest"


def test_rechained_money_is_still_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    rows = _read(d)
    row = _first(rows, "booking", path="shadow", price_source="settled")
    _rewrite(
        d,
        lambda rows: _first(rows, "booking", path="shadow", price_source="settled").update(
            money_eur="-1.0"
        ),
        rechain=True,
    )
    r = V.verify_export(d)
    iso = V.quarter_iso(int(row["quarter_ts"]))
    assert not r.ok and r.lines[0] == f"money mismatch at quarter {iso} path shadow"


def test_rechained_price_is_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    terms = _terms(d)
    q = _first(_read(d), "booking", price_source="settled")["quarter_ts"]

    def edit(rows):  # the three settled rows of the quarter, one price, money to match
        for r in rows:
            if r["kind"] == "booking" and r["quarter_ts"] == q and r["price_source"] == "settled":
                r["buy"] = "250.0"
                _reprice(r, terms)

    _rewrite(d, edit, rechain=True)
    r = V.verify_export(d)
    assert r.lines[0] == f"price mismatch at quarter {V.quarter_iso(int(q))} path shadow"


def test_one_path_repriced_is_a_mixed_price(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    terms = _terms(d)
    q = int(_first(_read(d), "booking", path="house", price_source="settled")["quarter_ts"])

    def edit(rows):
        row = _first(rows, "booking", path="house", price_source="settled")
        row["buy"] = "250.0"
        _reprice(row, terms)

    _rewrite(d, edit, rechain=True)
    assert V.verify_export(d).lines[0] == f"mixed price sources at quarter {V.quarter_iso(q)}"


def test_total_mismatch(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    m = _manifest(d)
    m["result_settled_eur"] += 0.01
    _set_manifest(d, m)
    r = V.verify_export(d)
    assert not r.ok and r.lines[0] == "total mismatch"


def test_rechained_energy_is_caught_only_with_flows(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    q = int(_first(_read(d), "booking", path="shadow", price_source="settled")["quarter_ts"])
    _rewrite(d, lambda rows: _shift_energy_and_money(d, rows), rechain=True)
    r = V.verify_export(d)
    assert r.lines[0] == f"energy mismatch at quarter {V.quarter_iso(q)}: shadow import_kwh"
    (d / "flows_minute.csv").unlink()
    r = V.verify_export(d)
    assert r.ok and "energies not checked (no flows_minute.csv)" in r.lines
    assert r.lines[0] == "chain intact, 8 quarters recomputed, totals match"


def test_rechained_decision_is_caught_only_with_flows(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    rows = _read(d)
    # the tenth charge decision: past the executor's dwell after the first minute's discharge,
    # and its setpoint takes effect inside the same quarter
    row = [r for r in rows if r["kind"] == "decision" and r["action"] == "charge"][9]
    q = int(row["ts"]) - int(row["ts"]) % 900
    assert int(row["ts"]) - q < 840
    seq = int(row["seq"])
    _rewrite(
        d, lambda rs: next(r for r in rs if int(r["seq"]) == seq).update(kw="6.0"), rechain=True
    )
    r = V.verify_export(d)
    assert r.lines[0] == f"energy mismatch at quarter {V.quarter_iso(q)}: shadow import_kwh"
    (d / "flows_minute.csv").unlink()
    assert V.verify_export(d).ok


def test_a_rebooking_keeps_the_energies_of_the_booking_it_rebooks(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    (d / "flows_minute.csv").unlink()
    terms = _terms(d)
    row = _first(_read(d), "booking", path="actual", price_source="settled")

    def edit(rows):  # the settled rebooking only, its money made to match
        r = _first(rows, "booking", path="actual", price_source="settled")
        r["export_kwh"] = repr(float(r["export_kwh"]) + 0.2)
        _reprice(r, terms)

    _rewrite(d, edit, rechain=True)
    iso = V.quarter_iso(int(row["quarter_ts"]))
    assert V.verify_export(d).lines[0] == f"energy mismatch at quarter {iso}: actual export_kwh"


def test_tampered_flows_are_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    with open(d / "flows_minute.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[3]["grid_import_kwh"] = repr(float(rows[3]["grid_import_kwh"]) + 0.05)
    with open(d / "flows_minute.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    q = int(rows[3]["ts"]) - int(rows[3]["ts"]) % 900
    r = V.verify_export(d)
    assert not r.ok
    assert r.lines[0] in (
        f"energy mismatch at quarter {V.quarter_iso(q)}: shadow import_kwh",
        f"energy mismatch at quarter {V.quarter_iso(q)}: house import_kwh",
    )


# -- exports that must pass ---------------------------------------------------------------


def test_day_exported_after_its_settlement_passes(tmp_path, monkeypatch):
    _negative_take(monkeypatch)
    site, store, start = _site()
    _run(site, start, 2 * 1440, flows=_charging)
    day2 = start + 86400
    rows_in = _settle_rows(store, start, range(96), published=start + 2 * 86400 - 600)
    assert len(site.settle(rows_in, start + 2 * 86400)) == 96
    _write(tmp_path, X.export_ledger(store, site.lc, start, day2, True))
    rows = _read(tmp_path)
    assert any(r["kind"] == "decision" and int(r["ts"]) >= day2 for r in rows)  # day 2 inside
    shadow = [r for r in rows if r["kind"] == "booking" and r["path"] == "shadow"]
    assert any(float(r["batt_charge_kwh"] or 0) > 0 for r in shadow)  # the shadow traded
    r = V.verify_export(tmp_path)
    assert r.ok and r.quarters == 96 and r.lines[0].endswith("energies match"), r.lines


def test_corrected_quarter_passes(tmp_path):
    site, store, start = _site()
    _run(site, start, 8 * 15)
    site.settle(_settle_rows(store, start, range(8)), start + 90000)
    assert site.settle(
        _settle_rows(store, start, [3], take=300.0, published=start + 99999), start + 100000
    ) == [start + 3 * 900]
    # a newer version stored but not yet booked is no mismatch
    store.put_settlement(_settle_rows(store, start, [5], take=400.0, published=start + 100500)[0])
    _write(tmp_path, X.export_ledger(store, site.lc, start, start + 7200, True))
    m = _manifest(tmp_path)
    quarters = [e["quarters"] for e in m["settles"]]
    assert quarters == [[start + 900 * k for k in range(8)], [start + 2700]]
    r = V.verify_export(tmp_path)
    assert r.ok, r.lines


def test_a_correction_moved_back_to_the_older_version_is_caught(tmp_path):
    site, store, start = _site()
    _run(site, start, 8 * 15)
    site.settle(_settle_rows(store, start, range(8)), start + 90000)
    corrected = _settle_rows(store, start, [3], take=300.0, published=start + 99999)
    site.settle(corrected, start + 100000)
    _write(tmp_path, X.export_ledger(store, site.lc, start, start + 7200, False))
    terms = _terms(tmp_path)
    q = start + 3 * 900
    total: dict[str, float] = {}

    def edit(rows):  # the corrected rebookings back at the first version's take, money to match
        for path, sign in (("shadow", -1.0), ("house", 1.0), ("actual", 0.0)):
            r = [x for x in rows if x["kind"] == "booking" and int(x["quarter_ts"]) == q
                 and x["path"] == path and x["price_source"] == "settled"][-1]
            old = float(r["money_eur"])
            r["buy"] = "100.0"
            _reprice(r, terms)
            total[path] = sign * (float(r["money_eur"]) - old)  # result = house - shadow

    _rewrite(tmp_path, edit, rechain=True)
    m = _manifest(tmp_path)
    m["result_settled_eur"] += total["shadow"] + total["house"]
    _set_manifest(tmp_path, m)
    r = V.verify_export(tmp_path)
    assert r.lines[0] == f"price mismatch at quarter {V.quarter_iso(q)} path shadow"


def test_corrected_quarter_without_settle_events_matches_by_price(tmp_path):
    site, store, start = _site()
    _run(site, start, 8 * 15)
    site.settle(_settle_rows(store, start, range(8)), start + 90000)
    corrected = _settle_rows(store, start, [3], take=300.0, published=start + 99999)
    site.settle(corrected, start + 100000)
    _write(tmp_path, X.export_ledger(store, site.lc, start, start + 7200, False))
    m = _manifest(tmp_path)
    del m["settles"]
    _set_manifest(tmp_path, m)
    assert V.verify_export(tmp_path).ok


def test_restart_gap_and_missing_minutes_pass(tmp_path, monkeypatch):
    _negative_take(monkeypatch)
    site, store, start = _site()
    gone = {start + 120, start + 180, start + 900 + 300}
    _run(site, start, 15 + 7, flows=lambda ts: None if ts in gone else _charging(ts))
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 900 * 3 + 60)  # quarter 1 half run: quarters 1 and 2 a gap
    for ts in range(start + 2700, start + 7200, 60):
        if (ts - start) % 900 == 60:
            site2.setup_quarter(ts)
        site2.on_minute(ts, _charging(ts))
    site2.settle(_settle_rows(store, start, range(8)), start + 90000)
    _write(tmp_path, X.export_ledger(store, site2.lc, start, start + 7200, True))
    assert len(_manifest(tmp_path)["events"]) == 2
    shadow = [r for r in _read(tmp_path) if r["kind"] == "booking" and r["path"] == "shadow"]
    assert any(float(r["batt_charge_kwh"] or 0) > 0 for r in shadow)
    r = V.verify_export(tmp_path)
    assert r.ok and r.lines[0].endswith("energies match"), r.lines


def test_a_never_stepped_shadow_re_anchored_at_the_sensor_passes(tmp_path, monkeypatch):
    _negative_take(monkeypatch)
    site, store, start = _site()
    _run(site, start, 2 * 15, flows=lambda ts: None)  # every minute missing: never stepped
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 2700, 70.0)  # the SoC sensor reads 70 %
    assert store.last_event("restart")["detail"]["soc_kwh"] == pytest.approx(0.7 * 48)
    for ts in range(start + 2700, start + 7200, 60):
        if (ts - start) % 900 == 60:
            site2.setup_quarter(ts)
        site2.on_minute(ts, _charging(ts))
    site2.settle(_settle_rows(store, start, range(8)), start + 90000)
    _write(tmp_path, X.export_ledger(store, site2.lc, start, start + 7200, True))
    shadow = [r for r in _read(tmp_path) if r["kind"] == "booking" and r["path"] == "shadow"]
    assert any(float(r["batt_charge_kwh"] or 0) > 0 for r in shadow)
    r = V.verify_export(tmp_path)
    assert r.ok and r.lines[0].endswith("energies match"), r.lines


def test_empty_range_passes(tmp_path):
    site, store, start = _site()
    _run(site, start, 15)
    _write(tmp_path, X.export_ledger(store, site.lc, start + 86400, start + 90000, True))
    r = V.verify_export(tmp_path)
    assert r.ok and r.quarters == 0


# -- the command line ---------------------------------------------------------------------


def test_main_prints_and_returns_the_status(tmp_path, monkeypatch, capsys):
    d = _export(tmp_path, monkeypatch)
    assert V.main([str(d)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("chain intact, 8 quarters recomputed")
    _rewrite(d, lambda rows: _first(rows, "decision").update(kw="9.9"))
    assert V.main([str(d)]) == 1
    assert capsys.readouterr().out.startswith("chain broken at seq")
    assert V.main([]) == 2


def test_module_runs_without_home_assistant(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    code = (
        "import sys, runpy; sys.modules['homeassistant'] = None; "
        "sys.argv = ['verify', sys.argv[1]]; "
        "runpy.run_module('betterybalance.verify', run_name='__main__')"
    )
    p = subprocess.run(
        [sys.executable, "-c", code, str(d)], capture_output=True, text=True, cwd=ROOT
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.startswith("chain intact, 8 quarters recomputed")
    assert "betterybalance.live" not in p.stdout


def test_verify_imports_neither_live_nor_home_assistant():
    code = (
        "import sys; import betterybalance.verify.check; "
        "print(sorted(m for m in sys.modules if m.startswith(('betterybalance.live', "
        "'homeassistant'))))"
    )
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "[]"


def test_quarter_iso_is_amsterdam_time():
    assert V.quarter_iso(1793401200) == "2026-10-31T00:00:00+01:00"


# -- ranges that start inside a run (the executor state comes from before the range) -------


def test_an_export_from_any_quarter_of_a_trading_run_passes(tmp_path, monkeypatch):
    _negative_take(monkeypatch)
    site, store, start = _site()
    _run(site, start, 8 * 15, flows=_charging)
    site.settle(_settle_rows(store, start, range(8)), start + 90000)
    for k in range(1, 8):
        d = tmp_path / f"from{k}"
        d.mkdir()
        _write(d, X.export_ledger(store, site.lc, start + 900 * k, start + 7200, True))
        r = V.verify_export(d)
        n = 8 - k
        assert r.ok and r.lines[0] == (
            f"chain intact, {n} quarter{'' if n == 1 else 's'} recomputed, totals match, "
            f"energies match on {n - 1} of {n} quarters"
        ), (k, r.lines)
        assert r.lines[1] == (
            "energies not checked for the first quarter (the range does not start at a restart)"
        )
    # a real energy edit in the second quarter of such a range is still caught
    d = tmp_path / "from1"
    q = start + 2 * 900
    _rewrite(d, lambda rows: _shift_energy_and_money(d, rows, str(q)), rechain=True)
    assert V.verify_export(d).lines[0] == (
        f"energy mismatch at quarter {V.quarter_iso(q)}: shadow import_kwh"
    )


def test_a_trade_across_midnight_passes_on_day_two(tmp_path, monkeypatch):
    site, store, start = _site()
    day2 = start + 86400
    _run(site, start, 1440 - 30)  # day 1 without trading
    _negative_take(monkeypatch)
    for ts in range(day2 - 1800, day2 + 3600, 60):  # trading from 23:30 into day 2
        if (ts - start) % 900 == 60:
            site.setup_quarter(ts)
        site.on_minute(ts, _flows(ts))
    site.settle(_settle_rows(store, start, range(96 + 4), published=day2 + 7200), day2 + 7200)
    _write(tmp_path, X.export_ledger(store, site.lc, day2, day2 + 3600, True))
    rows = _read(tmp_path)
    assert any(r["kind"] == "decision" and r["action"] == "charge" for r in rows)
    r = V.verify_export(tmp_path)
    assert r.ok and r.quarters == 4, r.lines
    assert r.lines[0].endswith("energies match on 3 of 4 quarters"), r.lines


def test_a_crash_before_midnight_with_the_restart_next_day_passes_on_day_one(
    tmp_path, monkeypatch
):
    site, store, start = _site()
    day2 = start + 86400
    _run(site, start, 1440 - 60)  # day 1 to 23:00 without trading
    _negative_take(monkeypatch)
    for ts in range(day2 - 3600, day2 - 3600 + 22 * 60, 60):  # trading, the box dies at 23:22
        if (ts - start) % 900 == 60:
            site.setup_quarter(ts)
        site.on_minute(ts, _flows(ts))
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(day2 + 1800)  # back at 00:30: 23:15 to 00:15 booked missing
    for ts in range(day2 + 1800, day2 + 3600, 60):
        if (ts - day2) % 900 == 60:
            site2.setup_quarter(ts)
        site2.on_minute(ts, _flows(ts))
    site2.settle(_settle_rows(store, start, range(96 + 4), published=day2 + 7000), day2 + 7200)
    _write(tmp_path, X.export_ledger(store, site2.lc, start, day2, True))
    m = _manifest(tmp_path)
    assert [e["ts"] for e in m["events"]][-1] == day2 + 1800  # the restart that booked the gap
    r = V.verify_export(tmp_path)
    assert r.ok and r.quarters == 96 and r.lines[0].endswith("energies match"), r.lines


# -- omissions the chain alone does not show ----------------------------------------------


def _settled(r, q=None, path=None) -> bool:
    return (
        r["kind"] == "booking" and r["price_source"] == "settled"
        and (q is None or r["quarter_ts"] == q) and (path is None or r["path"] == path)
    )


def test_a_deleted_settled_rebooking_is_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    q = _first(_read(d), "booking", price_source="settled")["quarter_ts"]
    _drop(d, lambda r: _settled(r, q))  # the three rows of one quarter
    r = V.verify_export(d)
    iso = V.quarter_iso(int(q))
    assert r.lines[0] == f"settled booking missing at quarter {iso} path shadow"


def test_a_deleted_settled_shadow_rebooking_is_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    q = _first(_read(d), "booking", price_source="settled")["quarter_ts"]
    _drop(d, lambda r: _settled(r, q, "shadow"))
    iso = V.quarter_iso(int(q))
    assert V.verify_export(d).lines[0] == f"settled booking missing at quarter {iso} path shadow"
    m = _manifest(d)
    del m["settles"]  # without the events the quarter's paths still disagree
    _set_manifest(d, m)
    assert V.verify_export(d).lines[0] == f"mixed price sources at quarter {iso}"


def test_a_deleted_quarter_is_caught(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    rows = _read(d)
    q = rows[0]["ts"]
    q = str(int(q) - int(q) % 900 + 3 * 900)  # quarter 3: decisions before and after it stay
    m = _manifest(d)
    m["settles"] = [
        dict(e, quarters=[x for x in e["quarters"] if str(x) != q],
             published_at=[p for x, p in zip(e["quarters"], e["published_at"], strict=True)
                           if str(x) != q])
        for e in m["settles"]
    ]
    m["n_quarters"] = 7
    _set_manifest(d, m)
    _drop(d, lambda r: r["kind"] == "booking" and r["quarter_ts"] == q)
    for flows in (True, False):
        if not flows:
            (d / "flows_minute.csv").unlink()
        r = V.verify_export(d)
        assert r.lines[0] == f"bookings missing at quarter {V.quarter_iso(int(q))}", r.lines


def test_a_deleted_last_quarter_disagrees_with_the_manifest_count(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    rows = _read(d)
    q = max(r["quarter_ts"] for r in rows if r["kind"] == "booking")
    m = _manifest(d)
    m["settles"] = [
        dict(e, quarters=e["quarters"][:-1], published_at=e["published_at"][:-1])
        for e in m["settles"]
    ]
    _set_manifest(d, m)
    _drop(d, lambda r: r["kind"] == "booking" and r["quarter_ts"] == q)
    r = V.verify_export(d)
    assert r.lines[0] == "bookings missing: the manifest names 8 quarters, the ledger holds 7"


# -- the unknown-state executor against a real one with a history before the range ---------

T0 = 1_800_000_000 - 1_800_000_000 % 900


def _wrongly_judged(p: ExecutionProfile, pre: list[float], post: list[float]) -> list[int]:
    """The quarters of ``post`` in which ``_Unsure`` called every setpoint known but one
    differs from a real executor that also ran ``pre`` (the minutes before the range)."""
    truth = Executor(p)
    for i, kw in enumerate(pre):
        truth.request(T0 - 60 * (len(pre) - i), kw)
    ex: object = V._Unsure(p, T0)
    known: dict[int, bool] = {}
    wrong: set[int] = set()
    for i, kw in enumerate(post):
        ts, q = T0 + 60 * i, i // 15
        t = truth.request(ts, kw)
        if isinstance(ex, V._Unsure):
            o, k = ex.request(ts, kw), ex.known
            one = ex.settled(ts + 60)
            if one is not None:
                ex = one
        else:
            o, k = ex.request(ts, kw), True
        known[q] = known.get(q, True) and k
        if k and o != t:
            wrong.add(q)
    return sorted(q for q in wrong if known[q])


def test_a_reversal_blocked_mid_quarter_is_not_judged_known():
    # the real executor charged just before the range, so the dwell blocks the reversals of
    # the first minutes; an unknown target kept by a blocked reversal is no change, so the
    # states do not agree on a setpoint the real executor would not deliver
    post = [
        -4.0, 4.0, 4.0, -4.0, 0.0, -4.0, -4.0, 0.0, 4.0, 4.0, 4.0, 4.0, -4.0, 4.0, -4.0,
        0.0, 4.0, -4.0, 0.0, 0.0, -4.0, 0.0, 4.0, 4.0, 4.0, -4.0, 0.0, -4.0, 0.0, 0.0,
    ]
    assert _wrongly_judged(ExecutionProfile(), [4.0], post) == []


@pytest.mark.parametrize(
    "profile",
    [
        ExecutionProfile(),  # the knob defaults: latency 60 s, dwell 300 s
        replace(DEYE_AMPS, latency_s=120, dwell_s=900),
        replace(DEYE_TOU, latency_s=300, dwell_s=600),
    ],
    ids=["defaults", "amps-120-900", "tou-300-600"],
)
def test_the_unknown_state_executor_never_judges_a_quarter_wrongly(profile):
    rng = random.Random(26)
    choices = (4.0, -4.0, 0.0, 0.5, 3.5, -3.5, 6.0)
    for _ in range(1500):
        style = rng.choice(("reversals", "slow", "mixed"))

        def stream(n: int) -> list[float]:
            out: list[float] = []
            for _ in range(n):
                if style == "reversals":
                    out.append(rng.choice((4.0, -4.0, 0.0, 4.0, -4.0)))
                elif style == "slow" and out and rng.random() < 0.8:
                    out.append(out[-1])
                else:
                    out.append(rng.choice(choices))
            return out

        pre, post = stream(rng.randint(1, 30)), stream(15 * 4)
        assert _wrongly_judged(profile, pre, post) == [], (pre, post)


def _knob_change_export(tmp_path, monkeypatch):
    """Four trading quarters on the shipped knobs, then a restart on a latency of 180 s and a
    dwell of 600 s (what an options change and its reload do) and four more quarters."""
    _negative_take(monkeypatch)
    site, store, start = _site()
    _run(site, start, 4 * 15, flows=_charging)
    knobs = {**C.knobs_dict(site.lc.cfg.knobs), "latency_s": 180, "dwell_s": 600}
    lc2 = replace(site.lc, cfg=C.with_knobs(site.lc.cfg, knobs))
    site2 = LS.LiveSite(store, lc2)
    site2.restore(start + 3600)
    for ts in range(start + 3600, start + 7200, 60):
        if (ts - start) % 900 == 60:
            site2.setup_quarter(ts)
        site2.on_minute(ts, _charging(ts))
    site2.settle(_settle_rows(store, start, range(8)), start + 90000)
    return store, site, site2, start


def test_a_knob_change_mid_range_verifies_clean(tmp_path, monkeypatch):
    store, site, site2, start = _knob_change_export(tmp_path, monkeypatch)
    _write(tmp_path, X.export_ledger(store, site2.lc, start, start + 7200, True))
    m = _manifest(tmp_path)
    assert m["site"]["knobs"]["latency_s"] == 180  # the knobs as now
    assert [e["knobs"]["latency_s"] for e in m["events"]] == [60, 180]
    r = V.verify_export(tmp_path)
    assert r.ok and r.lines[0].endswith("energies match"), r.lines
    # the same export read with one set of knobs (the manifest's) fails on the first run:
    # the per-run knobs carry the check
    for e in m["events"]:
        e["knobs"] = None
    _set_manifest(tmp_path, m)
    assert V.verify_export(tmp_path).lines[0].startswith("energy mismatch")
    # a range that starts inside the second run takes that run's knobs from knobs_at_start
    d = tmp_path / "late"
    d.mkdir()
    _write(d, X.export_ledger(store, site2.lc, start + 4 * 900, start + 7200, True))
    late = _manifest(d)
    assert late["events"] == [] or late["events"][0]["knobs"]["latency_s"] == 180
    assert late["knobs_at_start"]["latency_s"] == 180
    d2 = tmp_path / "early"
    d2.mkdir()
    _write(d2, X.export_ledger(store, site2.lc, start + 900, start + 3600, True))
    early = _manifest(d2)
    assert early["knobs_at_start"]["latency_s"] == 60
    assert V.verify_export(d2).ok, V.verify_export(d2).lines


def test_a_wrong_knob_in_the_manifest_for_a_run_is_an_energy_mismatch(tmp_path, monkeypatch):
    store, site, site2, start = _knob_change_export(tmp_path, monkeypatch)
    _write(tmp_path, X.export_ledger(store, site2.lc, start, start + 7200, True))
    m = _manifest(tmp_path)
    m["events"][1]["knobs"]["latency_s"] = 60  # the second run claims the old latency
    _set_manifest(tmp_path, m)
    r = V.verify_export(tmp_path)
    assert not r.ok and r.lines[0].startswith("energy mismatch at quarter"), r.lines


def test_an_export_without_per_run_knobs_reads_the_manifest_knobs(tmp_path, monkeypatch):
    d = _export(tmp_path, monkeypatch)
    m = _manifest(d)
    m.pop("knobs_at_start")
    for e in m["events"]:
        e.pop("knobs")
    _set_manifest(d, m)
    r = V.verify_export(d)
    assert r.ok and r.lines[0].endswith("energies match"), r.lines
