import csv
import hashlib
import io
import math

import numpy as np
import pytest
from betterybalance.core import ledger as LG
from betterybalance.core.rule.trader import Decision
from betterybalance.core.rule.window import Nowcast
from betterybalance.core.site.contract import Terms

T = Terms()
P = {"buy": 100.0, "sell": 50.0, "state": 1}
Q = 1788300000
NAN = float("nan")


def _book(path, net, batt, soc, prices=P, missing=False):
    return LG.booking_row(Q, path, net, batt, soc, prices, T, missing)


def test_booking_and_result_for_one_quarter():
    house = _book("house", np.full(15, 1.0), np.zeros(15), NAN)
    shadow = _book("shadow", np.zeros(15), np.full(15, -1.0), 49.5)
    assert house["import_kwh"] == pytest.approx(0.25) and house["price_source"] == "provisional"
    assert house["money_eur"] == pytest.approx(0.03524125) and shadow["money_eur"] == 0.0
    assert shadow["batt_discharge_kwh"] == pytest.approx(0.25) and shadow["batt_charge_kwh"] == 0.0
    assert LG.quarter_result(house, shadow) == pytest.approx(0.03524125)
    no_price = {"buy": NAN, "sell": NAN, "state": 0}
    none = _book("house", np.full(15, 1.0), np.zeros(15), NAN, no_price)
    assert none["price_source"] == "none" and math.isnan(none["money_eur"])


def test_resettle_keeps_energies():
    b = _book("house", np.full(15, 1.0), np.zeros(15), NAN)
    s = LG.resettle(b, 200.0, 50.0, 1, T)
    assert s["price_source"] == "settled" and s["import_kwh"] == b["import_kwh"]
    assert s["money_eur"] == pytest.approx(0.25 * (0.2 + 0.0165) * 1.21)


def test_canonical_json_is_sorted_and_nan_free():
    row = {f: 0 for f in LG.DECISION_FIELDS} | {
        "margin_w": NAN, "action": "hold", "reason": "wait", "committed": False,
        "learning": True, "model_version": "m",
    }
    c = LG.canonical_json("decision", 1, {**row, "prev_hash": "a"})
    assert c.startswith(b'{"action":"hold"') and b'"margin_w":null' in c
    assert b'"kind":"decision"' in c and b'"seq":1' in c


def _three():
    g = LG.genesis_hash("install-1")
    rows = []
    prev = g
    for i, kw in enumerate((0.0, 3.0, -3.0), start=1):
        row = {f: 0 for f in LG.DECISION_FIELDS} | {
            "ts": 60 * i, "kw": kw, "action": "hold", "reason": "default", "model_version": "m",
        }
        r = LG.chain_row("decision", i, row, prev)
        rows.append(("decision", r))
        prev = r["hash"]
    return g, rows


def test_chain_verifies_and_names_the_first_bad_row():
    g, rows = _three()
    assert g == hashlib.sha256(b"install-1").hexdigest()
    assert LG.verify_chain(rows, g) == LG.ChainCheck(True, 3, None, "")
    tampered = [rows[0], ("decision", dict(rows[1][1], kw=4.0)), rows[2]]
    assert LG.verify_chain(tampered, g) == LG.ChainCheck(False, 3, 2, "hash mismatch")
    swapped = [rows[0], rows[2], rows[1]]
    check = LG.verify_chain(swapped, g)
    assert not check.ok and check.first_bad_seq == 3 and check.reason == "seq gap"


def test_repeat_hold_rule():
    h = Decision(0, "hold", 0.0, False, "wait", 0.0, 0.0, 0.0, 0.0, False)
    assert LG.is_repeat_hold(h, Decision(60, "hold", 0.0, False, "wait", 0.1, 0.0, 0.0, 0.0, False))
    late = Decision(60, "hold", 0.0, False, "late", 0.0, 0.0, 0.0, 0.0, False)
    assert not LG.is_repeat_hold(h, late)
    assert not LG.is_repeat_hold(None, h)
    charge = Decision(60, "charge", 3.0, True, "wait", 0.0, 0.0, 0.0, 0.0, False)
    assert not LG.is_repeat_hold(h, charge) and not LG.is_repeat_hold(charge, h)


def test_chain_anchor_and_prev_hash_mismatch():
    g, rows = _three()
    assert LG.verify_chain(rows, LG.genesis_hash("install-2")) == LG.ChainCheck(
        False, 3, 1, "prev_hash mismatch"
    )
    relinked = [rows[0], ("decision", dict(rows[1][1], prev_hash="0" * 64)), rows[2]]
    assert LG.verify_chain(relinked, g) == LG.ChainCheck(False, 3, 2, "prev_hash mismatch")
    assert LG.verify_chain([], g) == LG.ChainCheck(True, 0, None, "")


def test_hash_survives_a_store_round_trip_and_mixed_kinds():
    g = LG.genesis_hash("install-1")
    d = Decision(1788300065, "charge", 3.0, True, "margin", 120.0, 95.0, 80.0, 140.0, True)
    band = {
        "take": {0.1: 50.0, 0.25: 70.0, 0.5: 90.0, 0.75: 110.0, 0.9: 130.0},
        "feed": {0.1: 40.0, 0.25: 60.0, 0.5: 80.0, 0.75: 100.0, 0.9: 120.0},
    }
    dr = LG.decision_row(d, band, None, 49.5, "m1")
    assert tuple(dr) == LG.DECISION_FIELDS and dr["quarter_ts"] == Q
    assert dr["band_take_q10"] == 50.0 and dr["band_feed_q90"] == 120.0
    assert math.isnan(dr["nowcast_take"]) and math.isnan(dr["nowcast_feed"])
    nc = Nowcast(101.0, 91.0, 1.0, 1.0, 0.1, 100.0, 90.0)
    dn = LG.decision_row(d, band, nc, 49.5, "m1")
    assert dn["nowcast_take"] == 101.0 and dn["nowcast_feed"] == 91.0
    b = _book("shadow", np.zeros(15), np.full(15, 2.0), 50.0)
    assert tuple(b) == LG.BOOKING_FIELDS
    r1 = LG.chain_row("decision", 1, dr, g)
    r2 = LG.chain_row("booking", 2, b, r1["hash"])
    # SQLite gives back 0/1 for bools, floats for REAL, numpy scalars from arrays.
    back1 = dict(r1, committed=1, learning=1, ts=np.int64(r1["ts"]), kw=np.float64(3))
    back2 = dict(r2, missing=0, state=np.int64(1), import_kwh=0)
    assert LG.verify_chain([("decision", back1), ("booking", back2)], g).ok
    assert LG.FIELD_TYPES["seq"] is int and LG.FIELD_TYPES["hash"] is str
    assert set(LG.DECISION_FIELDS) | set(LG.BOOKING_FIELDS) <= set(LG.FIELD_TYPES)


def test_missing_booking_keeps_energies_and_has_no_result():
    m = _book("house", np.full(15, 1.0), np.full(15, 0.5), NAN, missing=True)
    assert m["import_kwh"] == pytest.approx(0.25) and m["batt_charge_kwh"] == pytest.approx(0.125)
    assert m["price_source"] == "provisional" and math.isnan(m["money_eur"])
    assert math.isnan(m["tax_eur"])
    s = LG.resettle(m, 200.0, 50.0, 1, T)
    assert s["price_source"] == "settled" and math.isnan(s["money_eur"]) and s["buy"] == 200.0
    shadow = _book("shadow", np.zeros(15), np.zeros(15), 50.0)
    assert math.isnan(LG.quarter_result(m, shadow))
    with pytest.raises(ValueError):
        _book("virtual", np.zeros(15), np.zeros(15), 50.0)


def _decision(**kw):
    row = {f: 0 for f in LG.DECISION_FIELDS} | {
        "ts": 60, "kw": 0.0, "action": "hold", "reason": "default", "model_version": "m",
    }
    return row | kw


def test_negative_zero_hashes_as_zero():
    g = LG.genesis_hash("install-1")
    neg = LG.chain_row("decision", 1, _decision(kw=-0.0), g)
    pos = LG.chain_row("decision", 1, _decision(kw=0.0), g)
    assert neg["hash"] == pos["hash"]
    assert b'"kw":0.0' in LG.canonical_json("decision", 1, {**_decision(kw=-0.0), "prev_hash": g})
    # SQLite gives -0.0 back as 0.0: the chain still verifies.
    assert LG.verify_chain([("decision", dict(neg, kw=0.0))], g).ok


def test_csv_round_trip_with_string_values_verifies():
    g = LG.genesis_hash("install-1")
    d = Decision(1788300065, "charge", 3.0, True, "margin", 120.5, -0.0, 80.25, 140.0, False)
    band = {
        "take": {0.1: 50.0, 0.25: 70.0, 0.5: 90.0, 0.75: 110.0, 0.9: 130.0},
        "feed": {0.1: 40.0, 0.25: 60.0, 0.5: 80.0, 0.75: 100.0, 0.9: 120.0},
    }
    r1 = LG.chain_row("decision", 1, LG.decision_row(d, band, None, 49.5, "m1"), g)
    b = _book("house", np.full(15, 0.7), np.full(15, -0.3), 48.25, missing=True)
    r2 = LG.chain_row("booking", 2, b, r1["hash"])
    cols = ("seq", "kind") + LG.DECISION_FIELDS
    cols += tuple(f for f in LG.BOOKING_FIELDS if f not in cols) + ("prev_hash", "hash")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for kind, r in (("decision", r1), ("booking", r2)):
        w.writerow({c: str(r[c]) if c in r else "" for c in cols} | {"kind": kind})
    back = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert back[0]["committed"] == "True" and back[0]["nowcast_take"] == "nan"
    assert back[1]["missing"] == "True" and back[1]["money_eur"] == "nan"
    items = [(r["kind"], r) for r in back]
    assert LG.verify_chain(items, g) == LG.ChainCheck(True, 2, None, "")
    # "1.0" is an integral int and "1"/"0" are bools; the hash does not change.
    alt = dict(back[1], state="1.0", missing="1", seq="2.0")
    assert LG.verify_chain([items[0], ("booking", alt)], g).ok


def test_missing_field_is_a_hash_mismatch():
    g, rows = _three()
    gone = dict(rows[1][1])
    del gone["reason"]
    check = LG.verify_chain([rows[0], ("decision", gone), rows[2]], g)
    assert check == LG.ChainCheck(False, 3, 2, "hash mismatch")
    nohash = dict(rows[2][1])
    del nohash["hash"]
    assert LG.verify_chain([rows[0], rows[1], ("decision", nohash)], g) == LG.ChainCheck(
        False, 3, 3, "hash mismatch"
    )


def test_unparseable_and_non_integral_values_are_rejected():
    g, rows = _three()
    for bad in (1.9, "1.5", "x"):
        with pytest.raises(ValueError):
            LG.canonical_json("decision", 1, {**_decision(ts=bad), "prev_hash": g})
    for bad in ("yes", 2, ""):
        with pytest.raises(ValueError):
            LG.canonical_json("decision", 1, {**_decision(committed=bad), "prev_hash": g})
    with pytest.raises(ValueError):
        LG.canonical_json("decision", 1, {**_decision(kw="fast"), "prev_hash": g})
    for field, bad in (("ts", 120.5), ("learning", "maybe"), ("kw", "fast")):
        broken = [rows[0], ("decision", dict(rows[1][1], **{field: bad})), rows[2]]
        assert LG.verify_chain(broken, g) == LG.ChainCheck(False, 3, 2, "hash mismatch")
    assert LG.verify_chain([("decision", dict(rows[0][1], seq="one"))], g) == LG.ChainCheck(
        False, 1, None, "seq gap"
    )
