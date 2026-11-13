import datetime as dt
import pathlib
import zoneinfo

import numpy as np
from betterybalance.core.feeds import ida
from betterybalance.core.feeds.base import DayCache

AMS = zoneinfo.ZoneInfo("Europe/Amsterdam")


def _e(y, m, d, h, mi=0):
    return int(dt.datetime(y, m, d, h, mi, tzinfo=AMS).timestamp())


def test_clearing_times():
    d = dt.date(2026, 10, 15)
    assert ida.clearing_time("IDA1", d) == _e(2026, 10, 14, 15)
    assert ida.clearing_time("IDA2", d) == _e(2026, 10, 14, 22)
    assert ida.clearing_time("IDA3", d) == _e(2026, 10, 15, 10)


def test_ida3_covers_afternoon_only_and_availability_respects_clearing():
    morning, afternoon = _e(2026, 10, 15, 9), _e(2026, 10, 15, 14)
    assert not ida.covers("IDA3", morning) and ida.covers("IDA3", afternoon)
    # 15 min after clearing, delay 1200 s
    assert not ida.available_at("IDA3", afternoon, _e(2026, 10, 15, 10, 15))
    assert ida.available_at("IDA3", afternoon, _e(2026, 10, 15, 10, 20))
    assert ida.available_at("IDA1", morning, _e(2026, 10, 14, 15, 20))
    assert not ida.available_at("IDA1", morning, _e(2026, 10, 14, 15, 19))


def test_parse_results_fixture():
    # one synthetic IDA1 day (invented prices, same schema as the portal document)
    fix = pathlib.Path(__file__).parent / "fixtures" / "synthetic_ida_IDA1_2026-10-15.json"
    rows = ida.parse_results(fix.read_bytes(), "IDA1", dt.date(2026, 10, 15))
    assert len(rows) == 96 and all(r["ts"] % 900 == 0 for r in rows)
    assert rows[0]["updated_at"] < rows[0]["ts"]  # published before the first slot of D


def test_parse_missing_document_is_empty():
    payload = b'{"status":404,"title":"Missing"}'
    assert ida.parse_results(payload, "IDA1", dt.date(2024, 6, 14)) == []


def test_availability_follows_dst_local_clearing_time():
    # Delivery day 2026-10-26 follows the clock change of 2026-10-25 (CEST to CET).
    day = dt.date(2026, 10, 26)
    assert ida.clearing_time("IDA1", day) == _e(2026, 10, 25, 15)
    assert ida.clearing_time("IDA2", day) == _e(2026, 10, 25, 22)
    # 15:00 and 22:00 on 25 Oct (CET after the change) lie 7 h apart.
    assert ida.clearing_time("IDA2", day) - ida.clearing_time("IDA1", day) == 7 * 3600
    # Delivery day after the spring change: D-1 is 29 Mar 2026 (23-hour day).
    spring = dt.date(2026, 3, 30)
    assert ida.clearing_time("IDA2", spring) == _e(2026, 3, 29, 22)
    target = _e(2026, 3, 30, 8)
    assert not ida.available_at("IDA2", target, _e(2026, 3, 29, 22, 19))
    assert ida.available_at("IDA2", target, _e(2026, 3, 29, 22, 20))


def test_fetch_ida_aligns_caches_and_skips_history(tmp_path, monkeypatch):
    fix = pathlib.Path(__file__).parent / "fixtures" / "synthetic_ida_IDA1_2026-10-15.json"
    calls = []

    def fake_get(url, params=None, **kwargs):
        calls.append(params)
        if params["date"] == "2026-10-15":
            return fix.read_bytes()
        return b'{"status":404,"title":"Missing"}'

    monkeypatch.setattr(ida, "http_get", fake_get)
    monkeypatch.setattr(ida, "_pace", lambda: None)
    cache = DayCache(tmp_path / "ida_IDA1.jsonl")
    table = ida.fetch_ida("IDA1", dt.date(2026, 10, 14), dt.date(2026, 10, 15), cache)
    assert len(table["ts"]) == 192
    assert np.isnan(table["ida_eur_mwh"][:96]).all()
    assert not np.isnan(table["ida_eur_mwh"][96:]).any()
    assert calls[0]["market"] == "SIDC_IntradayAuction1" and calls[0]["deliveryArea"] == "NL"
    n = len(calls)
    ida.fetch_ida("IDA1", dt.date(2026, 10, 15), dt.date(2026, 10, 15), cache, offline=True)
    assert len(calls) == n
    ida.fetch_ida("IDA1", dt.date(2024, 6, 14), dt.date(2024, 6, 14), cache)
    assert len(calls) == n  # before HISTORY_START: not requested


def test_parse_results_empty_body_is_missing_document():
    assert ida.parse_results(b"", "IDA1", dt.date(2026, 10, 31)) == []


def test_available_mask_matches_available_at_across_dst():
    # quarters over both 2026 DST changes, forecast times from hours before to after clearing
    rng = np.random.default_rng(7)
    spans = ((_e(2026, 3, 27, 0), _e(2026, 3, 31, 0)), (_e(2026, 10, 23, 0), _e(2026, 10, 28, 0)))
    for lo, hi in spans:
        ts = np.arange(lo, hi, 900, dtype=np.int64)
        tf = ts - rng.integers(-3600, 40 * 3600, len(ts))
        for auction in ida.AUCTIONS:
            want = [ida.available_at(auction, int(t), int(f)) for t, f in zip(ts, tf, strict=True)]
            assert ida.available_mask(auction, ts, tf).tolist() == want
    # the publication instant itself and one second before it
    ts = np.arange(_e(2026, 10, 25, 0), _e(2026, 10, 26, 0), 900, dtype=np.int64)
    for auction in ida.AUCTIONS:
        gate = ida.clearing_time(auction, dt.date(2026, 10, 25)) + 1200
        for tf in (np.full(len(ts), gate), np.full(len(ts), gate - 1)):
            want = [ida.available_at(auction, int(t), int(f)) for t, f in zip(ts, tf, strict=True)]
            assert ida.available_mask(auction, ts, tf).tolist() == want
    assert ida.available_mask("IDA1", np.zeros(0, dtype=np.int64), np.zeros(0)).shape == (0,)
