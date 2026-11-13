import datetime as dt
import json
import pathlib

import numpy as np
from betterybalance.core.feeds import base
from betterybalance.core.feeds import sidc_continuous as sc

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_idc_stats_NL_2026-10-28.json"


def test_parse_statistics_quarter_hour_contracts():
    rows = sc.parse_statistics(json.loads(FIX.read_text()))
    qh = [r for r in rows if r["duration_min"] == 15]
    assert len(qh) == 96 and all(r["ts"] % 900 == 0 for r in qh)
    assert all(np.isnan(r["vwap"]) or r["low"] <= r["vwap"] <= r["high"] for r in qh)


def test_fetch_day_uses_cache_and_keeps_quarter_hours(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "http_get", lambda url, params=None, **k: FIX.read_bytes())
    cache = base.DayCache(tmp_path / "idc.jsonl")
    t = sc.fetch_continuous_day(dt.date(2026, 10, 28), cache)
    assert len(t["ts"]) == 96 and "vwap_last1h" in t
    t2 = sc.fetch_continuous_day(dt.date(2026, 10, 28), cache, offline=True)
    assert np.array_equal(t["ts"], t2["ts"])


def test_non_json_or_non_object_body_is_a_feed_error(tmp_path, monkeypatch):
    import pytest

    cache = base.DayCache(tmp_path / "idc.jsonl")
    for body in (b"<html>maintenance</html>", b"[1, 2]"):
        monkeypatch.setattr(sc, "http_get", lambda url, params=None, body=body, **k: body)
        with pytest.raises(base.FeedError, match="maintenance|not an object"):
            sc.fetch_continuous_day(dt.date(2026, 10, 28), cache)
