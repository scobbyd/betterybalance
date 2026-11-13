import datetime as dt
import pathlib

import numpy as np
from betterybalance.core.feeds import base
from betterybalance.core.feeds import energy_charts as ec

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_energy_charts_2026-10-01.json"


def test_fetch_day_ahead_dedupes_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, **k):
        calls.append(params)
        return FIX.read_bytes()

    monkeypatch.setattr(ec, "http_get", fake_get)
    cache = base.DayCache(tmp_path / "da.jsonl")
    t = ec.fetch_day_ahead(dt.date(2026, 10, 1), dt.date(2026, 10, 1), cache)
    assert len(t["ts"]) == 96 and np.all(np.diff(t["ts"]) == 900) and len(calls) == 1
    t2 = ec.fetch_day_ahead(dt.date(2026, 10, 1), dt.date(2026, 10, 1), cache, offline=True)
    assert np.array_equal(t["da_eur_mwh"], t2["da_eur_mwh"])


def test_hourly_points_expand_to_quarters():
    pts = [(0, 10.0), (3600, 20.0), (7200, 30.0), (10800, 40.0)]
    t = ec.to_quarter_table(pts)
    assert len(t["ts"]) == 16 and t["da_eur_mwh"][3] == 10.0 and t["da_eur_mwh"][4] == 20.0


def test_non_json_or_non_object_body_is_a_feed_error(tmp_path, monkeypatch):
    import pytest

    cache = base.DayCache(tmp_path / "da.jsonl")
    for body in (b"<html>maintenance</html>", b"[1, 2]"):
        monkeypatch.setattr(ec, "http_get", lambda url, params=None, body=body, **k: body)
        with pytest.raises(base.FeedError):
            ec.fetch_day_ahead(dt.date(2026, 10, 1), dt.date(2026, 10, 1), cache)
