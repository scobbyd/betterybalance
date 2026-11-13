import datetime as dt
import json
import pathlib

import numpy as np
import pytest
from betterybalance.core.feeds import base
from betterybalance.core.feeds import openmeteo as OM

FIX = pathlib.Path(__file__).parent / "fixtures"
M15 = FIX / "synthetic_openmeteo_minutely15.json"
HOURLY = FIX / "synthetic_openmeteo_archive_hourly.json"


def test_relabel_shifts_end_labels_to_start_labels():
    doc = json.loads(M15.read_text())
    t = OM.parse_minutely15(doc)
    assert t["ts"][0] == doc["minutely_15"]["time"][0] - 900
    assert np.all(np.diff(t["ts"]) == 900)
    assert t["ghi_w_m2"][40] == doc["minutely_15"]["shortwave_radiation"][40]


def test_forecast_uses_verified_params_and_health(monkeypatch):
    seen = {}

    def fake_get(url, params=None, **k):
        seen.update(params)
        return M15.read_bytes()

    monkeypatch.setattr(OM, "http_get", fake_get)
    src = OM.OpenMeteo()
    t = src.forecast(52.1, 5.2, now=1788307200)
    assert seen["forecast_days"] == 3 and seen["past_days"] == 1
    assert seen["timeformat"] == "unixtime"
    assert "direct_normal_irradiance" in seen["minutely_15"] and len(t["ts"]) == 96 * 4
    assert src.health().ok and src.health().note == ""


def test_archive_falls_back_to_hourly_and_caches(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, params=None, **k):
        calls.append(url)
        if url == OM.HISTORICAL_URL:
            return b'{"error": true, "reason": "no data"}'
        return HOURLY.read_bytes()

    monkeypatch.setattr(OM, "http_get", fake_get)
    cache = base.DayCache(tmp_path / "om.jsonl")
    day = dt.date(2026, 2, 1)
    t = OM.OpenMeteo().archive(52.1, 5.2, day, day, cache)
    assert calls == [OM.HISTORICAL_URL, OM.ARCHIVE_URL] and len(t["ts"]) == 96
    assert t["ghi_w_m2"][32] == t["ghi_w_m2"][33] == t["ghi_w_m2"][34] == t["ghi_w_m2"][35]
    t2 = OM.OpenMeteo().archive(52.1, 5.2, day, day, cache)
    assert len(calls) == 2 and np.array_equal(t["ghi_w_m2"], t2["ghi_w_m2"])


def test_interpolated_response_is_reported_in_health(monkeypatch):
    doc = json.loads(M15.read_text())
    sw = doc["minutely_15"]["shortwave_radiation"]
    doc["minutely_15"]["shortwave_radiation"] = [sw[i - i % 4] for i in range(len(sw))]
    monkeypatch.setattr(OM, "http_get", lambda url, params=None, **k: json.dumps(doc).encode())
    src = OM.OpenMeteo()
    src.forecast(52.1, 5.2, now=1788307200)
    assert src.health().note == "interpolated"


def test_stale_fallback_scales_clear_sky():
    assert OM.stale_fallback(np.array([0.0, 500.0, 800.0]), 0.6).tolist() == [0.0, 300.0, 480.0]


def test_non_json_body_is_feed_error(monkeypatch):
    monkeypatch.setattr(OM, "http_get", lambda url, params=None, **k: b"<html>")
    src = OM.OpenMeteo()
    with pytest.raises(base.FeedError):
        src.forecast(52.1, 5.2, now=1788307200)
    assert not src.health().ok


def test_irradiance_csv_roundtrip(tmp_path):
    t = OM.parse_minutely15(json.loads(M15.read_text()))
    t["temp_c"][3] = np.nan
    OM.write_irradiance_csv(t, tmp_path / "irr.csv")
    back = OM.read_irradiance_csv(tmp_path / "irr.csv")
    assert set(back) == {"ts", *OM.IRRADIANCE_COLS} and back["ts"].dtype == np.int64
    assert np.array_equal(back["ts"], t["ts"]) and np.isnan(back["temp_c"][3])
    assert np.allclose(back["ghi_w_m2"], t["ghi_w_m2"])


def _m15_doc(first, n_days, value=1.0):
    """A minutely_15 response with the real labels: day D runs from 00:00 (the interval that
    ends there) to 23:45, so the last quarter of D arrives with D + 1."""
    start = int(dt.datetime(first.year, first.month, first.day, tzinfo=dt.UTC).timestamp())
    times = [start + 900 * i for i in range(96 * n_days)]
    block = {"time": times}
    for name in ("shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation"):
        block[name] = [value if value is None else float(i % 7 + 1) for i in range(len(times))]
    block["temperature_2m"] = [value] * len(times)
    return {"minutely_15": block}


def _requested(params):
    first = dt.date.fromisoformat(params["start_date"])
    return first, (dt.date.fromisoformat(params["end_date"]) - first).days + 1


def test_archive_primary_path_slices_days_and_marks_src(tmp_path, monkeypatch):
    first = dt.date(2026, 2, 1)
    monkeypatch.setattr(
        OM,
        "http_get",
        lambda url, params=None, **k: json.dumps(_m15_doc(*_requested(params))).encode(),
    )
    cache = base.DayCache(tmp_path / "om.jsonl")
    t = OM.OpenMeteo().archive(52.1, 5.2, first, dt.date(2026, 2, 2), cache)
    midnight = int(dt.datetime(2026, 2, 1, tzinfo=dt.UTC).timestamp())
    assert len(t["ts"]) == 192 and t["ts"][0] == midnight and t["ts"][96] == midnight + 86400
    assert {r["src"] for r in cache.get(first)} == {"m15"}


def test_archive_spans_months_and_reuses_cached_days(tmp_path, monkeypatch):
    jan31, feb1 = dt.date(2026, 1, 31), dt.date(2026, 2, 1)
    seen = []

    def fake_get(url, params=None, **k):
        seen.append((url, params["start_date"], params["end_date"]))
        return json.dumps(_m15_doc(*_requested(params))).encode()

    monkeypatch.setattr(OM, "http_get", fake_get)
    cache = base.DayCache(tmp_path / "om.jsonl")
    prior = OM.parse_minutely15(_m15_doc(jan31, 2))
    OM._cache_day(prior, "m15", jan31, cache)
    t = OM.OpenMeteo().archive(52.1, 5.2, jan31, feb1, cache)
    assert seen == [(OM.HISTORICAL_URL, "2026-02-01", "2026-02-02")]
    assert len(t["ts"]) == 192 and np.all(np.diff(t["ts"]) == 900)


def test_archive_all_null_day_is_not_cached_and_raises(tmp_path, monkeypatch):
    day = dt.date(2026, 2, 1)
    hourly = {
        "hourly": {
            "time": [1769904000 + 3600 * i for i in range(48)],
            **{n: [None] * 48 for n in OM._VARIABLES},
        }
    }

    def fake_get(url, params=None, **k):
        if url == OM.HISTORICAL_URL:
            return json.dumps(_m15_doc(day, 1, value=None)).encode()
        return json.dumps(hourly).encode()

    monkeypatch.setattr(OM, "http_get", fake_get)
    cache = base.DayCache(tmp_path / "om.jsonl")
    with pytest.raises(base.FeedError, match="2026-02-01"):
        OM.OpenMeteo().archive(52.1, 5.2, day, day, cache)
    assert cache.get(day) is None


def _hourly_doc(params):
    """An hourly archive response with the real labels: day D runs from 00:00 to 23:00."""
    first, n_days = _requested(params)
    t0 = int(dt.datetime(first.year, first.month, first.day, tzinfo=dt.UTC).timestamp())
    block = {"time": [t0 + 3600 * i for i in range(24 * n_days)]}
    for name in OM._VARIABLES:
        block[name] = [float(i % 7 + 1) for i in range(24 * n_days)]
    return {"hourly": block}


def test_archive_last_requested_day_is_complete_with_real_end_labels(tmp_path, monkeypatch):
    # open-meteo labels an interval at its end, so the last quarter of a day carries the next
    # day's 00:00 stamp; the request must reach one day past the last day it caches
    seen = []

    def fake_get(url, params=None, **k):
        seen.append((url, params["start_date"], params["end_date"]))
        return json.dumps(_m15_doc(*_requested(params))).encode()

    monkeypatch.setattr(OM, "http_get", fake_get)
    day = dt.date(2026, 2, 28)
    cache = base.DayCache(tmp_path / "om.jsonl")
    t = OM.OpenMeteo().archive(52.1, 5.2, day, day, cache)
    assert seen == [(OM.HISTORICAL_URL, "2026-02-28", "2026-03-01")]
    assert len(t["ts"]) == 96 and {r["src"] for r in cache.get(day)} == {"m15"}


def test_archive_hourly_fallback_last_day_is_complete_with_real_end_labels(tmp_path, monkeypatch):
    def fake_get(url, params=None, **k):
        if url == OM.HISTORICAL_URL:
            return json.dumps({"error": True, "reason": "no 15 minute data"}).encode()
        return json.dumps(_hourly_doc(params)).encode()

    monkeypatch.setattr(OM, "http_get", fake_get)
    day = dt.date(2026, 2, 28)
    cache = base.DayCache(tmp_path / "om.jsonl")
    t = OM.OpenMeteo().archive(52.1, 5.2, day, day, cache)
    assert len(t["ts"]) == 96 and {r["src"] for r in cache.get(day)} == {"hourly"}
