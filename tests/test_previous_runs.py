# tests/test_previous_runs.py
import dataclasses
import datetime as dt
import json
import pathlib

import numpy as np
import pytest
from betterybalance.core.feeds import openmeteo as OM
from betterybalance.core.feeds.base import DayCache, FeedError
from betterybalance.core.timegrid import ZONE
from betterybalance.replay import irradiance as IR
from betterybalance.replay import steps as ST
from betterybalance.replay.bands import CachedBandSource
from betterybalance.replay.irradiance import IrrByLead
from betterybalance.replay.phases import SiteModels

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_previous_runs.json"
# 3 days hourly in the real response shape, lead d carries ghi = 100 * (d + 1); day 0 under
# the plain variable name, as the API answers it


def _t(s):
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZONE).timestamp())


def _by():
    return OM.parse_previous_runs(json.loads(FIX.read_text()), (0, 1, 2, 3))


def _archive(ts, ghi=5.0):
    n = len(ts)
    return {
        "ts": np.asarray(ts, dtype=np.int64),
        **{c: np.full(n, ghi) for c in OM.IRRADIANCE_COLS[:3]},
        "temp_c": np.full(n, 10.0),
    }


def test_parse_previous_runs_splits_the_leads():
    by = _by()
    assert set(by) == {0, 1, 2, 3} and np.nanmax(by[2]["ghi_w_m2"]) == 300.0
    assert len(by[0]["ts"]) == 3 * 96
    # the hourly start relabel: label 00:00 UTC holds [23:00, 00:00), four quarters
    assert int(by[1]["ts"][0]) == 1792454400 - 3600 and np.all(np.diff(by[1]["ts"]) == 900)
    assert by[3]["temp_c"][0] == 8.0 and by[1]["dni_w_m2"][0] == 20.0


def test_slots_take_the_lead_of_their_day():
    by = _by()
    src = IrrByLead(by, _archive(by[0]["ts"]), day0_lead=1)
    now = _t("2026-10-20T10:01:00")
    slots = np.array([_t("2026-10-20T12:00:00"), _t("2026-10-21T12:00:00")])
    out = src.at(now, slots)
    # today's 12:00 from lead 1; tomorrow's 12:00 (hour end 13:00, minus 1 day plus 6 h lies
    # after 10:01) from lead 2
    assert out["ghi_w_m2"].tolist() == [200.0, 300.0]
    assert out["ts"].tolist() == slots.tolist() and src.fallback_rows == 0
    assert "day 0 from lead 1" in src.note()


def test_lead_boundary_is_inclusive():
    by = _by()
    src = IrrByLead(by, _archive(by[0]["ts"]), day0_lead=1)
    slot1 = _t("2026-10-21T12:00:00")  # hour end 2026-10-21 13:00 local
    slot2 = _t("2026-10-22T12:00:00")  # hour end 2026-10-22 13:00 local
    now = slot1 + 3600 - 86400 + IR.LAG_S  # 2026-10-20 19:00 local
    assert now == slot2 + 3600 - 2 * 86400 + IR.LAG_S
    assert src.at(now, np.array([slot1, slot2]))["ghi_w_m2"].tolist() == [200.0, 300.0]
    assert src.at(now - 1, np.array([slot1, slot2]))["ghi_w_m2"].tolist() == [300.0, 400.0]


def test_slots_beyond_lead_3_take_the_archive():
    by = _by()
    src = IrrByLead(by, _archive(by[0]["ts"]), day0_lead=1)
    now = _t("2026-10-20T00:01:00")
    slots = np.array([_t("2026-10-20T12:00:00"), _t("2026-10-22T22:00:00")])
    out = src.at(now, slots)
    assert out["ghi_w_m2"].tolist() == [200.0, 5.0] and out["temp_c"].tolist() == [6.0, 10.0]
    assert src.fallback_rows == 1 and src.fallbacks["beyond_lead_3"] == 1 and src.reads == 2


def test_fallback_counts_split_by_reason():
    by = _by()
    hole = by[1]["ts"] == _t("2026-10-20T13:00:00")
    by[1]["temp_c"][hole] = np.nan
    src = IrrByLead(by, _archive(by[0]["ts"]))
    assert src.day0_lead == 1 and src.note().endswith(
        "archive for 0,0 % of reads: 0,0 % before 2024, 0,0 % beyond lead 3, 0,0 % missing"
    )
    slots = np.array(
        [_t(f"2026-10-{s}:00") for s in ("20T12:00", "20T13:00", "22T22:00", "22T23:00")]
    )
    out = src.at(_t("2026-10-20T00:01:00"), slots)
    assert out["ghi_w_m2"].tolist() == [200.0, 5.0, 5.0, 5.0]
    assert src.reads == 4 and src.fallback_rows == 3
    assert src.fallbacks == {"pre_2024": 0, "beyond_lead_3": 2, "missing_value": 1}
    src.at(_t("2026-10-20T10:01:00"), slots[:1])
    assert src.reads == 5 and src.fallback_rows == 3
    assert src.note() == (
        "previous runs (day 0 from lead 1); archive for 60,0 % of reads: 0,0 % before 2024, "
        "40,0 % beyond lead 3, 20,0 % missing"
    )


def test_rows_without_a_value_take_the_archive():
    by = _by()
    by[1]["ghi_w_m2"][:] = np.nan
    src = IrrByLead(by, _archive(by[0]["ts"]), day0_lead=1)
    out = src.at(_t("2026-10-20T10:01:00"), np.array([_t("2026-10-20T12:00:00")]))
    assert out["ghi_w_m2"].tolist() == [5.0] and out["temp_c"].tolist() == [10.0]
    assert src.fallback_rows == 1 and src.fallbacks["missing_value"] == 1


def test_days_before_2024_take_the_archive():
    # by-lead tables empty for 2023-12-31: at() returns the archive values and fallback_rows
    # counts every slot
    empty = {"ts": np.zeros(0, dtype=np.int64), **{c: np.zeros(0) for c in OM.IRRADIANCE_COLS}}
    slots = np.arange(_t("2023-12-31T08:00:00"), _t("2024-01-01T00:00:00"), 900, dtype=np.int64)
    archive = _archive(slots, ghi=42.0)
    src = IrrByLead({d: empty for d in (1, 2, 3)}, archive, day0_lead=1)
    out = src.at(_t("2023-12-31T07:01:00"), slots)
    assert np.all(out["ghi_w_m2"] == 42.0) and np.all(out["temp_c"] == 10.0)
    assert src.fallback_rows == len(slots) and src.fallbacks["pre_2024"] == len(slots)
    assert "archive for 100,0 % of reads: 100,0 % before 2024" in src.note()
    # values a lead table holds before PREVIOUS_RUNS_START are not read either
    full = {"ts": slots, **{c: np.full(len(slots), 1.0) for c in OM.IRRADIANCE_COLS}}
    src = IrrByLead({d: full for d in (1, 2, 3)}, archive, day0_lead=1)
    assert np.all(src.at(_t("2023-12-31T07:01:00"), slots)["ghi_w_m2"] == 42.0)
    assert src.fallback_rows == len(slots) and IR.PREVIOUS_RUNS_START == dt.date(2024, 1, 1)


def test_previous_runs_requests_months_caches_days_and_never_day0(tmp_path, monkeypatch):
    calls = []

    def fake(url, params):
        calls.append((url, params))
        return json.loads(FIX.read_text())

    monkeypatch.setattr(OM, "_request", fake)
    cache = DayCache(tmp_path / "pr.jsonl")
    by = OM.previous_runs(52.1, 5.2, dt.date(2026, 10, 20), dt.date(2026, 10, 21), cache=cache)
    assert len(calls) == 1 and calls[0][0] == OM.PREVIOUS_RUNS_URL
    p = calls[0][1]
    assert p["start_date"] == "2026-10-20" and p["end_date"] == "2026-10-22"
    assert "previous_day0" not in p["hourly"] and "shortwave_radiation_previous_day3" in p["hourly"]
    assert set(by) == {1, 2, 3} and len(by[1]["ts"]) == 2 * 96
    assert int(by[1]["ts"][0]) == 1792454400 and np.all(by[2]["ghi_w_m2"] == 300.0)
    again = OM.previous_runs(52.1, 5.2, dt.date(2026, 10, 20), dt.date(2026, 10, 21), cache=cache)
    assert len(calls) == 1 and np.array_equal(again[3]["ghi_w_m2"], by[3]["ghi_w_m2"])


def test_previous_runs_refuses_lead_0_and_no_leads():
    for leads in ((0, 1, 2, 3), ()):
        with pytest.raises(ValueError):
            OM.previous_runs(52.1, 5.2, dt.date(2026, 10, 20), dt.date(2026, 10, 21), leads=leads)


def test_previous_runs_and_archive_rows_never_mix_in_one_cache(tmp_path, monkeypatch):
    cache = DayCache(tmp_path / "shared.jsonl")
    day = dt.date(2026, 10, 20)
    archive_rows = [
        {"ts": 1792454400 + 900 * i, "ghi": 1.0, "dni": 1.0, "dhi": 1.0, "temp": 1.0, "src": "m15"}
        for i in range(96)
    ]
    cache.put(day, archive_rows)
    cache.put(dt.date(2026, 10, 21), [dict(r, ts=r["ts"] + 86400) for r in archive_rows])
    calls = []

    def fake(url, params):
        calls.append(url)
        return json.loads(FIX.read_text())

    monkeypatch.setattr(OM, "_request", fake)
    by = OM.previous_runs(52.1, 5.2, day, dt.date(2026, 10, 21), cache=cache)
    assert calls == [OM.PREVIOUS_RUNS_URL] and np.all(by[1]["ghi_w_m2"] == 200.0)
    assert all(r["src"] == OM.PREVIOUS_RUNS_SRC for r in cache.get(day))

    # the archive does not read the Previous Runs line: it fetches its own day again
    def fake_month(lat, lon, days, c):
        calls.append("archive " + ",".join(d.isoformat() for d in days))
        for d in days:
            c.put(d, archive_rows)

    monkeypatch.setattr(OM, "_fill_month", fake_month)
    table = OM.OpenMeteo().archive(52.1, 5.2, day, day, cache)
    assert calls[-1] == "archive 2026-10-20" and np.all(table["ghi_w_m2"] == 1.0)


def test_previous_runs_refuses_an_incomplete_day(monkeypatch):
    monkeypatch.setattr(OM, "_request", lambda url, params: json.loads(FIX.read_text()))
    with pytest.raises(FeedError):
        OM.previous_runs(52.1, 5.2, dt.date(2026, 10, 20), dt.date(2026, 10, 22))


def test_quarter_inputs_reads_the_pv_regressors_from_irr_forecast(monkeypatch):
    from tests.test_replay_engine import _inputs

    inp = _inputs()
    site = SiteModels(inp.flows_quarter, inp.irr, inp.start, inp.end, inp.cfg)
    real = site.at(inp.start)
    view = dataclasses.replace(real, state=dataclasses.replace(real.state, champion=object()))
    seen = []

    def fake_pv(champion, irr, lat, lon):
        seen.append(irr)
        return np.zeros(len(irr["ts"]))

    monkeypatch.setattr(ST, "predict_pv_kw", fake_pv)
    by = {d: _archive(inp.irr["ts"], ghi=100.0 * (d + 1)) for d in (1, 2, 3)}
    fc = IrrByLead(by, inp.irr, day0_lead=1)
    now = inp.start + 60
    source = CachedBandSource(inp.cache)
    qi = ST.quarter_inputs(now, source, inp.cfg, view, inp.irr, irr_forecast=fc)
    assert qi is not None and seen[-1]["ghi_w_m2"][0] == 200.0
    assert seen[-1]["ts"].tolist() == qi.window.ts.tolist()
    ST.quarter_inputs(now, source, inp.cfg, view, inp.irr)
    assert np.all(seen[-1]["ghi_w_m2"] == 0.0)  # without irr_forecast: the archive
