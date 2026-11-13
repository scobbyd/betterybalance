import datetime as dt

import numpy as np
import pytest
from betterybalance.core.site import ev as EV
from betterybalance.core.site import load as L
from betterybalance.core.site.contract import EvPlan
from betterybalance.core.timegrid import ZONE


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZONE).timestamp())


def _grid(first: str, days: int) -> np.ndarray:
    start = _ts(first + "T00:00:00")
    end = _ts((dt.date.fromisoformat(first) + dt.timedelta(days=days)).isoformat() + "T00:00:00")
    return np.arange(start, end, 900, dtype=np.int64)


def test_wall_slot_handles_the_dst_end_day():
    ts = _grid("2026-10-25", 1)
    ws = L.wall_slot(ts)
    assert len(ts) == 100 and ws.max() == 95 and (ws == 9).sum() == 2 and (ws == 50).sum() == 1


def test_load_median_is_per_wall_slot_and_falls_back_to_the_mean():
    ts = _grid("2026-10-19", 7)
    load = np.where(L.wall_slot(ts) == 72, 2.0, 0.3)  # 18:00 slot is 2 kW every day
    m = L.load_median(load, ts, now=int(ts[-1] + 900))
    assert m.profile_kw[72] == 2.0 and m.profile_kw[10] == 0.3 and m.trusted.all()
    short = L.load_median(load[-96 * 2 :], ts[-96 * 2 :], now=int(ts[-1] + 900))
    assert not short.trusted.any() and short.profile_kw[72] == pytest.approx(short.mean_kw)
    fut = L.forecast_load_kw(m, _grid("2026-10-26", 1))
    assert fut[72] == 2.0 and fut[40] == 0.3


def test_load_median_ignores_nan_and_masked_rows_and_splits_weekdays():
    ts = _grid("2026-10-12", 14)  # Monday start
    ws = L.wall_slot(ts)
    cal_dow = (((ts + 3600) // 86400) + 3) % 7
    load = np.where(cal_dow >= 5, 1.0, 0.5)
    load[(ws == 20)] = np.nan
    mask = np.zeros(len(ts), bool)
    mask[ws == 30] = True
    m = L.load_median(load, ts, now=int(ts[-1] + 900), mask=mask, days=14, weekday_split=True)
    assert m.profile_kw.shape == (2, 96)
    assert not m.trusted[:, 20].any() and not m.trusted[:, 30].any()
    assert m.profile_kw[0, 40] == 0.5 and m.profile_kw[1, 40] == 1.0
    fut = L.forecast_load_kw(m, _grid("2026-10-31", 1))  # a Saturday
    assert fut[40] == 1.0


def test_detector_fires_on_eleven_kw_for_an_hour_and_ignores_ovens_and_short_spikes():
    ts = _grid("2026-10-19", 8)
    base = np.full(len(ts), 0.3)
    model = L.load_median(base, ts, now=int(ts[-1] + 900))
    load = base.copy()
    day7 = slice(96 * 7, 96 * 8)
    load[day7][72:76] += 10.5
    load[day7][30:38] += 3.5
    load[day7][84:86] += 10.5
    s = EV.detect_sessions(load, ts, model, charger_kw=11.0, min_minutes=45)
    assert len(s) == 1 and s[0].start_ts == ts[96 * 7 + 72] and s[0].end_ts == ts[96 * 7 + 76]
    assert s[0].energy_kwh == pytest.approx(10.5) and s[0].mean_kw == pytest.approx(10.5)
    assert EV.session_mask(s, ts).sum() == 4
    assert (
        EV.detect_sessions(load, ts, model, charger_kw=3.7, min_minutes=45)[0].start_ts
        == ts[96 * 7 + 30]
    )


def test_stats_and_expectation_by_weekday():
    ts = _grid("2026-10-05", 14)
    s = [
        EV.Session(int(ts[72]), int(ts[76]), 11.0, 11.0),
        EV.Session(int(ts[96 * 7 + 72]), int(ts[96 * 7 + 76]), 11.0, 11.0),
    ]
    st = EV.ev_stats(s, int(ts[0]), int(ts[-1]))
    assert st.p_session[0] == 1.0 and np.isnan(st.p_session[1])
    exp = EV.expected_ev_kw(st, _grid("2026-10-19", 1), 11.0)
    assert exp[72] == 11.0 and exp[76] == 0.0 and exp[40] == 0.0
    # three sessions on one of the two Mondays: p counts days, the medians count sessions
    three = [
        EV.Session(int(ts[30]), int(ts[34]), 5.0, 5.0),
        EV.Session(int(ts[50]), int(ts[54]), 11.0, 11.0),
        EV.Session(int(ts[72]), int(ts[76]), 20.0, 20.0),
    ]
    st3 = EV.ev_stats(three, int(ts[0]), int(ts[-1]))
    assert st3.p_session[0] == 0.5 and st3.median_energy_kwh[0] == 11.0
    assert st3.median_start_slot[0] == 50
    # a night session from 22:00 for 12 quarters spills into Tuesday slots 0 to 3
    night = EV.EvStats(
        p_session=np.array([1.0] + [np.nan] * 6),
        median_start_slot=np.array([88.0] + [np.nan] * 6),
        median_energy_kwh=np.array([33.0] + [np.nan] * 6),
    )
    e = EV.expected_ev_kw(night, _grid("2026-10-19", 2), 11.0)  # Monday, Tuesday
    assert (e[88:96] == 11.0).all() and (e[96:100] == 11.0).all()
    assert e[100] == 0.0 and e[:88].sum() == 0.0 and e[101:].sum() == 0.0


def test_plan_deadline_and_cheapest_slots():
    plan = EvPlan(energy_kwh=22.0, by_local="07:00", days=(0, 1, 2, 3, 4))
    now = _ts("2026-10-24T18:00:00")
    dl = EV.plan_deadline(plan, now)
    assert dl == _ts("2026-10-26T07:00:00")
    slots = np.arange(now, dl + 8 * 3600, 900, dtype=np.int64)
    cost = np.full(len(slots), 100.0)
    cost[10:20] = 10.0
    m = EV.plan_slots(cost, slots, now, dl, 22.0, 11.0)
    assert m.sum() == 8 and m[10:18].all() and not m[slots >= dl].any()
    assert EV.plan_deadline(EvPlan(1.0, "07:00", ()), now) is None
