import datetime as dt

import numpy as np
from betterybalance.core.timegrid import align, isp_grid, isp_start, local_calendar, nl_holidays


def _epoch(s):  # "2026-03-29T00:00+01:00" style ISO string -> epoch s
    return int(dt.datetime.fromisoformat(s).timestamp())


def test_grid_is_900s_and_floors_start():
    g = isp_grid(_epoch("2026-10-01T00:07:00+02:00"), _epoch("2026-10-01T01:00:00+02:00"))
    assert g.dtype == np.int64 and g[0] % 900 == 0 and len(g) == 4 and np.all(np.diff(g) == 900)


def test_dst_days_have_92_and_100_slots():
    short = isp_grid(_epoch("2026-03-29T00:00:00+01:00"), _epoch("2026-03-30T00:00:00+02:00"))
    long_ = isp_grid(_epoch("2026-10-25T00:00:00+02:00"), _epoch("2026-10-26T00:00:00+01:00"))
    assert len(short) == 92 and len(long_) == 100
    cal = local_calendar(long_)
    assert cal["slot_of_day"].max() == 99 and cal["hour"][-1] == 23


def test_holidays_2026():
    h = nl_holidays(2026)
    assert dt.date(2026, 4, 5) in h and dt.date(2026, 4, 6) in h  # Pasen
    assert dt.date(2026, 5, 14) in h and dt.date(2026, 5, 25) in h  # Hemelvaart, 2e Pinksterdag
    assert dt.date(2026, 4, 27) in h and dt.date(2026, 12, 26) in h
    assert dt.date(2025, 4, 26) in nl_holidays(2025)  # Koningsdag on a Saturday 2025


def test_align_fills_nan():
    grid = isp_grid(0, 3600)
    t = {"ts": np.array([0, 1800], dtype=np.int64), "x": np.array([1.0, 2.0])}
    a = align(t, grid, ["x"])
    assert np.isnan(a["x"][1]) and a["x"][2] == 2.0 and isp_start(np.array([1799]))[0] == 900
