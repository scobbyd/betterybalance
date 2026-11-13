# tests/test_rule_window.py
import datetime as dt

import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core.rule import window as W
from betterybalance.core.site.contract import Knobs
from betterybalance.core.timegrid import ZONE


def _ts(s):
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZONE).timestamp())


class Source:
    def __init__(self, tiers=("A", "B", "C", "D"), fail=(), nan=()):
        self.tiers, self.fail, self.calls = tiers, set(fail), []
        self.nan = set(nan)

    def bands(self, tier, h, e, target_ts):
        self.calls.append((tier, h, e))
        if tier not in self.tiers or (tier, h) in self.fail:
            return None
        base = 100.0 + {"A": 0, "B": 1, "C": 2, "D": 3}[tier]
        q = {0.1: base - 20, 0.25: base - 10, 0.5: base, 0.75: base + 10, 0.9: base + 20}
        if (tier, h) in self.nan:
            q[0.9] = float("nan")
        return {"take": q, "feed": {k: v - 5 for k, v in q.items()}, "p_dual": 0.3, "cov": 2.0}


def _tables():
    return {"dual_rate": [[0.5] * 12] * 4, "dual_spread": [10.0, 20.0, 30.0, 40.0]}


def test_window_runs_to_midnight_of_the_day_after_tomorrow_across_dst():
    slots = W.window_slots(_ts("2026-10-24T12:07:00"))
    assert (
        slots[0] == _ts("2026-10-24T12:00:00")
        and slots[-1] == _ts("2026-10-26T23:45:00")
        and len(slots) == 244
    )


def test_tier_choice_per_slot_and_fitted_horizons():
    now = _ts("2026-10-20T12:03:00")
    src = Source()
    w = W.build_window(now, src, ("A", "B", "C", "D"), np.ones(300, bool), _tables(), Knobs())
    assert (
        w.elapsed_min == 1
        and w.tier[0] == "D"
        and w.tier[1] == "D"
        and w.tier[9] == "D"
        and w.tier[30] == "A"
    )
    assert (
        ("D", 8, None) in src.calls
        and ("D", 9, None) not in src.calls
        and ("A", 24, None) in src.calls
    )
    assert W.fitted_horizon(11) == 12 and W.fitted_horizon(23) == 24 and W.fitted_horizon(12) == 12
    # coverage factor 2 on the half-width
    assert w.take[2, 0] == 103.0 and w.take[0, 0] == 103.0 - 20 * 2.0
    # 12:00 is block 2
    assert w.p_dual[0] == 0.3 and w.p_dual[5] == 0.5 and w.dual_spread[0] == 30.0


def test_falls_back_to_lower_tiers_and_to_tier_a_alone():
    now = _ts("2026-10-20T12:06:00")
    w = W.build_window(
        now,
        Source(fail=(("D", 0), ("C", 0))),
        ("A", "B", "C", "D"),
        np.ones(300, bool),
        _tables(),
        Knobs(),
    )
    assert w.elapsed_min == 5 and w.tier[0] == "B" and w.tier[1] == "D"
    nan_d = W.build_window(
        now, Source(nan=(("D", 1),)), ("C", "D"), np.ones(300, bool), _tables(), Knobs()
    )
    assert nan_d.tier[1] == "C" and np.isfinite(nan_d.take).all()
    only_a = W.build_window(
        now, Source(tiers=("A",)), ("A",), np.ones(300, bool), _tables(), Knobs()
    )
    assert set(only_a.tier) == {"A"} and len(only_a.ts) == len(W.window_slots(now))


def test_unknown_day_ahead_uses_same_clock_band_widened():
    now = _ts("2026-10-20T12:01:00")
    n = len(W.window_slots(now))
    known = np.ones(n, bool)
    known[100:] = False
    w = W.build_window(now, Source(), ("A",), known, _tables(), Knobs())
    assert w.predicted[150] and not w.predicted[50]
    assert w.take[2, 150] == w.take[2, 54] and (w.take[4, 150] - w.take[2, 150]) == pytest.approx(
        1.5 * (w.take[4, 54] - w.take[2, 54])
    )


class CutSource(Source):
    def __init__(self, cut_ts, **k):
        super().__init__(**k)
        self.cut_ts = cut_ts

    def bands(self, tier, h, e, target_ts):
        return None if target_ts >= self.cut_ts else super().bands(tier, h, e, target_ts)


def test_slots_without_any_band_take_the_same_clock_fallback():
    now = _ts("2026-10-20T12:01:00")
    slots = W.window_slots(now)
    w = W.build_window(
        now, CutSource(int(slots[150])), ("A",), np.ones(len(slots), bool), _tables(), Knobs()
    )
    # da_known is True, but no tier answers
    assert w.predicted[150] and w.predicted[200] and not w.predicted[149]
    assert w.take[2, 150] == w.take[2, 54] and (w.take[4, 150] - w.take[2, 150]) == pytest.approx(
        1.5 * (w.take[4, 54] - w.take[2, 54])
    )


def test_nowcast_points_and_half_widths():
    w = W.build_window(
        _ts("2026-10-20T12:11:00"),
        Source(),
        ("A", "B", "C", "D"),
        np.ones(300, bool),
        _tables(),
        Knobs(),
    )
    nc = W.nowcast(w, Knobs())
    assert w.elapsed_min == 10 and nc.take_point == w.take[1, 0] and nc.feed_point == w.feed[3, 0]
    assert nc.w_take == pytest.approx((w.take[3, 0] - w.take[1, 0]) / 2) and nc.take_q50 == 103.0


def test_window_refuses_the_first_minute_of_the_quarter():
    with pytest.raises(ValueError, match="60 s"):
        W.build_window(
            _ts("2026-10-20T12:00:30"), Source(), ("A",), np.ones(300, bool), _tables(), Knobs()
        )


def test_tier_a_neighbour_prices_need_the_next_four_quarters_known():
    now = _ts("2026-10-20T12:01:00")
    known = np.ones(len(W.window_slots(now)), bool)
    known[48:] = False
    w = W.build_window(now, Source(), ("A",), known, _tables(), Knobs())
    assert w.predicted[44:48].all() and not w.predicted[43]


class RecordingSource(Source):
    def __init__(self, **k):
        super().__init__(**k)
        self.record = []

    def bands(self, tier, h, e, target_ts):
        self.record.append((tier, h, e, target_ts))
        return super().bands(tier, h, e, target_ts)


def test_every_band_call_is_honest():
    now = _ts("2026-10-20T12:00:00") + 60
    slots = W.window_slots(now)
    known = np.ones(len(slots), bool)
    known[48:] = False
    src = RecordingSource()
    w = W.build_window(now, src, ("A", "B", "C", "D"), known, _tables(), Knobs())
    assert src.record and w.predicted[44:].all()
    for tier, h, e, target in src.record:
        j = (target - int(slots[0])) // 900
        assert not w.predicted[j]
        if (tier, h) == ("A", 24) and j > 24:
            continue
        assert h >= j and F.forecast_time(target, h, e) <= now
