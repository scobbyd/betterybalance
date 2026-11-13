# tests/test_rule_dp.py
import time

import numpy as np
import pytest
from betterybalance.core import battery as B
from betterybalance.core.rule import thresholds as T
from betterybalance.core.site.contract import Knobs, Pack, Terms

from tests.test_rule_thresholds import NOW, _path, _window

FEES = Terms(fee_import_eur_kwh=0.0, fee_export_eur_kwh=0.0)
LOSSLESS = B.flat_profile(1.0)


def test_one_slot_window_values_a_kwh_at_the_terminal_value():
    pack = Pack(10, 4, 4, 0.0, 1.0, eta_rt=1.0)
    w = _window([100], [100])
    for q in (0, 1, 3, 4):  # a flat three-point distribution: every scenario is 100
        w.take[q, 0] = 100.0
        w.feed[q, 0] = 100.0
    knobs = Knobs(impl="dp", wear_eur_cycle=0.0, hurdle_ct_kwh=0.0)
    th = T.dp(w, _path(1, 5.0), pack, FEES, knobs, need=0.0, now=NOW, loss=LOSSLESS)
    # terminal value = the cheapest buy cost in the window = 0,10 EUR/kWh,
    # so one more kWh is worth 0,10
    assert (
        th.value_stored[0] == pytest.approx(0.10)
        and th.charge[0] == pytest.approx(100.0)
        and th.discharge[0] == pytest.approx(100.0)
    )
    assert th.impl == "dp"


def test_empty_pack_values_energy_more_than_a_full_pack_and_a_spike_raises_the_value():
    # only slots 0 and 1 can refill before the spike at slots 2 to 5,
    # so a stored kWh is worth the spike price
    take = [50] * 2 + [400] * 4 + [50] * 26
    w = _window(take, [t - 5 for t in take])
    knobs = Knobs(impl="dp", wear_eur_cycle=0.0, hurdle_ct_kwh=0.0)
    pack = Pack(48, 12, 12, 0.1, 1.0, eta_rt=1.0)
    empty = T.dp(w, _path(32, 6.0), pack, FEES, knobs, need=30.0, now=NOW, loss=LOSSLESS)
    full = T.dp(w, _path(32, 46.0), pack, FEES, knobs, need=0.0, now=NOW, loss=LOSSLESS)
    assert empty.value_stored[0] > full.value_stored[0]
    assert empty.value_stored[0] > 0.10
    assert empty.sale_slots[2:6].all() or empty.charge_slots[:2].any()


def test_dp_runtime_within_budget():
    rng = np.random.default_rng(0)
    take = rng.uniform(20, 200, 288)
    w = _window(take, take - 5)
    pack = Pack(48, 12, 12, 0.1, 1.0)
    t0 = time.perf_counter()
    T.dp(
        w,
        _path(288, 20.0, net_kw=0.5),
        pack,
        Terms(),
        Knobs(impl="dp"),
        need=10.0,
        now=NOW,
        loss=B.flat_profile(0.88),
    )
    assert time.perf_counter() - t0 < 0.2


def test_thresholds_dispatch():
    w = _window([100] * 4, [95] * 4)
    pack = Pack(48, 12, 12, 0.1, 1.0, eta_rt=1.0)
    dp_th = T.thresholds(w, _path(4, 10.0), pack, FEES, Knobs(impl="dp"), 0.0, NOW, loss=LOSSLESS)
    assert dp_th.impl == "dp"
    ranking = Knobs(impl="ranking", kappa=0.5)  # the pre-Task-25 defaults, pinned
    assert T.thresholds(w, _path(4, 10.0), pack, FEES, ranking, 0.0, NOW).impl == "ranking"


def test_nan_net_load_counts_as_zero_and_a_nan_price_raises():
    take = [50] * 2 + [400] * 4 + [50] * 26
    w = _window(take, [t - 5 for t in take])
    knobs = Knobs(impl="dp", wear_eur_cycle=0.0, hurdle_ct_kwh=0.0)
    pack = Pack(48, 12, 12, 0.1, 1.0, eta_rt=1.0)
    zero, gap = _path(32, 20.0, net_kw=1.0), _path(32, 20.0, net_kw=1.0)
    zero.net_kw[3] = 0.0
    gap.net_kw[3] = np.nan
    a = T.dp(w, zero, pack, FEES, knobs, need=0.0, now=NOW, loss=LOSSLESS)
    b = T.dp(w, gap, pack, FEES, knobs, need=0.0, now=NOW, loss=LOSSLESS)
    assert np.isfinite(b.value_stored).all()
    np.testing.assert_array_equal(a.value_stored, b.value_stored)
    w.feed = w.feed.astype(float)  # _window keeps the int dtype of an int price list
    w.feed[4, 7] = np.nan  # the q90 row
    with pytest.raises(ValueError, match="window has non-finite prices"):
        T.dp(w, zero, pack, FEES, knobs, need=0.0, now=NOW, loss=LOSSLESS)


def test_dp_ignores_the_need_so_the_learning_widening_is_inert_under_dp():
    # pins the documented limit (I4 of the final review): the learning-state widening scales
    # the need, which dp never reads; a dp-side equivalent is an open item
    take = [50] * 2 + [400] * 4 + [50] * 26
    w = _window(take, [t - 5 for t in take])
    knobs = Knobs(impl="dp")
    pack = Pack(48, 12, 12, 0.1, 1.0)
    base = T.dp(w, _path(32, 20.0), pack, FEES, knobs, need=4.0, now=NOW, loss=LOSSLESS)
    wide = T.dp(w, _path(32, 20.0), pack, FEES, knobs, need=4.0 * 1.5, now=NOW, loss=LOSSLESS)
    assert np.array_equal(base.charge, wide.charge)
    assert np.array_equal(base.discharge, wide.discharge)
    assert np.array_equal(base.sale_slots, wide.sale_slots)
