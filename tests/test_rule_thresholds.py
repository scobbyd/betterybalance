# tests/test_rule_thresholds.py
import numpy as np
import pytest
from betterybalance.core.rule import need as N
from betterybalance.core.rule import thresholds as T
from betterybalance.core.rule import window as W
from betterybalance.core.site.contract import Knobs, Pack, Terms
from betterybalance.core.site.socpath import SocPath

# the knob defaults these scenarios pin, from before the Task 25 ladder changed them
RANKING = {"impl": "ranking", "kappa": 0.5}

PACK = Pack(48, 12, 12, 0.1, 1.0, eta_rt=1.0)  # ec = ed = 1 keeps the arithmetic readable
NOW = 1793000700  # 2026-10-26 08:45 local, an ISP start
FEES = Terms(fee_import_eur_kwh=0.0, fee_export_eur_kwh=0.0)


def _window(take, feed, p_dual=0.0, spread=0.0):
    n = len(take)
    tq = np.vstack([np.array(take) + d for d in (-20, -10, 0, 10, 20)])
    fq = np.vstack([np.array(feed) + d for d in (-20, -10, 0, 10, 20)])
    return W.Window(
        np.arange(NOW, NOW + 900 * n, 900),
        tq,
        fq,
        np.full(n, p_dual),
        np.full(n, spread),
        np.zeros(n, bool),
        ["A"] * n,
        1,
    )


def _path(n, soc_kwh, net_kw=0.0):
    soc = np.full(n + 1, soc_kwh)
    room_charge, room_discharge = 48 - soc[:-1], soc[:-1] - 4.8
    return SocPath(np.arange(n), soc, room_charge, room_discharge, np.zeros(n), np.full(n, net_kw))


def test_need_and_slot_count():
    p = _path(8, 10.0, net_kw=2.0)  # 8 slots at 2 kW = 4 kWh demand, pack holds 5,2 above floor
    assert N.need_kwh(p, 10.0, PACK) == 0.0
    assert N.need_kwh(p, 5.0, PACK) == pytest.approx(4.0 - 0.2)
    assert N.need_kwh(p, 5.0, PACK, widening=1.5) == pytest.approx(5.7)
    assert N.charge_slots_needed(5.7, PACK) == 2 and N.charge_slots_needed(0.0, PACK) == 0
    lossy = Pack(48, 12, 12, 0.1, 1.0, eta_rt=0.81)  # ed 0,9: 4 AC kWh draw 4 / 0,9 cells kWh
    assert N.need_kwh(p, 5.0, lossy) == pytest.approx(4.0 / 0.9 - 0.2)


def test_charge_threshold_is_the_nth_cheapest_take_at_q25():
    w = _window(take=[80, 50, 120, 30, 90, 200, 60, 70], feed=[70, 40, 110, 20, 80, 190, 50, 60])
    knobs = Knobs(**RANKING, wear_eur_cycle=0.0, hurdle_ct_kwh=100.0)
    th = T.ranking(w, _path(8, 4.8), PACK, FEES, knobs, need=6.0, now=NOW)
    # need 6 kWh at 3 kWh per slot = 2 slots; the 2nd cheapest take at q25 is 50 - 10 = 40
    assert th.n_charge == 2 and th.charge[0] == 40.0 and np.isinf(th.discharge[0])
    assert th.charge_slots.sum() == 2
    assert th.value_stored[0] == pytest.approx(0.040)  # below need: the n-th buy cost in EUR/kWh


def test_need_beyond_the_window_values_stored_energy_at_the_dearest_slot():
    w = _window(take=[80, 50, 120, 30, 90, 200, 60, 70], feed=[70, 40, 110, 20, 80, 190, 50, 60])
    knobs = Knobs(**RANKING, wear_eur_cycle=0.0, hurdle_ct_kwh=100.0)
    th = T.ranking(w, _path(8, 4.8), PACK, FEES, knobs, need=100.0, now=NOW)  # 34 slots > 8
    assert th.n_charge == 8 and th.value_stored[0] == pytest.approx(0.190)  # 200 - 10 at q25


def test_free_refill_needs_the_ac_surplus_over_the_charge_leg():
    # ec = ed = 0,9; 10,8 kW for one quarter is 3,0 cells kWh per sale
    pack = Pack(48, 12, 10.8, 0.1, 1.0, eta_rt=0.81)
    w = _window([100, 100], [100, 100])
    knobs = Knobs(**RANKING, wear_eur_cycle=0.0)

    def sales(ac_surplus_after_slot_0):
        # pack at the floor (nothing unpromised), cells room to sell, no room to charge
        s2c = np.full(2, ac_surplus_after_slot_0)  # all of it in slot 1
        path = SocPath(
            np.arange(2), np.full(3, 4.8), np.zeros(2), np.full(2, 43.2), s2c, np.zeros(2)
        )
        return T.ranking(w, path, pack, FEES, knobs, need=0.0, now=NOW).sale_slots

    assert sales(3.0 / 0.9).tolist() == [True, False]  # 3,0 / ec AC kWh refills 3,0 cells kWh
    assert not sales(3.0).any()  # 3,0 AC kWh puts back only 2,7 cells kWh


def test_sale_slot_needs_the_spread_to_beat_the_hurdle_and_adds_a_charge_slot():
    take = [30, 30, 30, 30, 300, 300, 30, 30]
    feed = [t - 5 for t in take]
    w = _window(take, feed)
    knobs = Knobs(**RANKING, wear_eur_cycle=0.0, hurdle_ct_kwh=3.0)
    th = T.ranking(w, _path(8, 4.8), PACK, FEES, knobs, need=0.0, now=NOW)
    # sale at q75 feed 305 vs buy at q25 take 20: 0,285 EUR/kWh beats the 0,03 hurdle;
    # two sale slots, two charge slots
    assert th.sale_slots.sum() == 2 and th.charge_slots.sum() == 2 and th.n_charge == 2
    assert th.discharge[0] == 305.0 and th.charge[0] == 20.0
    knobs = Knobs(**RANKING, wear_eur_cycle=0.0, hurdle_ct_kwh=30.0)
    tight = T.ranking(w, _path(8, 4.8), PACK, FEES, knobs, need=0.0, now=NOW)
    assert tight.sale_slots.sum() == 0 and np.isinf(tight.discharge[0])
    assert np.isneginf(tight.charge[0])


def test_dual_penalty_only_when_enabled():
    take = [30, 30, 30, 30, 90, 90, 30, 30]
    feed = [t - 5 for t in take]
    w = _window(take, feed, p_dual=0.5, spread=100.0)  # penalty 0,05 + 0,05 = 0,10 EUR/kWh
    knobs_on = Knobs(**RANKING, wear_eur_cycle=0.0, hurdle_ct_kwh=3.0)
    knobs_off = Knobs(**RANKING, wear_eur_cycle=0.0, hurdle_ct_kwh=3.0, dual_penalty=False)
    on = T.ranking(w, _path(8, 4.8), PACK, FEES, knobs_on, need=0.0, now=NOW)
    off = T.ranking(w, _path(8, 4.8), PACK, FEES, knobs_off, need=0.0, now=NOW)
    assert off.sale_slots.sum() == 2 and on.sale_slots.sum() == 0  # 0,075 clears 0,03, not 0,13


def test_greed_scenarios():
    night = [100 - 2 * j for j in range(32)]  # a night sliding from 100 to 38 EUR/MWh
    w = _window(night, [t - 5 for t in night])
    empty = T.ranking(
        w, _path(32, 4.8, net_kw=1.0), PACK, FEES, Knobs(**RANKING), need=52.0, now=NOW
    )
    # 18 slots needed: the 18th cheapest q25 take is 62, above the night's median q25 take (59)
    assert empty.n_charge == 18 and empty.charge[0] == 62.0
    assert empty.charge[0] >= np.median(w.take[1])
    full_path = _path(32, 48.0, net_kw=-12.0)  # 96 kWh of surplus to come refills every sale
    full_path.surplus_to_come_kwh[:] = np.cumsum(np.full(32, 3.0)[::-1])[::-1]
    full = T.ranking(
        w, full_path, PACK, FEES, Knobs(**RANKING, wear_eur_cycle=0.0), need=0.0, now=NOW
    )
    assert full.n_charge == 0 and full.sale_slots.sum() == 32 and full.discharge[0] == 43.0
    assert full.discharge[0] <= np.median(w.feed[3])
    assert full.value_stored[0] < empty.value_stored[0]


def test_tax_mode_none_adds_tax_to_the_buy_leg_only():
    w = _window([100] * 4, [100] * 4)
    terms = Terms(fee_import_eur_kwh=0.0, fee_export_eur_kwh=0.0, tax_eur_kwh=0.10, tax_mode="none")
    buy, sale = T.slot_costs(w, PACK, terms, Knobs(**RANKING, wear_eur_cycle=0.0), NOW)
    assert buy[0] == pytest.approx(0.09 + 0.10) and sale[0] == pytest.approx(0.11)
    with pytest.raises(ValueError, match="maybe"):  # Terms built directly skip from_dict
        T.slot_costs(w, PACK, Terms(tax_mode="maybe"), Knobs(**RANKING), NOW)
