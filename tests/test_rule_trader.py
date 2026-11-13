# tests/test_rule_trader.py
import pytest
from betterybalance.core import battery as B
from betterybalance.core.rule import trader as TR
from betterybalance.core.rule import window as W
from betterybalance.core.site import contract as C

from tests.test_rule_thresholds import NOW, RANKING, _path, _window


def _cfg(**knobs):
    # a high hurdle keeps the sale pairing out of these scenarios,
    # so the thresholds are the need slots alone
    return C.from_dict(
        {
            "latitude": 52.1,
            "longitude": 5.2,
            "pack": {
                "capacity_kwh": 48,
                "charge_kw": 12,
                "discharge_kw": 12,
                "soc_floor": 0.1,
                "soc_ceiling": 1.0,
                "eta_rt": 1.0,
            },
            "terms": {"fee_import_eur_kwh": 0.0, "fee_export_eur_kwh": 0.0},
            "knobs": {"wear_eur_cycle": 0.0, "hurdle_ct_kwh": 100.0, **RANKING, **knobs},
        }
    )


def _nc(take, w=10.0, feed=None):
    feed = take - 5 if feed is None else feed
    return W.Nowcast(take, feed, w, w, 0.1, take, feed)


def _ranked(cfg, take, need, soc=4.8):
    t = TR.Trader(cfg, B.flat_profile(1.0))
    w = _window(take, [x - 5 for x in take])
    t.rank(w, _path(len(take), soc), need, NOW)
    return t


def test_clear_margin_commits_at_minute_one_and_holds_the_quarter():
    # charge threshold 40 (q25 of the 2nd cheapest)
    t = _ranked(_cfg(), [80, 50, 120, 30, 90, 200, 60, 70], need=6.0)
    d = t.minute(NOW + 60, _nc(20.0, w=10.0), 0.0, 4.8, False, False)
    assert (
        d.action == "charge"
        and d.kw == 12.0
        and d.committed
        and d.reason == "commit"
        and d.margin_w == pytest.approx(2.0)
    )
    later = t.minute(NOW + 8 * 60, _nc(60.0, w=10.0), 0.0, 6.0, False, False)
    assert later.action == "charge" and later.committed


def test_marginal_decision_waits_for_the_band_to_narrow():
    t = _ranked(_cfg(), [80, 50, 120, 30, 90, 200, 60, 70], need=6.0)
    # margin 0,4 w
    assert t.minute(NOW + 60, _nc(36.0, w=10.0), 0.0, 4.8, False, False).reason == "wait"
    # margin 0,67 w
    assert t.minute(NOW + 5 * 60, _nc(36.0, w=6.0), 0.0, 4.8, False, False).reason == "commit"
    late = _ranked(_cfg(), [80, 50, 120, 30, 90, 200, 60, 70], need=6.0)
    assert late.minute(NOW + 11 * 60, _nc(10.0, w=1.0), 0.0, 4.8, False, False).reason == "late"


def test_negative_take_price_charges_regardless_of_the_band():
    t = _ranked(_cfg(), [300] * 8, need=0.0)
    d = t.minute(NOW + 60, _nc(-5.0, w=50.0), 0.0, 20.0, False, False)
    assert d.action == "charge" and d.reason == "negative_take"


def test_boundary_precommit_and_revoke():
    take = [30, 10, 30, 30, 300, 300, 30, 30]
    # charge slots 0 and 1, sale slots 4, 5, 2, 3, 6 and 7
    t = _ranked(_cfg(hurdle_ct_kwh=3.0), take, need=0.0, soc=40.0)
    # slot 1 (q25 take 0) sits well under the charge threshold (20, the 2nd cheapest q25);
    # minute 0 pre-commits a charge
    d0 = t.minute(NOW + 900, None, 0.0, 40.0, False, False)
    assert d0.reason == "precommit" and d0.action == "charge"
    d1 = t.minute(NOW + 960, _nc(500.0, w=10.0), 0.0, 40.0, False, False)
    assert d1.reason == "revoked" and d1.action == "hold"
    off = _ranked(_cfg(boundary_precommit=False, hurdle_ct_kwh=3.0), take, need=0.0, soc=40.0)
    assert off.minute(NOW + 900, None, 0.0, 40.0, False, False).reason == "default"


def test_default_policy_value_tests_and_ev_plan_slot():
    t = _ranked(_cfg(), [80, 50, 120, 30, 90, 200, 60, 70], need=6.0)  # value_stored 0,040 EUR/kWh
    absorb = t.minute(NOW + 11 * 60, _nc(60.0, feed=30.0), -3.0, 20.0, False, False)
    assert absorb.action == "charge" and absorb.kw == 3.0 and absorb.reason == "late"
    export = t.minute(NOW + 11 * 60, _nc(60.0, feed=70.0), -3.0, 20.0, False, False)
    assert export.kw == 0.0  # 0,070 beats storing at 0,040
    negative = t.minute(NOW + 11 * 60, _nc(60.0, feed=-20.0), -3.0, 20.0, False, False)
    # a negative feed price never beats storing
    assert negative.action == "charge" and negative.kw == 3.0
    cover = t.minute(NOW + 11 * 60, _nc(60.0), 2.0, 20.0, False, False)
    assert cover.action == "discharge" and cover.kw == -2.0
    cheap = t.minute(NOW + 11 * 60, _nc(30.0), 2.0, 20.0, False, False)
    assert cheap.kw == 0.0  # importing at 0,030 beats discharging 0,040 worth
    ev = t.minute(NOW + 11 * 60, _nc(60.0), 12.0, 20.0, True, False)
    assert ev.kw == -1.0  # 12 kW load minus the 11 kW car: the pack covers the house only


def test_expected_gain_rule_waits_at_small_margins_and_commits_at_large_ones():
    t = _ranked(_cfg(commit_rule="expected_gain"), [80, 50, 120, 30, 90, 200, 60, 70], need=6.0)
    # margin 0,4 w: P = Phi(0,27) = 0,606; 0,606 * 0,4 - 0,394 = -0,15 -> wait
    # margin 0,5 w (kappa would commit): P = Phi(0,337) = 0,632; 0,632 * 0,5 - 0,368 < 0 -> wait
    # margin 0,9 w: P = 0,728; 0,655 - 0,272 > 0 -> commit
    assert t.minute(NOW + 60, _nc(36.0, w=10.0), 0.0, 4.8, False, False).reason == "wait"
    assert t.minute(NOW + 120, _nc(35.0, w=10.0), 0.0, 4.8, False, False).reason == "wait"
    assert t.minute(NOW + 180, _nc(31.0, w=10.0), 0.0, 4.8, False, False).reason == "commit"


def test_stale_ranking_holds_and_learning_flag_passes_through():
    t = _ranked(_cfg(), [80] * 8, need=0.0)
    d = t.minute(NOW + 3700, _nc(80.0), 1.0, 20.0, False, True)
    assert d.reason == "stale" and d.learning
    # the default policy covers the 1,0 kW load: action follows the kw sign, not committed
    assert d.action == "discharge" and d.kw == -1.0 and not d.committed
