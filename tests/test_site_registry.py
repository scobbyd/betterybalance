import numpy as np
import pytest
from betterybalance.core.site import registry as R
from betterybalance.core.site import separate as SEP
from betterybalance.core.site.contract import Plane

from tests.test_site_separate import LAT, LON, _cfg, synthetic_house


def _model(holdout, null, stability=(1.0, 0.0), planes=(Plane(30, 180),)):
    return SEP.PvModel(planes, np.array([5.0]), np.ones(12), None, holdout, null, stability, 0, 30)


def test_holdout_pass_rule():
    assert R.holdout_passes(_model(0.6, 1.0))
    assert not R.holdout_passes(_model(0.8, 1.0))
    assert not R.holdout_passes(_model(0.6, 1.0, stability=(1.3, 0.0)))
    assert not R.holdout_passes(_model(0.6, 1.0, stability=(1.0, 45.0)))
    assert not R.holdout_passes(_model(None, None))


def test_initial_state_with_planes_is_usable_on_day_one():
    st = R.initial_state(
        _cfg(planes=[{"tilt_deg": 35, "azimuth_deg": 200, "kwp": 4.0}], correction=0.9)
    )
    assert (
        not st.learning
        and st.champion.scales_kwp[0] == pytest.approx(3.6)
        and st.champion.month_factor.tolist() == [1.0] * 12
    )
    assert R.initial_state(_cfg()).learning and R.initial_state(_cfg()).champion is None
    assert not R.initial_state(_cfg(none=True)).learning


def test_weekly_update_promotes_on_margin_and_leaves_learning_when_check_passes():
    q, irr, pv, _ = synthetic_house((Plane(30, 180),), (5.0,), days=45)
    cfg = _cfg()
    st = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now=int(q["ts"][-1] + 900))
    assert st.champion is not None and not st.learning and st.last_fit_ts == int(q["ts"][-1] + 900)
    worse = R.SiteModelState(
        champion=_model(st.champion.holdout_error * 1.02, st.champion.null_error)
    )
    assert R.weekly_update(
        worse, q, irr, cfg, now=int(q["ts"][-1] + 900)
    ).champion.holdout_error == pytest.approx(worse.champion.holdout_error)
    much_worse = R.SiteModelState(
        champion=_model(st.champion.holdout_error * 1.5, st.champion.null_error)
    )
    assert R.weekly_update(
        much_worse, q, irr, cfg, now=int(q["ts"][-1] + 900)
    ).champion.holdout_error == pytest.approx(st.champion.holdout_error)


def test_too_little_daylight_data_keeps_the_state():
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=5)
    cfg = _cfg()
    st = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now=int(q["ts"][-1] + 900))
    assert st.champion is None and st.learning and R.need_widening(st, cfg.knobs) == 1.5


def test_drift_alarm_fires_on_a_twenty_percent_bias():
    q, irr, pv, _ = synthetic_house((Plane(30, 180),), (5.0,), days=20)
    now = int(q["ts"][-1] + 900)
    assert not R.drift_alarm(q, pv, irr, LAT, LON, now)
    assert R.drift_alarm(q, pv * 0.8, irr, LAT, LON, now)


def test_state_json_roundtrip():
    st = R.SiteModelState(champion=_model(0.5, 1.0), learning=False, learned_days=20, last_fit_ts=5)
    back = R.state_from_json(R.state_to_json(st))
    assert (
        back.champion.planes == st.champion.planes and back.learned_days == 20 and not back.learning
    )


def test_state_json_roundtrip_with_none_errors_and_nan_stability():
    nan = float("nan")
    st = R.SiteModelState(
        champion=_model(None, None, stability=(nan, nan)), challenger=_model(0.5, 1.0)
    )
    back = R.state_from_json(R.state_to_json(st))
    assert back.champion.holdout_error is None and np.isnan(back.champion.stability[0])
    assert back.challenger.holdout_error == 0.5 and back.last_fit_ts is None


def _house45():
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=45)
    return q, irr, int(q["ts"][-1] + 900)


def test_pv_none_skips_the_fit_and_keeps_the_empty_champion():
    q, irr, now = _house45()
    cfg = _cfg(none=True)
    st = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now=now)
    assert st.champion.planes == () and not st.learning and st.challenger is None
    assert st.last_fit_ts == now and st.learned_days == 45


def test_planes_given_first_passing_challenger_replaces_hints_champion():
    q, irr, now = _house45()
    cfg = _cfg(planes=[{"tilt_deg": 30, "azimuth_deg": 180, "kwp": 5.0}], correction=1.0)
    start = R.initial_state(cfg)
    assert start.champion.holdout_error is None
    st = R.weekly_update(start, q, irr, cfg, now=now)
    assert st.champion.holdout_error is not None and np.isfinite(st.champion.holdout_error)
    assert not st.learning


def test_hints_champion_stays_when_challenger_does_not_pass(monkeypatch):
    q, irr, now = _house45()
    cfg = _cfg(planes=[{"tilt_deg": 30, "azimuth_deg": 180, "kwp": 5.0}], correction=1.0)
    start = R.initial_state(cfg)
    monkeypatch.setattr(R, "holdout_passes", lambda m: False)
    st = R.weekly_update(start, q, irr, cfg, now=now)
    assert st.champion is start.champion and st.challenger is not None
    assert not st.learning  # planes given: learning stays False with None-error champion


def test_failing_champion_is_replaced_by_a_passing_challenger_without_the_margin():
    q, irr, now = _house45()
    cfg = _cfg()
    fresh = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now=now)
    assert R.holdout_passes(fresh.challenger)
    err, null = fresh.challenger.holdout_error, fresh.challenger.null_error
    # the champion's error is only 1 % above the challenger's, but its ratio 0,9 fails the rule
    failing = _model(err * 1.01, err * 1.01 / 0.9)
    assert not R.holdout_passes(failing) and err > (1.0 - cfg.knobs.promotion_margin) * err * 1.01
    st = R.weekly_update(R.SiteModelState(champion=failing), q, irr, cfg, now=now)
    assert st.champion.holdout_error == pytest.approx(err) and not st.learning
    assert st.champion.null_error == pytest.approx(null)


def test_passing_champion_keeps_the_promotion_margin():
    q, irr, now = _house45()
    cfg = _cfg()
    fresh = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now=now)
    err, null = fresh.challenger.holdout_error, fresh.challenger.null_error
    passing = _model(err * 1.01, null)
    assert R.holdout_passes(passing)
    st = R.weekly_update(R.SiteModelState(champion=passing), q, irr, cfg, now=now)
    assert st.champion.holdout_error == pytest.approx(err * 1.01)


def test_untested_stability_passes_on_the_error_ratio_alone():
    nan = float("nan")
    assert R.holdout_passes(_model(0.6, 1.0, stability=(nan, nan)))
    assert not R.holdout_passes(_model(0.8, 1.0, stability=(nan, nan)))
    assert not R.holdout_passes(_model(None, None, stability=(nan, nan)))
    assert not R.holdout_passes(_model(0.6, None, stability=(nan, nan)))


def test_fourteen_lit_day_house_leaves_learning_on_its_error_ratio():
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=14)
    cfg = _cfg()
    st = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now=int(q["ts"][-1] + 900))
    champ = st.champion
    assert champ is not None and all(np.isnan(x) for x in champ.stability)  # halves of 4 and 3
    assert champ.holdout_error < R.HOLDOUT_RATIO * champ.null_error and not st.learning
