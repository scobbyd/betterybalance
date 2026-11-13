# tests/test_live_views.py
from dataclasses import replace

import numpy as np
import pytest
from betterybalance.core.rule import thresholds as TV
from betterybalance.live import views as V

from tests.test_live_site import _run, _site


def _ranked():
    site, store, start = _site()
    _run(site, start, 2)
    return site, site.status()["ranked"], site.lc.cfg


def _plan(site, ranked, cfg):
    return V.plan_path(ranked, cfg.pack, site.loss, cfg.terms, cfg.knobs)


def test_ct_conversion():
    assert V.ct_kwh(123.4) == pytest.approx(12.34)


def test_plan_view_shape_and_range():
    site, store, start = _site()
    _run(site, start, 2)
    ranked = site.status()["ranked"]
    cfg = site.lc.cfg
    planned = _plan(site, ranked, cfg)
    state, attrs = V.plan_view(ranked, planned, cfg.terms, cfg.pack, cfg.knobs)
    n = min(144, len(ranked.window.ts))
    assert len(attrs["ts"]) == len(attrs["take_q50"]) == len(attrs["soc_pct"]) == n
    assert attrs["result_low"] <= state <= attrs["result_high"]
    assert attrs["attribution"] == "Weather data by Open-Meteo.com (CC BY 4.0)"


def _in_band(planned, pack):
    cap = pack.capacity_kwh
    assert (planned.kw <= pack.charge_kw + 1e-9).all()
    assert (planned.kw >= -pack.discharge_kw - 1e-9).all()
    assert (planned.soc_kwh >= pack.soc_floor * cap - 1e-9).all()
    assert (planned.soc_kwh <= pack.soc_ceiling * cap + 1e-9).all()


def test_dp_plan_walks_the_policy_at_the_planned_soc_within_the_pack():
    site, ranked, cfg = _ranked()
    assert ranked.th.impl == "dp"
    planned = _plan(site, ranked, cfg)
    slots = len(ranked.window.ts)
    assert len(planned.action) == len(planned.kw) == slots and len(planned.soc_kwh) == slots + 1
    _in_band(planned, cfg.pack)
    policy = TV.dp_policy(
        ranked.window, ranked.path, cfg.pack, cfg.terms, cfg.knobs, 0.0,
        V.ranking_now(ranked), site.loss,
    )
    grid = TV.soc_grid(cfg.pack)
    for j in range(slots):  # each slot takes the policy's action at the planned SoC
        s = int(np.abs(grid - planned.soc_kwh[j]).argmin())
        assert planned.action[j] == TV.DP_ACTIONS[policy[j, s]]
    # the policy charges where it pays from the SoC it reached, not every marked slot:
    # 11 of the 29 charge slots of the first 36 hours (the q50 masks at the default path)
    m = V.PLAN_SLOTS
    charges = planned.action[:m] == "charge"
    assert 0 < charges.sum() < ranked.th.charge_slots[:m].sum()
    assert (planned.kw[:m][charges] > 0).all()


def test_dp_policy_at_the_default_path_is_the_dp_ranking():
    site, ranked, cfg = _ranked()
    policy = TV.dp_policy(
        ranked.window, ranked.path, cfg.pack, cfg.terms, cfg.knobs, 0.0,
        V.ranking_now(ranked), site.loss,
    )
    grid = TV.soc_grid(cfg.pack)
    soc = ranked.path.soc_kwh[: len(ranked.window.ts)]
    nearest = np.abs(soc[:, None] - grid[None, :]).argmin(axis=1)
    codes = policy[np.arange(len(soc)), nearest]
    assert np.array_equal(codes == 1, ranked.th.charge_slots)
    assert np.array_equal(codes == 2, ranked.th.sale_slots)


def test_ranking_plan_keeps_the_slot_approximation_within_the_pack():
    site, ranked, cfg = _ranked()
    ranked = replace(ranked, th=replace(ranked.th, impl="ranking"))
    planned = V.plan_path(ranked, cfg.pack, site.loss)
    th = ranked.th
    assert set(planned.action[th.charge_slots]) <= {"charge"}
    assert set(planned.action[th.sale_slots & ~th.charge_slots]) <= {"discharge"}
    _in_band(planned, cfg.pack)


def test_a_dp_plan_needs_terms_and_knobs():
    site, ranked, cfg = _ranked()
    with pytest.raises(ValueError):
        V.plan_path(ranked, cfg.pack, site.loss)


def test_plan_value_is_zero_without_a_battery_move():
    site, store, start = _site()
    _run(site, start, 2)
    ranked = site.status()["ranked"]
    slots = len(ranked.window.ts)
    idle = V.PlannedPath(
        np.array(["hold"] * slots), np.zeros(slots), np.full(slots + 1, 10.0)
    )
    cfg = site.lc.cfg
    assert V.plan_value(idle, ranked, 0.5, cfg.terms, cfg.pack, cfg.knobs) == pytest.approx(0.0)


def test_energy_left_in_the_pack_counts_at_the_rules_terminal_value():
    # the synthetic window charges inside its 36 hours: money alone is a cost (-1,23 EUR at
    # q50 under the dp policy); the terminal term adds the kWh left times the rule's terminal
    # value incl. BTW, the same at every quantile
    site, ranked, cfg = _ranked()
    planned = _plan(site, ranked, cfg)
    m = min(V.PLAN_SLOTS, len(ranked.window.ts))
    assert planned.soc_kwh[m] > planned.soc_kwh[0]
    flat = V.PlannedPath(planned.action, planned.kw, np.full(len(planned.soc_kwh), 20.0))
    tv = TV.dp_terminal_value(ranked.window, cfg.pack, cfg.terms, cfg.knobs, V.ranking_now(ranked))
    left = planned.soc_kwh[m] - planned.soc_kwh[0]
    for q in (0.1, 0.5, 0.9):
        money = V.plan_value(flat, ranked, q, cfg.terms, cfg.pack, cfg.knobs, n=m)
        full = V.plan_value(planned, ranked, q, cfg.terms, cfg.pack, cfg.knobs, n=m)
        assert full == pytest.approx(money + left * tv * (1 + cfg.terms.btw))


def test_charging_now_to_sell_later_does_not_read_negative_under_the_dp_policy():
    site, ranked, cfg = _ranked()
    planned = _plan(site, ranked, cfg)
    state, attrs = V.plan_view(ranked, planned, cfg.terms, cfg.pack, cfg.knobs)
    assert state >= 0.0 and attrs["result_low"] <= state <= attrs["result_high"]


def test_the_terminal_term_leaves_a_plan_ending_at_its_start_soc_unchanged(monkeypatch):
    site, ranked, cfg = _ranked()
    planned = _plan(site, ranked, cfg)
    m = min(V.PLAN_SLOTS, len(ranked.window.ts))
    flat = V.PlannedPath(planned.action, planned.kw, np.full(len(planned.soc_kwh), 20.0))
    args = (cfg.terms, cfg.pack, cfg.knobs)
    before = V.plan_value(flat, ranked, 0.5, *args, n=m)
    moved = V.plan_value(planned, ranked, 0.5, *args, n=m)
    monkeypatch.setattr(V, "dp_terminal_value", lambda *a: 1000.0)
    assert V.plan_value(flat, ranked, 0.5, *args, n=m) == pytest.approx(before)
    left = planned.soc_kwh[m] - planned.soc_kwh[0]
    assert V.plan_value(planned, ranked, 0.5, *args, n=m) == pytest.approx(
        moved
        + left
        * (1000.0 - TV.dp_terminal_value(
            ranked.window, cfg.pack, cfg.terms, cfg.knobs, V.ranking_now(ranked)
        ))
        * (1 + cfg.terms.btw)
    )


def test_next_window_is_the_first_run_of_trade_slots():
    site, store, start = _site()
    _run(site, start, 2)
    ranked = site.status()["ranked"]
    th, ts = ranked.th, ranked.window.ts
    w = V.next_window(ranked, int(ts[0]))
    trade = th.charge_slots | th.sale_slots
    if not trade.any():
        assert w is None
        return
    j = int(np.flatnonzero(trade)[0])
    assert w["start"] == int(ts[j]) and w["side"] in ("charge", "discharge")
    assert w["end"] > w["start"] and (w["end"] - w["start"]) % 900 == 0


def test_state_view_carries_the_decision_and_the_flags():
    site, store, start = _site()
    _run(site, start, 30)
    st = site.status()
    state, attrs = V.state_view(st, st["ranked"], "en")
    d = st["decision"]
    assert state == d.action and attrs["reason"] == d.reason and attrs["committed"] == d.committed
    assert isinstance(attrs["verdict"], str) and attrs["verdict"]
    assert attrs["status"] == st["flags"] and len(attrs["status_text"]) == len(st["flags"])
    assert attrs["ledger_head"]["seq"] == st["head"][0]


def test_actual_path_never_reaches_a_view():
    site, store, start = _site()
    _run(site, start, 30)
    state, attrs = V.state_view(site.status(), site.status()["ranked"], "en")
    assert "actual" not in repr(attrs)


def test_pv_now_splits_the_total_per_string():
    from betterybalance.core.site.contract import Plane
    from betterybalance.core.site.separate import PvModel, predict_pv_kw

    slot = 1793620800  # 2026-11-02 13:00 Amsterdam
    irr = {
        "ts": np.array([slot], dtype=np.int64), "ghi_w_m2": np.array([400.0]),
        "dni_w_m2": np.array([300.0]), "dhi_w_m2": np.array([150.0]),
        "temp_c": np.array([10.0]),
    }
    planes = (Plane(35, 180, string="pv1"), Plane(30, 90, string="pv2"))

    def model(ac_max):
        return PvModel(
            planes, np.array([4.0, 2.0]), np.ones(12), ac_max, None, None, (1.0, 0.0),
            1793500000, 30,
        )

    for ac_max in (None, 1.0):
        m = model(ac_max)
        total = float(predict_pv_kw(m, irr, 52.1, 5.2)[0])
        assert total > 1.0 if ac_max is None else total == pytest.approx(1.0)
        out = V.pv_now(m, irr, slot, 52.1, 5.2, total)
        assert set(out["per_string"]) == {"pv1", "pv2"}
        assert sum(out["per_string"].values()) == pytest.approx(total, abs=2e-3)
        assert out["model_version"] == "pv-1793500000" and out["kw"] == total
    assert V.pv_now(None, irr, slot, 52.1, 5.2, 0.0) == {
        "kw": 0.0, "per_string": {}, "model_version": None,
    }


def test_next_window_follows_the_dp_plan_and_the_masks_without_one():
    site, ranked, cfg = _ranked()
    planned = _plan(site, ranked, cfg)
    ts = ranked.window.ts
    w = V.next_window(ranked, int(ts[0]), planned)
    trade = np.flatnonzero(planned.action != "hold")
    assert w is not None and w["start"] == int(ts[trade[0]])
    j, e = int(trade[0]), int(np.searchsorted(ts, w["end"] - 900))
    assert (planned.action[j : e + 1] == w["side"]).all()
    assert e + 1 == len(ts) or planned.action[e + 1] != w["side"]
    # the q50 masks mark far more charge slots than the plan uses: the windows differ
    masks = V.next_window(ranked, int(ts[0]))
    assert masks is not None and masks != w
    # the verdict names the plan's window
    st = dict(site.status(), decision=None)
    _, attrs = V.state_view(st, ranked, "en", planned)
    assert attrs["next_window"] == w
