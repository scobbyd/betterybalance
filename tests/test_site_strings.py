import dataclasses

import numpy as np
import pytest
from betterybalance.core.battery import pv_dc_to_ac_stage
from betterybalance.core.site import contract as C
from betterybalance.core.site import registry as R
from betterybalance.core.site import separate as SEP
from betterybalance.core.site import strings as ST
from betterybalance.core.site.plant import plane_shape
from betterybalance.replay.disaggregate import flatten_to_hours, hourly_to_quarters

from tests.test_site_separate import LAT, LON, _cfg, synthetic_house

EAST, WEST = C.Plane(35, 135), C.Plane(30, 225)
GEN = C.Plane(15, 180)


def _measured(irr, plane, kwp, rng, noise=0.03):
    """kWh per quarter of one measured string: kwp times the plane shape, 3 % noise, at least 0."""
    kw = kwp * plane_shape(irr, plane, LAT, LON)
    return np.clip(kw * (1.0 + noise * rng.normal(0.0, 1.0, len(kw))), 0.0, None) / 4.0


def _house(seed=3, days=40, gen_kwp=None, first="2026-05-01"):
    """Synthetic irradiance (fixture route of tests/test_site_separate.py) and two measured strings:
    35/135 at 4,0 kWp and 30/225 at 3,0 kWp, noise 3 %, seeded. With gen_kwp a third plane
    (15/180) feeds the GEN port: gen_kwh carries it and the grid export grows by it."""
    q, irr, _, _ = synthetic_house(
        (EAST, WEST), (4.0, 3.0), first=first, days=days, seed=seed, irr_noise=0.0
    )
    rng = np.random.default_rng(seed + 100)
    q["pv1_kwh"] = _measured(irr, EAST, 4.0, rng)
    q["pv2_kwh"] = _measured(irr, WEST, 3.0, rng)
    for i in range(3, 9):
        q[f"pv{i}_kwh"] = np.full(len(q["ts"]), np.nan)
    if gen_kwp is not None:
        gen = _measured(irr, GEN, gen_kwp, rng)
        r = q["grid_import_kwh"] - q["grid_export_kwh"] - gen
        q["grid_import_kwh"], q["grid_export_kwh"] = np.clip(r, 0, None), np.clip(-r, 0, None)
        q["gen_kwh"] = gen
    return q, irr, _cfg(), int(q["ts"][-1] + 900)


def test_two_strings_recover_geometry_and_size():
    q, irr, cfg, now = _house()
    m = ST.fit_strings(q, irr, cfg, now)
    assert [round(p.azimuth_deg) for p in m.planes] == pytest.approx([135, 225], abs=15)
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0], rel=0.05)
    assert [p.string for p in m.planes] == ["pv1", "pv2"]
    assert m.ac_max_kw is None and m.holdout_error is not None
    assert m.holdout_error < 0.1 * m.null_error and R.holdout_passes(m)


def test_fill_in_geometry_and_scale_check():
    q, irr, cfg, now = _house()
    given = (C.Plane(35, 135, 4.0), C.Plane(30, 225, 4.5))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert [p.azimuth_deg for p in m.planes] == [135, 225] and ST.scale_check(m, given) == [1]
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0], rel=0.05)


def test_fill_in_pairs_planes_with_strings_by_name_not_position():
    q, irr, cfg, now = _house()
    # entered in reverse order, each naming its string
    given = (C.Plane(30, 225, 3.0, "pv2"), C.Plane(35, 135, 4.0, "pv1"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert [p.string for p in m.planes] == ["pv2", "pv1"]
    assert m.scales_kwp.tolist() == pytest.approx([3.0, 4.0], rel=0.05)
    assert ST.scale_check(m, given) == []
    assert ST.scale_check(m, (C.Plane(35, 135, 2.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"))) == [0]


def test_two_strings_with_identical_geometry_keep_one_plane_each():
    q, irr, cfg, now = _house()
    rng = np.random.default_rng(7)
    q["pv2_kwh"] = _measured(irr, EAST, 2.0, rng)  # the same roof as pv1, a smaller string
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(35, 135, 2.0, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert len(m.planes) == 2 and m.scales_kwp.tolist() == pytest.approx([4.0, 2.0], rel=0.05)
    blind = ST.fit_strings(q, irr, cfg, now)
    assert len(blind.planes) == 2 and blind.planes[0].string == "pv1"
    assert blind.scales_kwp.tolist() == pytest.approx([4.0, 2.0], rel=0.05)


def test_gen_port_is_one_more_string_and_no_strings_means_blind():
    q, irr, cfg, now = _house()
    assert ST.strings_present(q) == ["pv1_kwh", "pv2_kwh"]
    # with gen_kwh carrying a third synthetic plane: strings_present ends with "gen_kwh" and
    # the model has 3 planes
    qg, irr, cfg, now = _house(gen_kwp=1.5)
    assert ST.strings_present(qg) == ["pv1_kwh", "pv2_kwh", "gen_kwh"]
    mg = ST.fit_strings(qg, irr, cfg, now)
    assert len(mg.planes) == 3 and mg.planes[2].string == "gen"
    assert mg.scales_kwp[2] == pytest.approx(1.5, rel=0.05)
    # with every pv{i}_kwh NaN: registry.weekly_update produces the plan-2 blind champion
    for i in range(1, 9):
        q[f"pv{i}_kwh"] = np.full(len(q["ts"]), np.nan)
    assert ST.strings_present(q) == []
    state = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now)
    blind, _, _ = SEP.fit_pv(q, irr, cfg, now)
    assert state.champion is state.challenger
    assert state.champion.planes == blind.planes
    assert state.champion.scales_kwp.tolist() == pytest.approx(blind.scales_kwp.tolist())
    assert state.champion.holdout_error == pytest.approx(blind.holdout_error)
    # the GEN port alone is no strings path: the unmetered DC strings would drop out
    qg, irr, cfg, now = _house(gen_kwp=1.5)
    for i in range(1, 9):
        qg[f"pv{i}_kwh"] = np.full(len(qg["ts"]), np.nan)
    assert ST.strings_present(qg) == ["gen_kwh"]
    state = R.weekly_update(R.initial_state(cfg), qg, irr, cfg, now)
    blind, _, _ = SEP.fit_pv(dict(qg, gen_kwh=np.zeros(len(qg["ts"]))), irr, cfg, now)
    assert state.challenger.planes == blind.planes
    assert state.challenger.scales_kwp.tolist() == pytest.approx(blind.scales_kwp.tolist())


def test_weekly_update_takes_the_strings_path_when_strings_are_present():
    q, irr, cfg, now = _house()
    state = R.weekly_update(R.initial_state(cfg), q, irr, cfg, now)
    m = ST.fit_strings(q, irr, cfg, now)
    assert [p.string for p in state.challenger.planes] == ["pv1", "pv2"]
    assert state.challenger.scales_kwp.tolist() == pytest.approx(m.scales_kwp.tolist())
    assert state.champion is state.challenger and not state.learning


def test_the_month_factor_and_holdout_use_the_measured_sum_with_gen():
    qg, irr, cfg, now = _house(gen_kwp=1.5)
    m = ST.fit_strings(qg, irr, cfg, now)
    # the measured sum includes the GEN port: a model without it would miss 1,5 kWp
    pred = SEP.predict_pv_kw(m, irr, LAT, LON)
    total = (qg["pv1_kwh"] + qg["pv2_kwh"] + qg["gen_kwh"]) * 4.0
    lit = total > 0.05
    assert np.median(pred[lit] / total[lit]) == pytest.approx(1.0, abs=0.03)
    assert m.month_factor[4] == pytest.approx(1.0, abs=0.03)  # May, the fixture's month


def test_strings_present_needs_seven_days_of_daytime_energy():
    q, _, _, _ = _house(days=20)
    day = (q["ts"] - q["ts"][0]) // 86400
    q["pv2_kwh"] = np.where(day < ST.MIN_STRING_DAYS - 1, q["pv2_kwh"], 0.0)
    q["pv3_kwh"] = np.where(day < ST.MIN_STRING_DAYS, q["pv1_kwh"], np.nan)
    assert ST.strings_present(q) == ["pv1_kwh", "pv3_kwh"]


def test_scale_check_skips_planes_without_a_kwp_and_names_strings():
    q, irr, cfg, now = _house()
    given = (C.Plane(35, 135, None, "pv1"), C.Plane(30, 225, 2.0, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert m.scales_kwp[0] == pytest.approx(4.0, rel=0.05)
    assert ST.scale_check(m, given) == [1]
    assert ST.scale_check(m, given, tolerance=0.6) == []
    assert ST.string_labels(given, [0, 1]) == "1, 2"
    assert ST.string_labels((C.Plane(15, 180, 1.0, "gen"), C.Plane(30, 225)), [0, 1]) == "GEN, 2"


def test_strings_survive_the_hourly_path():
    q, irr, cfg, now = _house()
    flat = flatten_to_hours(q)
    assert flat["pv1_kwh"][:4].tolist() == pytest.approx([q["pv1_kwh"][:4].mean()] * 4)
    # the hourly rows: the first quarter of every hour, its mean energy times four
    h = {k: np.asarray(v)[::4] for k, v in flat.items()}
    for c in h:
        if c.endswith("_kwh"):
            h[c] = h[c] * 4.0
    h["step_s"] = np.full(len(h["ts"]), 3600)
    back = hourly_to_quarters(h)
    assert ST.strings_present(back) == ["pv1_kwh", "pv2_kwh"]
    assert back["pv1_kwh"].sum() == pytest.approx(np.nansum(q["pv1_kwh"]))
    m = ST.fit_strings(back, irr, cfg, now)
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0], rel=0.05)


def test_fit_is_deterministic():
    q, irr, cfg, now = _house()
    a, b = ST.fit_strings(q, irr, cfg, now), ST.fit_strings(q, irr, cfg, now)
    assert SEP.to_dict(a) == SEP.to_dict(b)


def test_an_entered_string_without_production_keeps_its_hint_and_is_not_checked():
    # the review probe: pv1 and pv2 measured, the GEN port reads 0 (fewer than 7 production days)
    q, irr, cfg, now = _house()
    q["gen_kwh"] = np.zeros(len(q["ts"]))
    given = (
        C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"), C.Plane(15, 180, 1.0, "gen")
    )
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0, 1.0], rel=0.05)
    assert m.checked == (True, True, False) and ST.scale_check(m, given) == []
    # the forecast keeps the entered GEN plane at its entered kWp
    with_gen = SEP.predict_pv_kw(m, irr, LAT, LON)
    m.scales_kwp[2] = 0.0
    assert (with_gen - SEP.predict_pv_kw(m, irr, LAT, LON)).max() > 0.5
    # a dead pv2 sensor: pv2 keeps its entered 2,0 kWp, pv1 is still checked
    q["pv2_kwh"] = np.full(len(q["ts"]), np.nan)
    given = (C.Plane(35, 135, 3.0, "pv1"), C.Plane(30, 225, 2.0, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert m.scales_kwp[1] == 2.0 and m.checked == (True, False)
    assert ST.scale_check(m, given) == [0]  # pv1: 4,0 fitted against 3,0 entered


def test_stringless_planes_keep_their_hints_when_every_string_is_claimed():
    q, irr, cfg, now = _house()
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"), C.Plane(15, 180, 2.0))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert m.scales_kwp[2] == 2.0 and m.checked == (True, True, False)
    assert ST.scale_check(m, given) == []


def test_scale_check_reads_strings_path_models_only():
    q, irr, cfg, now = _house()
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 4.5, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert m.supervised and ST.scale_check(m, given) == [1]
    blind = dataclasses.replace(m, supervised=False)  # planes that pair, scales that disagree
    assert ST.scale_check(blind, given) == []
    back = SEP.from_dict(SEP.to_dict(m))
    assert back.supervised and back.checked == m.checked and back.dc_to_ac == m.dc_to_ac
    assert "supervised" not in SEP.to_dict(blind)  # blind dicts stay as plan 2 wrote them


def test_dc_strings_pass_the_inverter_stage_and_the_gen_port_does_not():
    stage = pv_dc_to_ac_stage("deye")
    assert stage == (37.9, 2.6) and pv_dc_to_ac_stage("flat") == (0.0, 0.0)
    q, irr, cfg, now = _house(gen_kwp=1.5)
    deye = dataclasses.replace(cfg, loss_profile="deye")
    dc = (q["pv1_kwh"] + q["pv2_kwh"]) * 4.0
    ac_truth = SEP.dc_to_ac_kw(dc, stage) + q["gen_kwh"] * 4.0
    m = ST.fit_strings(q, irr, deye, now)
    assert m.dc_to_ac == stage
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0, 1.5], rel=0.05)  # DC kWp, GEN AC
    pred = SEP.predict_pv_kw(m, irr, LAT, LON)
    lit = ac_truth > 0.05
    assert pred[lit].sum() / ac_truth[lit].sum() == pytest.approx(1.0, abs=0.01)
    measured_dc = dc + q["gen_kwh"] * 4.0
    assert pred[lit].sum() / measured_dc[lit].sum() < 0.97  # the stage takes 3 to 6 %
    flat = ST.fit_strings(q, irr, cfg, now)  # the flat profile: no DC-to-AC loss
    pred = SEP.predict_pv_kw(flat, irr, LAT, LON)
    assert pred[lit].sum() / measured_dc[lit].sum() == pytest.approx(1.0, abs=0.01)
    back = SEP.from_dict(SEP.to_dict(m))
    assert SEP.predict_pv_kw(back, irr, LAT, LON).tolist() == SEP.predict_pv_kw(
        m, irr, LAT, LON
    ).tolist()
