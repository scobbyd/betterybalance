import dataclasses
import json

import numpy as np
import pytest
from betterybalance.core.site import contract as C
from betterybalance.core.site import separate as SEP
from betterybalance.core.site import shading as SH
from betterybalance.core.site import solar
from betterybalance.core.site import strings as ST
from betterybalance.core.site.plant import plane_shape

from tests.test_site_strings import _house


def _clear_year(lat=52.1, lon=5.2):
    ts = np.arange(1767225600, 1767225600 + 365 * 86400, 900, dtype=np.int64)  # 2026, quarters
    zen, az = solar.solar_position(ts, lat, lon)
    ghi = solar.clear_sky_ghi(zen)
    return ts, zen, az, ghi


def _bins(zen, az):
    el = 90 - zen
    return (az // 10).astype(int), np.clip(el // 10, 0, 8).astype(int)


def _irr(ts, zen, ghi):
    dhi = 0.15 * ghi
    cosz = np.cos(np.radians(np.minimum(zen, 89.0)))
    return {
        "ts": ts,
        "ghi_w_m2": ghi,
        "dni_w_m2": np.where(zen < 89, (ghi - dhi) / cosz, 0.0),
        "dhi_w_m2": dhi,
        "temp_c": np.full(len(ts), 15.0),
    }


def test_a_shaded_bin_is_found_and_open_sky_is_one():
    ts, zen, az, ghi = _clear_year()
    el = 90 - zen
    modelled = np.clip(ghi / 1000 * 4.0, 0, None)[:, None]
    shade = (az >= 240) & (az < 250) & (el >= 10) & (el < 20)
    measured = modelled * np.where(shade, 0.5, 1.0)[:, None]
    s = SH.fit_shading(measured, modelled, ts, 52.1, 5.2, ghi, int(ts[-1]))
    assert s.factors.shape == (1, 36, 9) and s.fitted_at == int(ts[-1])
    assert s.factors[0, 24, 1] == pytest.approx(0.5, abs=0.02)
    assert s.factors[0, 18, 3] == pytest.approx(1.0)


def test_factors_are_clipped_sparse_bins_keep_their_own_and_empty_bins_inherit():
    ts, zen, az, ghi = _clear_year()
    ab, eb = _bins(zen, az)
    modelled = np.clip(ghi / 1000 * 4.0, 0, None)
    factor = np.ones(len(ts))
    factor[(ab == 12) & (eb == 1)] = 0.0  # measured 0 in one bin: the clip gives 0,2
    # around the empty bin (18, 2): the row below at 0,5, its own row at 0,6, the row above at
    # 0,7, so the 8 neighbours have the median 0,6; the empty bin's own quarters read 1,0
    for a in (17, 18, 19):
        for e, f in ((1, 0.5), (2, 0.6), (3, 0.7)):
            if (a, e) != (18, 2):
                factor[(ab == a) & (eb == e)] = f
    # the sparse bin (14, 2) reads 1,0 on its 3 clear quarters, its neighbours 0,5
    for a in (13, 14, 15):
        for e in (1, 2, 3):
            if (a, e) != (14, 2):
                factor[(ab == a) & (eb == e)] = 0.5
    ghi = ghi.copy()
    ghi[(ab == 18) & (eb == 2)] = 0.0  # no clear quarter in the empty bin
    sparse = np.flatnonzero((ab == 14) & (eb == 2) & (ghi > 0))
    assert len(sparse) > 3
    ghi[sparse[3:]] = 0.0  # only 3 clear quarters left in the sparse bin
    measured = modelled * factor
    s = SH.fit_shading(measured[:, None], modelled[:, None], ts, 52.1, 5.2, ghi, int(ts[-1]))
    assert s.factors[0, 12, 1] == pytest.approx(0.2)
    assert s.factors[0, 18, 2] == pytest.approx(0.6)  # empty: the neighbours' median
    assert s.factors[0, 14, 2] == pytest.approx(1.0)  # sparse: its own median
    assert s.factors[0, 13, 2] == pytest.approx(0.5)
    assert s.factors[0, 18, 4] == pytest.approx(1.0)  # populated and open
    # a bin the sun never reaches and no populated neighbour: 1,0
    assert s.factors[0, 0, 8] == 1.0


def test_an_empty_bin_takes_its_populated_neighbours_and_a_sparse_bin_its_own():
    # every bin of azimuth 230 to 260 is shaded to 0,4 except bin (24, 2), which reads 1,0;
    # bin (24, 1) has no clear quarter (takes its neighbours' 0,4), bin (24, 2) has 3 (keeps
    # its own 1,0)
    ts, zen, az, ghi = _clear_year()
    ab, eb = _bins(zen, az)
    modelled = np.clip(ghi / 1000 * 4.0, 0, None)
    band = (ab >= 23) & (ab <= 25) & ~((ab == 24) & (eb == 2))
    measured = modelled * np.where(band, 0.4, 1.0)
    ghi = ghi.copy()
    ghi[(ab == 24) & (eb == 1)] = 0.0
    sparse = np.flatnonzero((ab == 24) & (eb == 2) & (ghi > 0))
    ghi[sparse[3:]] = 0.0
    s = SH.fit_shading(measured[:, None], modelled[:, None], ts, 52.1, 5.2, ghi, int(ts[-1]))
    assert s.factors[0, 24, 1] == pytest.approx(0.4)
    assert s.factors[0, 24, 2] == pytest.approx(1.0)
    assert s.factors[0, 23, 1] == pytest.approx(0.4)
    assert s.factors.min() >= 0.2 and s.factors.max() <= 1.0


def test_the_fit_reads_only_quarters_before_fitted_at():
    ts, zen, az, ghi = _clear_year()
    ab, eb = _bins(zen, az)
    modelled = np.clip(ghi / 1000 * 4.0, 0, None)[:, None]
    later = ts >= 1767225600 + 200 * 86400
    measured = modelled * np.where(later & (ab == 18), 0.3, 1.0)[:, None]
    s = SH.fit_shading(measured, modelled, ts, 52.1, 5.2, ghi, 1767225600 + 200 * 86400)
    assert (s.factors == 1.0).all()


def _model(shading=None, dc_to_ac=None):
    return SEP.PvModel(
        (C.Plane(30, 225, string="pv1"),), np.array([4.0]), np.ones(12), None, None, None,
        (1.0, 0.0), 1767225600, 30, supervised=True, checked=(True,), dc_to_ac=dc_to_ac,
        shading=shading,
    )


def test_model_roundtrip_and_prediction_apply_the_profile():
    ts, zen, az, ghi = _clear_year()
    irr = _irr(ts, zen, ghi)
    ab, eb = _bins(zen, az)
    factors = np.ones((1, 36, 9))
    factors[0, 24, 1] = 0.4
    m = _model(SH.Shading(factors, int(ts[0])))
    back = SEP.from_dict(json.loads(json.dumps(SEP.to_dict(m), allow_nan=False)))
    assert SEP.to_dict(back) == SEP.to_dict(m)
    assert back.shading.fitted_at == m.shading.fitted_at
    assert np.array_equal(back.shading.factors, m.shading.factors)
    shaded = SEP.predict_pv_kw(m, irr, 52.1, 5.2)
    plain = SEP.predict_pv_kw(_model(), irr, 52.1, 5.2)
    k = np.flatnonzero((ab == 24) & (eb == 1) & (plain > 0.1))[0]
    assert shaded[k] == pytest.approx(plain[k] * 0.4)
    open_sky = (ab != 24) | (eb != 1)
    assert np.array_equal(shaded[open_sky], plain[open_sky])
    assert 0.0 < SH.share_removed(m, irr, 52.1, 5.2) < 0.05
    assert SH.share_removed(_model(), irr, 52.1, 5.2) == 0.0
    assert SH.share_by_string(m, irr, 52.1, 5.2)["pv1"] == pytest.approx(
        SH.share_removed(m, irr, 52.1, 5.2)
    )


def test_the_profile_shades_the_dc_shape_before_the_inverter_stage():
    ts, zen, az, ghi = _clear_year()
    irr = _irr(ts, zen, ghi)
    factors = np.full((1, 36, 9), 0.5)
    stage = (37.9, 2.6)
    m = _model(SH.Shading(factors, int(ts[0])), dc_to_ac=stage)
    dc = 4.0 * plane_shape(irr, m.planes[0], 52.1, 5.2) * 0.5
    assert SEP.predict_pv_kw(m, irr, 52.1, 5.2) == pytest.approx(SEP.dc_to_ac_kw(dc, stage))


def test_the_profile_applies_on_every_quarter_the_model_predicts():
    # fitted_at is the fit time, not a gate: quarters before it read the profile as well,
    # like the scales fitted with it
    ts, zen, az, ghi = _clear_year()
    irr = _irr(ts, zen, ghi)
    fit_at = int(ts[0]) + 100 * 86400 + 13 * 3600  # 13:00 UTC on the 101st day
    m = _model(SH.Shading(np.full((1, 36, 9), 0.5), fit_at))
    shaded = SEP.predict_pv_kw(m, irr, 52.1, 5.2)
    plain = SEP.predict_pv_kw(_model(), irr, 52.1, 5.2)
    assert (ts < fit_at).any() and (plain > 0).any()
    assert shaded == pytest.approx(plain * 0.5)


def test_a_plan2_dict_without_shading_loads_and_predicts_unchanged():
    ts, zen, az, ghi = _clear_year()
    irr = _irr(ts[:9600], zen[:9600], ghi[:9600])
    blind = SEP.PvModel(
        (C.Plane(35, 180),), np.array([5.0]), np.ones(12), 4.0, 0.1, 1.0, (1.0, 0.0),
        1767225600, 30,
    )
    d = SEP.to_dict(blind)
    assert "shading" not in d  # blind dicts stay as plan 2 wrote them
    back = SEP.from_dict(d)
    assert back.shading is None
    assert SEP.predict_pv_kw(back, irr, 52.1, 5.2).tobytes() == (
        SEP.predict_pv_kw(blind, irr, 52.1, 5.2).tobytes()
    )
    sup = _model()
    assert SEP.to_dict(sup)["shading"] is None and SEP.from_dict(SEP.to_dict(sup)).shading is None


def _shaded_house(first="2026-05-23"):
    """The two-string house of test_site_strings over 60 days, by default around the solstice
    (the holdout's sun positions recur in the fit window); pv2 (30/225) loses 60 % of its
    power when the sun stands between azimuth 230 and 290 below 30 degrees (a tree)."""
    q, irr, cfg, now = _house(days=60, first=first)
    zen, az = solar.solar_position(q["ts"], 52.1, 5.2)
    tree = (az >= 230) & (az < 290) & (90 - zen < 30)
    q["pv2_kwh"] = q["pv2_kwh"] * np.where(tree, 0.4, 1.0)
    return q, irr, cfg, now, tree


def test_fit_strings_fits_the_profile_on_the_shaded_string():
    q, irr, cfg, now, tree = _shaded_house()
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    assert m.shading is not None and m.shading.fitted_at == now
    assert m.shading.factors.shape == (2, 36, 9)
    f1, f2 = m.shading.factors
    # in May and June the sun stands below 30 degrees in the west from azimuth 260 on; the
    # tree bins the fit saw clear come out at 0,4, the ones it never saw clear stay 1,0
    tree_bins = f2[23:29, 0:3]
    cut = tree_bins[tree_bins < 1.0]
    assert len(cut) >= 3 and float(np.median(cut)) == pytest.approx(0.4, abs=0.05)
    assert cut.max() < 0.8  # a sparse edge bin takes its neighbours' median
    assert f2[18:24].min() > 0.95 and f1.min() > 0.95
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0], rel=0.06)
    shares = SH.share_by_string(m, irr, 52.1, 5.2)
    assert shares["pv2"] > 0.05 and shares["pv1"] < shares["pv2"]
    # the profile is fitted on the days before the holdout, so the holdout sees it honestly
    plain = ST.fit_strings(q, irr, cfg, now, planes=given, shading=False)
    assert plain.shading is None and m.holdout_error < plain.holdout_error


def test_an_unshaded_house_keeps_its_fit():
    q, irr, cfg, now = _house(days=60)
    m = ST.fit_strings(q, irr, cfg, now)
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 3.0], rel=0.05)
    assert all(s < 0.03 for s in SH.share_by_string(m, irr, 52.1, 5.2).values())


def test_a_history_use_with_the_profile_matches_measured_energy_better():
    # the load estimate and the drift alarm read the champion on past quarters: the scales
    # lifted by the shading refit must meet their profile there
    q, irr, cfg, now, tree = _shaded_house()
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    plain = ST.fit_strings(q, irr, cfg, now, planes=given, shading=False)
    lifted = dataclasses.replace(m, shading=None)  # the lifted scales without their profile
    measured = float(np.nansum(q["pv1_kwh"] + q["pv2_kwh"]) * 4.0)
    past = np.asarray(q["ts"]) < now

    def err(model):
        return abs(float(SEP.predict_pv_kw(model, irr, 52.1, 5.2)[past].sum()) - measured)

    def tree_bias(model):
        pred = SEP.predict_pv_kw(model, irr, 52.1, 5.2)
        return abs(float((pred - (q["pv1_kwh"] + q["pv2_kwh"]) * 4.0)[tree].mean()))

    assert err(m) < err(plain) < err(lifted)
    assert tree_bias(m) < 0.25 * tree_bias(plain)


def test_a_decision_reads_only_the_fit_of_its_moment():
    # the window-at-decision-time rule: the PV expectation of a decision at t1 comes from the
    # model fitted at t1; data after t1 (here a new shade from t1 on) and the later refit at
    # t2 do not change it
    q, irr, cfg, now, _ = _shaded_house()
    ts = np.asarray(q["ts"])
    t1, t2 = int(ts[0]) + 40 * 86400, now
    window = {k: np.asarray(v)[(ts >= t1) & (ts < t1 + 86400)] for k, v in irr.items()}
    m1 = ST.fit_strings(q, irr, cfg, t1)
    decided = SEP.predict_pv_kw(m1, window, 52.1, 5.2)
    later = dict(q, pv1_kwh=np.where(ts >= t1, q["pv1_kwh"] * 0.5, q["pv1_kwh"]))
    again = ST.fit_strings(later, irr, cfg, t1)
    assert SEP.to_dict(again) == SEP.to_dict(m1)
    assert SEP.predict_pv_kw(again, window, 52.1, 5.2).tobytes() == decided.tobytes()
    m2 = ST.fit_strings(later, irr, cfg, t2)
    assert not np.array_equal(SEP.predict_pv_kw(m2, window, 52.1, 5.2), decided)
    assert m1.shading.fitted_at == t1 and m2.shading.fitted_at == t2


def test_a_may_fit_does_not_carry_the_tree_into_sparse_bins():
    # regression (Task 22 fix 2): fitted on May and early June, scored on the last 14 days of
    # June, the holdout sun stands in bins (26, 3), (29, 0) and (29, 1) that the fit saw clear
    # 6, 7 and 1 times; they keep their own median instead of the tree's 0,4
    q, irr, cfg, now, _ = _shaded_house(first="2026-05-01")
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    plain = ST.fit_strings(q, irr, cfg, now, planes=given, shading=False)
    assert m.holdout_error < 0.5 * plain.holdout_error
    assert m.shading.factors[1][26, 3] > 0.9 and m.shading.factors[1][29, 1] > 0.9


def test_shading_does_not_move_the_scale_check():
    q, irr, cfg, now, _ = _shaded_house()
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 2.5, "pv2"))
    m = ST.fit_strings(q, irr, cfg, now, planes=given)
    plain = ST.fit_strings(q, irr, cfg, now, planes=given, shading=False)
    assert np.array_equal(m.scales_check_kwp, plain.scales_kwp)
    assert not np.array_equal(m.scales_kwp, plain.scales_kwp)  # the refit lifts the scales
    assert ST.scale_check(m, given) == ST.scale_check(plain, given)
    back = SEP.from_dict(json.loads(json.dumps(SEP.to_dict(m))))
    assert np.array_equal(back.scales_check_kwp, m.scales_check_kwp)
    assert ST.scale_check(back, given) == ST.scale_check(m, given)
    assert "scales_check_kwp" not in SEP.to_dict(plain)


def test_the_site_report_states_the_share_per_string():
    from betterybalance.replay import report as RP

    lines = RP._shading_lines({"shading_share": {"pv1": 0.012, "pv2": 0.087}})
    assert lines == [
        "- Shading removes 1,2 % of the modelled energy of pv1",
        "- Shading removes 8,7 % of the modelled energy of pv2",
    ]
    assert RP._shading_lines({}) == []


def test_the_per_string_split_reads_the_profile():
    from betterybalance.live import views as V

    slot = 1793620800  # 2026-11-02 13:00 Amsterdam
    irr = {
        "ts": np.array([slot], dtype=np.int64), "ghi_w_m2": np.array([400.0]),
        "dni_w_m2": np.array([300.0]), "dhi_w_m2": np.array([150.0]),
        "temp_c": np.array([10.0]),
    }
    planes = (C.Plane(35, 180, string="pv1"), C.Plane(30, 90, string="pv2"))
    factors = np.ones((2, 36, 9))
    factors[0] = 0.5  # pv1 under shade at every sun position
    m = SEP.PvModel(
        planes, np.array([4.0, 2.0]), np.ones(12), None, None, None, (1.0, 0.0), slot - 86400,
        30, shading=SH.Shading(factors, slot - 86400),
    )
    plain = dataclasses.replace(m, shading=None)
    total = float(SEP.predict_pv_kw(m, irr, 52.1, 5.2)[0])
    split = V.pv_now(m, irr, slot, 52.1, 5.2, total)["per_string"]
    base = V.pv_now(plain, irr, slot, 52.1, 5.2, float(SEP.predict_pv_kw(plain, irr, 52.1, 5.2)[0]))
    assert sum(split.values()) == pytest.approx(total, abs=2e-3)
    assert split["pv1"] == pytest.approx(base["per_string"]["pv1"] * 0.5, abs=2e-3)
    assert split["pv2"] == pytest.approx(base["per_string"]["pv2"], abs=2e-3)


def test_the_site_summary_carries_the_share_only_with_a_profile():
    from types import SimpleNamespace

    from betterybalance.replay import phases as PH

    from tests.test_site_separate import _cfg

    ts, zen, az, ghi = _clear_year()
    irr = _irr(ts[:9600], zen[:9600], ghi[:9600])
    factors = np.full((1, 36, 9), 0.9)

    def site(model):
        state = SimpleNamespace(champion=model, learned_days=30)
        return SimpleNamespace(state=state, drift_seen=[False], irr_q=irr)

    shaded = PH.site_summary(_cfg(), site(_model(SH.Shading(factors, int(ts[-1])))), None, 0, 0)
    assert shaded["shading_share"]["pv1"] == pytest.approx(0.1)
    assert "shading_share" not in PH.site_summary(_cfg(), site(_model()), None, 0, 0)


def test_hourly_rows_give_the_profile_at_hour_resolution():
    from betterybalance.replay.disaggregate import flatten_to_hours, hourly_to_quarters

    q, irr, cfg, now, tree = _shaded_house()
    flat = flatten_to_hours(q)
    h = {k: np.asarray(v)[::4] for k, v in flat.items()}
    for c in h:
        if c.endswith("_kwh"):
            h[c] = h[c] * 4.0
    h["step_s"] = np.full(len(h["ts"]), 3600)
    back = hourly_to_quarters(h)
    assert SEP.hourly_origin_rows(back).all()
    given = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(30, 225, 3.0, "pv2"))
    m = ST.fit_strings(back, irr, cfg, now, planes=given)
    plain = ST.fit_strings(back, irr, cfg, now, planes=given, shading=False)
    # the tree's hours are smeared over neighbouring bins, but the core of it is found
    assert float(m.shading.factors[1][27:29, 0:3].min()) < 0.6
    assert SH.share_by_string(m, irr, 52.1, 5.2)["pv2"] > 0.04
    assert m.holdout_error < plain.holdout_error
