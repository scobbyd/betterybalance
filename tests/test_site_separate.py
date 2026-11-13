import datetime as dt
import json
import math

import numpy as np
import pytest
from betterybalance.core.site import contract as C
from betterybalance.core.site import plant as P
from betterybalance.core.site import separate as SEP
from betterybalance.core.site import solar as S
from betterybalance.core.site.contract import Plane
from betterybalance.core.site.load import forecast_load_kw, load_median
from betterybalance.core.timegrid import ZONE, local_calendar

LAT, LON = 52.1, 5.2


def _grid(first: str, days: int) -> np.ndarray:
    start = int(dt.datetime.fromisoformat(first + "T00:00:00").replace(tzinfo=ZONE).timestamp())
    end_day = (dt.date.fromisoformat(first) + dt.timedelta(days=days)).isoformat()
    end = int(dt.datetime.fromisoformat(end_day + "T00:00:00").replace(tzinfo=ZONE).timestamp())
    return np.arange(start, end, 900, dtype=np.int64)


def synthetic_house(
    planes,
    kwps,
    first="2026-05-01",
    days=60,
    seed=1,
    load_base=0.3,
    noise=0.2,
    ac_cap_kw=None,
    irr_noise=0.0,
):
    """A synthetic house: quarters q, the irradiance the fit gets, the true PV and load.

    irr_noise > 0 multiplies the returned irradiance by a seeded factor 1 + irr_noise * N(0, 1)
    per quarter (clipped at 0), a stand-in for a forecast archive; the true PV uses the clean
    irradiance.
    """
    rng = np.random.default_rng(seed)
    ts = _grid(first, days)
    z, _ = S.solar_position(ts, LAT, LON)
    cs = S.clear_sky_ghi(z)
    day_index = (ts - ts[0]) // 86400
    kt = rng.uniform(0.3, 1.0, size=days)[day_index] * (1 - 0.3 * rng.random(len(ts)))
    ghi = cs * kt
    dhi = ghi * np.where(kt > 0.6, 0.25, 0.7)
    cosz = np.cos(np.radians(np.minimum(z, 89.0)))
    irr = {
        "ts": ts,
        "ghi_w_m2": ghi,
        "dni_w_m2": np.where(z < 89, (ghi - dhi) / cosz, 0.0),
        "dhi_w_m2": dhi,
        "temp_c": np.full(len(ts), 18.0),
    }
    pv = sum(k * P.plane_shape(irr, pl, LAT, LON) for pl, k in zip(planes, kwps, strict=True))
    if ac_cap_kw is not None:
        pv = np.minimum(pv, ac_cap_kw)
    cal = local_calendar(ts)
    load = (
        load_base
        + np.where((cal["hour"] >= 7) & (cal["hour"] < 9), 1.0, 0.0)
        + np.where((cal["hour"] >= 18) & (cal["hour"] < 22), 1.5, 0.0)
        + rng.normal(0, noise, len(ts)).clip(-0.25)
    )
    r = load - pv
    n = len(ts)
    if irr_noise > 0.0:
        factor = np.clip(1.0 + irr_noise * rng.normal(0.0, 1.0, n), 0.0, None)
        irr = {k: (v * factor if k.endswith("_w_m2") else v) for k, v in irr.items()}
    q = {
        "ts": ts,
        "step_s": np.full(n, 900),
        "missing": np.zeros(n, bool),
        "grid_import_kwh": np.clip(r, 0, None) * 0.25,
        "grid_export_kwh": np.clip(-r, 0, None) * 0.25,
        "batt_charge_kwh": np.zeros(n),
        "batt_discharge_kwh": np.zeros(n),
        "soc_pct": np.full(n, 50.0),
        "soc_min_pct": np.full(n, 50.0),
        "soc_max_pct": np.full(n, 50.0),
        "load_kwh": np.full(n, np.nan),
        "gen_kwh": np.full(n, np.nan),
        "pv_kwh": np.full(n, np.nan),
    }
    return q, irr, pv, load


def _cfg(**pv):
    return C.from_dict(
        {
            "latitude": LAT,
            "longitude": LON,
            "pack": {
                "capacity_kwh": 48,
                "charge_kw": 12,
                "discharge_kw": 12,
                "soc_floor": 0.1,
                "soc_ceiling": 1.0,
            },
            "pv": pv,
        }
    )


def _within(a, b, deg):
    return abs((a - b + 180) % 360 - 180) <= deg


def test_two_plane_house_is_recovered_blind():
    q, irr, pv, load = synthetic_house((Plane(30, 180), Plane(40, 90)), (5.0, 3.0))
    model, lm, sessions = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert len(model.planes) == 2 and sessions == []
    got = sorted(zip(model.planes, model.scales_kwp), key=lambda x: x[0].azimuth_deg)
    (pe, ke), (ps, ks) = got
    assert _within(pe.azimuth_deg, 90, 15) and abs(pe.tilt_deg - 40) <= 15
    assert ke == pytest.approx(3.0, rel=0.15)
    assert _within(ps.azimuth_deg, 180, 15) and abs(ps.tilt_deg - 30) <= 15
    assert ks == pytest.approx(5.0, rel=0.15)
    est = SEP.predict_pv_kw(model, irr, LAT, LON)
    day = pv > 0.05
    assert np.mean(np.abs(est[day] - pv[day])) <= 0.2 * pv[day].mean()
    assert abs(np.mean(est[day] - pv[day])) <= 0.05 * pv[day].mean()
    assert model.holdout_error < 0.7 * model.null_error
    assert np.all(np.abs(model.month_factor - 1.0) <= 0.15)


def test_single_plane_house_keeps_one_plane_and_given_planes_fit_only_scales():
    q, irr, pv, _ = synthetic_house((Plane(35, 200),), (4.0,))
    model, _, _ = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert len(model.planes) == 1 and model.scales_kwp[0] == pytest.approx(4.0, rel=0.15)
    fixed, _, _ = SEP.fit_pv(
        q,
        irr,
        _cfg(planes=[{"tilt_deg": 35, "azimuth_deg": 200}], correction=0.5),
        now=int(q["ts"][-1] + 900),
    )
    assert fixed.planes == (Plane(35, 200),)
    assert fixed.scales_kwp[0] == pytest.approx(4.0, rel=0.1)
    assert fixed.stability == (1.0, 0.0)
    q2, irr2, _, _ = synthetic_house((Plane(30, 180), Plane(40, 90)), (5.0, 3.0))
    two = [{"tilt_deg": 30, "azimuth_deg": 180}, {"tilt_deg": 40, "azimuth_deg": 90}]
    moved, _, _ = SEP.fit_pv(
        q2, irr2, _cfg(planes=two, correction=1.0), now=int(q2["ts"][-1] + 900)
    )
    assert moved.scales_kwp.tolist() == pytest.approx([5.0, 3.0], rel=0.05)


def test_pv_none_skips_the_fit_and_load_equals_residual():
    q, irr, _, load = synthetic_house((Plane(35, 200),), (0.0,))
    model, lm, _ = SEP.fit_pv(q, irr, _cfg(none=True), now=int(q["ts"][-1] + 900))
    assert model.planes == () and SEP.predict_pv_kw(model, irr, LAT, LON).sum() == 0.0
    at_20 = np.median(load[local_calendar(q["ts"])["hour"] == 20])
    assert lm.profile_kw[80] == pytest.approx(at_20, abs=0.15)


def test_masks():
    q = {"soc_max_pct": np.array([96.0, 96.0, 50.0]), "grid_export_kwh": np.array([0.0, 0.1, 0.0])}
    assert SEP.curtailment_mask(q).tolist() == [True, False, False]
    pv = np.array([0.5, 1.0, 2.0, 2.99, 3.0, 2.98, 3.01, 1.0])
    mask, level = SEP.clipping_plateau(pv)
    assert mask.tolist() == [False, False, False, True, True, True, True, False]
    assert level == pytest.approx(2.995, abs=0.01)
    assert SEP.clipping_plateau(np.array([1.0, 2.0, 3.0]))[1] is None


def test_month_factor_without_data_is_one_and_huber_fit_is_non_negative():
    f = SEP.month_factors(np.array([2.0, 2.0]), np.array([1.0, 1.0]), np.array([6, 6]))
    assert f[5] > 1.0 and f[0] == 1.0 and f[11] == 1.0
    scales = SEP.huber_fit(
        np.column_stack([np.ones(10), -np.ones(10)]), np.full(10, 2.0), delta=1.0
    )
    assert scales[0] == pytest.approx(2.0, abs=0.05) and scales[1] == 0.0


def _oracle_deficit(q, load, pv):
    """d = L - r with L the full-history slot median of the true load."""
    ts = q["ts"]
    lm = load_median(load, ts, int(ts[-1] + 900), days=61)
    return forecast_load_kw(lm, ts) - (load - pv)


def test_clipping_is_learned_only_where_the_model_expects_a_peak():
    house = ((Plane(30, 180),), (5.0,))
    q, irr, free, load = synthetic_house(*house, noise=0.02)
    cap = 0.6 * free.max()
    qc, _, capped, load_c = synthetic_house(*house, noise=0.02, ac_cap_kw=cap)
    ts = q["ts"]
    z, _ = S.solar_position(ts, LAT, LON)
    lit, none = S.daylight(z), np.zeros(len(ts), bool)
    mask, ac_max = SEP.clipping_mask(_oracle_deficit(qc, load_c, capped), free, ts, lit, none)
    assert ac_max == pytest.approx(cap, rel=0.05)
    assert (
        mask.sum() >= 20 and np.mean(free[mask] >= cap) >= 0.95 and free[mask].min() >= 0.95 * cap
    )
    mask, ac_max = SEP.clipping_mask(_oracle_deficit(q, load, free), free, ts, lit, none)
    assert ac_max is None and not mask.any()
    model, _, _ = SEP.fit_pv(qc, irr, _cfg(), now=int(ts[-1] + 900))
    assert model.ac_max_kw == pytest.approx(cap, rel=0.05)
    assert SEP.fit_pv(q, irr, _cfg(), now=int(ts[-1] + 900))[0].ac_max_kw is None


def test_short_history_has_no_holdout_and_reports_unknown_stability():
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=6)
    model, _, _ = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert len(model.planes) >= 1
    assert model.holdout_error is None and model.null_error is None
    assert all(math.isnan(x) for x in model.stability)
    text = json.dumps(SEP.to_dict(model), allow_nan=False)
    back = SEP.from_dict(json.loads(text))
    assert back.holdout_error is None and back.null_error is None
    assert all(math.isnan(x) for x in back.stability) and back.planes == model.planes


def test_ten_day_history_validates_on_a_five_day_holdout():
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=10)
    model, _, _ = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert model.n_days == 10
    assert math.isfinite(model.holdout_error) and math.isfinite(model.null_error)
    assert model.holdout_error < 0.7 * model.null_error
    from betterybalance.core.site.registry import holdout_passes

    # halves of 3 and 2 lit days: not tested
    assert all(math.isnan(x) for x in model.stability)
    assert holdout_passes(model)


def test_stability_runs_only_on_holdout_halves_of_five_lit_days():
    assert SEP.MIN_STABILITY_HALF_DAYS == 5
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=20)  # holdout 10: halves 5, 5
    model, _, _ = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert all(math.isfinite(x) for x in model.stability)
    q, irr, _, _ = synthetic_house((Plane(30, 180),), (5.0,), days=18)  # holdout 9: halves 5, 4
    model, _, _ = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert math.isfinite(model.holdout_error) and all(math.isnan(x) for x in model.stability)


def test_flat_split_hourly_house_recovers_the_scale():
    from betterybalance.replay.disaggregate import hourly_to_quarters

    q, irr, _, _ = synthetic_house((Plane(40, 90),), (4.0,), days=40)
    n = len(q["ts"]) // 4
    h = {
        c: np.asarray(v, dtype=float).reshape(n, 4).sum(axis=1)
        for c, v in q.items()
        if c not in ("ts", "step_s", "missing")
    }
    for c in ("soc_pct", "soc_min_pct", "soc_max_pct"):
        h[c] = h[c] / 4.0
    h["ts"], h["step_s"] = q["ts"][::4], np.full(n, 3600)
    flat = hourly_to_quarters(h)
    assert flat["hourly_origin"].all() and len(flat["ts"]) == len(q["ts"])
    model, _, _ = SEP.fit_pv(flat, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert len(model.planes) == 1 and model.scales_kwp.sum() == pytest.approx(4.0, rel=0.05)
    peaked = SEP.predict_pv_kw(model, irr, LAT, LON)
    hourly = SEP.predict_pv_kw(model, irr, LAT, LON, hour_mean_rows=np.ones(len(q["ts"]), bool))
    i = 4 * int(np.argmax(peaked[::4]))  # the first quarter of a morning hour of the east plane
    assert hourly[i : i + 4].tolist() == pytest.approx([peaked[i : i + 4].mean()] * 4)
    assert np.ptp(peaked[i : i + 4]) > 0.0


def test_noisy_irradiance_does_not_attenuate_the_scale():
    # 25 % irradiance noise per quarter: the slot-centred scale alone gave 3,27 of 5 kWp; the
    # night-anchored level fit gives the scale within 10 %. A noisy two-plane house still
    # merges to one plane (geometry, not scale), so this test uses one plane.
    q, irr, pv, _ = synthetic_house((Plane(30, 180),), (5.0,), irr_noise=0.25)
    model, _, _ = SEP.fit_pv(q, irr, _cfg(), now=int(q["ts"][-1] + 900))
    assert len(model.planes) == 1 and abs(model.planes[0].azimuth_deg - 180) <= 15
    assert model.scales_kwp.sum() == pytest.approx(5.0, rel=0.10)
    given, _, _ = SEP.fit_pv(
        q, irr, _cfg(planes=[{"tilt_deg": 30, "azimuth_deg": 180}]), now=int(q["ts"][-1] + 900)
    )
    assert given.scales_kwp[0] == pytest.approx(5.0, rel=0.10)


def test_pv_model_dicts_keep_the_string_and_load_without_it():
    named = SEP.PvModel(
        planes=(Plane(35, 135, 4.0, "pv1"), Plane(30, 225)),
        scales_kwp=np.array([4.0, 3.0]),
        month_factor=np.ones(12),
        ac_max_kw=None,
        holdout_error=None,
        null_error=None,
        stability=(np.nan,),
        fitted_at=0,
        n_days=1,
    )
    d = SEP.to_dict(named)
    assert d["planes"][0]["string"] == "pv1" and "string" not in d["planes"][1]
    assert SEP.from_dict(d).planes == named.planes
