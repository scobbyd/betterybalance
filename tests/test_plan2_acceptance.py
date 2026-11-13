import csv
import json
from pathlib import Path

import numpy as np
import pytest
from betterybalance.core.site.contract import Plane
from betterybalance.core.site.flows import COLUMNS

from tests.test_site_separate import LAT, LON, _cfg, synthetic_house
from tools.study import plan2_acceptance as A


def test_merged_residual_is_blind_and_flags_gaps():
    q, irr, pv, load = synthetic_house((Plane(35, 135),), (4.0,), days=20)
    q["load_kwh"] = load * 0.25
    q["gen_kwh"] = pv * 0.25
    q["gen_kwh"][5] = np.nan
    merged, truth = A.merged_residual(q, scale=0.5)
    assert np.isnan(merged["gen_kwh"]).all() and np.isnan(merged["pv_kwh"]).all()
    assert (merged["soc_pct"] == 50.0).all()
    assert merged["missing"][5] and not merged["missing"][6]
    assert truth[100] == pytest.approx(pv[100] * 0.5)
    assert merged["grid_import_kwh"][100] - merged["grid_export_kwh"][100] == pytest.approx(
        (load[100] - 0.5 * pv[100]) * 0.25
    )


def test_compare_geometry_and_hourly_error():
    from betterybalance.core.site.separate import PvModel

    m = PvModel((Plane(30, 145),), np.array([2.5]), np.ones(12), None, 0.5, 1.0, (1.0, 0.0), 0, 20)
    g = A.compare_geometry(m, [(35.0, 135.0, 4.0), (35.0, 250.0, 6.0)])  # synthetic planes
    assert g[0]["tilt_delta"] == pytest.approx(5.0)
    assert g[0]["azimuth_delta"] == pytest.approx(10.0)
    assert g[0]["scale_ratio"] == pytest.approx(2.5 / 4.0)
    ts = np.arange(0, 8 * 3600, 900)
    truth = np.tile([0, 1, 2, 1], 8).astype(float)
    err = A.hourly_error(truth * 1.1, truth, ts, truth > 0)
    assert err["bias_share"] == pytest.approx(0.1) and err["mae_share"] == pytest.approx(0.1)


def test_effective_kwp_recovers_the_synthetic_scale():
    q, irr, pv, _ = synthetic_house((Plane(35, 135),), (4.0,), days=20)
    eff = A.effective_kwp(pv, irr, Plane(35, 135), LAT, LON, q["ts"])
    assert eff == pytest.approx(4.0, rel=1e-3)
    assert A.effective_kwp(pv * 0.5, irr, Plane(35, 135), LAT, LON, q["ts"]) == pytest.approx(
        2.0, rel=1e-3
    )


def test_learning_pass_day_on_a_synthetic_house():
    q, irr, pv, load = synthetic_house((Plane(35, 135),), (4.0,), first="2026-09-02", days=18)
    q["load_kwh"] = load * 0.25
    q["gen_kwh"] = pv * 0.25
    merged, _ = A.merged_residual(q)
    trace = []
    day = A.learning_pass_day(merged, irr, _cfg(), "2026-09-02", max_days=18, trace=trace)
    assert day is not None and "2026-09-15" <= day <= "2026-09-19"
    assert trace[0]["day"] == "2026-09-15" and trace[-1]["day"] == day
    assert trace[-1]["champion_passes"] and trace[-1]["holdout_ratio"] < 0.7
    assert A.learning_pass_day(merged, irr, _cfg(), "2026-09-02", max_days=10) is None


def _write_flows(table, path):
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(COLUMNS)
        for i in range(len(table["ts"])):
            w.writerow(
                [
                    int(table[c][i]) if c in ("ts", "step_s") else repr(float(table[c][i]))
                    for c in COLUMNS
                ]
            )


def _house_flows(first, days, step_s):
    # a plane on the search grid: an off-grid plane may split into two grid planes
    q, irr, pv, load = synthetic_house((Plane(30, 225),), (4.0,), first=first, days=days, noise=0.1)
    n = len(q["ts"])
    t = {c: np.full(n, np.nan) for c in COLUMNS}
    t["ts"], t["step_s"] = q["ts"], np.full(n, 900)
    for c in ("grid_import_kwh", "grid_export_kwh", "batt_charge_kwh", "batt_discharge_kwh"):
        t[c] = np.zeros(n)
    t["soc_pct"] = np.full(n, 70.0)
    t["load_kwh"], t["gen_kwh"], t["pv_kwh"] = load * 0.25, pv * 0.25, np.zeros(n)
    if step_s == 3600:
        k = n // 4
        t = {c: (v[::4] if c == "ts" else v.reshape(k, 4).sum(axis=1)) for c, v in t.items()}
        t["step_s"] = np.full(k, 3600)
        t["soc_pct"] = np.full(k, 70.0)
    return t, irr


def test_merged_micro_end_to_end_on_synthetic_files(tmp_path):
    minute, irr_a = _house_flows("2026-09-02", 16, 900)
    hourly, irr_b = _house_flows("2026-06-01", 16, 3600)
    _write_flows(minute, tmp_path / "minute.csv")
    _write_flows(hourly, tmp_path / "hourly.csv")
    with open(tmp_path / "irr.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        cols = ["ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c"]
        w.writerow(["ts", *cols])
        for irr in (irr_b, irr_a):
            for i in range(len(irr["ts"])):
                w.writerow([int(irr["ts"][i]), *(repr(float(irr[c][i])) for c in cols)])
    planes = {"reference": {"micro": {"tilt_deg": 30, "azimuth_deg": 225}}}
    (tmp_path / "planes.json").write_text(json.dumps(planes))
    out_md = tmp_path / "report.md"
    out_md.write_text("# Report\n\nKept text.\n")
    res = A.merged_micro(
        tmp_path / "minute.csv",
        tmp_path / "hourly.csv",
        tmp_path / "irr.csv",
        tmp_path / "planes.json",
        LAT,
        LON,
        tmp_path / "out.json",
        out_md,
        hourly_to="2026-06-17",
    )
    checks = {c["name"]: c for c in res["checks"]}
    assert checks["plane count (minute)"]["pass"] and checks["azimuth (minute)"]["pass"]
    assert checks["scale (minute)"]["pass"] and checks["hourly error (hourly)"]["pass"]
    # proportional scale 0,5 is a diagnostic row: the night anchor holds part of this house's
    # evening peak, so at half the PV it reads about 0,59 (the known risk of separate.py)
    assert checks["proportional scale 2"]["pass"]
    assert checks["curtailment"]["pass"] and checks["learning by 2026-09-15"]["pass"]
    assert res["detection_floor"] is not None
    saved = json.loads((tmp_path / "out.json").read_text())
    assert saved["runs"]["minute_1"]["planes"] == 1
    md = out_md.read_text()
    assert md.startswith("# Report") and "Kept text." in md and A.ATTRIBUTION in md
    assert "52.1" not in md and "52,1" not in md and "pass" in md
    A.merged_micro(  # a re-run replaces its own section and keeps the rest
        tmp_path / "minute.csv",
        tmp_path / "hourly.csv",
        tmp_path / "irr.csv",
        tmp_path / "planes.json",
        LAT,
        LON,
        tmp_path / "out.json",
        out_md,
        hourly_to="2026-06-17",
    )
    assert out_md.read_text() == md


def test_blind_fit_report_renders_without_fingerprints():
    from betterybalance.replay.report import ATTRIBUTION

    from tests.test_shipped_model import _scan

    result = {
        "planes": [
            {"tilt_deg": 35.0, "azimuth_deg": 135.0},
            {"tilt_deg": 35.0, "azimuth_deg": 250.0},
        ],  # synthetic reference planes
        "scale_ratio": [0.97, 0.97],
        "hourly_error": 0.12,
        "learning_pass_day": "2026-09-15",
    }
    text = A.render_blind_fit(result)
    assert ATTRIBUTION in text and "0,97" in text and "2026-09-15" in text and _scan(text) == []


# --- Task 25: replay-windows -----------------------------------------------------------------


def _minute_flows(first_ts: int, days: int, soc_nan_minutes: int = 0) -> dict:
    """Synthetic minute flows: 0,5 kW import, SoC 50 %, NaN SoC on the first minutes."""
    mts = np.arange(first_ts, first_ts + days * 86400, 60, dtype=np.int64)
    n = len(mts)
    t = {c: np.zeros(n) for c in COLUMNS}
    t["ts"], t["step_s"] = mts, np.full(n, 60, dtype=np.int64)
    t["grid_import_kwh"] = np.full(n, 0.5 / 60)
    for c in ("soc_pct", "soc_min_pct", "soc_max_pct"):
        t[c] = np.full(n, 50.0)
        t[c][:soc_nan_minutes] = np.nan
    return t


def test_replay_windows_bounds_cache_paths_and_start_anchor(tmp_path):
    import datetime as dt

    from betterybalance.core.site.flows import to_quarters
    from betterybalance.replay import __main__ as M
    from betterybalance.replay import bands as BC
    from betterybalance.replay.disaggregate import HOUR_S

    from tests.test_features import _synthetic_master

    live, ab = A.REPLAY_WINDOWS
    assert (live.name, live.first, live.last, live.tiers) == (
        "live",
        "2026-09-02",
        None,  # the last local day with settled prices in the master
        ("A", "B", "C", "D"),
    )
    assert (ab.name, ab.first, ab.last, ab.tiers) == ("ab", "2026-02-01", "2026-08-31", ("A", "B"))
    start, end = M.window_bounds(ab.first, ab.last)
    assert end - start == 212 * 86400 - 3600  # 212 local days, the DST start in March
    assert A.window_dir(tmp_path, "ab") == tmp_path / "ab"

    master = _synthetic_master(days=20)
    s, e = int(master["ts"][96 * 10]), int(master["ts"][96 * 12])
    key4, path4 = M.cache_path(master, s, e, tmp_path / "cache", ("A", "B", "C", "D"))
    key2, path2 = M.cache_path(master, s, e, tmp_path / "cache", ("A", "B"))
    lo, hi = M.cache_bounds(master, s, e)
    assert key2 == BC.cache_key(master, lo, hi, 7, 365, tiers=("A", "B")) and key2 != key4
    assert path2 == tmp_path / "cache" / f"bands_{key2[:16]}.npz" and path2 != path4

    # the start anchor: a NaN SoC at the first local midnight moves the start a day on
    first = "2026-09-02"
    t0 = A.local_midnight(first)
    minute = _minute_flows(t0 + 240, 3, soc_nan_minutes=600)
    q = to_quarters(minute)
    assert A.anchored_first_day(q, minute, first, "2026-09-03", 48.0) == "2026-09-03"
    assert A.anchored_first_day(q, minute, "2026-09-03", "2026-09-04", 48.0) == "2026-09-03"
    with pytest.raises(ValueError, match="end anchor"):
        A.anchored_first_day(q, minute, first, "2026-09-05", 48.0)
    # hourly rows: the anchor is the quarter that ends at the start
    hts = np.arange(t0, t0 + 2 * 86400, HOUR_S, dtype=np.int64)
    hourly = {c: np.zeros(len(hts)) for c in COLUMNS}
    hourly["ts"], hourly["step_s"] = hts, np.full(len(hts), HOUR_S, dtype=np.int64)
    hourly["soc_pct"] = np.full(len(hts), 50.0)
    hq = A.ab_quarters(hourly, None, [], 52.1, 5.2)
    assert hq["hourly_origin"].all() and len(hq["ts"]) == 4 * len(hts)
    assert A.anchored_first_day(hq, None, first, "2026-09-03", 48.0) == "2026-09-03"
    assert dt.date.fromisoformat(A._plus_days(first, 1)) == dt.date(2026, 9, 3)


def test_ab_quarters_shape_the_pv_columns_by_the_plant_planes():
    from betterybalance.replay.disaggregate import HOUR_S

    from tests.test_site_separate import synthetic_house

    q, irr, pv, _ = synthetic_house((Plane(35, 180),), (4.0,), days=2)
    hts = np.asarray(q["ts"], dtype=np.int64)[::4]
    hourly = {c: np.zeros(len(hts)) for c in COLUMNS}
    hourly["ts"], hourly["step_s"] = hts, np.full(len(hts), HOUR_S, dtype=np.int64)
    hourly["soc_pct"] = np.full(len(hts), 50.0)
    hourly["pv_kwh"] = pv.reshape(-1, 4).sum(axis=1) * 0.25
    shaped = A.ab_quarters(hourly, irr, [Plane(35, 180, 4.0)], LAT, LON)
    flat = A.ab_quarters(hourly, irr, [], LAT, LON)
    noon = int(np.argmax(hourly["pv_kwh"]))
    rows = slice(4 * noon, 4 * noon + 4)
    assert shaped["pv_kwh"][rows].sum() == pytest.approx(hourly["pv_kwh"][noon])
    assert np.allclose(flat["pv_kwh"][rows], hourly["pv_kwh"][noon] / 4)
    shape = A.plane_shape(
        A.align(irr, shaped["ts"], list(A.IRRADIANCE_COLS)), Plane(35, 180), LAT, LON
    )
    want = shape[rows] / shape[rows].sum() * hourly["pv_kwh"][noon]
    assert np.allclose(shaped["pv_kwh"][rows], want) and not np.allclose(want, want[0])
    assert np.array_equal(shaped["grid_import_kwh"], flat["grid_import_kwh"])


def test_greed_failure_text_names_the_threshold_of_the_impl_that_ran():
    fail = {"applies": 1, "ok": 0, "first_fail": "2026-09-03", "fails": []}
    fail["fails"] = [
        {
            "day": "2026-09-03",
            "soc_kwh": 48.0,
            "sale_threshold": 150.0,
            "night_median_feed": 100.0,
            "gap": 50.0,
        }
    ]
    ranking = "\n".join(A._greed_fails(A.REPLAY_WINDOWS[0], "high", fail, "ranking"))
    dp = "\n".join(A._greed_fails(A.REPLAY_WINDOWS[0], "high", fail, "dp"))
    assert "minimum over the ranked sale slots" in ranking and "value function" in dp
    assert "open question" in ranking and "open question" in dp and "50,0" in dp


def test_greed_scenarios_read_the_summer_rows_and_the_low_soc_misses():
    g = {
        "day": np.array(["2026-07-01", "2026-07-02", "2026-07-03", "2026-08-01"], dtype=object),
        "soc_kwh": np.array([45.0, 46.0, 6.0, 44.0]),
        "charge_threshold": np.array([50.0, 40.0, 30.0, 45.0]),
        "sale_threshold": np.array([120.0, 130.0, 150.0, 300.0]),
        "night_median_take": np.array([60.0, 70.0, 80.0, 60.0]),
        "night_median_feed": np.array([140.0, 150.0, 90.0, 100.0]),
        "surplus_tomorrow_kwh": np.array([40.0, 30.0, 10.0, 35.0]),
        "low_soc_applies": np.array([False, False, True, False]),
        "low_soc_ok": np.array([True, True, False, True]),
        "high_soc_applies": np.array([True, True, False, True]),
        "high_soc_ok": np.array([True, True, True, False]),
    }
    s = A.greed_scenarios(g, month=7)
    assert s["summer"]["applies"] == 2 and s["summer"]["ok"] == 2
    assert [r["day"] for r in s["summer"]["examples"]] == ["2026-07-01", "2026-07-02"]
    assert s["low"]["applies"] == 1 and s["low"]["ok"] == 0
    assert s["low"]["fails"][0]["day"] == "2026-07-03"
    assert s["low"]["fails"][0]["gap"] == pytest.approx(50.0)  # night take - charge threshold
    assert s["high"]["applies"] == 3 and s["high"]["ok"] == 2
    assert [f["day"] for f in s["high"]["fails"]] == ["2026-08-01"]
    assert s["high"]["fails"][0]["gap"] == pytest.approx(200.0)  # sale threshold - night feed


def test_combined_ladder_and_report_assembly(tmp_path):
    from betterybalance.replay import engine as E
    from betterybalance.replay import report as R

    from tests.test_replay_engine import _inputs
    from tests.test_shipped_model import _scan

    inputs = _inputs()
    result = E.replay(inputs)
    rungs = [
        r for r in R.ladder_rungs() if r[0] in (R.DEFAULT_RUNG, "dp q25/q75 kappa 0,5 penalty none")
    ]
    rows = R.run_ladder(inputs, rungs=rungs)
    ab_rows = [
        dict(r, days=np.array(["2026-07-0%d" % (i + 1) for i in range(3)], dtype=object))
        for r in rows
    ]
    combined = A.combine_ladders(rows, ab_rows)
    assert [r["name"] for r in combined] == [r["name"] for r in rows]
    assert len(combined[0]["days"]) == 6
    assert combined[0]["diff_eur"] == pytest.approx(rows[0]["diff_eur"] + ab_rows[0]["diff_eur"])
    bias = R.measure_hourly_bias(inputs, result)
    states = {
        "live": A.window_state(
            A.REPLAY_WINDOWS[0],
            "2026-11-18",
            "2026-11-20",
            result,
            rows,
            bias,
            {"key": "k", "folds": 2, "build_s": 60.0, "reused": False},
            {"rung_s": [1.0, 2.0], "replay_s": 1.0},
        ),
        "ab": A.window_state(
            A.REPLAY_WINDOWS[1],
            "2026-11-18",
            "2026-11-20",
            result,
            ab_rows,
            R.HOURLY_INPUT,
            {"key": "k", "folds": 2, "build_s": 60.0, "reused": True},
            {"rung_s": [1.0, 2.0], "replay_s": 1.0},
        ),
    }
    A.save_state(tmp_path / "out", states["ab"])
    loaded = A.load_state(tmp_path / "out", "ab")
    assert loaded["spec"] is A.REPLAY_WINDOWS[1] and loaded["first"] == "2026-11-18"
    report = tmp_path / "report.md"
    out = A.assemble_windows(states, inputs.cfg, tmp_path / "out", report)
    text = report.read_text()
    assert out["selection"]["chosen"] in {r["name"] for r in rows}
    for needle in (
        R.ATTRIBUTION,
        "## Headline",
        "Live window",
        "A/B window",
        "## Ladder and the chosen defaults",
        out["selection"]["reason"],
        "## Greed check",
        "## Hourly bias",
        R.HOURLY_INPUT_SENTENCE,
        "## Pooled summary",
        "## Site models",
    ):
        assert needle in text, needle
    assert "- Plane 1:" not in text  # no absolute plane geometry in the report
    assert "On its own, the chosen rung" in text
    assert "chosen on the same live days that score them" in text
    assert "the only out-of-sample check" in text
    assert "With the default rung of the ladder run" in text
    assert "come from replays on the knobs" in text
    # C1: the PV expectation reads the day-0 archive, and the report says so
    assert A.PV_EXPECTATION_NOTE in text and "favours the virtual path" in text
    assert "- Irradiance: archive (day-0 model output; optimistic)" in text
    assert "No number in this report comes from" not in text
    # M2: the terminal valuation's units and place
    assert A.TERMINAL_NOTE in text and "excluding fees, energy tax and BTW" in A.TERMINAL_NOTE
    assert "roughly" not in A.TERMINAL_NOTE
    # M8: no wall-clock time in the generated section
    method = text.split("## Windows and method", 1)[1].split("\n## ", 1)[0]
    assert "cache build" not in method and "per rung" not in method
    assert "byte-identical" in method
    # M3: one house's two windows are pooled as windows
    pooled = text.split("## Pooled summary", 1)[1].split("\n## ", 1)[0]
    assert "- Windows: 2" in pooled and "### Per window" in pooled and "ouses:" not in pooled
    assert "| class | windows |" in pooled and "| window | days | mean EUR/day |" in pooled
    # I3: the drift alarm is a reported flag, not a guard
    assert A.DRIFT_NOTE in text and "live guard" not in text
    assert "live guard" not in A.ESTIMATOR_NOTE and A.DRIFT_NOTE in A.ESTIMATOR_NOTE
    assert "\n## A/B window in detail" in text and "\n\n## A/B window in detail" in text
    assert _scan(text) == []
    for name in ("live", "ab"):
        assert (tmp_path / "out" / name / "results.json").exists()
        assert (tmp_path / "out" / name / "ladder.csv").exists()
    first = text
    A.assemble_windows(states, inputs.cfg, tmp_path / "out", report)
    assert report.read_text() == first


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_window_ladder_in_worker_processes_matches_the_sequential_run():
    from betterybalance.replay import report as R

    from tests.test_replay_engine import _inputs

    inputs = _inputs()
    logged: list[str] = []
    one, rows, t1 = A.run_window_ladder(inputs, False, 1, logged.append)
    two, _, _ = A.run_window_ladder(inputs, False, 2, logged.append)
    assert rows is None and t1["rung_s"] == [] and len(logged) == 2
    assert R.headline(one) == R.headline(two)
    assert [r.money_eur for r in one.virtual] == [r.money_eur for r in two.virtual]


def test_cache_dir_defaults_under_the_output_and_can_be_shared(tmp_path):
    assert A.cache_dir_for(tmp_path / "out", None) == tmp_path / "out" / "cache"
    assert A.cache_dir_for(tmp_path / "out", tmp_path / "shared") == tmp_path / "shared"


def test_with_ladders_takes_the_ladder_and_its_baseline_from_the_ladder_states():
    from types import SimpleNamespace

    from betterybalance.core.site.contract import Knobs

    shipped = SimpleNamespace(knobs=Knobs(), pairs={"day": np.array(["2026-09-03"])})
    start = SimpleNamespace(
        knobs=Knobs(impl="ranking", kappa=0.5), pairs={"day": np.array(["2026-09-03"])}
    )
    states = {"live": {"result": shipped, "ladder": None, "timings": {"rung_s": []}}}
    ladder_states = {"live": {"result": start, "ladder": ["row"], "timings": {"rung_s": [1.0]}}}
    got = A.with_ladders(states, ladder_states)["live"]
    assert got["result"] is shipped and got["ladder"] == ["row"]
    assert got["baseline_knobs"].impl == "ranking" and got["ladder_timings"]["rung_s"] == [1.0]
    own = A.with_ladders(states, None)["live"]
    assert own["baseline_knobs"] is shipped.knobs and own["baseline_days"] == {"2026-09-03"}


def test_live_window_ends_on_the_last_fully_settled_day_of_the_master():
    from tests.test_features import _synthetic_master

    master = _synthetic_master(days=10)  # 2026-10-14 to 2026-10-23 local
    assert A.last_settled_day(master) == "2026-10-23"
    buy = np.asarray(master["buy_eur_mwh"], dtype=float).copy()
    buy[-3:] = np.nan  # the last day's final quarters are not settled yet
    partly = dict(master, buy_eur_mwh=buy)
    assert A.last_settled_day(partly) == "2026-10-22"
    live, ab = A.REPLAY_WINDOWS
    got = A.resolve_window(live, partly)
    assert (got.last, got.first, got.last_source) == ("2026-10-22", live.first, "master")
    over = A.resolve_window(live, partly, live_to="2026-10-20")
    assert (over.last, over.last_source) == ("2026-10-20", "override")
    assert A.resolve_window(ab, partly) == ab and ab.last == "2026-08-31"
    same = A._live_end_lines(
        {"live": {"last": "2026-10-22", "last_source": "master"}}, "2026-10-22"
    )
    assert "ends on 2026-10-22, the last local day with settled prices" in same[0]
    hand = A._live_end_lines(
        {"live": {"last": "2026-10-20", "last_source": "override"}}, "2026-10-22"
    )
    assert "set by hand" in hand[0] and "2026-10-22" in hand[0]


def test_with_ladders_refuses_ladder_states_of_another_window():
    from types import SimpleNamespace

    from betterybalance.core.site.contract import Knobs

    res = SimpleNamespace(knobs=Knobs(), pairs={"day": np.array(["2026-09-03"])})

    def st(first, last, key):
        return {
            "result": res,
            "ladder": [],
            "timings": {"rung_s": []},
            "first": first,
            "last": last,
            "cache": {"key": key},
        }

    shipped = {"live": st("2026-09-03", "2026-10-31", "k1")}
    with pytest.raises(ValueError, match="live.*last"):
        A.with_ladders(shipped, {"live": st("2026-09-03", "2026-10-30", "k1")})
    with pytest.raises(ValueError, match="cache key"):
        A.with_ladders(shipped, {"live": st("2026-09-03", "2026-10-31", "k2")})
    assert A.with_ladders(shipped, {"live": st("2026-09-03", "2026-10-31", "k1")})["live"]


def test_summer_failures_are_named_and_the_live_bias_is_not_called_indicative():
    from types import SimpleNamespace

    from betterybalance.core.site.contract import Knobs
    from betterybalance.replay import report as R

    g = {
        "day": np.array(["2026-07-01", "2026-07-02", "2026-07-03"], dtype=object),
        "soc_kwh": np.array([45.0, 46.0, 47.0]),
        "charge_threshold": np.array([50.0, 40.0, 30.0]),
        "sale_threshold": np.array([120.0, 160.0, 150.0]),
        "night_median_take": np.array([60.0, 70.0, 80.0]),
        "night_median_feed": np.array([140.0, 150.0, 140.0]),
        "surplus_tomorrow_kwh": np.array([40.0, 30.0, 35.0]),
        "low_soc_applies": np.array([False, False, False]),
        "low_soc_ok": np.array([True, True, True]),
        "high_soc_applies": np.array([True, True, True]),
        "high_soc_ok": np.array([True, False, False]),
    }
    s = A.greed_scenarios(g)
    assert s["summer"]["fail_days"] == ["2026-07-02", "2026-07-03"]
    spec_ab = A.REPLAY_WINDOWS[1]
    ran = SimpleNamespace(knobs=Knobs())
    text = "\n".join(A._greed_section({"ab": {"spec": spec_ab, "result": ran}}, {"ab": s}))
    assert "2026-07-02, 2026-07-03" in text
    bias = {
        "quarter_eur": 1.0,
        "hourly_eur": 1.4,
        "bias_eur": 0.4,
        "bias_share": 0.4,
        "indicative_only": True,
    }
    states = {
        "live": {"spec": A.REPLAY_WINDOWS[0], "bias": bias},
        "ab": {"spec": spec_ab, "bias": R.HOURLY_INPUT},
    }
    out = "\n".join(A._bias_section(states))
    assert R.INDICATIVE_SENTENCE not in out and "0,40" in out


def test_report_says_not_applicable_in_english():
    assert A._english("Stability: n.v.t., n.v.t.") == "Stability: not applicable, not applicable"


def test_replay_windows_refuses_knobs_off_the_ladder_grid_before_the_master(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "synthetic_site.json"
    site = json.loads(fixture.read_text())
    site["knobs"] = {"impl": "ranking", "kappa": 0.75}
    (tmp_path / "site.json").write_text(json.dumps(site))
    none = tmp_path / "absent"
    with pytest.raises(ValueError, match="not a ladder rung"):
        A.replay_windows(tmp_path / "site.json", none, none, none, none, none, none)
    with pytest.raises(OSError):  # without the ladder the knobs pass; the missing master fails
        A.replay_windows(tmp_path / "site.json", none, none, none, none, none, none, ladder=False)
