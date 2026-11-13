# tests/test_replay_report.py
import json

import numpy as np
import pytest
from betterybalance.core.site.contract import Knobs
from betterybalance.replay import report as R


def _row(name, daily, days, wrong=3, cycles=0.4):
    return {
        "name": name,
        "knobs": Knobs(),
        "diff_eur": float(np.sum(daily)),
        "wrong_side": wrong,
        "cycles_per_day": cycles,
        "daily": np.asarray(daily, float),
        "days": np.asarray(days, object),
    }


def test_ladder_has_twenty_named_rungs():
    rungs = R.ladder_rungs()
    assert (
        len(rungs) == 20
        and R.DEFAULT_RUNG in dict(rungs)
        and any("expected-gain" in n for n, _ in rungs)
    )
    assert dict(rungs)["dp q10/q90 kappa 0,25 penalty none"].impl == "dp"


def test_selection_keeps_default_on_ties_and_switches_on_a_clear_gain():
    days = [f"2026-09-{d:02d}" for d in range(1, 31)]
    rng = np.random.default_rng(0)
    base = rng.normal(1.0, 0.5, 30)
    rows = [
        _row(R.DEFAULT_RUNG, base, days),
        _row("tie", base, days),
        _row("better", base + 0.5, days),
    ]  # tie: paired differences all zero, interval [0, 0]
    sel = R.select_default(rows, R.DEFAULT_RUNG, live_days=set(days), ab_days=set())
    assert sel["chosen"] == "better" and any(
        r["name"] == "tie" and not r["candidate"] for r in sel["table"]
    )
    only_tie = R.select_default(rows[:2], R.DEFAULT_RUNG, live_days=set(days), ab_days=set())
    assert only_tie["chosen"] == R.DEFAULT_RUNG


def test_selection_rejects_a_rung_that_loses_on_the_ab_window():
    live = [f"2026-09-{d:02d}" for d in range(1, 16)]
    ab = [f"2026-03-{d:02d}" for d in range(1, 16)]
    days = live + ab
    base = np.ones(30)
    cand = np.concatenate([np.ones(15) * 2.0, np.ones(15) * 0.5])  # wins live, loses A/B
    sel = R.select_default(
        [_row(R.DEFAULT_RUNG, base, days), _row("mixed", cand, days)],
        R.DEFAULT_RUNG,
        set(live),
        set(ab),
    )
    assert sel["chosen"] == R.DEFAULT_RUNG
    # with A/B days given, a rung without a paired A/B day fails the A/B condition
    sel = R.select_default(
        [_row(R.DEFAULT_RUNG, base[:15], live), _row("live only", cand[:15], live)],
        R.DEFAULT_RUNG,
        set(live),
        set(ab),
    )
    row = next(t for t in sel["table"] if t["name"] == "live only")
    assert not row["candidate"] and np.isnan(row["ab_mean"]) and sel["chosen"] == R.DEFAULT_RUNG


def test_report_has_attribution_european_numbers_and_indicative_sentence(tmp_path):
    from betterybalance.replay import engine as E

    from tests.test_replay_engine import _inputs

    result = E.replay(_inputs())
    bias = {
        "quarter_eur": 1.0,
        "hourly_eur": 1.3,
        "bias_eur": 0.3,
        "bias_share": 0.3,
        "indicative_only": True,
    }
    text = R.render_report(result, {"site": "synthetic house", "window": "3 days"}, bias=bias)
    assert (
        R.ATTRIBUTION in text
        and "indicative only" in text
        and "0,30" in text
        and "0.30" not in text
    )
    files = R.write_outputs(result, tmp_path, {"site": "synthetic house", "window": "3 days"})
    first = {f.name: f.read_bytes() for f in files}
    R.write_outputs(result, tmp_path, {"site": "synthetic house", "window": "3 days"})
    assert (
        {f.name: f.read_bytes() for f in files} == first
        and "report.md" in first
        and "greed.csv" in first
    )


def test_greed_check_counts_and_names_the_first_failure():
    g = {
        "day": np.array(["2026-09-01", "2026-09-02"], object),
        "low_soc_ok": np.array([True, False]),
        "high_soc_ok": np.array([True, True]),
        "low_soc_applies": np.array([True, True]),
        "high_soc_applies": np.array([False, True]),
    }
    c = R.greed_check(g)
    assert (
        c["low"]["applies"] == 2
        and c["low"]["ok"] == 1
        and c["low"]["first_fail"] == "2026-09-02"
        and c["high"]["applies"] == 1
    )


# -- beyond the brief: the CLI helpers, the hourly re-ledger and the load_inputs move ----


def test_load_inputs_moved_to_core_and_stays_importable_from_the_study():
    from betterybalance.core import inputs as I

    from tools.study import october_2026 as O

    assert O.load_inputs is I.load_inputs and O.NPZ_FILE == I.NPZ_FILE == "master_inputs.npz"


def test_hourly_ledgers_see_netting_inside_the_hour_and_pair_missing_quarters():
    from types import SimpleNamespace

    from betterybalance.core.ledger import rows_for_quarter
    from betterybalance.core.site.contract import Terms
    from betterybalance.replay.phases import Prices

    from tests.test_features import _synthetic_master

    master = _synthetic_master(days=37)
    n = len(master["ts"])
    q = {"ts": master["ts"], "step_s": np.full(n, 900)}
    for c in (
        "batt_charge_kwh",
        "batt_discharge_kwh",
        "load_kwh",
        "gen_kwh",
        "pv_kwh",
        "soc_pct",
        "soc_min_pct",
        "soc_max_pct",
    ):
        q[c] = np.zeros(n)
    imp = np.where(np.arange(n) % 2 == 0, 0.5, 0.0)  # import and export quarters alternate
    q["grid_import_kwh"], q["grid_export_kwh"] = imp, 0.5 - imp
    q["missing"] = np.zeros(n, bool)
    k0 = 96 * 35 + 4  # a whole hour; quarters k0 .. k0 + 7 are two hours
    q["missing"][k0 + 5] = True  # one quarter of the second hour
    inp = SimpleNamespace(flows_quarter=q, master=master, cfg=SimpleNamespace(terms=Terms()))
    prices = Prices(inp.master)
    actual = [
        rows_for_quarter(
            int(q["ts"][k]),
            "actual",
            np.full(15, (imp[k] - (0.5 - imp[k])) * 4),
            prices.at(int(q["ts"][k])),
            inp.cfg.terms,
            missing=bool(q["missing"][k]),
        )
        for k in range(k0, k0 + 8)
    ]
    booked, hourly = R.hourly_ledgers(inp, SimpleNamespace(actual=actual))
    assert [b.missing for b in booked] == [h.missing for h in hourly] == [False] * 4 + [True] * 4
    assert all(h.import_kwh == 0.0 and h.export_kwh == 0.0 for h in hourly[:4])
    assert sum(b.import_kwh for b in booked[:4]) == pytest.approx(1.0)
    assert all(np.isnan(b.money_eur) and np.isnan(b.tax_eur) for b in booked[4:])
    assert all(np.isnan(h.money_eur) for h in hourly[4:])


def test_cli_window_flows_and_embargo_helpers(tmp_path):
    import datetime as dt

    from betterybalance.core.site.flows import COLUMNS
    from betterybalance.replay import __main__ as M
    from betterybalance.replay.phases import local_midnight

    from tests.test_features import _synthetic_master

    start, end = M.window_bounds("2026-10-24", "2026-10-25")  # the DST end on 10-25: 49 h
    assert end - start == 49 * 3600 and start == local_midnight(dt.date(2026, 10, 24))
    with pytest.raises(ValueError):
        M.window_bounds("2026-10-25", "2026-10-24")
    hourly = {c: np.zeros(2) for c in COLUMNS}
    hourly["ts"] = np.array([start, start + 3600], dtype=np.int64)
    hourly["step_s"] = np.full(2, 3600, dtype=np.int64)
    hourly["soc_pct"] = np.full(2, 50.0)
    quarters, resolution = M.quarter_flows(hourly)
    assert resolution == "hourly" and len(quarters["ts"]) == 8
    master = _synthetic_master(days=10)  # starts 2026-10-14 00:00 local
    assert M.earliest_from(master) == dt.date(2026, 10, 20)
    with pytest.raises(ValueError, match="earliest valid start is 2026-10-20"):
        M.check_start(master, local_midnight(dt.date(2026, 10, 19)))
    M.check_start(master, local_midnight(dt.date(2026, 10, 20)))
    with pytest.raises(SystemExit):
        M.parse_args(
            ["--site", "s", "--flows", "f", "--data", "d", "--from", "x", "--to", "y", "--out", "o"]
        )  # neither --irradiance nor --lat --lon


def test_bias_paragraph_names_hourly_input_and_never_says_indicative_for_it():
    text = "\n".join(R.bias_paragraph(R.HOURLY_INPUT))
    assert "input flows are hourly" in text and "indicative only" not in text
    assert R.bias_paragraph(None) == []


def test_hourly_input_report_carries_the_reference_bias_and_indicative_sentence():
    from betterybalance.replay import engine as E

    from tests.test_replay_engine import _inputs

    result = E.replay(_inputs(days_replay=1))
    text = R.render_report(result, {"window": "1 day"}, bias=R.HOURLY_INPUT)
    context = (
        "The quarter-versus-hour bias could not be measured for this house; on the reference "
        "house it was 93,2 % of the live headline."
    )
    headline = text.split("## Headline", 1)[1].split("\n## ", 1)[0]
    assert context in headline and R.INDICATIVE_SENTENCE in headline
    assert R.HOURLY_INPUT_SENTENCE in text
    measured = {"quarter_eur": 1.0, "hourly_eur": 1.01, "bias_eur": 0.01, "bias_share": 0.01}
    assert context not in R.render_report(result, {}, bias=measured)


def test_cli_feed_error_prints_no_url_and_no_location(monkeypatch, capsys):
    from betterybalance.core.feeds.base import FeedError
    from betterybalance.replay import __main__ as M

    def fail(a):
        raise FeedError(
            "request failed: https://archive-api.open-meteo.com/v1/archive"
            "?latitude=52.1&longitude=5.2&start_date=2026-09-01: timed out"
        )

    monkeypatch.setattr(M, "run", fail)
    args = ["--site", "s", "--flows", "f", "--data", "d", "--from", "2026-09-01"]
    args += ["--to", "2026-09-02", "--out", "o", "--lat", "52.1", "--lon", "5.2", "--cache", "c"]
    assert M.main(args) == 2
    err = capsys.readouterr().err
    assert "FeedError" in err and M.FEED_ERROR_SENTENCE in err
    assert "52.1" not in err and "5.2" not in err and "latitude" not in err and "http" not in err


def test_default_rung_name_follows_the_knobs():
    assert R.default_rung_name(Knobs()) == R.DEFAULT_RUNG
    base = Knobs(impl="ranking", q_take=0.25, q_feed=0.75, kappa=0.5)
    assert R.default_rung_name(base) == "ranking q25/q75 kappa 0,5 penalty on"
    assert R.default_rung_name(Knobs(impl="ranking", kappa=0.5, dual_penalty=False)) == (
        "ranking q25/q75 kappa 0,5 penalty off"
    )
    assert R.default_rung_name(Knobs(impl="ranking", kappa=0.5, commit_rule="expected_gain")) == (
        "ranking q25/q75 kappa 0,5 expected-gain"
    )
    assert R.default_rung_name(Knobs(impl="dp", kappa=1.0)) == "dp q25/q75 kappa 1 penalty none"
    names = dict(R.ladder_rungs())
    assert all(R.default_rung_name(k) == n for n, k in R.ladder_rungs()) and len(names) == 20


def test_ladder_refuses_knobs_off_the_grid_before_any_replay(tmp_path, capsys):
    from betterybalance.replay import __main__ as M

    R.check_ladder_default(Knobs())
    R.check_ladder_default(Knobs(impl="ranking", kappa=0.5, dual_penalty=False))
    with pytest.raises(ValueError, match="kappa 0,75 penalty none.*not a ladder rung") as exc:
        R.check_ladder_default(Knobs(kappa=0.75))
    assert all(name in str(exc.value) for name, _ in R.ladder_rungs())
    site = {
        "latitude": 52.1,
        "longitude": 5.2,
        "pack": {
            "capacity_kwh": 48,
            "charge_kw": 12,
            "discharge_kw": 12,
            "soc_floor": 0.1,
            "soc_ceiling": 1.0,
        },
        "knobs": {"kappa": 0.75},
    }
    (tmp_path / "site.json").write_text(json.dumps(site))
    args = ["--site", str(tmp_path / "site.json"), "--flows", str(tmp_path / "none.csv")]
    args += ["--data", str(tmp_path / "no-data"), "--from", "2026-09-01", "--to", "2026-09-02"]
    args += ["--out", str(tmp_path / "out"), "--irradiance", str(tmp_path / "irr.csv")]
    assert M.main([*args, "--ladder"]) == 2
    err = capsys.readouterr().err
    assert "not a ladder rung" in err and R.DEFAULT_RUNG in err
    assert not (tmp_path / "out").exists()
    assert M.main(args) == 2  # without --ladder the knobs pass; the missing master fails
    assert "not a ladder rung" not in capsys.readouterr().err


def _statistics_like_the_fixture(first_hour: int, hours: int) -> dict:
    """Hourly HA statistics in the shape of ``synthetic_ha_statistics.json`` (same ids and
    fields: counters with ``sum``, means with ``mean``, ``min`` and ``max``), ``hours`` long."""
    import pathlib

    fixture = pathlib.Path(__file__).parent / "fixtures" / "synthetic_ha_statistics.json"
    template = json.loads(fixture.read_text())
    out: dict[str, list[dict]] = {}
    for sid, rows in template.items():
        counter = "sum" in rows[0]
        level = rows[0]["sum"] if counter else rows[0]["mean"]
        series = []
        for h in range(hours):
            t = (first_hour + 3600 * h) * 1000
            row = {"start": t, "end": t + 3_600_000}
            if counter:
                level += 0.5 if sid.startswith("import") else 0.1
                row["sum"] = round(level, 3)
            else:
                mean = level + (10.0 * np.sin(h / 24 * 2 * np.pi) if sid == "soc" else 0.0)
                row.update(mean=mean, min=mean - 2.5, max=mean + 2.5)
            series.append(row)
        out[sid] = series
    return out


def test_cli_end_to_end_on_an_hourly_export_that_starts_at_from(monkeypatch, tmp_path, capsys):
    """export_site -> flows.csv (hourly, from --from 00:00) -> python -m betterybalance.replay.

    The export drops its extra counter hour, so the quarter that ends at --from 00:00 is
    absent: the CLI starts the replay a day later and says so, and the report carries the
    reference bias and the indicative-only sentence of an hourly replay."""
    import datetime as dt

    from betterybalance.core.site.flows import read_flows
    from betterybalance.replay import __main__ as M
    from betterybalance.replay.phases import local_midnight

    from tests.test_replay_engine import _inputs
    from tools import export_site as X

    base = _inputs(days_replay=1)  # synthetic master, tier-A band cache, start 2026-11-18
    first, last = dt.date(2026, 11, 18), dt.date(2026, 11, 19)
    pre = local_midnight(first) - 3600
    hours = (local_midnight(last + dt.timedelta(days=1)) - pre) // 3600
    stats = _statistics_like_the_fixture(pre, hours)

    async def fake_stats(url, token, ids, s, e):
        return {i: stats[i] for i in ids if i in stats}

    async def fake_loc(url, token):
        return 52.1, 5.2

    monkeypatch.setattr(X, "fetch_statistics", fake_stats)
    monkeypatch.setattr(X, "fetch_location", fake_loc)
    entities = tmp_path / "entities.yaml"
    entities.write_text(
        "grid_import_kwh: [import_t1, import_t2]\ngrid_export_kwh: export_t1\n"
        "battery_power_w: batt\nsoc_pct: soc\n"
    )
    export = tmp_path / "export"
    X.main(
        ["--url", "http://ha.invalid", "--token", "t", "--from", first.isoformat()]
        + ["--to", last.isoformat(), "--out", str(export), "--entities", str(entities)]
    )
    flows = read_flows(export / "flows.csv")
    assert int(flows["ts"][0]) == local_midnight(first) and (flows["step_s"] == 3600).all()
    site = json.loads((export / "site.json").read_text())
    site["pack"].update(capacity_kwh=48, charge_kw=12, discharge_kw=12)
    site["pv"] = {"none": True}
    (export / "site.json").write_text(json.dumps(site))
    irr = tmp_path / "irradiance.csv"
    ts = np.asarray(base.irr["ts"], dtype=np.int64)
    irr.write_text(
        "ts,ghi_w_m2,dni_w_m2,dhi_w_m2,temp_c\n" + "".join(f"{t},0,0,0,10\n" for t in ts)
    )
    monkeypatch.setattr(M, "load_inputs", lambda data: base.master)
    monkeypatch.setattr(M, "band_cache", lambda master, s, e, cache_dir, **kw: base.cache)
    capsys.readouterr()
    out = tmp_path / "out"
    argv = ["--site", str(export / "site.json"), "--flows", str(export / "flows.csv")]
    argv += ["--data", str(tmp_path / "data"), "--from", first.isoformat()]
    argv += ["--to", last.isoformat(), "--irradiance", str(irr), "--out", str(out)]
    assert M.main(argv) == 0
    printed = capsys.readouterr()
    assert "no recorded SoC at the start of 2026-11-18 00:00; the replay starts 2026-11-19" in (
        printed.out
    ), printed.err
    assert R.INDICATIVE_SENTENCE in printed.out
    report = (out / "report.md").read_text()
    headline = report.split("## Headline", 1)[1].split("\n## ", 1)[0]
    assert "on the reference house it was 93,2 % of the live headline" in headline
    assert R.INDICATIVE_SENTENCE in headline and "- window: 2026-11-19 to 2026-11-19" in report
    assert "over 1 days" in report
    written = {p.name for p in out.iterdir()}
    assert {"report.md", "results.json", "decisions.csv", "greed.csv"} <= written


def test_a_role_without_statistics_logs_a_warning(caplog):
    import logging

    from betterybalance.core import inputs as I

    with caplog.at_level(logging.WARNING, logger=I.__name__):
        assert I._sum_role({}, "load_power", "sensor.synthetic_load", lambda rows: rows) == {}
    assert "no statistics for role load_power" in caplog.text
