import numpy as np
import pytest

from tools import owner_site_archive as OA


def test_minute_rows_split_by_sign_and_convert_to_kwh():
    ts = [0, 60, 120]
    series = {
        "grid": dict(zip(ts, [600.0, -600.0, 0.0])),
        "batt": dict(zip(ts, [1200.0, -1200.0, 0.0])),
        "soc": dict(zip(ts, [50.0, 50.5, 50.5])),
        "load": dict(zip(ts, [300.0, 300.0, 300.0])),
        "gen": {},
        "pv": {},
    }
    t = OA.minute_rows(series, grid_sign=1, batt_sign=-1)
    assert t["ts"].tolist() == ts and (t["step_s"] == 60).all()
    assert t["grid_import_kwh"][0] == pytest.approx(0.01)
    assert t["grid_export_kwh"][1] == pytest.approx(0.01)
    # sign -1 flips: a positive reading is a discharge
    assert t["batt_discharge_kwh"][0] == pytest.approx(0.02)
    assert t["batt_charge_kwh"][1] == pytest.approx(0.02)
    assert t["load_kwh"][2] == pytest.approx(0.005) and np.isnan(t["gen_kwh"]).all()
    assert t["soc_min_pct"][0] == 50.0


def test_sign_check_reports_charge_while_soc_rises():
    ts = list(range(0, 600, 60))
    soc = [50 + 0.1 * i for i in range(10)]
    series = {
        "grid": dict(zip(ts, [1000.0] * 10)),
        "batt": dict(zip(ts, [2000.0] * 10)),
        "soc": dict(zip(ts, soc)),
        "load": {},
        "gen": {},
        "pv": {},
    }
    t = OA.minute_rows(series, grid_sign=1, batt_sign=1)
    chk = OA.sign_check(t)
    assert chk["batt_kw_while_soc_rises"] > 0 and chk["minutes_rising"] == 9


def test_minute_without_a_required_sample_is_nan():
    series = {"grid": {0: 100.0, 60: 100.0}, "batt": {0: 50.0}, "soc": {60: 40.0}}
    t = OA.minute_rows(series, grid_sign=1, batt_sign=1)
    assert t["ts"].tolist() == [0, 60]
    assert np.isnan(t["soc_pct"][0]) and np.isnan(t["batt_charge_kwh"][1])
    assert t["grid_import_kwh"][1] == pytest.approx(100 / 60000)
    assert np.isnan(t["pv_kwh"]).all()


def test_sign_check_on_a_flipped_sensor_is_negative_while_soc_rises():
    ts = [0, 60, 120, 180]
    series = {
        "grid": {},
        "batt": dict.fromkeys(ts, -3000.0),
        "soc": dict(zip(ts, [10, 11, 12, 11])),
    }
    chk = OA.sign_check(OA.minute_rows(series, grid_sign=1, batt_sign=1))
    assert chk["batt_kw_while_soc_rises"] == pytest.approx(-3.0)
    assert chk["minutes_rising"] == 2 and chk["minutes_falling"] == 1


def test_irradiance_archive_writes_the_quarter_csv(tmp_path, monkeypatch):
    import datetime as dt

    from betterybalance.core.feeds.openmeteo import read_irradiance_csv

    from tools import irradiance_archive as IA

    calls = []

    class FakeSource:
        def archive(self, lat, lon, start, end, cache):
            calls.append((lat, lon, start, end, type(cache).__name__))
            ts = np.arange(4, dtype=np.int64) * 900
            col = np.array([0.0, 10.0, np.nan, 30.0])
            return {"ts": ts, "ghi_w_m2": col, "dni_w_m2": col, "dhi_w_m2": col, "temp_c": col}

    monkeypatch.setattr(IA, "OpenMeteo", FakeSource)
    out = tmp_path / "irr.csv"
    IA.main(
        ["--lat", "52.1", "--lon", "5.2", "--from", "2026-02-01", "--to", "2026-02-02"]
        + ["--cache", str(tmp_path / "c.jsonl"), "--out", str(out)]
    )
    assert calls == [(52.1, 5.2, dt.date(2026, 2, 1), dt.date(2026, 2, 2), "DayCache")]
    back = read_irradiance_csv(out)
    assert back["ts"].tolist() == [0, 900, 1800, 2700] and np.isnan(back["ghi_w_m2"][2])


def test_grid_fallback_fills_only_minutes_without_a_direct_sample():
    grid = {60: 500.0}
    import_kw = {0: 1.5, 60: 9.0, 120: 0.0}
    export_kw = {0: 0.25, 60: 9.0, 120: 2.0}
    # minute 0: fallback only; minute 60: both, the direct sample wins; minute 120: export
    filled = OA.grid_with_fallback(grid, import_kw, export_kw, grid_sign=1)
    assert filled == {0: pytest.approx(1250.0), 60: 500.0, 120: pytest.approx(-2000.0)}
    # the fallback is import positive; a grid entity with sign -1 gets it in its own convention
    flipped = OA.grid_with_fallback({}, {0: 1.0}, {0: 0.0}, grid_sign=-1)
    t = OA.minute_rows({"grid": flipped, "batt": {0: 0.0}, "soc": {0: 50.0}}, -1, 1)
    assert t["grid_import_kwh"][0] == pytest.approx(1000 / 60000)


def test_grid_fallback_needs_both_meter_series_at_the_minute():
    assert OA.grid_with_fallback({}, {0: 1.0}, {60: 1.0}, grid_sign=1) == {}


def test_irradiance_archive_clips_the_end_at_yesterday_utc(tmp_path, monkeypatch, capsys):
    import datetime as dt

    from tools import irradiance_archive as IA

    day = dt.date(2026, 10, 31)
    assert IA.clipped_end(dt.date(2026, 11, 2), today=day) == dt.date(2026, 10, 30)
    assert IA.clipped_end(dt.date(2026, 10, 1), today=day) == dt.date(2026, 10, 1)
    calls = []

    class FakeSource:
        def archive(self, lat, lon, start, end, cache):
            calls.append(end)
            return {c: np.zeros(0) for c in ("ts", "ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c")}

    monkeypatch.setattr(IA, "OpenMeteo", FakeSource)
    args = ["--lat", "52.1", "--lon", "5.2", "--cache", str(tmp_path / "c.jsonl")]
    args += ["--out", str(tmp_path / "irr.csv")]
    IA.main([*args, "--from", "2026-02-01", "--to", "2099-01-01"])
    yesterday = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)
    assert calls == [yesterday]
    assert f"clipped to {yesterday.isoformat()}" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        IA.main([*args, "--from", "2099-01-01", "--to", "2099-01-02"])
    assert calls == [yesterday]
