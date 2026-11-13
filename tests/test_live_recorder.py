import math

import pytest
from betterybalance.live.recorder import (
    MIN_SAMPLES,
    OPTIONAL_SERIES,
    REQUIRED_SERIES,
    MinuteRecorder,
)

MAP = {"grid_power": "g", "battery_power": "b", "soc": "s", "load_power": "l"}
SIGNS = {"grid_power": 1, "battery_power": -1}
T0 = 1788300000


def _feed(rec, eid, value, every=5, start=T0, n=12):
    for i in range(n):
        rec.sample(eid, start + i * every, value)


def test_series_names():
    assert MIN_SAMPLES == 4
    assert REQUIRED_SERIES == ("grid_power", "battery_power", "soc")
    assert OPTIONAL_SERIES == (
        "load_power",
        "load_ups_power",
        "pv_power",
        *(f"pv{i}_power" for i in range(1, 9)),
        "gen_power",
    )


def test_twelve_samples_integrate_to_energy():
    rec = MinuteRecorder(MAP, SIGNS)
    _feed(rec, "g", 1200.0)
    _feed(rec, "b", -3000.0)
    _feed(rec, "s", 50.0)
    _feed(rec, "l", 600.0)
    row = rec.close(T0)
    assert row["grid_import_kwh"] == pytest.approx(0.02) and row["grid_export_kwh"] == 0.0
    assert row["batt_charge_kwh"] == pytest.approx(0.05) and row["soc_pct"] == 50.0
    assert (
        row["load_kwh"] == pytest.approx(0.01)
        and math.isnan(row["pv_kwh"])
        and row["n_samples"] == 12
    )
    assert row["soc_min_pct"] == row["soc_max_pct"] == 50.0


def test_three_samples_make_a_missing_minute():
    rec = MinuteRecorder(MAP, SIGNS)
    _feed(rec, "g", 1200.0, every=20, n=3)
    _feed(rec, "b", 0.0)
    _feed(rec, "s", 50.0)
    assert rec.close(T0) is None


def test_unavailable_inside_the_minute_is_missing_never_zero():
    rec = MinuteRecorder(MAP, SIGNS)
    _feed(rec, "g", 1200.0)
    _feed(rec, "s", 50.0)
    _feed(rec, "b", 0.0, n=6)
    rec.sample("b", T0 + 30, None)
    _feed(rec, "b", 0.0, start=T0 + 35, n=5)
    assert rec.close(T0) is None
    _feed(rec, "g", 1200.0, start=T0 + 60)
    _feed(rec, "b", 0.0, start=T0 + 60)
    _feed(rec, "s", 50.0, start=T0 + 60)
    assert rec.close(T0 + 60) is not None


def test_optional_series_missing_leaves_nan():
    rec = MinuteRecorder(MAP, SIGNS)
    _feed(rec, "g", 0.0)
    _feed(rec, "b", 0.0)
    _feed(rec, "s", 50.0)
    _feed(rec, "l", 600.0, n=2)
    assert math.isnan(rec.close(T0)["load_kwh"])


def test_sample_and_hold_from_the_value_before_the_minute():
    rec = MinuteRecorder(MAP, SIGNS)
    # the held -1200 W covers [T0, T0 + 30), then +1200 W: import and export of 0,01 kWh each
    rec.sample("g", T0 - 2, -1200.0)
    _feed(rec, "g", 1200.0, every=5, start=T0 + 30, n=6)
    _feed(rec, "b", 3000.0)
    _feed(rec, "s", 50.0)
    row = rec.close(T0)
    assert row["grid_import_kwh"] == pytest.approx(0.01)
    assert row["grid_export_kwh"] == pytest.approx(0.01)
    # battery sign -1: +3000 W reads as a discharge
    assert row["batt_discharge_kwh"] == pytest.approx(0.05) and row["batt_charge_kwh"] == 0.0
    assert row["n_samples"] == 6


def test_no_value_before_the_first_sample_is_missing():
    rec = MinuteRecorder(MAP, SIGNS)
    _feed(rec, "g", 1200.0, start=T0 + 5, n=11)
    _feed(rec, "b", 0.0)
    _feed(rec, "s", 50.0)
    assert rec.close(T0) is None


def test_soc_mean_min_max_over_the_held_values():
    rec = MinuteRecorder(MAP, SIGNS)
    _feed(rec, "g", 0.0)
    _feed(rec, "b", 0.0)
    _feed(rec, "s", 50.0, n=9)
    _feed(rec, "s", 51.0, start=T0 + 45, n=3)
    row = rec.close(T0)
    assert row["soc_pct"] == pytest.approx(50.25)
    assert row["soc_min_pct"] == 50.0 and row["soc_max_pct"] == 51.0


def test_strings_gen_and_load_alone():
    mapping = {
        **MAP,
        "load_ups_power": "u",
        "pv_power": "p",
        "pv1_power": "p1",
        "pv2_power": "p2",
        "gen_power": "x",
    }
    rec = MinuteRecorder(mapping, SIGNS)
    for eid, w in (
        ("g", 0.0),
        ("b", 0.0),
        ("s", 50.0),
        ("l", 600.0),
        ("u", 300.0),
        ("p", 1800.0),
        ("p1", 1200.0),
        ("p2", 600.0),
        ("x", 60.0),
    ):
        _feed(rec, eid, w)
    row = rec.close(T0)
    # Load Power already holds the UPS port: the UPS series is not added
    assert row["load_kwh"] == pytest.approx(0.01)
    assert row["pv_kwh"] == pytest.approx(0.03) and row["gen_kwh"] == pytest.approx(0.001)
    assert row["pv1_kwh"] == pytest.approx(0.02) and row["pv2_kwh"] == pytest.approx(0.01)
    assert all(math.isnan(row[f"pv{i}_kwh"]) for i in range(3, 9))


def test_samples_of_the_next_minute_wait_and_older_samples_drop():
    rec = MinuteRecorder(MAP, SIGNS)
    for eid, w in (("g", 600.0), ("b", 0.0), ("s", 50.0)):
        _feed(rec, eid, w, n=24)  # two minutes before the first close
    first = rec.close(T0)
    assert first["n_samples"] == 12
    rec.sample("g", T0 + 59, 99999.0)  # older than the closed minute: dropped
    second = rec.close(T0 + 60)
    assert second["ts"] == T0 + 60 and second["n_samples"] == 12
    assert second["grid_import_kwh"] == pytest.approx(0.01)


def test_unknown_entity_and_non_finite_values():
    rec = MinuteRecorder(MAP, SIGNS)
    rec.sample("not_mapped", T0, 5.0)
    _feed(rec, "g", 0.0)
    _feed(rec, "s", 50.0)
    _feed(rec, "b", 0.0, n=6)
    rec.sample("b", T0 + 30, math.nan)
    _feed(rec, "b", 0.0, start=T0 + 35, n=5)
    assert rec.close(T0) is None


def test_missing_required_mapping_raises():
    with pytest.raises(ValueError, match="soc"):
        MinuteRecorder({"grid_power": "g", "battery_power": "b"}, SIGNS)


# grid from energy counters (grid_source p1 or custom)

COUNTERS = {"import": ["i1", "i2"], "export": ["e1", "e2"]}
CMAP = {"battery_power": "b", "soc": "s", "grid_power": "g"}


def _counter_rec():
    rec = MinuteRecorder(CMAP, SIGNS, grid_counters=COUNTERS)
    for eid, v in (("i1", 100.0), ("i2", 200.0), ("e1", 50.0), ("e2", 60.0)):
        rec.sample(eid, T0 - 5, v)
    _feed(rec, "b", 0.0)
    _feed(rec, "s", 50.0)
    return rec


def test_counters_two_tariffs_summed_without_grid_power():
    rec = _counter_rec()
    rec.sample("i1", T0 + 30, 100.01)
    rec.sample("i2", T0 + 50, 200.02)
    rec.sample("e2", T0 + 20, 60.0)  # state_reported: unchanged
    rec.sample("e2", T0 + 40, 60.005)
    row = rec.close(T0)
    assert row["grid_import_kwh"] == pytest.approx(0.03)
    assert row["grid_export_kwh"] == pytest.approx(0.005)
    assert row["n_samples"] == 12


def test_counter_unchanged_all_minute_is_valid():
    rec = _counter_rec()  # no counter sample inside the minute
    row = rec.close(T0)
    assert row["grid_import_kwh"] == 0.0 and row["grid_export_kwh"] == 0.0
    # the next minute holds the same counters
    _feed(rec, "b", 0.0, start=T0 + 60)
    _feed(rec, "s", 50.0, start=T0 + 60)
    rec.sample("i1", T0 + 70, 100.002)
    assert rec.close(T0 + 60)["grid_import_kwh"] == pytest.approx(0.002)


def test_counter_unavailable_inside_the_minute_is_missing():
    rec = _counter_rec()
    rec.sample("e1", T0 + 30, None)
    rec.sample("e1", T0 + 35, 50.0)
    assert rec.close(T0) is None


def test_counter_reset_is_missing_not_negative():
    rec = _counter_rec()
    rec.sample("i1", T0 + 30, 0.5)
    rec.sample("i2", T0 + 30, 200.5)  # the sum still rises: the reset of i1 still counts
    assert rec.close(T0) is None


def test_counter_never_seen_is_missing():
    rec = MinuteRecorder(CMAP, SIGNS, grid_counters=COUNTERS)
    for eid, v in (("i1", 100.0), ("i2", 200.0), ("e1", 50.0)):
        rec.sample(eid, T0 - 5, v)
    _feed(rec, "b", 0.0)
    _feed(rec, "s", 50.0)
    assert rec.close(T0) is None
