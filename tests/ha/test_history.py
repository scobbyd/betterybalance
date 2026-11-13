import datetime as dt
import math
from unittest.mock import patch

import pytest
from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.betterybalance import history as H

MAPPING = {
    "grid_import_total": "sensor.synthetic_grid_import_total",
    "grid_export_total": "sensor.synthetic_grid_export_total",
    "battery_charge_total": "sensor.synthetic_battery_charge_total",
    "battery_discharge_total": "sensor.synthetic_battery_discharge_total",
    "soc": "sensor.synthetic_battery",
    "load_power": "sensor.synthetic_load_power",
}
SIGNS = {"battery_power": 1, "grid_power": 1}
P1 = {
    "import": ["sensor.synthetic_p1_import_t1", "sensor.synthetic_p1_import_t2"],
    "export": ["sensor.synthetic_p1_export_t1", "sensor.synthetic_p1_export_t2"],
}


def _import(hass, statistic_id: str, unit: str, rows: list[dict], has_sum: bool) -> None:
    meta = {
        "mean_type": StatisticMeanType.NONE if has_sum else StatisticMeanType.ARITHMETIC,
        "has_sum": has_sum,
        "name": None,
        "source": "recorder",
        "statistic_id": statistic_id,
        "unit_class": None,
        "unit_of_measurement": unit,
    }
    async_import_statistics(hass, meta, rows)


def _hours(n: int) -> list[dt.datetime]:
    top = dt_util.utcnow().replace(minute=0, second=0, microsecond=0)
    return [top - dt.timedelta(hours=n + 1 - i) for i in range(n)]


def _counter(hass, sid: str, hours, sums) -> None:
    _import(
        hass, sid, "kWh", [{"start": h, "sum": s, "state": s} for h, s in zip(hours, sums)], True
    )


def _power(hass, sid: str, unit: str, hours, means, lows=None, highs=None) -> None:
    lows, highs = lows or means, highs or means
    rows = [
        {"start": h, "mean": m, "min": lo, "max": hi}
        for h, m, lo, hi in zip(hours, means, lows, highs)
    ]
    _import(hass, sid, unit, rows, False)


async def _site(hass, hours) -> None:
    _counter(hass, MAPPING["grid_import_total"], hours, [10.0, 11.0])
    _counter(hass, MAPPING["grid_export_total"], hours, [4.0, 4.5])
    _counter(hass, MAPPING["battery_charge_total"], hours, [20.0, 22.0])
    _counter(hass, MAPPING["battery_discharge_total"], hours, [30.0, 30.25])
    _power(hass, MAPPING["soc"], "%", hours, [50.0, 60.0], [48.0, 55.0], [52.0, 64.0])
    _power(hass, MAPPING["load_power"], "W", hours, [400.0, 800.0])
    await async_wait_recording_done(hass)


async def test_backfill_builds_hourly_flows(hass, recorder_mock):
    hours = _hours(2)
    await _site(hass, hours)
    table = await H.backfill_hours(
        hass, MAPPING, SIGNS, hours[0], hours[-1] + dt.timedelta(hours=1)
    )
    assert table["ts"].tolist() == [int(h.timestamp()) for h in hours]
    assert table["step_s"].tolist() == [3600, 3600]
    # the counter delta belongs to the later hour: the first hour has none
    assert math.isnan(table["grid_import_kwh"][0])
    assert table["grid_import_kwh"][1] == 1.0 and table["grid_export_kwh"][1] == 0.5
    assert table["batt_charge_kwh"][1] == 2.0 and table["batt_discharge_kwh"][1] == 0.25
    assert table["soc_pct"][0] == 50.0 and table["soc_pct"][1] == 60.0
    assert table["soc_min_pct"][1] == 55.0 and table["soc_max_pct"][1] == 64.0
    assert table["load_kwh"].tolist() == pytest.approx([0.4, 0.8])
    assert math.isnan(table["pv_kwh"][1])


async def test_backfill_reads_the_hour_before_for_the_first_delta(hass, recorder_mock):
    hours = _hours(2)
    await _site(hass, hours)
    table = await H.backfill_hours(hass, MAPPING, SIGNS, hours[1], hours[1] + dt.timedelta(hours=1))
    assert table["ts"].tolist() == [int(hours[1].timestamp())]
    assert table["grid_import_kwh"][0] == 1.0


async def test_backfill_sums_the_grid_counters(hass, recorder_mock):
    hours = _hours(2)
    await _site(hass, hours)
    _counter(hass, P1["import"][0], hours, [100.0, 100.75])
    _counter(hass, P1["import"][1], hours, [200.0, 200.5])
    _counter(hass, P1["export"][0], hours, [10.0, 10.0])
    _counter(hass, P1["export"][1], hours, [20.0, 20.25])
    await async_wait_recording_done(hass)
    table = await H.backfill_hours(
        hass, MAPPING, SIGNS, hours[0], hours[-1] + dt.timedelta(hours=1), grid_counters=P1
    )
    assert table["grid_import_kwh"][1] == 1.25 and table["grid_export_kwh"][1] == 0.25
    assert table["batt_charge_kwh"][1] == 2.0


async def test_backfill_chunks_through_the_recorder_executor(hass, recorder_mock):
    hours = _hours(3)
    _counter(hass, MAPPING["grid_import_total"], hours, [10.0, 11.0, 13.0])
    _counter(hass, MAPPING["grid_export_total"], hours, [4.0, 4.5, 4.5])
    _counter(hass, MAPPING["battery_charge_total"], hours, [20.0, 22.0, 22.0])
    _counter(hass, MAPPING["battery_discharge_total"], hours, [30.0, 30.25, 31.0])
    _power(hass, MAPPING["soc"], "%", hours, [50.0, 60.0, 55.0])
    _power(hass, MAPPING["load_power"], "W", hours, [400.0, 800.0, 600.0])
    await async_wait_recording_done(hass)
    start, end = hours[0], hours[-1] + dt.timedelta(hours=1)
    whole = await H.backfill_hours(hass, MAPPING, SIGNS, start, end)
    calls = []
    real = H.statistics_during_period

    def spy(*args, **kwargs):
        calls.append(args[1:3])
        return real(*args, **kwargs)

    with (
        patch("custom_components.betterybalance.history.CHUNK", dt.timedelta(hours=1)),
        patch("custom_components.betterybalance.history.statistics_during_period", spy),
    ):
        chunked = await H.backfill_hours(hass, MAPPING, SIGNS, start, end)
    # one extra hour before the window, then one call per hour
    assert len(calls) == 4 and calls[0][0] == start - dt.timedelta(hours=1)
    assert all(a[1] == b[0] for a, b in zip(calls, calls[1:])) and calls[-1][1] == end
    assert chunked.keys() == whole.keys()
    for key in whole:
        assert chunked[key].tolist() == pytest.approx(whole[key].tolist(), nan_ok=True)
    assert chunked["grid_import_kwh"][1:].tolist() == [1.0, 2.0]


async def test_backfill_reads_wh_counters_in_kwh(hass, recorder_mock):
    hours = _hours(2)
    await _site(hass, hours)
    for sid, sums in ((P1["import"][0], [100000.0, 100750.0]), (P1["export"][0], [0.0, 250.0])):
        rows = [{"start": h, "sum": s, "state": s} for h, s in zip(hours, sums)]
        meta = {
            "mean_type": StatisticMeanType.NONE, "has_sum": True, "name": None,
            "source": "recorder", "statistic_id": sid, "unit_class": "energy",
            "unit_of_measurement": "Wh",
        }
        async_import_statistics(hass, meta, rows)
    await async_wait_recording_done(hass)
    counters = {"import": [P1["import"][0]], "export": [P1["export"][0]]}
    table = await H.backfill_hours(
        hass, MAPPING, SIGNS, hours[0], hours[-1] + dt.timedelta(hours=1), grid_counters=counters
    )
    assert table["grid_import_kwh"][1] == 0.75 and table["grid_export_kwh"][1] == 0.25


async def test_backfill_builds_the_table_off_the_event_loop(hass, recorder_mock):
    import threading

    hours = _hours(2)
    await _site(hass, hours)
    threads = []
    real = H.flows_from_statistics

    def spy(*args, **kwargs):
        threads.append(threading.current_thread())
        return real(*args, **kwargs)

    with patch("custom_components.betterybalance.history.flows_from_statistics", spy):
        table = await H.backfill_hours(
            hass, MAPPING, SIGNS, hours[0], hours[-1] + dt.timedelta(hours=1)
        )
    assert table["grid_import_kwh"][1] == 1.0
    assert threads and threading.main_thread() not in threads
