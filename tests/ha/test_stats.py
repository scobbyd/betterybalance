import datetime as dt

import pytest
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import (
    list_statistic_ids,
    statistics_during_period,
)
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.betterybalance import stats as S
from custom_components.betterybalance.const import STAT_RESULT

ZONE = "Europe/Amsterdam"


def _midnight(day: dt.date) -> int:
    from zoneinfo import ZoneInfo

    return int(dt.datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(ZONE)).timestamp())


def _sums(start: int, increments: list[float]) -> list[tuple[int, float]]:
    out, total = [], 0.0
    for k, inc in enumerate(increments):
        total += inc
        out.append((start + 3600 * k, total))
    return out


async def _days(hass, start: int, end: int) -> list[dict]:
    got = await get_instance(hass).async_add_executor_job(
        statistics_during_period,
        hass,
        dt.datetime.fromtimestamp(start, dt.UTC),
        dt.datetime.fromtimestamp(end, dt.UTC),
        {STAT_RESULT},
        "day",
        None,
        {"change"},
    )
    return got.get(STAT_RESULT, [])


async def test_import_result_gives_daily_changes_and_a_correction_changes_its_day_only(hass):
    await hass.config.async_set_time_zone(ZONE)
    start = _midnight(dt.date(2026, 10, 20))
    end = _midnight(dt.date(2026, 10, 22))
    assert end - start == 48 * 3600
    increments = [0.01 * (k % 7) for k in range(48)]
    await S.import_result(hass, _sums(start, increments))
    await async_wait_recording_done(hass)
    rows = await _days(hass, start, end)
    assert [int(r["start"]) for r in rows] == [start, start + 86400]
    assert [r["change"] for r in rows] == [
        pytest.approx(sum(increments[:24])), pytest.approx(sum(increments[24:]))
    ]
    # a TenneT correction of hour 5 of the first day: the hours from 5 on are imported again
    corrected = list(increments)
    corrected[5] += 2.5
    await S.import_result(hass, _sums(start, corrected)[5:])
    await async_wait_recording_done(hass)
    again = await _days(hass, start, end)
    assert again[0]["change"] == pytest.approx(rows[0]["change"] + 2.5)
    assert again[1]["change"] == pytest.approx(rows[1]["change"])
    ids = await get_instance(hass).async_add_executor_job(
        list_statistic_ids, hass, {STAT_RESULT}
    )
    assert ids[0]["source"] == "betterybalance" and ids[0]["has_sum"]
    assert ids[0]["statistics_unit_of_measurement"] == "EUR"
    assert ids[0]["mean_type"] == 0  # StatisticMeanType.NONE


async def test_import_result_with_no_sums_imports_nothing(hass):
    await S.import_result(hass, [])
    await async_wait_recording_done(hass)
    ids = await get_instance(hass).async_add_executor_job(
        list_statistic_ids, hass, {STAT_RESULT}
    )
    assert ids == []
