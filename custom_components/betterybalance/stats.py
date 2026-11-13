"""The external result statistic ``betterybalance:result``: the settled result per hour.

The result bars (design 13.1) come from this statistic, not from the statistics of the
``sensor.betterybalance_result`` entity: a settlement lands a day after its quarters, and an
import of past hours into the entity's statistic would collide with the recorder's own hourly
compile of the same id. An external id is the integration's alone. Each row's ``sum`` (and
``state``) is the cumulative settled result in EUR to the hour's end (``live.results``), so
the statistics card draws a day's bar as the day's change. Hours imported again (a TenneT
correction) replace the stored rows of the same start.
"""

from __future__ import annotations

import datetime as dt

from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import async_add_external_statistics
from homeassistant.core import HomeAssistant

from .const import DOMAIN, STAT_RESULT

UNIT = "EUR"
NAME = "BetteryBalance result"


def metadata() -> StatisticMetaData:
    return StatisticMetaData(
        mean_type=StatisticMeanType.NONE,
        has_sum=True,
        name=NAME,
        source=DOMAIN,
        statistic_id=STAT_RESULT,
        unit_class=None,  # no unit converter for a currency
        unit_of_measurement=UNIT,
    )


async def import_result(hass: HomeAssistant, sums: list[tuple[int, float]]) -> None:
    """Queue ``(hour start, cumulative EUR)`` rows into the recorder; nothing for none.

    ``async_add_external_statistics`` only queues the import, so this never blocks the loop.
    """
    if not sums:
        return
    rows = [
        StatisticData(start=dt.datetime.fromtimestamp(int(h), dt.UTC), sum=s, state=s)
        for h, s in sums
    ]
    async_add_external_statistics(hass, metadata(), rows)
