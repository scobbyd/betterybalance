"""Hourly backfill: HA long-term statistics to hourly flows (design 6.2).

The recorder's ``statistics_during_period`` runs in the recorder executor, hourly, in chunks of
``CHUNK``, with ``sum`` for the counters and ``mean``, ``min``, ``max`` for power and SoC. The
rows go through ``core.inputs.flows_from_statistics`` as ``tools/export_site.py`` does: the
counter delta belongs to the later hour, so the read starts one hour before ``start`` and the
table keeps the hours in ``[start, end)``.

The grid comes from the inverter's own totals, or, with ``grid_counters`` (grid source ``p1``
or ``custom``), from those energy counters, summed per direction. The recorder returns energy
in kWh and power in W (``UNITS``) whatever unit each sensor counts in.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.core import HomeAssistant

from .core import solarman as S
from .core.inputs import HOUR_S, Rows, _ids, flows_from_statistics
from .core.timegrid import Table

CHUNK = dt.timedelta(days=31)
TYPES = {"sum", "mean", "min", "max"}
# the recorder converts each statistic to these units (a P1 reader may count in Wh)
UNITS = {"energy": "kWh", "power": "W"}


async def _statistics(
    hass: HomeAssistant, ids: set[str], start: dt.datetime, end: dt.datetime
) -> list[dict]:
    """The recorder's hourly statistics of the ids over ``[start, end)``, one dict per chunk."""
    chunks = []
    lo = start
    while lo < end:
        hi = min(lo + CHUNK, end)
        chunks.append(
            await get_instance(hass).async_add_executor_job(
                statistics_during_period, hass, lo, hi, ids, "hour", UNITS, TYPES
            )
        )
        lo = hi
    return chunks


def _rows(chunks: list[dict]) -> dict[str, Rows]:
    """The chunks in the core.inputs shape (``start`` in ms)."""
    out: dict[str, Rows] = {}
    for stats in chunks:
        for sid, rows in stats.items():
            out.setdefault(sid, []).extend(
                {**row, "start": int(round(float(row["start"]) * 1000))} for row in rows
            )
    return out


def _table(chunks: list[dict], roles: dict, battery_sign: int, lo: int, hi: int) -> Table:
    """Hourly flows of the hours in ``[lo, hi)`` (blocking: runs in the executor)."""
    table = flows_from_statistics(_rows(chunks), roles, battery_sign=battery_sign)
    keep = (table["ts"] >= lo) & (table["ts"] < hi)
    return {k: np.asarray(v)[keep] for k, v in table.items()}


async def backfill_hours(
    hass: HomeAssistant,
    mapping: dict[str, str],
    signs: dict[str, int],
    start: dt.datetime,
    end: dt.datetime,
    grid_counters: dict[str, list[str]] | None = None,
) -> Table:
    """Hourly flows (step_s 3600) for the hours in ``[start, end)`` (module docstring).

    The statistics read runs in the recorder's executor, the table build in HA's executor:
    nothing of the year-long conversion runs on the event loop.
    """
    roles = S.statistic_roles(mapping)
    if grid_counters is not None:
        roles["grid_import_kwh"] = list(grid_counters["import"])
        roles["grid_export_kwh"] = list(grid_counters["export"])
    ids = {i for spec in roles.values() for i in _ids(spec)}
    pre = start - dt.timedelta(seconds=HOUR_S)
    chunks = await _statistics(hass, ids, pre, end)
    return await hass.async_add_executor_job(
        _table, chunks, roles, signs["battery_power"], int(start.timestamp()), int(end.timestamp())
    )
