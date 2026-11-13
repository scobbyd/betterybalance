"""Solarman detection inside Home Assistant: registry entries, battery settings, statistics.

The registry and the states are in memory and read on the event loop. The recorder's statistics
are read in the recorder's executor. The pure work (role matching, pack derivation, sign
inference) is `core.solarman`.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er

from .core import solarman as S
from .core.inputs import counter_deltas

DSMR = "dsmr"
# the dsmr integration's energy counters by translation key: one total per direction, or one
# counter per tariff (summed)
P1_TOTALS = {
    "grid_import_kwh": "electricity_imported_total",
    "grid_export_kwh": "electricity_exported_total",
}
P1_TARIFFS = {
    "grid_import_kwh": tuple(f"electricity_used_tariff_{i}" for i in range(1, 5)),
    "grid_export_kwh": tuple(f"electricity_delivered_tariff_{i}" for i in range(1, 5)),
}
VOLTAGE_HOURS = 24
HISTORY_DAYS = 14  # the onboarding backtest needs this much hourly history of the grid counters
SIGN_HOURS = 48
SETTINGS = ("capacity_ah", "shutdown_soc", "low_soc")
CURRENTS = ("max_charge_a", "max_discharge_a")


def _as_dict(e: er.RegistryEntry) -> dict:
    return {
        "entity_id": e.entity_id,
        "platform": e.platform,
        "original_name": e.original_name,
        "unique_id": e.unique_id,
        "device_id": e.device_id,
        "config_entry_id": e.config_entry_id,
    }


def registry_entries(hass: HomeAssistant) -> list[dict]:
    """Every entity registry entry of the solarman platform, as a core.solarman entry dict."""
    reg = er.async_get(hass)
    return [_as_dict(e) for e in reg.entities.values() if e.platform == S.PLATFORM]


def dsmr_entries(hass: HomeAssistant) -> list[dict]:
    """Every enabled dsmr entity registry entry with its translation key and state class."""
    reg = er.async_get(hass)
    return [
        {
            "entity_id": e.entity_id,
            "platform": e.platform,
            "translation_key": e.translation_key,
            "capabilities": dict(e.capabilities or {}),
            "config_entry_id": e.config_entry_id,
        }
        for e in reg.entities.values()
        if e.platform == DSMR and e.disabled_by is None
    ]


def _is_counter(e: dict) -> bool:
    return e.get("capabilities", {}).get("state_class") == "total_increasing"


def dsmr_counter_ids(hass: HomeAssistant) -> set[str]:
    """Entity ids of the enabled dsmr `total_increasing` counters."""
    return {e["entity_id"] for e in dsmr_entries(hass) if _is_counter(e)}


def p1_counters(entries: list[dict]) -> dict[str, list[str]]:
    """The P1 meter's import and export counters: {"grid_import_kwh": [...], ...}, or {}.

    Only `total_increasing` counters count. Per direction the meter's total wins, else the
    tariff counters, summed by the consumer. The first dsmr config entry with both directions
    wins; one direction alone is no grid source.
    """
    by_entry: dict[str, list[dict]] = {}
    for e in entries:
        if e.get("platform") != DSMR:
            continue
        if not _is_counter(e):
            continue
        by_entry.setdefault(str(e.get("config_entry_id")), []).append(e)
    for found in by_entry.values():
        keys: dict[str, list[str]] = {}
        for e in found:
            keys.setdefault(e.get("translation_key"), []).append(e["entity_id"])
        out = {}
        for role, total in P1_TOTALS.items():
            ids = keys.get(total) or [i for k in P1_TARIFFS[role] for i in keys.get(k, [])]
            if ids:
                out[role] = sorted(ids)
        if len(out) == len(P1_TOTALS):
            return out
    return {}


def missing_tariffs(entries: list[dict], grid: dict[str, list[str]]) -> list[str]:
    """The dsmr tariff counters a grid pick leaves out, import first, each direction sorted.

    A direction counts when the pick holds a tariff counter of that direction: then every other
    `total_increasing` tariff counter of the same direction and dsmr config entry is missing
    from the pick (one tariff alone misses the other tariff's energy).
    """
    out: list[str] = []
    for role, tariffs in P1_TARIFFS.items():
        chosen = set(grid.get(role) or [])
        by_entry: dict[str, set[str]] = {}
        for e in entries:
            if e.get("platform") == DSMR and _is_counter(e) and e.get("translation_key") in tariffs:
                by_entry.setdefault(str(e.get("config_entry_id")), set()).add(e["entity_id"])
        for ids in by_entry.values():
            if ids & chosen:
                out.extend(sorted(ids - chosen))
    return out


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _role_name(key: str) -> str:
    return next(r.name for r in S.SOLARMAN_ROLES if r.key == key)


async def read_settings(hass: HomeAssistant, mapping: dict[str, str]) -> dict:
    """The battery settings as floats, and the BMS current limits of the SoC sensor.

    Capacity, shutdown and low SoC must have a value (else Refusal missing_entity). A current
    setting or BMS limit may be NaN: derive_pack takes the other one, or refuses.
    """
    out: dict[str, float] = {}
    for key in SETTINGS:
        state = hass.states.get(mapping[key])
        value = _float(state.state if state else None)
        if not math.isfinite(value):
            raise S.Refusal("missing_entity", f"{_role_name(key)} has no value")
        out[key] = value
    for key in CURRENTS:
        state = hass.states.get(mapping[key])
        out[key] = _float(state.state if state else None)
    soc = hass.states.get(mapping["soc"])
    attributes = soc.attributes if soc else {}
    for key, attribute in S.BMS_ATTRIBUTES.items():
        out[key] = _float(attributes.get(attribute))
    return out


async def io_mode(hass: HomeAssistant, mapping: dict[str, str]) -> str | None:
    """The state of the IO Mode select when it is one of the profile's IO modes."""
    state = hass.states.get(mapping["io_mode"]) if "io_mode" in mapping else None
    return state.state if state is not None and state.state in S.IO_MODES else None


async def _hourly(
    hass: HomeAssistant, ids: set[str], start: dt.datetime, end: dt.datetime, types: set[str]
) -> dict[str, list[dict]]:
    """Hourly statistics rows in the core.inputs shape (`start` in milliseconds)."""
    stats = await get_instance(hass).async_add_executor_job(
        statistics_during_period, hass, start, end, ids, "hour", None, types
    )
    return {
        sid: [{**row, "start": int(round(float(row["start"]) * 1000))} for row in rows]
        for sid, rows in stats.items()
    }


async def voltage_mean_v(hass: HomeAssistant, entity_id: str, now: dt.datetime) -> float | None:
    """Mean of the hourly means of the last 24 hours; None without one."""
    start = now - dt.timedelta(hours=VOLTAGE_HOURS)
    rows = (await _hourly(hass, {entity_id}, start, now, {"mean"})).get(entity_id, [])
    means = [_float(r.get("mean")) for r in rows]
    means = [m for m in means if math.isfinite(m)]
    return float(np.mean(means)) if means else None


def _sign(stats: dict[str, list[dict]], power_id: str, plus_id: str, minus_id: str) -> int | None:
    """infer_sign of a power's hourly means against the hourly deltas of two counters."""
    plus = dict(zip(*(a.tolist() for a in counter_deltas(stats.get(plus_id, [])))))
    minus = dict(zip(*(a.tolist() for a in counter_deltas(stats.get(minus_id, [])))))
    hours = sorted(set(plus) & set(minus))
    means = {int(r["start"]) // 1000: _float(r.get("mean")) for r in stats.get(power_id, [])}
    power = np.asarray([means.get(h, np.nan) for h in hours], dtype=float)
    return S.infer_sign(power, [plus[h] for h in hours], [minus[h] for h in hours])


async def inferred_signs(
    hass: HomeAssistant, mapping: dict[str, str], now: dt.datetime
) -> dict[str, int]:
    """The power signs from 48 hours of hourly statistics; the profile's SIGNS without evidence.

    battery_power against the battery charge and discharge counters, grid_power against the
    inverter's import and export counters.
    """
    pairs = {
        "battery_power": ("battery_power", "battery_charge_total", "battery_discharge_total"),
        "grid_power": ("grid_power", "grid_import_total", "grid_export_total"),
    }
    ids = {mapping[k] for roles in pairs.values() for k in roles}
    # one extra hour so the first hour in the window has a counter difference
    start = now - dt.timedelta(hours=SIGN_HOURS + 1)
    stats = await _hourly(hass, ids, start, now, {"mean", "sum"})
    signs = dict(S.SIGNS)
    for key, (power, plus, minus) in pairs.items():
        sign = _sign(stats, mapping[power], mapping[plus], mapping[minus])
        if sign is not None:
            signs[key] = sign
    return signs


async def history_days(hass: HomeAssistant, ids: list[str], now: dt.datetime) -> float:
    """Days of hourly statistics the counters `ids` hold, the shortest one, up to one day more
    than HISTORY_DAYS (the query looks no further back); 0 when one has none."""
    start = now - dt.timedelta(days=HISTORY_DAYS + 1)
    stats = await _hourly(hass, set(ids), start, now, {"sum"})
    days = []
    for sid in ids:
        rows = stats.get(sid, [])
        if not rows:
            return 0.0
        first = min(int(r["start"]) for r in rows) / 1000.0
        days.append((now.timestamp() - first) / 86400.0)
    return min(days) if days else 0.0
