"""Solarman mapping: find a Deye hybrid of the ha-solarman `deye_p3` profile and its entities.

Works on Home Assistant entity registry entries (`config/entity_registry/list`) as plain dicts.
A role names the profile's entity by its `original_name` and its platform, because the profile
reuses a name across platforms ("PV Power" is a sensor and a number). The GEN port entities take
their name from the IO mode (Generator, Microinverter or SmartLoad), and a mode change leaves the
old entities in the registry, so the gen roles list the source variants, Generator and
Microinverter. A SmartLoad port is a load output, not a source, so no gen role takes its
names. The pack derives from the
inverter's battery settings and the BMS limits. Everything outside the supported profile raises
`Refusal`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .fmt import eu

PLATFORM = "solarman"
DEFAULT_VOLTAGE_V = 51.2
MAX_VOLTAGE_V = 100.0
ETA_RT = 0.88
MIN_SIGN_HOURS = 6
# kW x kWh: one watt against one watt-hour in one hour is 1e-6, so this is float noise only
SIGN_COV_EPS = 1e-9
INVERTER_KW = 12.0  # the SUN-12K, the supported model
IO_MODES = ("Generator", "Microinverter", "SmartLoad")
# the BMS current limits are attributes of the SoC sensor ("Battery"), not entities: the profile
# lists them under that sensor's attributes and drops the "Battery " prefix
BMS_ATTRIBUTES = {
    "bms_charge_a": "BMS Max Charging Current",
    "bms_discharge_a": "BMS Max Discharging Current",
}


@dataclass(frozen=True)
class Role:
    key: str
    name: str
    platform: str
    required: bool
    alt: tuple[str, ...] = ()  # other names of the same entity, in order of preference


SOLARMAN_ROLES: tuple[Role, ...] = (
    Role("grid_power", "Grid Power", "sensor", True),
    Role("grid_l2_power", "Grid L2 Power", "sensor", True),  # three-phase marker
    Role("grid_import_total", "Total Energy Import", "sensor", True),
    Role("grid_export_total", "Total Energy Export", "sensor", True),
    Role("battery_power", "Battery Power", "sensor", True),
    Role("soc", "Battery", "sensor", True),
    Role("battery_voltage", "Battery Voltage", "sensor", True),
    Role("battery_charge_total", "Total Battery Charge", "sensor", True),
    Role("battery_discharge_total", "Total Battery Discharge", "sensor", True),
    Role("load_power", "Load Power", "sensor", True),
    Role("load_ups_power", "Load UPS Power", "sensor", False),
    Role("load_total", "Total Load Consumption", "sensor", False),
    Role("pv_power", "PV Power", "sensor", False),
    *(Role(f"pv{i}_power", f"PV{i} Power", "sensor", False) for i in range(1, 9)),
    Role("pv_total", "Total Production", "sensor", False),
    # SmartLoad Power / SmartLoad Energy are a load output: never generation
    Role("gen_power", "Generator Power", "sensor", False, ("Microinverter Power",)),
    Role("gen_total", "Generator Energy", "sensor", False, ("Microinverter Energy",)),
    Role("io_mode", "IO Mode", "select", False),
    Role("capacity_ah", "Battery Capacity", "number", True),
    Role("max_charge_a", "Battery Max Charging Current", "number", True),
    Role("max_discharge_a", "Battery Max Discharging Current", "number", True),
    Role("shutdown_soc", "Battery Shutdown SOC", "number", True),
    Role("low_soc", "Battery Low SOC", "number", True),
)

# the profile's sign conventions: Battery Power reads positive while discharging, Grid Power
# reads positive on import
SIGNS = {"battery_power": -1, "grid_power": 1}

REASONS = (
    "no_solarman",
    "no_battery",
    "old_integration",
    "missing_entity",
    "single_phase",
    "high_voltage",
)


class Refusal(Exception):
    """The house is outside the supported Solarman profile. `reason` is one of REASONS."""

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in REASONS:
            raise ValueError(f"unknown refusal reason: {reason}")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _slug(text: str) -> str:
    """Lowercase, runs of non-alphanumerics to one underscore, stripped."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def uid_suffix(name: str, platform: str) -> str:
    """The unique id tail ha-solarman writes for an entity: slug(name) + "_" + platform.

    The whole string is slugged, so an empty name gives the platform alone (no double
    underscore), as the integration writes it.
    """
    return _slug(f"{name}_{platform}")


def _domain(entry: dict) -> str:
    return str(entry.get("entity_id", "")).split(".", 1)[0]


def devices(entries: list[dict]) -> list[str]:
    """Device ids of Solarman entries that have a Battery Power sensor, sorted."""
    return sorted(
        {
            e["device_id"]
            for e in entries
            if e.get("platform") == PLATFORM
            and e.get("original_name") == "Battery Power"
            and _domain(e) == "sensor"
            and e.get("device_id")
        }
    )


def _names(role: Role, io_mode: str | None) -> tuple[str, ...]:
    """The role's names in order of preference.

    With an IO mode, a GEN port role (its names start with an IO mode) keeps only the names of
    that mode, so the stale entities of an earlier mode never match; in SmartLoad mode it keeps
    none.
    """
    names = (role.name, *role.alt)
    if io_mode is None or not any(n.split(" ", 1)[0] in IO_MODES for n in names):
        return names
    return tuple(n for n in names if n.startswith(io_mode + " "))


def match_entities(
    entries: list[dict], device_id: str, io_mode: str | None = None
) -> dict[str, str]:
    """Role key to entity id for one Solarman device.

    An entry fills a role when its original_name equals one of the role's names and its entity
    id is on the role's platform; the first name that matches wins. `io_mode` (the state of the
    IO Mode select) limits the GEN port roles to the names of that mode (none for SmartLoad).
    Raises Refusal: old_integration
    when a unique id predates the current format, single_phase without Grid L2 Power,
    missing_entity for another required role.
    """
    own = [e for e in entries if e.get("platform") == PLATFORM and e.get("device_id") == device_id]
    for e in own:
        name = e.get("original_name") or ""
        expected = _slug(str(e.get("config_entry_id", ""))) + "_" + uid_suffix(name, _domain(e))
        if e.get("unique_id") != expected:
            raise Refusal("old_integration", f"the unique id of {name or 'an entity'} is old")
    mapping: dict[str, str] = {}
    for role in SOLARMAN_ROLES:
        found = (
            e["entity_id"]
            for name in _names(role, io_mode)
            for e in own
            if e.get("original_name") == name and _domain(e) == role.platform
        )
        entity_id = next(found, None)
        if entity_id is not None:
            mapping[role.key] = entity_id
        else:
            if role.key == "grid_l2_power":
                raise Refusal("single_phase", role.name)
            if role.required:
                raise Refusal("missing_entity", role.name)
    return mapping


def _usable(current_a: float | None) -> float | None:
    if current_a is None or not math.isfinite(current_a) or current_a <= 0:
        return None
    return float(current_a)


def _limit_a(setting_a: float | None, bms_a: float | None, name: str) -> float:
    """The smaller of the setting and the BMS limit; a value <= 0 or NaN does not count.

    A 0 A setting is an EMS holding the battery, not a rating.
    """
    usable = [a for a in (_usable(setting_a), _usable(bms_a)) if a is not None]
    if not usable:
        raise Refusal("missing_entity", name)
    return min(usable)


def derive_pack(
    capacity_ah: float,
    voltage_mean_v: float | None,
    max_charge_a: float,
    max_discharge_a: float,
    shutdown_pct: float,
    low_pct: float,
    bms_charge_a: float | None = None,
    bms_discharge_a: float | None = None,
    inverter_kw: float = INVERTER_KW,
) -> dict:
    """The site.json pack from the battery settings at the mean voltage (51,2 V without one).

    The charge and discharge currents are the smaller of the setting and the BMS limit, a value
    <= 0 or NaN ignored; the power is capped at the inverter's `inverter_kw`. A capacity
    setting <= 0 Ah or NaN refuses (missing_entity Battery Capacity).
    """
    if not (math.isfinite(capacity_ah) and capacity_ah > 0):
        raise Refusal("missing_entity", "Battery Capacity")
    v = DEFAULT_VOLTAGE_V
    if voltage_mean_v is not None and math.isfinite(voltage_mean_v):
        v = float(voltage_mean_v)
    if v > MAX_VOLTAGE_V:
        raise Refusal("high_voltage", f"battery voltage {eu(v, 1)} V")
    charge_a = _limit_a(max_charge_a, bms_charge_a, "Battery BMS Max Charging Current")
    discharge_a = _limit_a(max_discharge_a, bms_discharge_a, "Battery BMS Max Discharging Current")
    return {
        "capacity_kwh": capacity_ah * v / 1000.0,
        "charge_kw": min(charge_a * v / 1000.0, inverter_kw),
        "discharge_kw": min(discharge_a * v / 1000.0, inverter_kw),
        "soc_floor": max(shutdown_pct, low_pct) / 100.0,
        "soc_ceiling": 1.0,
        "eta_rt": ETA_RT,
        "side": "dc",
    }


def statistic_roles(mapping: dict[str, str]) -> dict:
    """The core.inputs.STAT_ROLES dict for flows_from_statistics from a role mapping."""
    roles: dict = {
        "grid_import_kwh": mapping["grid_import_total"],
        "grid_export_kwh": mapping["grid_export_total"],
        "battery_charge_kwh": mapping["battery_charge_total"],
        "battery_discharge_kwh": mapping["battery_discharge_total"],
        "soc_pct": mapping["soc"],
        # Load Power already holds the UPS port (it equals Total Load Consumption): no sum
        "load_power_w": [mapping["load_power"]],
    }
    for stat, key in (("gen_power_w", "gen_power"), ("pv_power_w", "pv_power")):
        if key in mapping:
            roles[stat] = mapping[key]
    for i in range(1, 9):
        if f"pv{i}_power" in mapping:
            roles[f"pv{i}_power_w"] = mapping[f"pv{i}_power"]
    return roles


def infer_sign(
    power_mean_w: np.ndarray, charge_kwh: np.ndarray, discharge_kwh: np.ndarray
) -> int | None:
    """+1 when the hourly power mean correlates positively with charge - discharge, -1 when
    negatively, None without evidence.

    Only hours where all three are finite count. Fewer than 6 such hours, or a covariance of
    zero up to float noise (a flat power or frozen counters), give None: the caller keeps the
    profile constant.
    """
    p = np.asarray(power_mean_w, dtype=float) / 1000.0
    net = np.asarray(charge_kwh, dtype=float) - np.asarray(discharge_kwh, dtype=float)
    ok = np.isfinite(p) & np.isfinite(net)
    if ok.sum() < MIN_SIGN_HOURS:
        return None
    p, net = p[ok], net[ok]
    # the correlation has the sign of the covariance; this form stays quiet on a flat series
    cov = float(np.sum((p - p.mean()) * (net - net.mean())))
    if abs(cov) < SIGN_COV_EPS:
        return None
    return 1 if cov > 0 else -1
