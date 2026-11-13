import copy
import datetime as dt
import json
import pathlib
from dataclasses import asdict
from unittest.mock import patch

import pytest
from homeassistant import config_entries
from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.components.recorder.statistics import async_import_statistics
from homeassistant.data_entry_flow import InvalidData
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.betterybalance import config_flow as F
from custom_components.betterybalance import detect as D
from custom_components.betterybalance.core import solarman as S
from custom_components.betterybalance.core.site import contract as C

FIX = pathlib.Path(__file__).parent.parent / "fixtures"
SOLARMAN_ENTRY = "01JSYNTHETICENTRY0000000000"
INVERTER_NOTE = (
    "your inverter's grid measurement; a P1 reader in Home Assistant gives better export figures"
)


def _second_inverter(reg: list[dict], states: list[dict]) -> tuple[list[dict], list[dict]]:
    """A copy of the synthetic inverter (dev1) under a second Solarman config entry."""
    entry_id = SOLARMAN_ENTRY[:-1] + "1"
    ids: dict[str, str] = {}
    more = []
    for e in reg:
        if e["platform"] != "solarman" or e["device_id"] != "dev1":
            continue
        e = dict(e)
        ids[e["entity_id"]] = e["entity_id"] + "_2"
        e["entity_id"] = ids[e["entity_id"]]
        e["unique_id"] = e["unique_id"].replace(SOLARMAN_ENTRY.lower(), entry_id.lower())
        e["device_id"], e["config_entry_id"] = "dev5", entry_id
        more.append(e)
    copies = [{**s, "entity_id": ids[s["entity_id"]]} for s in states if s["entity_id"] in ids]
    return reg + more, states + copies


async def _install_solarman(hass, drop=(), p1=False, second=False):
    """Register the synthetic registry fixture under MockConfigEntries, set the states fixture.

    Every fixture config entry becomes a MockConfigEntry with the fixture's entry id (the
    Solarman unique ids carry it), every fixture device a registry device, every entity a
    registry entry with the fixture's entity id, and every state a state. Entities whose
    original name is in `drop` are left out. The dsmr P1 meter is installed only with `p1`;
    `second` adds a copy of the inverter under a second Solarman entry. Returns the fixture
    device id to Home Assistant device id map.
    """
    reg = json.loads((FIX / "synthetic_solarman_registry.json").read_text())
    states = json.loads((FIX / "synthetic_solarman_states.json").read_text())
    if not p1:
        reg = [e for e in reg if e["platform"] != "dsmr"]
    if second:
        reg, states = _second_inverter(reg, states)
    reg = [e for e in reg if e.get("original_name") not in drop]
    entries: dict[str, MockConfigEntry] = {}
    devices: dict[str, str] = {}
    dev_reg, ent_reg = dr.async_get(hass), er.async_get(hass)
    for e in reg:
        ce = e["config_entry_id"]
        if ce not in entries:
            entries[ce] = MockConfigEntry(domain=e["platform"], entry_id=ce, title="Synthetic")
            entries[ce].add_to_hass(hass)
        if e["device_id"] not in devices:
            device = dev_reg.async_get_or_create(
                config_entry_id=ce,
                identifiers={(e["platform"], e["device_id"])},
                name=f"Synthetic {e['device_id']}",
            )
            devices[e["device_id"]] = device.id
        domain, object_id = e["entity_id"].split(".", 1)
        ent_reg.async_get_or_create(
            domain,
            e["platform"],
            e["unique_id"],
            suggested_object_id=object_id,
            config_entry=entries[ce],
            device_id=devices[e["device_id"]],
            original_name=e.get("original_name"),
            translation_key=e.get("translation_key"),
            capabilities=e.get("capabilities"),
        )
    kept = {e["entity_id"] for e in reg}
    for s in states:
        if s["entity_id"] in kept:
            hass.states.async_set(s["entity_id"], s["state"], s["attributes"])
    await hass.async_block_till_done()
    return devices


async def _start(hass):
    return await hass.config_entries.flow.async_init(
        "betterybalance", context={"source": config_entries.SOURCE_USER}
    )


def _suggested(r) -> dict:
    return {
        k.schema: k.description["suggested_value"]
        for k in r["data_schema"].schema
        if k.description and "suggested_value" in k.description
    }


def _defaults(r) -> dict:
    return {k.schema: k.default() for k in r["data_schema"].schema}


def _import_hourly(hass, statistic_id: str, unit: str, rows: list[dict], has_sum: bool) -> None:
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


def _hours(now: dt.datetime, n: int) -> list[dt.datetime]:
    top = now.replace(minute=0, second=0, microsecond=0)
    return [top - dt.timedelta(hours=n - i) for i in range(n)]


async def test_no_solarman_aborts_with_the_reason(hass):
    r = await _start(hass)
    assert r["type"] == "abort" and r["reason"] == "no_solarman"
    assert r["description_placeholders"]["issue_url"]


async def test_no_battery_aborts_with_the_reason(hass):
    await _install_solarman(hass, drop=("Battery Power",))
    r = await _start(hass)
    assert r["type"] == "abort" and r["reason"] == "no_battery"


async def test_one_device_prefills_the_pack(hass):
    await _install_solarman(hass)
    r = await _start(hass)
    assert r["type"] == "form" and r["step_id"] == "inverter"
    defaults = _defaults(r)
    assert defaults["capacity_kwh"] == pytest.approx(30.72) and defaults["soc_floor_pct"] == 20
    # 240 A setting under the 300 A BMS limit at 51,2 V is 12,288 kW, capped at the 12 kW inverter
    assert defaults["charge_kw"] == pytest.approx(12.0)
    assert defaults["discharge_kw"] == pytest.approx(12.0)


async def test_single_phase_profile_is_refused(hass):
    await _install_solarman(hass, drop=("Grid L2 Power",))
    r = await _start(hass)
    assert r["type"] == "abort" and r["reason"] == "single_phase"


async def test_missing_entity_names_it_in_the_placeholders(hass):
    await _install_solarman(hass, drop=("Battery Capacity",))
    r = await _start(hass)
    assert r["type"] == "abort" and r["reason"] == "missing_entity"
    placeholders = r["description_placeholders"]
    assert placeholders["detail"] == "Battery Capacity" and placeholders["issue_url"]


async def test_unavailable_setting_is_a_missing_entity(hass):
    await _install_solarman(hass)
    hass.states.async_set("number.synthetic_battery_low_soc", "unavailable")
    r = await _start(hass)
    assert r["type"] == "abort" and r["reason"] == "missing_entity"
    assert "Battery Low SOC" in r["description_placeholders"]["detail"]


async def test_old_unique_ids_are_refused(hass):
    await _install_solarman(hass)
    ent_reg = er.async_get(hass)
    ent_reg.async_update_entity("sensor.synthetic_battery_power", new_unique_id="old_style_id")
    r = await _start(hass)
    assert r["type"] == "abort" and r["reason"] == "old_integration"


async def test_registry_entries_are_the_solarman_ones(hass):
    devices = await _install_solarman(hass, p1=True)
    entries = D.registry_entries(hass)
    assert entries and {e["platform"] for e in entries} == {"solarman"}
    grid = next(e for e in entries if e["entity_id"] == "sensor.synthetic_grid_power")
    assert grid == {
        "entity_id": "sensor.synthetic_grid_power",
        "platform": "solarman",
        "original_name": "Grid Power",
        "unique_id": "01jsyntheticentry0000000000_grid_power_sensor",
        "device_id": devices["dev1"],
        "config_entry_id": SOLARMAN_ENTRY,
    }


async def test_read_settings_takes_the_bms_limits_from_the_soc_sensor(hass):
    devices = await _install_solarman(hass)
    mapping = S.match_entities(D.registry_entries(hass), devices["dev1"])
    settings = await D.read_settings(hass, mapping)
    assert settings == {
        "capacity_ah": 600.0,
        "max_charge_a": 240.0,
        "max_discharge_a": 240.0,
        "shutdown_soc": 10.0,
        "low_soc": 20.0,
        "bms_charge_a": 300.0,
        "bms_discharge_a": 300.0,
    }


async def test_io_mode_picks_the_gen_entities_of_that_mode(hass):
    await _install_solarman(hass)
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await _start(hass)
        r = await hass.config_entries.flow.async_configure(r["flow_id"], _pack_input())
    entities = r["description_placeholders"]["data"]["entities"]
    # the states fixture puts the IO mode on Microinverter
    assert entities["gen_power"] == "sensor.synthetic_microinverter_power"


async def test_several_devices_ask_which_one(hass):
    devices = await _install_solarman(hass, second=True)
    r = await _start(hass)
    assert r["type"] == "form" and r["step_id"] == "device"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"device": devices["dev5"]})
    assert r["type"] == "form" and r["step_id"] == "inverter"
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await hass.config_entries.flow.async_configure(r["flow_id"], _pack_input())
    data = r["description_placeholders"]["data"]
    assert data["device_id"] == devices["dev5"]
    assert data["entities"]["battery_power"] == "sensor.synthetic_battery_power_2"


@pytest.mark.parametrize("device", ["dev1", "dev5"])
async def test_a_second_entry_is_refused(hass, device):
    devices = await _install_solarman(hass, second=True)
    MockConfigEntry(domain="betterybalance", unique_id=devices[device]).add_to_hass(hass)
    r = await _start(hass)  # the configured inverter or another one: one entry per HA
    assert r["type"] == "abort" and r["reason"] == "single_instance_allowed"


async def test_a_flow_started_before_the_first_entry_aborts_at_its_first_step(hass):
    from custom_components.betterybalance.config_flow import BetteryConfigFlow

    devices = await _install_solarman(hass)
    MockConfigEntry(domain="betterybalance", unique_id=devices["dev1"]).add_to_hass(hass)
    flow = BetteryConfigFlow()
    flow.hass = hass
    r = await flow.async_step_user()
    assert r["type"] == "abort" and r["reason"] == "single_instance_allowed"


def test_the_manifest_allows_one_entry():
    import json
    import pathlib

    manifest = pathlib.Path(__file__).parents[2] / "custom_components/betterybalance/manifest.json"
    assert json.loads(manifest.read_text())["single_config_entry"] is True


def _pack_input() -> dict:
    return {"capacity_kwh": 30.0, "charge_kw": 10.0, "discharge_kw": 11.0, "soc_floor_pct": 15}


async def _capture(self):
    """Stand-in for the next screen: hand the collected data back in the abort placeholders."""
    return self.async_abort(reason="captured", description_placeholders={"data": self._data})


async def test_without_p1_the_grid_comes_from_the_inverter(hass):
    await _install_solarman(hass)
    r = await _start(hass)
    defaults = _defaults(r)
    assert defaults["grid_import_kwh"] == [] and defaults["grid_export_kwh"] == []
    assert r["description_placeholders"]["grid_note"] == INVERTER_NOTE
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await hass.config_entries.flow.async_configure(r["flow_id"], _pack_input())
    data = r["description_placeholders"]["data"]
    assert data["grid_source"] == "inverter"
    assert data["entities"]["grid_import_kwh"] == ["sensor.synthetic_total_energy_import"]
    assert data["entities"]["grid_export_kwh"] == ["sensor.synthetic_total_energy_export"]
    pack = data["site"]["pack"]
    assert pack["capacity_kwh"] == 30.0 and pack["charge_kw"] == 10.0
    assert pack["discharge_kw"] == 11.0 and pack["soc_floor"] == pytest.approx(0.15)
    assert pack["side"] == "dc" and pack["eta_rt"] == pytest.approx(0.88)
    assert data["signs"] == S.SIGNS


async def test_a_p1_meter_is_prefilled_as_the_grid_source(hass):
    await _install_solarman(hass, p1=True)
    r = await _start(hass)
    defaults = _defaults(r)
    imports = [
        "sensor.synthetic_p1_energy_import_tariff_1",
        "sensor.synthetic_p1_energy_import_tariff_2",
    ]
    exports = [
        "sensor.synthetic_p1_energy_export_tariff_1",
        "sensor.synthetic_p1_energy_export_tariff_2",
    ]
    assert defaults["grid_import_kwh"] == imports and defaults["grid_export_kwh"] == exports
    assert r["description_placeholders"]["grid_note"] != INVERTER_NOTE
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await hass.config_entries.flow.async_configure(
            r["flow_id"], {**_pack_input(), "grid_import_kwh": imports, "grid_export_kwh": exports}
        )
    data = r["description_placeholders"]["data"]
    assert data["grid_source"] == "p1"
    assert data["entities"]["grid_import_kwh"] == imports
    assert data["entities"]["grid_export_kwh"] == exports
    # the inverter's own counters stay mapped under their role keys
    assert data["entities"]["grid_import_total"] == "sensor.synthetic_total_energy_import"


async def test_a_grid_meter_needs_both_directions(hass):
    await _install_solarman(hass, p1=True)
    r = await _start(hass)
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"],
        {
            **_pack_input(),
            "grid_import_kwh": ["sensor.synthetic_p1_energy_import_tariff_1"],
            "grid_export_kwh": [],
        },
    )
    assert r["type"] == "form" and r["errors"] == {"base": "grid_pair"}
    # the re-shown form keeps the user's edits, not the derived defaults
    assert _suggested(r) == {
        **_pack_input(),
        "grid_import_kwh": ["sensor.synthetic_p1_energy_import_tariff_1"],
        "grid_export_kwh": [],
    }


async def test_the_floor_must_leave_room_below_the_ceiling(hass):
    await _install_solarman(hass)
    r = await _start(hass)
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"], {**_pack_input(), "soc_floor_pct": 100}
    )
    assert r["type"] == "form" and r["errors"] == {"soc_floor_pct": "floor_range"}
    suggested = _suggested(r)
    assert suggested["capacity_kwh"] == 30.0 and suggested["soc_floor_pct"] == 100
    # a corrected resubmit of the re-shown form goes through
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await hass.config_entries.flow.async_configure(
            r["flow_id"], {**_pack_input(), "soc_floor_pct": 25}
        )
    assert r["description_placeholders"]["data"]["site"]["pack"]["soc_floor"] == 0.25


async def test_voltage_mean_reads_the_last_day_of_hourly_means(hass):
    await _install_solarman(hass)
    now = dt_util.utcnow()
    hours = _hours(now, 30)
    # the six oldest hours lie outside the last 24 h and read 60 V
    rows = [{"start": h, "mean": 60.0 if i < 6 else 48.0} for i, h in enumerate(hours)]
    _import_hourly(hass, "sensor.synthetic_battery_voltage", "V", rows, has_sum=False)
    await async_wait_recording_done(hass)
    mean = await D.voltage_mean_v(hass, "sensor.synthetic_battery_voltage", now)
    assert mean == pytest.approx(48)
    assert await D.voltage_mean_v(hass, "sensor.synthetic_load_power", now) is None
    r = await _start(hass)
    assert _defaults(r)["capacity_kwh"] == pytest.approx(600 * 48 / 1000)


async def test_inferred_signs_follow_the_counters(hass):
    devices = await _install_solarman(hass)
    mapping = S.match_entities(D.registry_entries(hass), devices["dev1"])
    now = dt_util.utcnow()
    assert await D.inferred_signs(hass, mapping, now) == S.SIGNS
    hours = _hours(now, 12)
    # this inverter reads battery power positive while charging: the opposite of the profile
    net = [2.0, -1.0, 3.0, -2.0, 1.0, 0.5, -3.0, 2.5, -0.5, 1.5, -1.5, 0.0]
    charge = discharge = 0.0
    power, charges, discharges = [], [], []
    for h, kwh in zip(hours, net, strict=True):
        charge += max(kwh, 0.0)
        discharge += max(-kwh, 0.0)
        power.append({"start": h, "mean": kwh * 1000.0})
        charges.append({"start": h, "state": charge, "sum": charge})
        discharges.append({"start": h, "state": discharge, "sum": discharge})
    _import_hourly(hass, mapping["battery_power"], "W", power, has_sum=False)
    _import_hourly(hass, mapping["battery_charge_total"], "kWh", charges, has_sum=True)
    _import_hourly(hass, mapping["battery_discharge_total"], "kWh", discharges, has_sum=True)
    await async_wait_recording_done(hass)
    assert await D.inferred_signs(hass, mapping, now) == {"battery_power": 1, "grid_power": 1}


def test_p1_counters_prefer_the_totals_and_skip_other_state_classes():
    reg = json.loads((FIX / "synthetic_solarman_registry.json").read_text())
    p1 = [e for e in reg if e["platform"] == "dsmr"]
    got = D.p1_counters(p1)
    assert got["grid_import_kwh"] == [
        "sensor.synthetic_p1_energy_import_tariff_1",
        "sensor.synthetic_p1_energy_import_tariff_2",
    ]
    totals = copy.deepcopy(p1[:2])
    totals[0]["translation_key"] = "electricity_imported_total"
    totals[0]["entity_id"] = "sensor.synthetic_p1_import_total"
    totals[1]["translation_key"] = "electricity_exported_total"
    totals[1]["entity_id"] = "sensor.synthetic_p1_export_total"
    assert D.p1_counters(p1 + totals) == {
        "grid_import_kwh": ["sensor.synthetic_p1_import_total"],
        "grid_export_kwh": ["sensor.synthetic_p1_export_total"],
    }
    # a counter that is not total_increasing does not count; one direction alone is no source
    measured = [dict(e, capabilities={"state_class": "measurement"}) for e in p1]
    assert D.p1_counters(measured) == {}
    assert D.p1_counters([e for e in p1 if "import" in e["entity_id"]]) == {}


async def _grid_choice(hass, imports: list[str], exports: list[str]) -> dict:
    r = await _start(hass)
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await hass.config_entries.flow.async_configure(
            r["flow_id"], {**_pack_input(), "grid_import_kwh": imports, "grid_export_kwh": exports}
        )
    return r["description_placeholders"]["data"]


async def test_the_inverter_totals_chosen_by_hand_are_the_inverter_source(hass):
    await _install_solarman(hass, p1=True)
    data = await _grid_choice(
        hass, ["sensor.synthetic_total_energy_import"], ["sensor.synthetic_total_energy_export"]
    )
    assert data["grid_source"] == "inverter"


async def test_other_counters_are_a_custom_source(hass):
    await _install_solarman(hass, p1=True)
    # one dsmr counter and one inverter counter: neither the P1 meter nor the inverter alone
    data = await _grid_choice(
        hass,
        ["sensor.synthetic_p1_energy_import_tariff_1"],
        ["sensor.synthetic_total_energy_export"],
    )
    assert data["grid_source"] == "custom"
    assert data["entities"]["grid_export_kwh"] == ["sensor.synthetic_total_energy_export"]


async def test_the_inverter_note_shows_only_for_the_inverter_source(hass):
    await _install_solarman(hass, p1=True)
    r = await _start(hass)
    # a custom pair with one direction missing: the re-shown form names the custom source
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"],
        {
            **_pack_input(),
            "grid_import_kwh": ["sensor.synthetic_load_power"],
            "grid_export_kwh": [],
        },
    )
    assert r["errors"] == {"base": "grid_pair"}
    assert r["description_placeholders"]["grid_note"] == F.CUSTOM_NOTE


async def test_frozen_counters_keep_the_profile_signs(hass):
    devices = await _install_solarman(hass)
    mapping = S.match_entities(D.registry_entries(hass), devices["dev1"])
    now = dt_util.utcnow()
    hours = _hours(now, 12)
    # the powers move but every counter is frozen: no evidence, so the grid sign stays +1 (a
    # zero covariance used to read as -1) and the battery sign stays -1
    power = [{"start": h, "mean": 1000.0 * (i % 3 - 1)} for i, h in enumerate(hours)]
    frozen = [{"start": h, "state": 7.0, "sum": 7.0} for h in hours]
    for key in ("battery_power", "grid_power"):
        _import_hourly(hass, mapping[key], "W", power, has_sum=False)
    for key in (
        "battery_charge_total",
        "battery_discharge_total",
        "grid_import_total",
        "grid_export_total",
    ):
        _import_hourly(hass, mapping[key], "kWh", frozen, has_sum=True)
    await async_wait_recording_done(hass)
    assert await D.inferred_signs(hass, mapping, now) == S.SIGNS
    with patch.object(F.BetteryConfigFlow, "_after_inverter", _capture):
        r = await _start(hass)
        r = await hass.config_entries.flow.async_configure(r["flow_id"], _pack_input())
    assert r["description_placeholders"]["data"]["signs"] == S.SIGNS


# --- screens 2 to 4 -----------------------------------------------------------------------------

CONTRACT = {
    "location": {"latitude": 52.149, "longitude": 5.231},
    "fee_import_ct": 1.65,
    "fee_export_ct": 1.65,
    "dual_pricing": True,
    "btw_pct": 21,
    "tax_mode": "net_exporter",
}
PLANES = {
    "tilt_1": 35,
    "azimuth_1": 135,
    "kwp_1": 4.0,
    "tilt_2": 30,
    "azimuth_2": 225,
    "kwp_2": 3.0,
    "tilt_gen": 20,
    "azimuth_gen": 180,
    "kwp_gen": 0.8,
}
NO_GEN = ("Generator Power", "Microinverter Power")


async def _to_location(hass, p1=False, grid: dict | None = None):
    await _install_solarman(hass, p1=p1)
    hass.config.latitude, hass.config.longitude = 52.149, 5.231
    r = await _start(hass)
    return await hass.config_entries.flow.async_configure(
        r["flow_id"], {**_pack_input(), **(grid or {})}
    )


async def test_full_flow_creates_the_entry(hass):
    await _install_solarman(hass)
    hass.config.latitude, hass.config.longitude = 52.149, 5.231
    flow = hass.config_entries.flow
    r = await flow.async_init("betterybalance", context={"source": config_entries.SOURCE_USER})
    r = await flow.async_configure(
        r["flow_id"],
        {"capacity_kwh": 30.72, "charge_kw": 12.0, "discharge_kw": 12.0, "soc_floor_pct": 20},
    )
    r = await flow.async_configure(r["flow_id"], CONTRACT)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "pv_fill"})
    r = await flow.async_configure(r["flow_id"], PLANES)
    # the fixture recorder holds no statistics
    assert r["step_id"] == "backtest" and r["description_placeholders"]["reason"]
    r = await flow.async_configure(r["flow_id"], {"run_backtest": False})
    d = r["data"]
    assert r["type"] == "create_entry" and r["title"] == "BetteryBalance"
    assert d["site"]["latitude"] == 52.1 and d["site"]["longitude"] == 5.2
    assert d["site"]["terms"]["fee_import_eur_kwh"] == pytest.approx(0.0165)
    assert d["site"]["execution_profile"] == "deye_amps" and d["site"]["loss_profile"] == "deye"
    assert d["site"]["pv"]["planes"][1] == {
        "tilt_deg": 30,
        "azimuth_deg": 225,
        "kwp": 3.0,
        "string": "pv2",
    }
    assert [p["string"] for p in d["site"]["pv"]["planes"]] == ["pv1", "pv2", "gen"]
    assert len(d["install_id"]) == 32
    assert d["pv_mode"] == "fill" and d["backtest"] is False and d["grid_source"] == "inverter"
    assert d["entities"]["grid_import_kwh"] == ["sensor.synthetic_total_energy_import"]
    assert d["device_id"] and d["signs"] == S.SIGNS
    # the stored site is a valid contract; the knobs start at the shipped defaults
    cfg = C.from_dict(d["site"])
    assert cfg.pack.capacity_kwh == pytest.approx(30.72) and cfg.terms.btw == pytest.approx(0.21)
    assert cfg.terms.balans_revenue_eur_month == 0.0 and cfg.terms.tax_mode == "net_exporter"
    assert r["result"].options == {
        "knobs": asdict(C.Knobs()),
        "tennet_key": None,
        "export_flows": False,
        "pool_results": False,
        "pack_source": None,
        "fetch_ida": False,
    }


async def test_location_and_contract_are_prefilled(hass):
    r = await _to_location(hass)
    assert r["type"] == "form" and r["step_id"] == "location_contract"
    defaults = _defaults(r)
    assert defaults["location"] == {"latitude": 52.1, "longitude": 5.2}
    assert defaults["fee_import_ct"] == 1.65 and defaults["fee_export_ct"] == 1.65
    assert defaults["dual_pricing"] is True and defaults["btw_pct"] == 21
    assert defaults["tax_mode"] == "net_exporter" and defaults["balans_revenue_eur_month"] == 0
    assert r["description_placeholders"]["grid_warning"] == ""


async def test_the_contract_fields_reach_the_terms(hass):
    r = await _to_location(hass)
    r = await hass.config_entries.flow.async_configure(
        r["flow_id"],
        {
            **CONTRACT,
            "location": {"latitude": 51.96, "longitude": 5.16},
            "fee_import_ct": 2.0,
            "fee_export_ct": 0.5,
            "dual_pricing": False,
            "btw_pct": 9,
            "tax_mode": "none",
            "balans_revenue_eur_month": 12.5,
        },
    )
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"next_step_id": "pv_learn"})
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"run_backtest": False})
    site = r["data"]["site"]
    assert site["latitude"] == 52.0 and site["longitude"] == 5.2
    assert site["terms"] == {
        "fee_import_eur_kwh": pytest.approx(0.02),
        "fee_export_eur_kwh": pytest.approx(0.005),
        "dual_pricing": False,
        "btw": pytest.approx(0.09),
        "tax_mode": "none",
        "balans_revenue_eur_month": 12.5,
    }


async def test_learning_the_panels_stores_no_planes(hass):
    r = await _to_location(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], CONTRACT)
    assert r["type"] == "menu" and r["step_id"] == "pv"
    assert r["menu_options"] == ["pv_learn", "pv_fill"]
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"next_step_id": "pv_learn"})
    assert r["step_id"] == "backtest"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"run_backtest": False})
    assert r["data"]["pv_mode"] == "learn" and r["data"]["site"]["pv"] == {}


async def test_fill_in_has_one_row_per_string(hass):
    r = await _to_location(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], CONTRACT)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"next_step_id": "pv_fill"})
    assert r["type"] == "form" and r["step_id"] == "pv_fill"
    # the synthetic inverter reports PV1 and PV2
    assert {k.schema for k in r["data_schema"].schema} == set(PLANES)


async def _to_fill(hass, drop=()):
    await _install_solarman(hass, drop=drop)
    r = await _start(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], _pack_input())
    r = await hass.config_entries.flow.async_configure(r["flow_id"], CONTRACT)
    return await hass.config_entries.flow.async_configure(r["flow_id"], {"next_step_id": "pv_fill"})


async def test_fill_in_without_strings_asks_for_one(hass):
    r = await _to_fill(hass, drop=("PV1 Power", "PV2 Power", *NO_GEN))
    assert {k.schema for k in r["data_schema"].schema} == {"tilt_1", "azimuth_1", "kwp_1"}


async def test_fill_in_without_a_gen_port_has_no_gen_row(hass):
    r = await _to_fill(hass, drop=NO_GEN)
    keys = {k.schema for k in r["data_schema"].schema}
    assert keys == {k for k in PLANES if not k.endswith("_gen")}


async def test_a_zero_kwp_row_is_skipped(hass):
    r = await _to_fill(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {**PLANES, "kwp_2": 0})
    assert r["step_id"] == "backtest"
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"run_backtest": False})
    planes = r["data"]["site"]["pv"]["planes"]
    assert [p["string"] for p in planes] == ["pv1", "gen"]
    assert planes[1] == {"tilt_deg": 20, "azimuth_deg": 180, "kwp": 0.8, "string": "gen"}


async def test_every_row_at_zero_kwp_is_a_form_error(hass):
    r = await _to_fill(hass)
    zero = {**PLANES, "kwp_1": 0, "kwp_2": 0, "kwp_gen": 0}
    r = await hass.config_entries.flow.async_configure(r["flow_id"], zero)
    assert r["type"] == "form" and r["step_id"] == "pv_fill"
    assert r["errors"] == {"base": "no_planes"}
    assert _suggested(r)["kwp_1"] == 0


async def test_the_backtest_is_on_with_two_weeks_of_statistics(hass):
    r = await _to_location(hass)
    now = dt_util.utcnow()
    rows = [
        {"start": h, "state": float(i), "sum": float(i)} for i, h in enumerate(_hours(now, 15 * 24))
    ]
    _import_hourly(hass, "sensor.synthetic_total_energy_import", "kWh", rows, has_sum=True)
    await async_wait_recording_done(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], CONTRACT)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"next_step_id": "pv_learn"})
    assert r["step_id"] == "backtest" and _defaults(r)["run_backtest"] is True
    assert r["description_placeholders"]["reason"] == ""
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"run_backtest": True})
    assert r["data"]["backtest"] is True


async def test_a_short_history_turns_the_backtest_off_with_the_reason(hass):
    r = await _to_location(hass)
    now = dt_util.utcnow()
    rows = [
        {"start": h, "state": float(i), "sum": float(i)} for i, h in enumerate(_hours(now, 5 * 24))
    ]
    _import_hourly(hass, "sensor.synthetic_total_energy_import", "kWh", rows, has_sum=True)
    await async_wait_recording_done(hass)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], CONTRACT)
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {"next_step_id": "pv_learn"})
    assert _defaults(r)["run_backtest"] is False
    assert "5" in r["description_placeholders"]["reason"]


P1_IMPORTS = [
    "sensor.synthetic_p1_energy_import_tariff_1",
    "sensor.synthetic_p1_energy_import_tariff_2",
]
P1_EXPORTS = [
    "sensor.synthetic_p1_energy_export_tariff_1",
    "sensor.synthetic_p1_energy_export_tariff_2",
]


async def test_one_tariff_counter_of_two_warns_on_the_next_screen(hass):
    r = await _to_location(
        hass,
        p1=True,
        grid={"grid_import_kwh": P1_IMPORTS[:1], "grid_export_kwh": P1_EXPORTS},
    )
    assert r["step_id"] == "location_contract"
    warning = r["description_placeholders"]["grid_warning"]
    assert "sensor.synthetic_p1_energy_import_tariff_2" in warning
    assert "export" not in warning  # both export tariffs were chosen
    # the warning does not block
    r = await hass.config_entries.flow.async_configure(r["flow_id"], CONTRACT)
    assert r["step_id"] == "pv"


async def test_both_tariff_counters_give_no_warning(hass):
    r = await _to_location(
        hass, p1=True, grid={"grid_import_kwh": P1_IMPORTS, "grid_export_kwh": P1_EXPORTS}
    )
    assert r["description_placeholders"]["grid_warning"] == ""


def test_missing_tariffs_names_the_unchosen_counters():
    reg = json.loads((FIX / "synthetic_solarman_registry.json").read_text())
    p1 = [e for e in reg if e["platform"] == "dsmr"]
    grid = {"grid_import_kwh": P1_IMPORTS[:1], "grid_export_kwh": P1_EXPORTS[1:]}
    assert D.missing_tariffs(p1, grid) == [P1_IMPORTS[1], P1_EXPORTS[0]]
    assert (
        D.missing_tariffs(p1, {"grid_import_kwh": P1_IMPORTS, "grid_export_kwh": P1_EXPORTS}) == []
    )
    # counters of another integration are not tariff counters
    other = {"grid_import_kwh": ["sensor.synthetic_total_energy_import"], "grid_export_kwh": []}
    assert D.missing_tariffs(p1, other) == []


# --- the options flow ---------------------------------------------------------------------------

OPTIONS = {
    "knobs": asdict(C.Knobs()),
    "tennet_key": None,
    "export_flows": False,
    "pool_results": False,
    "pack_source": None,
    "fetch_ida": False,
}


async def _options(hass, step: str, options: dict | None = None):
    entry = MockConfigEntry(
        domain="betterybalance", data={}, options=options or OPTIONS, title="BetteryBalance"
    )
    entry.add_to_hass(hass)
    r = await hass.config_entries.options.async_init(entry.entry_id)
    assert r["type"] == "menu"
    assert r["menu_options"] == ["advanced", "feeds", "export", "pack", "view", "backtest"]
    r = await hass.config_entries.options.async_configure(r["flow_id"], {"next_step_id": step})
    return entry, r


async def test_advanced_prefills_every_knob(hass):
    _, r = await _options(hass, "advanced")
    assert r["type"] == "form" and r["step_id"] == "advanced"
    assert _defaults(r) == asdict(C.Knobs())


async def test_advanced_saves_the_changed_knobs(hass):
    entry, r = await _options(hass, "advanced")
    knobs = {**asdict(C.Knobs()), "hurdle_ct_kwh": 4.5, "impl": "ranking", "dwell_s": 600}
    r = await hass.config_entries.options.async_configure(r["flow_id"], knobs)
    assert r["type"] == "create_entry"
    assert entry.options["knobs"]["hurdle_ct_kwh"] == 4.5
    assert entry.options["knobs"]["impl"] == "ranking" and entry.options["knobs"]["dwell_s"] == 600
    assert isinstance(entry.options["knobs"]["dwell_s"], int)
    assert entry.options["export_flows"] is False and entry.options["pack_source"] is None


@pytest.mark.parametrize(
    "change,error",
    [
        ({"latest_commit_min": 15}, {"latest_commit_min": "commit_minute_range"}),
        ({"predicted_widen": 0.9}, {"predicted_widen": "widen_range"}),
        ({"learning_widen": 0.5}, {"learning_widen": "widen_range"}),
        ({"latency_s": 60.5}, {"latency_s": "knob_integer"}),
        ({"latency_s": 301}, {"latency_s": "latency_range"}),
        ({"dwell_s": 901}, {"dwell_s": "dwell_range"}),
    ],
)
async def test_advanced_refuses_knobs_out_of_bounds(hass, change, error):
    _, r = await _options(hass, "advanced")
    r = await hass.config_entries.options.async_configure(
        r["flow_id"], {**asdict(C.Knobs()), **change}
    )
    assert r["type"] == "form" and r["errors"] == error


@pytest.mark.parametrize("change", [{"latency_s": -1}, {"dwell_s": -60}])
async def test_advanced_schema_refuses_negative_timing(hass, change):
    _, r = await _options(hass, "advanced")
    with pytest.raises(InvalidData):
        await hass.config_entries.options.async_configure(
            r["flow_id"], {**asdict(C.Knobs()), **change}
        )


async def test_advanced_accepts_the_bounds_themselves(hass):
    entry, r = await _options(hass, "advanced")
    edge = {
        **asdict(C.Knobs()), "latest_commit_min": 14, "predicted_widen": 1.0,
        "latency_s": 300, "dwell_s": 900,
    }
    r = await hass.config_entries.options.async_configure(r["flow_id"], edge)
    assert r["type"] == "create_entry" and entry.options["knobs"]["latest_commit_min"] == 14
    assert entry.options["knobs"]["latency_s"] == 300 and entry.options["knobs"]["dwell_s"] == 900


async def test_advanced_refuses_quantiles_outside_zero_one(hass):
    _, r = await _options(hass, "advanced")
    knobs = {**asdict(C.Knobs()), "q_take": 1.5}
    r = await hass.config_entries.options.async_configure(r["flow_id"], knobs)
    assert r["type"] == "form" and r["errors"] == {"q_take": "knob_range"}


def test_the_advanced_descriptions_are_the_knob_help():
    strings = json.loads(
        (FIX.parent.parent / "custom_components/betterybalance/strings.json").read_text()
    )
    advanced = strings["options"]["step"]["advanced"]
    assert advanced["data_description"] == C.KNOB_HELP
    assert set(advanced["data"]) == set(C.KNOB_HELP)


async def test_feeds_stores_the_key_and_an_empty_key_as_none(hass):
    entry, r = await _options(hass, "feeds")
    r = await hass.config_entries.options.async_configure(r["flow_id"], {"tennet_key": " abc "})
    assert r["type"] == "create_entry" and entry.options["tennet_key"] == "abc"
    assert entry.options["knobs"] == asdict(C.Knobs())
    r = await hass.config_entries.options.async_init(entry.entry_id)
    r = await hass.config_entries.options.async_configure(r["flow_id"], {"next_step_id": "feeds"})
    r = await hass.config_entries.options.async_configure(r["flow_id"], {})
    assert entry.options["tennet_key"] is None


async def test_export_defaults_off_and_saves(hass):
    entry, r = await _options(hass, "export")
    assert _defaults(r) == {"export_flows": False, "pool_results": False}
    r = await hass.config_entries.options.async_configure(
        r["flow_id"], {"export_flows": True, "pool_results": False}
    )
    assert entry.options["export_flows"] is True and entry.options["pool_results"] is False


async def test_pack_source_must_hold_a_release(hass, tmp_path):
    from tests.ha.test_coordinator import _pack_release

    src, _ = _pack_release(tmp_path / "releases")
    entry, r = await _options(hass, "pack")
    for bad in (tmp_path / "absent", tmp_path):  # missing, and a folder without manifest.json
        r = await hass.config_entries.options.async_configure(
            r["flow_id"], {"pack_source": str(bad)}
        )
        assert r["type"] == "form" and r["errors"] == {"pack_source": "not_a_pack_source"}
    r = await hass.config_entries.options.async_configure(r["flow_id"], {"pack_source": str(src)})
    assert r["type"] == "create_entry" and entry.options["pack_source"] == str(src)
    flow = hass.config_entries.options

    async def again(value):
        r = await flow.async_init(entry.entry_id)
        r = await flow.async_configure(r["flow_id"], {"next_step_id": "pack"})
        return await flow.async_configure(r["flow_id"], {"pack_source": value})

    r = await again(src.as_uri())  # a file:// URL
    assert r["type"] == "create_entry" and entry.options["pack_source"] == src.as_uri()
    r = await again("")  # empty: no source, no updates
    assert r["type"] == "create_entry" and entry.options["pack_source"] is None


async def test_view_shows_the_strategy_snippet(hass):
    _, r = await _options(hass, "view")
    assert r["type"] == "form" and r["step_id"] == "view"
    assert "custom:betterybalance" in r["description_placeholders"]["snippet"]
    assert r["description_placeholders"]["path"].endswith("dashboard/view.yaml")


async def test_backtest_without_the_service_aborts_with_the_reason(hass):
    _, r = await _options(hass, "backtest")
    assert r["type"] == "abort" and r["reason"] == "backtest_unavailable"
    assert r["description_placeholders"]["reason"]


async def test_backtest_without_a_pack_aborts_with_the_reason(hass):
    def refuse(call):
        raise ServiceValidationError(
            "no data pack", translation_domain="betterybalance", translation_key="no_pack"
        )

    hass.services.async_register("betterybalance", "run_backtest", refuse)
    _, r = await _options(hass, "backtest")
    assert r["type"] == "form" and r["step_id"] == "backtest"
    r = await hass.config_entries.options.async_configure(r["flow_id"], {})
    assert r["type"] == "abort" and r["reason"] == "backtest_refused"
    assert r["description_placeholders"]["reason"] == F.REFUSALS["no_pack"]


@pytest.mark.parametrize("key", ["backtest_running", "backtest_stopping", "no_end_soc"])
async def test_backtest_refusals_read_the_flow_sentence(hass, key):
    def refuse(call):
        raise ServiceValidationError("refused", translation_domain="betterybalance",
                                     translation_key=key)

    hass.services.async_register("betterybalance", "run_backtest", refuse)
    _, r = await _options(hass, "backtest")
    r = await hass.config_entries.options.async_configure(r["flow_id"], {})
    assert r["type"] == "abort" and r["reason"] == "backtest_refused"
    assert r["description_placeholders"]["reason"] == F.REFUSALS[key] == F.COPY["en"][key]


async def test_backtest_calls_the_service(hass):
    calls = []
    hass.services.async_register("betterybalance", "run_backtest", calls.append)
    entry, r = await _options(hass, "backtest")
    r = await hass.config_entries.options.async_configure(r["flow_id"], {})
    assert len(calls) == 1 and r["type"] == "create_entry"
    assert entry.options == OPTIONS


async def test_the_screen_sentences_follow_the_language_of_home_assistant(hass):
    hass.config.language = "nl"
    await _install_solarman(hass, p1=True)
    hass.config.latitude, hass.config.longitude = 52.149, 5.231
    r = await _start(hass)
    assert r["description_placeholders"]["grid_note"] == F.COPY["nl"]["grid_p1"]
    grid = {"grid_import_kwh": P1_IMPORTS[:1], "grid_export_kwh": P1_EXPORTS}
    r = await hass.config_entries.flow.async_configure(r["flow_id"], {**_pack_input(), **grid})
    warning = r["description_placeholders"]["grid_warning"]
    assert warning.startswith("Je keuze")
    assert "sensor.synthetic_p1_energy_import_tariff_2" in warning
    assert set(F.COPY["nl"]) == set(F.COPY["en"])
    assert all(F.COPY["nl"][k] != F.COPY["en"][k] for k in F.COPY["en"])


async def test_a_dutch_backtest_refusal_reads_dutch(hass):
    hass.config.language = "nl"

    def refuse(call):
        raise ServiceValidationError(
            "no data pack", translation_domain="betterybalance", translation_key="no_pack"
        )

    hass.services.async_register("betterybalance", "run_backtest", refuse)
    _, r = await _options(hass, "backtest")
    r = await hass.config_entries.options.async_configure(r["flow_id"], {})
    assert r["description_placeholders"]["reason"] == F.COPY["nl"]["no_pack"]


async def test_the_backtest_screen_says_a_pack_is_needed_when_none_is_installed(hass, tmp_path):
    from custom_components.betterybalance.config_flow import COPY

    hass.config.config_dir = str(tmp_path)
    await _install_solarman(hass)
    r = await _start(hass)
    flow = hass.config_entries.flow
    r = await flow.async_configure(r["flow_id"], _pack_input())
    r = await flow.async_configure(r["flow_id"], CONTRACT)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "pv_learn"})
    assert r["step_id"] == "backtest"
    assert r["description_placeholders"]["start"] == COPY["en"]["start_no_pack"]
    assert "after this screen closes" not in r["description_placeholders"]["start"]
