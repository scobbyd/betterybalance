import json
import pathlib

import numpy as np
import pytest
from betterybalance.core import solarman as S

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_solarman_registry.json"


def _entries():
    # device "dev1": deye_p3 with PV1, PV2; device "dev2": a solarman meter without Battery Power
    return json.loads(FIX.read_text())


def test_devices_pick_the_one_with_battery_power():
    assert S.devices(_entries()) == ["dev1"]


def test_match_resolves_name_collisions_by_platform():
    m = S.match_entities(_entries(), "dev1")
    assert m["pv_power"] == "sensor.synthetic_pv_power"
    assert m["capacity_ah"] == "number.synthetic_battery_capacity"
    assert m["pv2_power"] == "sensor.synthetic_pv2_power" and "pv3_power" not in m


def test_missing_required_entity_is_named():
    e = [x for x in _entries() if x["original_name"] != "Battery Voltage"]
    with pytest.raises(S.Refusal, match="Battery Voltage") as exc:
        S.match_entities(e, "dev1")
    assert exc.value.reason == "missing_entity"


def test_single_phase_and_old_integration_are_refused():
    e = [x for x in _entries() if x["original_name"] != "Grid L2 Power"]
    with pytest.raises(S.Refusal) as exc:
        S.match_entities(e, "dev1")
    assert exc.value.reason == "single_phase"
    old = [dict(x, unique_id="1234567890_" + x["original_name"].lower()) for x in _entries()]
    with pytest.raises(S.Refusal) as exc:
        S.match_entities(old, "dev1")
    assert exc.value.reason == "old_integration"


def test_uid_suffix_slugs_name_and_platform():
    assert S.uid_suffix("Battery Max Charging Current", "number") == (
        "battery_max_charging_current_number"
    )
    assert S.uid_suffix("PV1 Power", "sensor") == "pv1_power_sensor"
    # an unnamed entity: the integration writes <entry>_<domain> with one underscore
    assert S.uid_suffix("", "binary_sensor") == "binary_sensor"


def test_derive_pack_uses_measured_voltage():
    p = S.derive_pack(600, 51.2, 240, 240, 10, 20, inverter_kw=20.0)
    assert p["capacity_kwh"] == pytest.approx(30.72) and p["charge_kw"] == pytest.approx(12.288)
    assert S.derive_pack(600, 51.2, 240, 240, 10, 20)["charge_kw"] == pytest.approx(12.0)
    assert p["soc_floor"] == pytest.approx(0.2) and p["side"] == "dc" and p["eta_rt"] == 0.88
    assert S.derive_pack(600, float("nan"), 240, 240, 10, 20)["capacity_kwh"] == pytest.approx(
        30.72
    )
    assert S.derive_pack(600, 48.0, 100, 100, 10, 10)["capacity_kwh"] == pytest.approx(28.8)
    with pytest.raises(S.Refusal) as exc:
        S.derive_pack(100, 400.0, 10, 10, 10, 10)
    assert exc.value.reason == "high_voltage"


def test_derive_pack_takes_the_bms_limit_when_an_ems_holds_the_setting_at_zero():
    # a 0 A setting is an EMS holding the battery: the 300 A BMS limit at 51,2 V is 15,36 kW,
    # capped at the 12 kW inverter
    p = S.derive_pack(600, 51.2, 0, 0, 10, 20, bms_charge_a=300, bms_discharge_a=300)
    assert p["charge_kw"] == pytest.approx(12.0) and p["discharge_kw"] == pytest.approx(12.0)
    p = S.derive_pack(600, 51.2, 100, 100, 10, 20, bms_charge_a=300, bms_discharge_a=300)
    assert p["charge_kw"] == pytest.approx(5.12) and p["discharge_kw"] == pytest.approx(5.12)
    p = S.derive_pack(600, 51.2, float("nan"), 100, 10, 20, bms_charge_a=50, bms_discharge_a=0)
    assert p["charge_kw"] == pytest.approx(2.56) and p["discharge_kw"] == pytest.approx(5.12)
    with pytest.raises(S.Refusal, match="Battery BMS Max Charging Current") as exc:
        S.derive_pack(600, 51.2, 0, 100, 10, 20, bms_charge_a=float("nan"))
    assert exc.value.reason == "missing_entity"
    with pytest.raises(S.Refusal, match="Battery BMS Max Discharging Current"):
        S.derive_pack(600, 51.2, 100, 0, 10, 20)


def test_gen_port_roles_follow_the_io_mode():
    # the fixture has Generator and Microinverter entities, as after an IO mode change
    m = S.match_entities(_entries(), "dev1")
    assert m["gen_power"] == "sensor.synthetic_generator_power"  # the first name wins
    m = S.match_entities(_entries(), "dev1", io_mode="Microinverter")
    assert m["gen_power"] == "sensor.synthetic_microinverter_power"
    assert m["gen_total"] == "sensor.synthetic_microinverter_energy"
    assert m["io_mode"] == "select.synthetic_io_mode"
    only_micro = [x for x in _entries() if not x["original_name"].startswith("Generator ")]
    m = S.match_entities(only_micro, "dev1")
    assert m["gen_power"] == "sensor.synthetic_microinverter_power"
    assert S.statistic_roles(m)["gen_power_w"] == "sensor.synthetic_microinverter_power"


def test_smartload_mode_maps_no_gen_role():
    # a SmartLoad port is a load output: in SmartLoad mode the stale Generator entities and the
    # SmartLoad entities fill no gen role, and gen_kwh stays empty
    dev1 = next(x for x in _entries() if x["device_id"] == "dev1")
    smartload = [
        dict(
            dev1,
            entity_id=f"sensor.synthetic_{slug}",
            original_name=name,
            unique_id=f"01jsyntheticentry0000000000_{slug}_sensor",
        )
        for name, slug in (
            ("SmartLoad Power", "smartload_power"),
            ("SmartLoad Energy", "smartload_energy"),
        )
    ]
    entries = [x for x in _entries() if not x["original_name"].startswith("Microinverter ")]
    m = S.match_entities(entries + smartload, "dev1", io_mode="SmartLoad")
    assert "gen_power" not in m and "gen_total" not in m
    assert "gen_power_w" not in S.statistic_roles(m)
    no_generator = [x for x in entries if not x["original_name"].startswith("Generator ")]
    m = S.match_entities(no_generator + smartload, "dev1")
    assert "gen_power" not in m and "gen_total" not in m


def test_infer_sign():
    p = np.array([-2000.0, 1000.0, 0.0, -500.0, 1500.0, 0.0])
    ch = np.array([2.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    dis = np.array([0.0, 1.0, 0.0, 0.0, 1.5, 0.0])
    assert S.infer_sign(p, ch, dis) == -1 and S.infer_sign(-p, ch, dis) == 1
    assert S.infer_sign(p[:5], ch[:5], dis[:5]) is None  # fewer than 6 hours


@pytest.mark.parametrize("capacity_ah", [0.0, -100.0, float("nan")])
def test_a_capacity_without_ah_is_refused(capacity_ah):
    with pytest.raises(S.Refusal) as exc:
        S.derive_pack(capacity_ah, 51.2, 240, 240, 10, 20)
    assert exc.value.reason == "missing_entity" and exc.value.detail == "Battery Capacity"


def test_infer_sign_without_evidence_is_none():
    # frozen counters: charge - discharge never moves, so the covariance is zero
    p = np.array([-2000.0, 1000.0, 0.0, -500.0, 1500.0, 300.0])
    assert S.infer_sign(p, np.zeros(6), np.zeros(6)) is None
    # a flat power against moving counters
    ch = np.array([2.0, 0.0, 0.0, 0.5, 0.0, 0.0])
    assert S.infer_sign(np.full(6, 700.0), ch, np.zeros(6)) is None
    # NaN hours do not count towards the six
    p_nan = np.array([np.nan, 1000.0, 0.0, -500.0, 1500.0, 0.0])
    assert S.infer_sign(p_nan, ch, np.zeros(6)) is None


def test_statistic_roles_take_load_power_alone():
    m = S.match_entities(_entries(), "dev1")
    r = S.statistic_roles(m)
    # Load Power already holds the UPS port; the UPS role stays mapped but is not summed
    assert r["load_power_w"] == ["sensor.synthetic_load_power"] and "load_ups_power" in m
    assert r["grid_import_kwh"] == "sensor.synthetic_total_energy_import" and "pv3_power_w" not in r
