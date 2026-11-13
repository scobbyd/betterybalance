import json
import pathlib
from dataclasses import fields

import pytest
from betterybalance.core.site import contract as C

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_site.json"


def test_from_dict_applies_defaults_and_reads_hints():
    cfg = C.from_dict(json.loads(FIX.read_text()))
    assert cfg.pack.capacity_kwh == 48.0 and cfg.pack.side == "dc"
    assert cfg.terms.fee_import_eur_kwh == 0.0165 and cfg.terms.btw == 0.21
    # the Task 25 ladder chose dp with kappa 1 (docs/reports/2026-10-31-plan2-replay.md)
    assert cfg.knobs.kappa == 1.0 and cfg.knobs.impl == "dp" and cfg.knobs.q_take == 0.25
    assert cfg.knobs.q_feed == 0.75 and cfg.knobs.dual_penalty
    assert cfg.ev.charger_kw == 11.0 and cfg.ev.min_minutes == 45
    assert cfg.pv.planes[0].azimuth_deg == 135.0 and cfg.pv.correction == 1.0


def test_percent_plan_converts_to_energy():
    d = json.loads(FIX.read_text())
    d["ev"] = {
        "present": True,
        "plan": {"percent": 50, "battery_kwh": 60, "by": "07:00", "days": [0, 1, 2, 3, 4]},
    }
    plan = C.from_dict(d).ev.plan
    assert plan.energy_kwh == 30.0 and plan.by_local == "07:00" and plan.days == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    "path,value",
    [
        (("latitude",), 95.0),
        (("pack", "soc_floor"), 1.0),
        (("pack", "soc_ceiling"), 1.2),
        (("pack", "side"), "hv"),
        (("terms", "tax_mode"), "maybe"),
        (("knobs", "impl"), "magic"),
        (("knobs", "commit_rule"), "coin"),
        (("ev", "charger_kw"), 0.0),
        # M6: the pack and the longitude
        (("longitude",), 181.0),
        (("longitude",), -180.5),
        (("pack", "capacity_kwh"), 0.0),
        (("pack", "capacity_kwh"), -48.0),
        (("pack", "charge_kw"), 0.0),
        (("pack", "discharge_kw"), -1.0),
        (("pack", "discharge_kw"), float("nan")),
        (("pack", "eta_rt"), 0.0),
        (("pack", "eta_rt"), 1.05),
    ],
)
def test_invalid_values_name_the_key(path, value):
    d = json.loads(FIX.read_text())
    node = d
    for k in path[:-1]:
        node = node.setdefault(k, {})
    node[path[-1]] = value
    with pytest.raises(ValueError, match=path[-1]):
        C.from_dict(d)


@pytest.mark.parametrize(
    "plane,key",
    [
        ({"tilt_deg": 35, "azimuth_deg": 360}, "azimuth_deg"),
        ({"tilt_deg": 95, "azimuth_deg": 0}, "tilt_deg"),
    ],
)
def test_invalid_plane_names_the_key(plane, key):
    d = json.loads(FIX.read_text())
    d["pv"] = {"planes": [plane]}
    with pytest.raises(ValueError, match=key):
        C.from_dict(d)


def test_a_plane_names_its_string_and_older_planes_still_load():
    d = json.loads(FIX.read_text())
    d["pv"] = {
        "planes": [
            {"tilt_deg": 35, "azimuth_deg": 135, "kwp": 4.0, "string": "pv1"},
            {"tilt_deg": 20, "azimuth_deg": 180, "kwp": 0.8, "string": "gen"},
            {"tilt_deg": 30, "azimuth_deg": 225},
        ]
    }
    planes = C.from_dict(d).pv.planes
    assert [p.string for p in planes] == ["pv1", "gen", None]
    d["pv"]["planes"][0]["string"] = "pv9"
    with pytest.raises(ValueError, match="string"):
        C.from_dict(d)


def test_tax_mode_switches_at_salderen_end():
    t = C.Terms()
    before = 1798758000 - 1  # 2027-01-01 00:00 Europe/Amsterdam is 1798758000
    assert C.tax_mode_at(t, before) == "net_exporter" and C.tax_mode_at(t, 1798758000) == "none"


def test_round_location():
    assert C.round_location(52.149, 5.231) == (52.1, 5.2)


def test_unknown_keys_are_ignored():
    base = C.from_dict(json.loads(FIX.read_text()))
    d = json.loads(FIX.read_text())
    d["_comment"] = ["x"]
    d["pack"]["colour"] = "blue"
    d["pv"]["planes"][0]["note"] = "y"
    d["knobs"] = {"future_knob": 1}
    assert C.from_dict(d) == base


@pytest.mark.parametrize(
    "path",
    [("latitude",), ("longitude",), ("pack",), ("pack", "capacity_kwh"), ("ev", "plan", "by")],
)
def test_missing_required_key_names_it(path):
    d = json.loads(FIX.read_text())
    d["ev"] = {"present": True, "plan": {"energy_kwh": 10, "by": "07:00", "days": [0]}}
    node = d
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    with pytest.raises(ValueError, match=path[-1]):
        C.from_dict(d)


def test_pack_bounds_are_inclusive_where_the_physics_allows():
    d = json.loads(FIX.read_text())
    d["longitude"] = -180.0
    d["pack"]["eta_rt"] = 1.0
    cfg = C.from_dict(d)
    assert cfg.longitude == -180.0 and cfg.pack.eta_rt == 1.0


def test_every_knob_has_help():
    assert set(C.KNOB_HELP) == {f.name for f in fields(C.Knobs)}
    for text in C.KNOB_HELP.values():
        assert text and "\n" not in text and len(text) < 120
