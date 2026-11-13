import asyncio
import json
import pathlib

import numpy as np
import pytest
from betterybalance.core import solarman as S
from betterybalance.core.site import flows as F
from betterybalance.core.site.contract import from_dict

from tools import export_site as X

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_ha_statistics.json"  # 4 hours, 7 ids
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = pathlib.Path(__file__).parent / "fixtures" / "synthetic_solarman_registry.json"
STATES = pathlib.Path(__file__).parent / "fixtures" / "synthetic_solarman_states.json"


def test_counter_deltas_mark_resets_and_gaps():
    rows = [
        {"start": 0, "sum": 10.0},
        {"start": 3600000, "sum": 11.5},
        {"start": 7200000, "sum": 11.0},
        {"start": 14400000, "sum": 12.0},
    ]
    ts, kwh = X.counter_deltas(rows)
    assert ts.tolist() == [3600, 7200, 14400] and kwh[0] == 1.5
    assert np.isnan(kwh[1]) and np.isnan(kwh[2])


def test_mean_power_to_kwh():
    ts, kwh = X.mean_power_to_kwh(
        [{"start": 3600000, "mean": 2500.0}, {"start": 7200000, "mean": None}]
    )
    assert ts.tolist() == [3600, 7200] and kwh[0] == 2.5 and np.isnan(kwh[1])


def test_flows_from_statistics_uses_counters_and_signs():
    stats = json.loads(FIX.read_text())
    roles = {
        "grid_import_kwh": ["import_t1", "import_t2"],
        "grid_export_kwh": ["export_t1"],
        "battery_power_w": "batt",
        "soc_pct": "soc",
        "load_power_w": "load",
        "pv_power_w": "pv",
    }
    t = X.flows_from_statistics(stats, roles, battery_sign=-1)
    assert set(t) >= set(F.COLUMNS) and (t["step_s"] == 3600).all()
    h = 1  # import t1 +0,4 and t2 +0,6, export +0,2, batt mean -2000 W (sign -1: charging 2 kWh)
    assert t["grid_import_kwh"][h] == pytest.approx(1.0)
    assert t["grid_export_kwh"][h] == pytest.approx(0.2)
    assert t["batt_charge_kwh"][h] == pytest.approx(2.0) and t["batt_discharge_kwh"][h] == 0.0
    assert t["soc_min_pct"][h] <= t["soc_pct"][h] <= t["soc_max_pct"][h]
    assert np.isnan(t["gen_kwh"]).all()
    assert np.isnan(t["grid_import_kwh"][0])  # the first counter row yields no delta
    assert t["load_kwh"][h] == pytest.approx(0.6)


def test_signed_power_fallback_splits_grid_and_battery():
    stats = json.loads(FIX.read_text())
    roles = {"grid_power_w": "batt", "battery_power_w": "batt"}
    t = X.flows_from_statistics(stats, roles, battery_sign=1, grid_sign=-1)
    assert t["grid_import_kwh"][1] == pytest.approx(2.0) and t["grid_export_kwh"][1] == 0.0
    assert t["grid_export_kwh"][0] == pytest.approx(0.5)
    assert t["batt_discharge_kwh"][1] == pytest.approx(2.0) and t["batt_charge_kwh"][
        0
    ] == pytest.approx(0.5)


def test_unknown_id_gives_nan_and_a_warning(caplog):
    # a logging warning (stderr through logging's last-resort handler when the CLI runs)
    with caplog.at_level("WARNING"):
        t = X.flows_from_statistics(
            json.loads(FIX.read_text()), {"load_power_w": "nope", "soc_pct": "soc"}
        )
    assert np.isnan(t["load_kwh"]).all()
    assert "load_power_w" in caplog.text and "nope" not in caplog.text


def test_the_cli_warning_reaches_stderr_without_a_logging_setup():
    import subprocess
    import sys

    code = (
        "import json, sys; from tools import export_site as X; "
        f"X.flows_from_statistics(json.loads(open({str(FIX)!r}).read()), "
        "{'load_power_w': 'nope', 'soc_pct': 'soc'})"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT_DIR, check=True
    )
    assert "load_power_w" in done.stderr and "nope" not in done.stderr


def test_site_template_rounds_location_and_names_no_entities(tmp_path):
    X.write_site_template(tmp_path / "site.json", 52.149, 5.231)
    text = (tmp_path / "site.json").read_text()
    assert "52.1" in text and "5.2" in text and "52.149" not in text and "sensor." not in text
    assert '"pv"' in text and '"ev"' in text
    data = json.loads(text)
    cfg = from_dict(data)
    assert (cfg.latitude, cfg.longitude) == (52.1, 5.2) and cfg.pack.side == "dc"
    comments = " ".join(data["_comment"])
    assert "dc" in comments and "Read both files before you send them" in comments
    for key in ("pack", "terms", "pv", "ev", "capacity_kwh", "side"):
        assert any(key in line for line in data["_comment"])


def test_solarman_site_json_carries_the_derived_pack(tmp_path):
    path = tmp_path / "site.json"
    X.write_site_template(path, 52.149, 5.231, pack=S.derive_pack(600, 51.2, 240, 240, 10, 20))
    text = path.read_text()
    assert '"capacity_kwh": 30.72' in text
    data = json.loads(text)
    assert data["pack"]["capacity_kwh"] == 30.72
    assert data["loss_profile"] == "deye" and data["execution_profile"] == "deye_amps"
    cfg = from_dict(data)
    assert cfg.loss_profile == "deye" and cfg.pack.soc_floor == pytest.approx(0.2)
    X.write_site_template(path, 52.149, 5.231)
    assert "loss_profile" not in json.loads(path.read_text())


def test_load_entities_rejects_unknown_keys(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("load_power_w: a\nbogus: b\nbattery_sign: -1\n")
    with pytest.raises(ValueError, match="bogus"):
        X.load_entities(p)
    p.write_text("load_power_w: [a, b]\nbattery_sign: -1\n")
    roles, bs, gs = X.load_entities(p)
    assert roles == {"load_power_w": ["a", "b"]} and bs == -1 and gs == 1


class FakeWS:
    def __init__(self, token_ok=True):
        self.sent, self.queue, self.token_ok = [], ["auth_required"], token_ok

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg["type"] == "auth":
            self.queue.append("auth_ok" if self.token_ok else "auth_invalid")
        elif msg["type"] == "recorder/statistics_during_period":
            rows = {i: [{"start": 0, "sum": 1.0}] for i in msg["statistic_ids"]}
            self.queue.append(("result", msg["id"], rows))
        elif msg["type"] == "get_config":
            self.queue.append(("result", msg["id"], {"latitude": 52.149, "longitude": 5.231}))
        elif msg["type"] == "config/entity_registry/list":
            self.queue.append(("result", msg["id"], json.loads(REGISTRY.read_text())))
        elif msg["type"] == "get_states":
            self.queue.append(("result", msg["id"], json.loads(STATES.read_text())))

    async def recv(self):
        item = self.queue.pop(0)
        if isinstance(item, str):
            return json.dumps({"type": item})
        _, i, res = item
        return json.dumps({"id": i, "type": "result", "success": True, "result": res})


def test_fetch_statistics_chunks_31_days_over_one_connection(monkeypatch):
    ws = FakeWS()
    opened = []
    monkeypatch.setattr(X, "_connect", lambda url: (opened.append(url), ws)[1])
    out = asyncio.run(
        X.fetch_statistics(
            "http://ha.invalid:8123",
            "tok",
            ["a"],
            "2026-01-01T00:00:00+00:00",
            "2026-03-15T00:00:00+00:00",
        )
    )
    reqs = [m for m in ws.sent if m["type"] == "recorder/statistics_during_period"]
    assert len(opened) == 1 and opened[0] == "ws://ha.invalid:8123/api/websocket"
    assert len(reqs) == 3 and reqs[0]["period"] == "hour" and reqs[0]["statistic_ids"] == ["a"]
    assert sorted(reqs[0]["types"]) == ["max", "mean", "min", "sum"]
    assert len(out["a"]) == 3


def test_fetch_location_and_bad_token(monkeypatch):
    monkeypatch.setattr(X, "_connect", lambda url: FakeWS())
    assert asyncio.run(X.fetch_location("https://ha.invalid", "tok")) == (52.149, 5.231)
    monkeypatch.setattr(X, "_connect", lambda url: FakeWS(token_ok=False))
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(X.fetch_location("https://ha.invalid", "secret-token"))
    assert "secret-token" not in str(exc.value)


def test_fetch_registry_reads_entries_and_states_over_one_connection(monkeypatch):
    ws = FakeWS()
    opened = []
    monkeypatch.setattr(X, "_connect", lambda url: (opened.append(url), ws)[1])
    entries, states = asyncio.run(X.fetch_registry("https://ha.invalid", "tok"))
    assert len(opened) == 1 and S.devices(entries) == ["dev1"]
    assert [m["type"] for m in ws.sent] == ["auth", "config/entity_registry/list", "get_states"]
    assert {st["entity_id"] for st in states} >= {"sensor.synthetic_battery_voltage"}


def test_cli_writes_both_files_and_prints_no_secrets(monkeypatch, tmp_path, capsys):
    stats = json.loads(FIX.read_text())

    async def fake_stats(url, token, ids, s, e):
        return {i: stats[i] for i in ids if i in stats}

    async def fake_loc(url, token):
        return 52.149, 5.231

    monkeypatch.setattr(X, "fetch_statistics", fake_stats)
    monkeypatch.setattr(X, "fetch_location", fake_loc)
    ent = tmp_path / "e.yaml"
    ent.write_text(
        "grid_import_kwh: [import_t1, import_t2]\ngrid_export_kwh: export_t1\n"
        "battery_power_w: batt\nsoc_pct: soc\n"
    )
    out = tmp_path / "out"
    X.main(
        [
            "--url",
            "http://ha.invalid",
            "--token",
            "secret-token",
            "--from",
            "2027-01-15",
            "--to",
            "2027-01-15",
            "--out",
            str(out),
            "--entities",
            str(ent),
        ]
    )
    printed = capsys.readouterr().out
    assert "secret-token" not in printed and "import_t1" not in printed and "52.149" not in printed
    assert "Read both files before you send them." in printed and "52,1" in printed
    assert len(F.read_flows(out / "flows.csv")["ts"]) == 4
    from_dict(json.loads((out / "site.json").read_text()))


def _solarman_stats(start_ts: int, hours: int) -> dict:
    """Hourly statistics of the fixture device: charge 1 kWh in even hours, discharge 0,5 in odd.

    Battery Power reads positive while discharging, so its mean is -1000 W or +500 W. The
    inverter imports 0,4 + 0,1 * (h % 3) kWh and exports 0,1 kWh per hour. The P1 entities read
    0,3 + 0,2 kWh import (two tariffs) and 0,15 kWh export, and a P1 power that reads positive
    on export (the inverter's net import times -1000 W).
    """
    ms = [(start_ts + h * 3600) * 1000 for h in range(hours)]
    ch = [1.0 if h % 2 == 0 else 0.0 for h in range(hours)]
    dis = [0.0 if h % 2 == 0 else 0.5 for h in range(hours)]
    imp = [0.4 + 0.1 * (h % 3) for h in range(hours)]

    def counter(step):
        return [{"start": m, "sum": float(step * (i + 1))} for i, m in enumerate(ms)]

    def cumulative(values):
        return [{"start": m, "sum": float(sum(values[: i + 1]))} for i, m in enumerate(ms)]

    def mean(values):
        return [{"start": m, "mean": float(v)} for m, v in zip(ms, values)]

    soc = [{"start": m, "mean": 50.0, "min": 49.0, "max": 51.0} for m in ms]
    power = [1000.0 * (d - c) for c, d in zip(ch, dis)]
    return {
        "sensor.synthetic_total_energy_import": cumulative(imp),
        "sensor.synthetic_total_energy_export": counter(0.1),
        "sensor.synthetic_total_battery_charge": cumulative(ch),
        "sensor.synthetic_total_battery_discharge": cumulative(dis),
        "sensor.synthetic_battery": soc,
        "sensor.synthetic_battery_power": mean(power),
        "sensor.synthetic_battery_voltage": mean([52.0] * hours),
        "sensor.synthetic_load_power": mean([600.0] * hours),
        "sensor.synthetic_load_ups_power": mean([100.0] * hours),
        "sensor.synthetic_pv_power": mean([300.0] * hours),
        "sensor.synthetic_pv1_power": mean([200.0] * hours),
        "sensor.synthetic_pv2_power": mean([100.0] * hours),
        "sensor.synthetic_p1_import_t1": counter(0.3),
        "sensor.synthetic_p1_import_t2": counter(0.2),
        "sensor.synthetic_p1_export_t1": counter(0.15),
        "sensor.synthetic_p1_power": mean([-1000.0 * (i - 0.1) for i in imp]),
    }


def _run_solarman(monkeypatch, tmp_path, capsys, *extra):
    """Run the --solarman CLI on the fixtures: (printed, statistic ids asked, out dir)."""
    pre = int(X._local_midnight("2027-01-15").timestamp()) - 3600
    stats = _solarman_stats(pre, 25)
    asked = []

    async def fake_registry(url, token):
        return json.loads(REGISTRY.read_text()), json.loads(STATES.read_text())

    async def fake_stats(url, token, ids, s, e):
        asked.extend(ids)
        return {i: stats[i] for i in ids if i in stats}

    async def fake_loc(url, token):
        return 52.149, 5.231

    monkeypatch.setattr(X, "fetch_registry", fake_registry)
    monkeypatch.setattr(X, "fetch_statistics", fake_stats)
    monkeypatch.setattr(X, "fetch_location", fake_loc)
    out = tmp_path / "out"
    X.main(
        [
            "--url",
            "http://ha.invalid",
            "--token",
            "secret-token",
            "--from",
            "2027-01-15",
            "--to",
            "2027-01-15",
            "--out",
            str(out),
            "--solarman",
            *extra,
        ]
    )
    return capsys.readouterr().out, asked, out


def test_cli_solarman_frozen_counters_keep_the_profile_sign(monkeypatch, tmp_path, capsys):
    # frozen battery counters: no evidence for a sign, so the tool says it keeps the constant
    real = _solarman_stats

    def frozen(start_ts, hours):
        stats = real(start_ts, hours)
        for sid in (
            "sensor.synthetic_total_battery_charge",
            "sensor.synthetic_total_battery_discharge",
        ):
            stats[sid] = [dict(r, sum=5.0) for r in stats[sid]]
        return stats

    monkeypatch.setitem(globals(), "_solarman_stats", frozen)
    printed, _, _ = _run_solarman(monkeypatch, tmp_path, capsys)
    assert "battery sign: no evidence, using the profile constant" in printed
    assert "battery sign inferred" not in printed


def test_cli_solarman_maps_derives_and_infers_the_sign(monkeypatch, tmp_path, capsys):
    printed, asked, out = _run_solarman(monkeypatch, tmp_path, capsys)
    assert "secret-token" not in printed and "synthetic" not in printed and "52.149" not in printed
    assert "battery sign inferred: -1 (matches the profile constant)" in printed
    assert "30,72" not in printed and "31,2" in printed  # 600 Ah at the 52,0 V mean
    assert "grid: inverter counters (export may read low at high discharge, see README)" in printed
    assert "P1" not in printed and "grid sign" not in printed
    assert "sensor.synthetic_battery_voltage" in asked
    assert not any("p1" in i for i in asked)
    t = F.read_flows(out / "flows.csv")
    assert len(t["ts"]) == 24 and "pv3_kwh" not in t
    assert t["pv1_kwh"][0] == pytest.approx(0.2) and t["pv2_kwh"][0] == pytest.approx(0.1)
    # Load Power alone (0,6 kWh): the UPS port is inside it and is not added
    assert t["load_kwh"][0] == pytest.approx(0.6) and t["grid_import_kwh"][0] == pytest.approx(0.5)
    assert t["grid_export_kwh"][0] == pytest.approx(0.1)
    assert t["batt_discharge_kwh"][0] == pytest.approx(0.5)  # the first kept hour is odd
    site = json.loads((out / "site.json").read_text())
    assert site["pack"]["capacity_kwh"] == pytest.approx(31.2) and site["loss_profile"] == "deye"
    assert site["execution_profile"] == "deye_amps" and site["pack"]["soc_floor"] == 0.2
    assert site["pack"]["charge_kw"] == 12.0  # 240 A at 52,0 V is 12,48 kW, capped at 12
    assert X.INVERTER_NOTE in site["_comment"] and X.P1_NOTE not in site["_comment"]
    assert "sensor." not in (out / "site.json").read_text()


def test_cli_solarman_p1_counters_override_the_inverter(monkeypatch, tmp_path, capsys):
    printed, asked, out = _run_solarman(
        monkeypatch,
        tmp_path,
        capsys,
        "--grid-import",
        "sensor.synthetic_p1_import_t1",
        "sensor.synthetic_p1_import_t2",
        "--grid-export",
        "sensor.synthetic_p1_export_t1",
    )
    assert "grid: P1 override" in printed and "inverter counters" not in printed
    assert "synthetic" not in printed
    assert {"sensor.synthetic_p1_import_t1", "sensor.synthetic_p1_import_t2"} <= set(asked)
    t = F.read_flows(out / "flows.csv")
    # the two tariff counters summed (0,3 + 0,2 every hour); the inverter's import varies
    # 0,4 .. 0,6 and its export is 0,1, so neither is used
    assert t["grid_import_kwh"] == pytest.approx(np.full(24, 0.5))
    assert t["grid_export_kwh"] == pytest.approx(np.full(24, 0.15))
    site = json.loads((out / "site.json").read_text())
    assert X.P1_NOTE in site["_comment"] and X.INVERTER_NOTE not in site["_comment"]
    assert "sensor." not in (out / "site.json").read_text()


def test_cli_solarman_p1_power_alone_splits_with_the_inferred_sign(monkeypatch, tmp_path, capsys):
    # the P1 power reads positive on export: inferred -1 against the inverter's counters, and the
    # split takes -1 so the import is the inverter's net import
    printed, asked, out = _run_solarman(
        monkeypatch, tmp_path, capsys, "--grid-power", "sensor.synthetic_p1_power"
    )
    assert "grid: P1 override" in printed
    assert "grid sign inferred: -1 (the split takes -1)" in printed
    assert "sensor.synthetic_total_energy_import" in asked  # fetched as the sign reference
    t = F.read_flows(out / "flows.csv")
    net = [0.4 + 0.1 * (h % 3) - 0.1 for h in range(1, 25)]
    assert t["grid_import_kwh"] == pytest.approx(np.asarray(net))
    assert (t["grid_export_kwh"] == 0.0).all()


def test_cli_needs_entities_or_solarman(tmp_path):
    base = ["--url", "u", "--token", "t", "--from", "2027-01-15", "--to", "2027-01-15"]
    with pytest.raises(SystemExit):
        X.main([*base, "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        X.main([*base, "--out", str(tmp_path), "--solarman", "--entities", "e.yaml"])
    with pytest.raises(SystemExit):
        X.main([*base, "--out", str(tmp_path), "--solarman", "--grid-import", "a"])
    with pytest.raises(SystemExit):
        X.main([*base, "--out", str(tmp_path), "--entities", "e.yaml", "--grid-power", "a"])


def test_solarman_setup_refuses_and_reads_settings():
    entries, states = json.loads(REGISTRY.read_text()), json.loads(STATES.read_text())
    mapping, settings = X.solarman_setup(entries, states, None)
    assert settings == {
        "capacity_ah": 600.0,
        "max_charge_a": 240.0,
        "max_discharge_a": 240.0,
        "shutdown_soc": 10.0,
        "low_soc": 20.0,
        "bms_charge_a": 300.0,
        "bms_discharge_a": 300.0,
    }
    # the IO Mode select reads Microinverter, so the GEN port role takes that name
    assert mapping["gen_power"] == "sensor.synthetic_microinverter_power"
    held = [dict(st, state="0") if "max_charging" in st["entity_id"] else st for st in states]
    assert X.solarman_setup(entries, held, None)[1]["max_charge_a"] == 0.0
    with pytest.raises(S.Refusal) as exc:
        X.solarman_setup([e for e in entries if e["platform"] != "solarman"], states, None)
    assert exc.value.reason == "no_solarman"
    with pytest.raises(S.Refusal) as exc:
        X.solarman_setup([e for e in entries if e["device_id"] == "dev2"], states, None)
    assert exc.value.reason == "no_battery"
    gone = [dict(st, state="unavailable") if "capacity" in st["entity_id"] else st for st in states]
    with pytest.raises(S.Refusal, match="Battery Capacity") as exc:
        X.solarman_setup(entries, gone, None)
    assert exc.value.reason == "missing_entity"
