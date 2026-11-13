"""Export hourly Home Assistant statistics to the flows format and write a site template.

Run this on your own machine against your own Home Assistant. It reads long-term statistics
over the websocket (`recorder/statistics_during_period`), writes `flows.csv` and a `site.json`
template. No token, entity id or exact location is written to a file or printed.

    python -m tools.export_site --url URL --token TOKEN --from 2026-01-01 --to 2026-10-31 \
        --out DIR (--entities entities.yaml | --solarman [--device N]) [--lat 52.1 --lon 5.2]

With `--solarman` the tool finds the Deye hybrid of the ha-solarman `deye_p3` profile in the
entity registry, maps its entities, derives the pack from the inverter's battery settings and
prints the battery sign the statistics show. The site.json then carries that pack. With a P1
reader in Home Assistant, `--grid-import ID [ID ...]` and `--grid-export ID [ID ...]` (energy
counters, summed: the tariff 1 and 2 counters of a DSMR meter) and `--grid-power ID` (signed,
import positive) take the grid from the meter instead of the inverter.

The entities file is YAML. It maps each role in ROLES to one statistic id or a list of ids
(summed), plus `battery_sign` (+1 when the battery sensor reads positive while charging, -1
otherwise, default +1) and `grid_sign` (+1 when the grid sensor reads positive on import,
default +1). `websockets` and `pyyaml` come from the tools extra and load only in the CLI path.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
from betterybalance.core import solarman as S
from betterybalance.core.fmt import eu
from betterybalance.core.inputs import (  # noqa: F401  (re-exported for the existing callers)
    HOUR_S,
    STAT_ROLES,
    Rows,
    _ids,
    counter_deltas,
    flows_from_statistics,
    mean_power_to_kwh,
)
from betterybalance.core.site.contract import round_location
from betterybalance.core.site.flows import write_flows
from betterybalance.core.timegrid import ZONE, Table

ROLES = STAT_ROLES
_SIGNS = ("battery_sign", "grid_sign")
CHUNK_DAYS = 31

_COMMENT = [
    "Fill in this file, then read both files before you send them. flows.csv and "
    "site.json hold no token, no entity id and no exact location.",
    "latitude, longitude: your location, rounded to 0,1 degree.",
    "pack.capacity_kwh: usable battery energy in kWh. The value below is a placeholder.",
    "pack.charge_kw, pack.discharge_kw: maximum charge and discharge power in kW. The"
    " values below are placeholders.",
    "pack.soc_floor, pack.soc_ceiling: lowest and highest state of charge you allow, "
    "as a fraction from 0 to 1.",
    "pack.eta_rt: round-trip efficiency as a fraction. 0,88 is a typical value.",
    "pack.side: the dc flag. Use dc when the battery sits on the DC side of the "
    "inverter with the panels. Use ac when it has its own inverter.",
    "terms: the Balans tariff terms. Fees and BTW are as of 2026. Change them if your"
    " contract differs.",
    "terms.balans_revenue_eur_month: your monthly Balans payout in euro. 0 means unknown.",
    "pv.none: true when you have no solar panels.",
    "pv.planes: one entry per roof plane with tilt_deg (0 is flat), azimuth_deg (0 "
    "north, 90 east, 180 south, 270 west) and kwp. Leave empty if unknown.",
    "pv.correction: multiplier on the modelled solar yield. 1 means no correction.",
    "ev.present: true when you charge an electric car at home.",
    "ev.charger_kw: charger power in kW. ev.min_minutes: shortest charge session in minutes.",
    "ev.plan: set to null if you have no fixed charge plan.",
    "Read both files before you send them.",
]


# comment lines that change when the pack is derived, keyed by the text before the colon
_DERIVED_COMMENT = {
    "pack.capacity_kwh": "pack.capacity_kwh: usable battery energy in kWh, derived from the"
    " inverter's battery capacity in Ah at the mean battery voltage. Check it.",
    "pack.charge_kw, pack.discharge_kw": "pack.charge_kw, pack.discharge_kw: maximum charge"
    " and discharge power in kW, from the smaller of the inverter's battery current setting and"
    " the BMS limit, capped at 12 kW (the SUN-12K). Check them.",
}
_PLACEHOLDER_PACK = {
    "capacity_kwh": 10.0,
    "charge_kw": 5.0,
    "discharge_kw": 5.0,
    "soc_floor": 0.1,
    "soc_ceiling": 1.0,
    "eta_rt": 0.88,
    "side": "dc",
}


def _comment(derived: bool) -> list[str]:
    if not derived:
        return list(_COMMENT)
    return [_DERIVED_COMMENT.get(line.split(":", 1)[0], line) for line in _COMMENT]


def write_site_template(
    path: Path, lat: float, lon: float, pack: dict | None = None, notes: tuple[str, ...] = ()
) -> None:
    """Write a site.json template: rounded location, the pack, Balans terms and comments.

    Without `pack` the pack holds placeholders. With it (the Solarman path) the template carries
    that pack, the `deye` loss profile and the `deye_amps` execution profile. `notes` are extra
    comment lines (the grid source).
    """
    lat_r, lon_r = round_location(lat, lon)
    template = {
        "_comment": [*_comment(pack is not None), *notes],
        "latitude": lat_r,
        "longitude": lon_r,
        "pack": dict(pack) if pack is not None else dict(_PLACEHOLDER_PACK),
        "terms": {
            "fee_import_eur_kwh": 0.0165,
            "fee_export_eur_kwh": 0.0165,
            "btw": 0.21,
            "dual_pricing": True,
            "tax_mode": "net_exporter",
            "salderen_end": "2027-01-01",
            "balans_revenue_eur_month": 0.0,
        },
        "pv": {"none": False, "planes": [], "correction": 1.0},
        "ev": {"present": False, "charger_kw": 11.0, "min_minutes": 45, "plan": None},
    }
    if pack is not None:
        template["loss_profile"] = "deye"
        template["execution_profile"] = "deye_amps"
    Path(path).write_text(json.dumps(template, indent=2) + "\n")


def _connect(url: str):
    """Open a websocket connection. Tests replace this."""
    import websockets

    return websockets.connect(url, max_size=2**27)


def _ws_url(url: str) -> str:
    base = url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return base + "/api/websocket"


async def _authenticate(ws, token: str) -> None:
    await ws.recv()  # auth_required
    await ws.send(json.dumps({"type": "auth", "access_token": token}))
    reply = json.loads(await ws.recv())
    if reply.get("type") != "auth_ok":
        raise RuntimeError("Home Assistant refused the token")


async def _call(ws, counter: list[int], message: dict):
    counter[0] += 1
    await ws.send(json.dumps({"id": counter[0], **message}))
    while True:
        reply = json.loads(await ws.recv())
        if reply.get("id") == counter[0]:
            break
    if not reply.get("success"):
        code = (reply.get("error") or {}).get("code", "unknown")
        raise RuntimeError(f"Home Assistant call failed: {code}")
    return reply["result"]


async def fetch_statistics(
    url: str, token: str, ids: list[str], start_iso: str, end_iso: str
) -> dict:
    """Hourly statistics of the ids over [start, end): sum, mean, min, max, in 31-day chunks."""
    start = dt.datetime.fromisoformat(start_iso)
    end = dt.datetime.fromisoformat(end_iso)
    merged: dict[str, list] = {}
    counter = [0]
    async with _connect(_ws_url(url)) as ws:
        await _authenticate(ws, token)
        lo = start
        while lo < end:
            hi = min(lo + dt.timedelta(days=CHUNK_DAYS), end)
            result = await _call(
                ws,
                counter,
                {
                    "type": "recorder/statistics_during_period",
                    "start_time": lo.isoformat(),
                    "end_time": hi.isoformat(),
                    "statistic_ids": list(ids),
                    "period": "hour",
                    "types": ["sum", "mean", "min", "max"],
                },
            )
            for sid, rows in result.items():
                merged.setdefault(sid, []).extend(rows)
            lo = hi
    return merged


async def fetch_registry(url: str, token: str) -> tuple[list[dict], list[dict]]:
    """Entity registry entries and current states, over one connection."""
    async with _connect(_ws_url(url)) as ws:
        await _authenticate(ws, token)
        counter = [0]
        entries = await _call(ws, counter, {"type": "config/entity_registry/list"})
        states = await _call(ws, counter, {"type": "get_states"})
    return entries, states


async def fetch_location(url: str, token: str) -> tuple[float, float]:
    """Latitude and longitude from the Home Assistant configuration."""
    async with _connect(_ws_url(url)) as ws:
        await _authenticate(ws, token)
        config = await _call(ws, [0], {"type": "get_config"})
    return float(config["latitude"]), float(config["longitude"])


def load_entities(path: Path) -> tuple[dict, int, int]:
    """Read the entities YAML: (roles, battery_sign, grid_sign). Unknown keys raise ValueError."""
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("the entities file must map roles to statistic ids")
    unknown = sorted(set(data) - set(ROLES) - set(_SIGNS))
    if unknown:
        raise ValueError(f"unknown keys in the entities file: {', '.join(unknown)}")
    roles = {k: data[k] for k in ROLES if data.get(k)}
    signs = {k: int(data.get(k, 1)) for k in _SIGNS}
    for k, v in signs.items():
        if v not in (1, -1):
            raise ValueError(f"{k} must be 1 or -1, got {v}")
    return roles, signs["battery_sign"], signs["grid_sign"]


def _local_midnight(day: str) -> dt.datetime:
    return dt.datetime.fromisoformat(day).replace(tzinfo=ZONE)


def _local_iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), ZONE).isoformat()


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _setting(states: dict[str, dict], mapping: dict[str, str], key: str) -> float:
    """The current value of a setting role; an unknown or unavailable state refuses."""
    value = _number(states.get(mapping[key], {}).get("state"))
    if not np.isfinite(value):
        name = next(r.name for r in S.SOLARMAN_ROLES if r.key == key)
        raise S.Refusal("missing_entity", f"{name} has no value")
    return value


def solarman_setup(
    entries: list[dict], states: list[dict], device: int | None
) -> tuple[dict[str, str], dict[str, float]]:
    """Role mapping and battery settings of the chosen Solarman device."""
    if not any(e.get("platform") == S.PLATFORM for e in entries):
        raise S.Refusal("no_solarman", "no ha-solarman entity in the registry")
    found = S.devices(entries)
    if not found:
        raise S.Refusal("no_battery", "no Solarman device has a Battery Power sensor")
    if device is None and len(found) > 1:
        raise SystemExit(
            f"Found {len(found)} Solarman devices with a battery. "
            f"Choose one with --device 0 to {len(found) - 1}."
        )
    index = device or 0
    if not 0 <= index < len(found):
        raise SystemExit(f"--device must be 0 to {len(found) - 1}.")
    by_id = {st["entity_id"]: st for st in states}
    mapping = S.match_entities(entries, found[index])
    if "io_mode" in mapping:
        io_mode = by_id.get(mapping["io_mode"], {}).get("state")
        if io_mode in S.IO_MODES:
            mapping = S.match_entities(entries, found[index], io_mode=io_mode)
    settings = {k: _setting(by_id, mapping, k) for k in ("capacity_ah", "shutdown_soc", "low_soc")}
    # a current may be unavailable: derive_pack takes the BMS limit then, or refuses
    for key in ("max_charge_a", "max_discharge_a"):
        settings[key] = _number(by_id.get(mapping[key], {}).get("state"))
    attributes = by_id.get(mapping["soc"], {}).get("attributes", {})
    for key, attribute in S.BMS_ATTRIBUTES.items():
        settings[key] = _number(attributes.get(attribute))
    return mapping, settings


def _hourly_means(rows: Rows | None) -> dict[int, float]:
    return {
        int(r["start"]) // 1000: (np.nan if r.get("mean") is None else float(r["mean"]))
        for r in rows or []
    }


def last_day_mean(rows: Rows | None, end_ts: int) -> float:
    """Mean of the hourly means in the 24 hours before end_ts; NaN without a finite one."""
    vals = [v for t, v in _hourly_means(rows).items() if end_ts - 86400 <= t < end_ts]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--url",
        required=True,
        help="Home Assistant URL, for example http://HOST:8123",
    )
    ap.add_argument("--token", required=True, help="long-lived access token (never printed)")
    ap.add_argument("--from", dest="first", required=True, help="first local day, YYYY-MM-DD")
    ap.add_argument("--to", dest="last", required=True, help="last local day, inclusive")
    ap.add_argument("--out", required=True, type=Path)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--entities", type=Path, help="YAML file that maps roles to ids")
    source.add_argument(
        "--solarman", action="store_true", help="map a Deye hybrid of ha-solarman deye_p3"
    )
    ap.add_argument("--device", type=int, help="with --solarman: which inverter, from 0")
    ap.add_argument(
        "--grid-import", nargs="+", metavar="ID", help="with --solarman: P1 import counters"
    )
    ap.add_argument(
        "--grid-export", nargs="+", metavar="ID", help="with --solarman: P1 export counters"
    )
    ap.add_argument(
        "--grid-power", metavar="ID", help="with --solarman: P1 grid power, import positive"
    )
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    args = ap.parse_args(argv)
    if not args.solarman and (args.grid_import or args.grid_export or args.grid_power):
        ap.error("--grid-import, --grid-export and --grid-power need --solarman")
    if bool(args.grid_import) != bool(args.grid_export):
        ap.error("--grid-import and --grid-export go together")

    try:
        _export(args)
    except S.Refusal as exc:
        print(
            f"Refused ({exc.reason}): {exc.detail}. This tool maps only the Deye hybrid of the "
            "ha-solarman deye_p3 profile. For another inverter, use --entities.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None


def _export(args: argparse.Namespace) -> None:
    mapping: dict[str, str] = {}
    p1: dict = {}
    inverter_grid: dict = {}
    if args.solarman:
        entries, states = asyncio.run(fetch_registry(args.url, args.token))
        mapping, settings = solarman_setup(entries, states, args.device)
        roles, battery_sign, grid_sign = S.statistic_roles(mapping), 1, 1
        inverter_grid = {k: roles.pop(k) for k in ("grid_import_kwh", "grid_export_kwh")}
        p1 = grid_overrides(args)
        roles.update(p1 or inverter_grid)
        extra = [mapping["battery_power"], mapping["battery_voltage"], *inverter_grid.values()]
    else:
        roles, battery_sign, grid_sign = load_entities(args.entities)
        extra = []
    ids = sorted({i for spec in roles.values() for i in _ids(spec)} | set(extra))
    start = _local_midnight(args.first)
    end = _local_midnight((dt.date.fromisoformat(args.last) + dt.timedelta(days=1)).isoformat())
    # one extra hour before the window so the first hour has a counter difference
    pre = start - dt.timedelta(seconds=HOUR_S)
    stats = asyncio.run(
        fetch_statistics(args.url, args.token, ids, pre.isoformat(), end.isoformat())
    )
    grid_inferred = None
    if args.solarman and "grid_power_w" in roles:
        # the grid power's sign against the counters in use (the P1 ones, else the inverter's)
        counters = {k: roles.get(k, v) for k, v in inverter_grid.items()}
        grid_inferred = grid_power_sign(stats, roles["grid_power_w"], counters)
        grid_sign = grid_inferred or 1
    table = flows_from_statistics(stats, roles, battery_sign, grid_sign)
    keep = (table["ts"] >= int(start.timestamp())) & (table["ts"] < int(end.timestamp()))
    table = {k: v[keep] for k, v in table.items()}
    pack = None
    if args.solarman:
        voltage = last_day_mean(stats.get(mapping["battery_voltage"]), int(end.timestamp()))
        pack = S.derive_pack(
            settings["capacity_ah"],
            voltage,
            settings["max_charge_a"],
            settings["max_discharge_a"],
            settings["shutdown_soc"],
            settings["low_soc"],
            bms_charge_a=settings["bms_charge_a"],
            bms_discharge_a=settings["bms_discharge_a"],
        )
    if args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
    else:
        lat, lon = asyncio.run(fetch_location(args.url, args.token))
    args.out.mkdir(parents=True, exist_ok=True)
    write_flows(table, args.out / "flows.csv")
    notes = ()
    if args.solarman:
        notes = (P1_NOTE if p1 else INVERTER_NOTE,)
    write_site_template(args.out / "site.json", lat, lon, pack=pack, notes=notes)
    n = len(table["ts"])
    lat_r, lon_r = round_location(lat, lon)
    print(f"Rows: {eu(n, 0)}")
    if n:
        print(f"First hour: {_local_iso(table['ts'][0])}")
        print(f"Last hour: {_local_iso(table['ts'][-1])}")
    print(f"Location in site.json: {eu(lat_r, 1)}, {eu(lon_r, 1)}")
    if pack is None:
        print(
            f"Wrote {args.out / 'flows.csv'} and {args.out / 'site.json'}. "
            "Fill in the pack placeholders."
        )
    else:
        _print_solarman(pack, settings, voltage, stats.get(mapping["battery_power"]), table)
        if p1:
            print("grid: P1 override")
        else:
            print("grid: inverter counters (export may read low at high discharge, see README)")
        if "grid_power_w" in roles:
            if grid_inferred is None:
                print("grid sign: no evidence, the split takes +1")
            else:
                verdict = "import positive" if grid_inferred == 1 else "the split takes -1"
                print(f"grid sign inferred: {grid_inferred:+d} ({verdict})")
        print(
            f"Wrote {args.out / 'flows.csv'} and {args.out / 'site.json'}. Check the derived pack."
        )
    print("Read both files before you send them.")


P1_NOTE = (
    "grid: from the P1 meter entities given with --grid-import, --grid-export or --grid-power,"
    " not from the inverter."
)
INVERTER_NOTE = (
    "grid: from the inverter's counters. The export may read low at high discharge (see README)."
)


def grid_overrides(args: argparse.Namespace) -> dict:
    """The P1 statistic roles of --grid-import, --grid-export and --grid-power; {} without them."""
    roles: dict = {}
    if args.grid_import:
        roles["grid_import_kwh"] = list(args.grid_import)
        roles["grid_export_kwh"] = list(args.grid_export)
    if args.grid_power:
        roles["grid_power_w"] = args.grid_power
    return roles


def grid_power_sign(stats: dict[str, Rows], power_id: str, counters: dict) -> int | None:
    """+1 when the grid power rises with import - export of the counters, -1 when it falls.

    None without evidence: fewer than 6 hours where both are known, or a zero covariance.
    """
    ref = flows_from_statistics(stats, counters)
    means = _hourly_means(stats.get(power_id))
    power = np.asarray([means.get(int(t), np.nan) for t in ref["ts"]], dtype=float)
    return S.infer_sign(power, ref["grid_import_kwh"], ref["grid_export_kwh"])


def _print_solarman(
    pack: dict, settings: dict, voltage: float, power_rows: Rows | None, table: Table
) -> None:
    """Print the derived pack and the battery sign the statistics show."""
    v = voltage if np.isfinite(voltage) else S.DEFAULT_VOLTAGE_V
    print(
        f"Pack in site.json: {eu(pack['capacity_kwh'], 2)} kWh, charge "
        f"{eu(pack['charge_kw'], 2)} kW, discharge {eu(pack['discharge_kw'], 2)} kW, SoC floor "
        f"{eu(pack['soc_floor'] * 100, 0)} %, from {eu(settings['capacity_ah'], 0)} Ah at "
        f"{eu(v, 1)} V"
        + ("" if np.isfinite(voltage) else " (no voltage statistics: the nominal value)")
    )
    means = _hourly_means(power_rows)
    power = np.asarray([means.get(int(t), np.nan) for t in table["ts"]], dtype=float)
    sign = S.infer_sign(power, table["batt_charge_kwh"], table["batt_discharge_kwh"])
    if sign is None:
        print("battery sign: no evidence, using the profile constant")
        return
    constant = S.SIGNS["battery_power"]
    verdict = (
        "matches the profile constant"
        if sign == constant
        else f"differs from the profile constant {constant:+d}"
    )
    print(f"battery sign inferred: {sign:+d} ({verdict})")


if __name__ == "__main__":
    main()
