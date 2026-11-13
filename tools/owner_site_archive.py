"""Export the owner's VictoriaMetrics minute archive of the site flows to a flows CSV.

Owner-only. Writes the flows format (core.site.flows.COLUMNS) with one row per minute,
step_s 60. The owner supplies the mapping at run time through environment variables:
VM_URL, BB_SITE_GRID_ENTITY, BB_SITE_GRID_SIGN, BB_SITE_BATT_ENTITY, BB_SITE_BATT_SIGN,
BB_SITE_SOC_ENTITY, and the optional BB_SITE_LOAD_ENTITY, BB_SITE_GEN_ENTITY, BB_SITE_PV_ENTITY,
BB_SITE_GRID_IMPORT_KW_ENTITY and BB_SITE_GRID_EXPORT_KW_ENTITY. No entity id is printed or written
to the CSV.

Grid fallback: a derived grid power sensor can start later than the raw meter series it is made
from. When BB_SITE_GRID_IMPORT_KW_ENTITY and BB_SITE_GRID_EXPORT_KW_ENTITY are both set (the raw
meter's import and export power, metric `homeassistant_sensor_power_kw`), a minute where the grid
entity has no sample takes (import - export) * 1000 W from them, turned into the grid entity's sign
convention. A minute with a direct sample keeps it.

Power comes from `homeassistant_sensor_power_w`, the state of charge from
`homeassistant_sensor_battery_percent`, both as `avg_over_time(...[1m])` relabelled to the minute
start (tools.vm_live_archive.start_labels): the row labelled t holds the minute [t, t + 60).

Signs: BB_SITE_GRID_SIGN is +1 when the grid sensor reads positive on import, BB_SITE_BATT_SIGN
is +1 when the battery sensor reads positive while charging. `--check-signs` prints the mean net
battery power (charge positive) while the SoC rises and while it falls; it must be positive while
the SoC rises.

Days are local Europe/Amsterdam calendar days, `--from` and `--to` both inclusive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.parse
import urllib.request

import numpy as np
from betterybalance.core.fmt import eu
from betterybalance.core.site.flows import write_flows
from betterybalance.core.timegrid import ZONE, Table

from tools.vm_live_archive import CHUNK_S, STEP_S, start_labels, window

POWER_METRIC = "homeassistant_sensor_power_w"
POWER_KW_METRIC = "homeassistant_sensor_power_kw"
SOC_METRIC = "homeassistant_sensor_battery_percent"
KEYS = ("grid", "batt", "soc", "load", "gen", "pv")
_REQUIRED_ENV = ("VM_URL", "BB_SITE_GRID_ENTITY", "BB_SITE_BATT_ENTITY", "BB_SITE_SOC_ENTITY")
_ENTITY_ENV = {
    "grid": "BB_SITE_GRID_ENTITY",
    "batt": "BB_SITE_BATT_ENTITY",
    "soc": "BB_SITE_SOC_ENTITY",
    "load": "BB_SITE_LOAD_ENTITY",
    "gen": "BB_SITE_GEN_ENTITY",
    "pv": "BB_SITE_PV_ENTITY",
}
_W_TO_KWH_MINUTE = 1.0 / 1000.0 / 60.0


def _column(stamps: np.ndarray, series: dict[int, float] | None) -> np.ndarray:
    series = series or {}
    return np.asarray([series.get(int(t), np.nan) for t in stamps], dtype=float)


def _split(signed_kwh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive part and negative part (as a positive number); NaN stays NaN."""
    return np.maximum(signed_kwh, 0.0), np.maximum(-signed_kwh, 0.0)


def grid_with_fallback(
    grid: dict[int, float],
    import_kw: dict[int, float],
    export_kw: dict[int, float],
    grid_sign: int,
) -> dict[int, float]:
    """Grid power in W in the grid entity's convention, gaps filled from the raw meter series.

    A minute without a grid sample, with both an import and an export sample, takes
    grid_sign * (import - export) * 1000: import - export is import positive, and minute_rows
    multiplies by grid_sign again. A minute with a grid sample keeps it.
    """
    out = dict(grid)
    for t in import_kw.keys() & export_kw.keys():
        if t not in out:
            out[t] = grid_sign * (import_kw[t] - export_kw[t]) * 1000.0
    return out


def minute_rows(series: dict[str, dict[int, float]], grid_sign: int, batt_sign: int) -> Table:
    """Minute flows (step_s 60) on the union of the minutes of all series.

    kwh = w / 1000 / 60. The grid splits by sign after grid_sign (positive is import), the
    battery after batt_sign (positive is charge). soc_pct = soc_min_pct = soc_max_pct = soc.
    A minute without a sample of a series is NaN in that series' columns; a missing or empty
    optional series (load, gen, pv) is NaN throughout.
    """
    stamps: set[int] = set()
    for key in KEYS:
        stamps.update(int(t) for t in (series.get(key) or {}))
    ts = np.asarray(sorted(stamps), dtype=np.int64)
    out: Table = {"ts": ts, "step_s": np.full(len(ts), STEP_S, dtype=np.int64)}
    grid_kwh = _column(ts, series.get("grid")) * grid_sign * _W_TO_KWH_MINUTE
    out["grid_import_kwh"], out["grid_export_kwh"] = _split(grid_kwh)
    batt_kwh = _column(ts, series.get("batt")) * batt_sign * _W_TO_KWH_MINUTE
    out["batt_charge_kwh"], out["batt_discharge_kwh"] = _split(batt_kwh)
    soc = _column(ts, series.get("soc"))
    out["soc_pct"], out["soc_min_pct"], out["soc_max_pct"] = soc, soc.copy(), soc.copy()
    for col, key in (("load_kwh", "load"), ("gen_kwh", "gen"), ("pv_kwh", "pv")):
        out[col] = _column(ts, series.get(key)) * _W_TO_KWH_MINUTE
    return out


def sign_check(table: Table) -> dict:
    """Mean net battery power in kW (charge positive) over the minutes where the SoC rises or falls.

    A pair of consecutive minutes (60 s apart, SoC and battery both present) counts as rising
    when the later minute's SoC is above the earlier one, falling when below; the battery power of
    the later minute is attributed to the pair. With the right sign the value while the SoC rises
    is positive. A mean over no minutes is NaN.
    """
    ts = np.asarray(table["ts"], dtype=np.int64)
    soc = np.asarray(table["soc_pct"], dtype=float)
    net_kw = (
        np.asarray(table["batt_charge_kwh"], dtype=float)
        - np.asarray(table["batt_discharge_kwh"], dtype=float)
    ) * 60.0
    if len(ts) < 2:
        rising = falling = np.zeros(0, dtype=bool)
        later = np.zeros(0)
    else:
        ok = (np.diff(ts) == STEP_S) & np.isfinite(soc[1:]) & np.isfinite(soc[:-1])
        ok &= np.isfinite(net_kw[1:])
        d_soc = np.diff(soc)
        rising, falling, later = ok & (d_soc > 0), ok & (d_soc < 0), net_kw[1:]

    def mean(mask: np.ndarray) -> float:
        return float(later[mask].mean()) if mask.any() else float("nan")

    return {
        "batt_kw_while_soc_rises": mean(rising),
        "batt_kw_while_soc_falls": mean(falling),
        "minutes_rising": int(rising.sum()),
        "minutes_falling": int(falling.sum()),
    }


def query_range(vm_url: str, metric: str, entity: str, start: int, end: int) -> dict[int, float]:
    """Start-labelled minute averages of one entity's metric, steps T in [start, end], in chunks."""
    out: dict[int, float] = {}
    query = f'avg_over_time({metric}{{entity="{entity}"}}[1m])'
    t = start
    while t <= end:
        stop = min(t + CHUNK_S - STEP_S, end)
        params = urllib.parse.urlencode({"query": query, "start": t, "end": stop, "step": STEP_S})
        url = f"{vm_url.rstrip('/')}/api/v1/query_range?{params}"
        with urllib.request.urlopen(url, timeout=120) as response:
            body = json.load(response)
        out.update(start_labels(body))
        t = stop + STEP_S
    return out


def _sign(name: str) -> int:
    value = int(os.environ.get(name, "1"))
    if value not in (1, -1):
        raise SystemExit(f"{name} must be 1 or -1, got {value}")
    return value


def _local_iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), ZONE).isoformat()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="start", required=True, help="first local day, inclusive")
    parser.add_argument("--to", dest="end", required=True, help="last local day, inclusive")
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-signs", action="store_true")
    args = parser.parse_args(argv)
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        parser.error(f"set the environment variables {', '.join(missing)}")
    grid_sign, batt_sign = _sign("BB_SITE_GRID_SIGN"), _sign("BB_SITE_BATT_SIGN")
    vm_url = os.environ["VM_URL"]
    # step T holds the minute labelled T - 60, so steps start + 60 to end give the labels start
    # to end - 60: every minute of the local days named, and none outside them.
    start, end = window(args.start, args.end)
    series: dict[str, dict[int, float]] = {}

    def fetch(metric: str, entity: str) -> dict[int, float]:
        raw = query_range(vm_url, metric, entity, start + STEP_S, end)
        return {t: v for t, v in raw.items() if start <= t < end}

    for key, env in _ENTITY_ENV.items():
        entity = os.environ.get(env)
        series[key] = fetch(SOC_METRIC if key == "soc" else POWER_METRIC, entity) if entity else {}
        print(f"{key}: {eu(len(series[key]), 0)} minutes")
    import_entity = os.environ.get("BB_SITE_GRID_IMPORT_KW_ENTITY")
    export_entity = os.environ.get("BB_SITE_GRID_EXPORT_KW_ENTITY")
    if import_entity and export_entity:
        direct = len(series["grid"])
        series["grid"] = grid_with_fallback(
            series["grid"],
            fetch(POWER_KW_METRIC, import_entity),
            fetch(POWER_KW_METRIC, export_entity),
            grid_sign,
        )
        print(f"grid from the meter fallback: {eu(len(series['grid']) - direct, 0)} minutes")
    table = minute_rows(series, grid_sign, batt_sign)
    write_flows(table, args.out)
    n = len(table["ts"])
    print(f"Rows: {eu(n, 0)}")
    if n:
        print(f"First minute: {_local_iso(table['ts'][0])}")
        print(f"Last minute: {_local_iso(table['ts'][-1])}")
        expected = (end - start) // STEP_S
        print(f"Minutes in the window: {eu(expected, 0)}")
        required = ("grid_import_kwh", "batt_charge_kwh", "soc_pct")
        incomplete = np.zeros(n, dtype=bool)
        for col in required:
            incomplete |= np.isnan(table[col])
        print(f"Minutes without a grid, battery or SoC sample: {eu(int(incomplete.sum()), 0)}")
    if args.check_signs:
        chk = sign_check(table)
        print(
            f"Battery net power while the SoC rises: {eu(chk['batt_kw_while_soc_rises'], 3)} kW"
            f" over {eu(chk['minutes_rising'], 0)} minutes"
        )
        print(
            f"Battery net power while the SoC falls: {eu(chk['batt_kw_while_soc_falls'], 3)} kW"
            f" over {eu(chk['minutes_falling'], 0)} minutes"
        )
        rises = chk["batt_kw_while_soc_rises"]
        verdict = "ok" if rises > 0 else "inverted: set BB_SITE_BATT_SIGN to the other value"
        print(f"Battery sign {eu(batt_sign, 0)}: {verdict}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
