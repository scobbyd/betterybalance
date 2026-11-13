"""Plan 2 acceptance on the owner's site: the site-model checks of design section 6.9.

Subcommands:
- merged-micro: the merged-micro dataset. The house is the metered load register and the only
  unmetered PV is the microinverter on the GEN port, merged as if it fed the load port, with no
  battery. The blind fit learns the plane and the scale, and the checks compare them with the
  micro's metered output and its reference plane.
- blind-fit: the full house, blind. The fit sees the grid, the battery (with the pack's `dc`
  flag) and the SoC; the GEN port reads 0 because the replay learns all PV. The checks compare
  the learned planes with the metered Deye strings plus the metered micro and with the
  reference planes, on the minute window and on the summer hourly window, and run the
  learning check of 6.8 from 2026-09-02.

The reference planes live in a git-ignored site file (`reference.micro`, `reference.planes`),
passed as a path; nothing in this module names the owner's planes or location. Reports carry
European numbers, deltas and ratios only, and no coordinates.

Run (background, log under data/results/):
python -m tools.study.plan2_acceptance merged-micro --minute ... --hourly ... --irradiance ...
--site-planes ... --lat ... --lon ... --out-json ... --out-md ...
python -m tools.study.plan2_acceptance blind-fit --minute ... --hourly ... --irradiance ...
--site ... --site-planes ... --out-json ... --out-md ...
Re-render only: add the flag --render-only. It reads the JSON at --out-json and rewrites the
section in --out-md; the other arguments must be given but are not read.
- replay-windows: the replay of the shadow trader on the live window (minute flows, tiers A
  to D) and the A/B window (hourly flows, tiers A and B), each with its walk-forward band
  cache and the 20-rung ladder; the selection rule on both windows, the greed scenarios, the
  hourly bias, the pooled summary and the report. `--windows live` (or `ab`) with
  `--no-assemble` runs one window and saves its state; `--windows ""` assembles the report
  from the saved states.
python -m tools.study.plan2_acceptance replay-windows --site ... --minute ... --hourly ...
--irradiance ... --data data --out data/results/plan2 --report docs/reports/... [--plant ...]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from betterybalance.core.feeds.openmeteo import IRRADIANCE_COLS, read_irradiance_csv
from betterybalance.core.fmt import NA as FMT_NA
from betterybalance.core.fmt import eu
from betterybalance.core.inputs import load_inputs
from betterybalance.core.ledger import bootstrap_interval
from betterybalance.core.site import contract as C
from betterybalance.core.site.contract import Plane, SiteConfig
from betterybalance.core.site.flows import read_flows, to_quarters
from betterybalance.core.site.load import local_fields
from betterybalance.core.site.plant import plane_shape
from betterybalance.core.site.registry import (
    HOLDOUT_RATIO,
    holdout_passes,
    initial_state,
    weekly_update,
)
from betterybalance.core.site.separate import (
    CLIP_TOL,
    PvModel,
    curtailment_mask,
    fit_pv,
    hour_mean,
    hourly_origin_rows,
    huber_fit,
    predict_pv_kw,
)
from betterybalance.core.site.solar import clear_sky_ghi, clearness, daylight, solar_position
from betterybalance.core.timegrid import ZONE, Table, align
from betterybalance.replay import __main__ as M
from betterybalance.replay import phases as P
from betterybalance.replay import report as R
from betterybalance.replay.bands import BandCache
from betterybalance.replay.disaggregate import HOUR_S, hourly_to_quarters
from betterybalance.replay.engine import ReplayInputs, ReplayResult, replay
from betterybalance.replay.report import ATTRIBUTION

CLEAR_KT = 0.8
GEOMETRY_MAX_DEG = 15.0
SCALE_TOL = 0.15  # design 6.9: within 15 % of the effective kWp
HOURLY_MAE_MAX = 0.20
PROPORTION_TOL = 0.10
SCALES = (1.0, 0.5, 2.0)
MINUTE_FROM = "2026-09-02"
HOURLY_FROM = "2026-06-01"
HOURLY_TO = "2026-09-01"
LEARN_BY = "2026-09-15"
LEARN_MAX_DAYS = 60
BLIND_PLANES = 2  # design 6.9: the owner's two main roof planes
BIAS_MAX = 0.05  # design 6.9: hourly bias within 5 % of the mean production
CURTAIL_MONTHS = ("2026-04", "2026-05", "2026-06")
NA_TEXT = "not applicable"
# merged-micro rows that are not design 6.9 checks of the merged-micro table
MERGED_DIAGNOSTICS = frozenset({"proportional scale", "learning", "clipping"})
_PORT_PLACEHOLDERS = {
    "grid_import_kwh": 0.0,
    "grid_export_kwh": 0.0,
    "batt_charge_kwh": 0.0,
    "batt_discharge_kwh": 0.0,
    "soc_pct": 50.0,
    "soc_min_pct": 50.0,
    "soc_max_pct": 50.0,
}
# the model piece a miss points at, per check kind
# the terminal valuation's units (M2 of the final review)
TERMINAL_NOTE = (
    "The terminal valuation follows the spec formula: it prices the SoC difference at the end "
    "of the window at the trailing mean buy price, excluding fees, energy tax and BTW, so it "
    "understates the bill value of a stored kWh. It sits beside the difference, not inside it."
)
# what the drift alarm is and is not (I3 of the final review): one text for every report part
DRIFT_NOTE = (
    "The drift alarm is not a guard for the night-anchor or the geometry risk. It compares the "
    "PV model with the deficit against the same night-anchored load on the clear quarters of "
    "the trailing week and flags a mean gap above 15 % of the modelled PV, so it cannot see "
    "the anchor's own error, and a rotated plane with a compensating scale stays quiet too. "
    "Nothing in plan 2 acts on its flag; the replay only reports it. It fired on both owner "
    "windows of the replay. Acting on it is plan-3 work."
)
# the scale estimator and its known risk, for the method sections of both report parts
ESTIMATOR_NOTE = (
    "The PV fit has two stages: slot-centred rounds choose the planes and their relative "
    "scales, and one level multiplier against a night-anchored load (the wall-slot median of "
    "the residual over night quarters, daytime slots at the night mean) sets the scale, "
    "because the slot-centred fit alone attenuates the scale on forecast irradiance. Known "
    "risk: a daytime load above the night mean pulls the scale low by that excess, and an "
    "evening peak inside the night rows (late sunset) lifts the anchor and pulls the scale "
    f"high. {DRIFT_NOTE}"
)
# the replay's PV expectation reads the irradiance archive (C1 of the final review)
PV_EXPECTATION_NOTE = (
    "The PV expectation is not held to that rule. It reads the irradiance archive, the "
    "open-meteo historical-forecast API's archived model output for the day itself (ERA5 "
    "reanalysis where it was missing), also for the slots 1 to 3 days ahead, where a live box "
    "would have only a multi-day forecast. The replay's PV expectation is therefore "
    "optimistic, and the bias favours the virtual path. A replay with as-issued multi-day "
    "forecasts is plan-3 data."
)
# the site-model fits read the same archive (method sections of both report parts)
IRRADIANCE_NOTE = (
    "The irradiance is the open-meteo archive: the historical-forecast API's archived model "
    "output for the day itself (ERA5 reanalysis where it was missing). The fits and their "
    "hourly errors read it for the same day, so they measure the plant model on day-0 "
    "weather, not on a forecast 1 to 3 days ahead as a live box would have it."
)
_MISS_PIECE = {
    "plane count": "the plane search (second-plane penalty knob, 2 %) and the plane grid",
    "azimuth": "the plane grid and the Hay-Davies transposition of the plant model",
    "tilt": "the plane grid and the Hay-Davies transposition of the plant model",
    "scale": "the slot-centred Huber fit and the month factors",
    "hourly error": "the plant model and the per-slot load profile",
    "curtailment": "the curtailment mask of 6.4 (SoC 95 % and export under 100 W)",
    "clipping": "the clipping detector of 6.4 (plateau share 0,9, model excess 8 %)",
    "proportional scale": "the Huber delta and the night-anchored load level",
    "learning": "the holdout rule of 6.8 (error ratio 0,7, stability 0,8 to 1,25, 30 degrees)",
    "hourly bias": "the plant model, the month factors and the per-slot load profile",
}


# measured causes of misses, from the readings of the report sections (the 2026-10-31 runs)
_RIDGE = (
    "geometry and scale trade off along a ridge of nearly equal loss, so the slot-centred stage "
    "settles on a rotated or tilted plane and the scale makes up the output"
)
MERGED_CAUSES: dict[str, str] = {
    "azimuth": _RIDGE,
    "tilt": _RIDGE,
    "hourly error": "the weather data: the reference plane at its own effective kWp misses too "
    "(46,2 % in the minute window, 38,0 % in the hourly window)",
    "proportional scale": "at half the PV the night anchor's offset (a fixed load error in kW) "
    "weighs twice as much; at twice the PV the fit takes a plane 49 degrees off and the scale "
    "makes up the output (" + _RIDGE + ")",
}
BLIND_CAUSES: dict[str, str] = {
    "azimuth": _RIDGE + "; the reference planes given score about the same hourly error",
    "tilt": _RIDGE + "; the reference planes given score about the same hourly error",
    "scale": "the learned planes' geometry is off, and a kWp on another geometry does not "
    "compare with the effective kWp: with the reference planes given the same estimator gives "
    "1,045 and 1,051 (minute) and 1,001 and 0,884 (summer)",
    "plane count": "in the summer hourly data the two roofs on opposite sides merge into one "
    "plane between them, and a second plane does not lower the loss by the 2 % penalty",
    "hourly error": "the weather data: the reference planes at their own effective kWp reach "
    "25,7 % (minute) and 29,4 % (summer), above the target",
    "hourly bias": "the reference planes' own bias (-10,1 % minute, -2,1 % summer), the month "
    "factors fitted on the anchored deficit (0,91 to 0,96) and the night anchor's known risk "
    "(a daytime load above the night mean pulls the scale low)",
}


# --- shared loaders and helpers -------------------------------------------------------------


def local_midnight(day: str) -> int:
    """Epoch s of 00:00 Europe/Amsterdam on an ISO day."""
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=ZONE).timestamp())


def local_day(ts: int) -> str:
    """ISO local day of an epoch second."""
    return dt.datetime.fromtimestamp(int(ts), ZONE).date().isoformat()


def _plus_days(day: str, k: int) -> str:
    return (dt.date.fromisoformat(day) + dt.timedelta(days=k)).isoformat()


def window(table: Table, lo: int | None = None, hi: int | None = None) -> Table:
    """Rows with lo <= ts < hi (an open end where None)."""
    ts = np.asarray(table["ts"], dtype=np.int64)
    keep = np.ones(len(ts), dtype=bool)
    if lo is not None:
        keep &= ts >= lo
    if hi is not None:
        keep &= ts < hi
    return {k: np.asarray(v)[keep] for k, v in table.items()}


def port_quarters(flows: Table) -> Table:
    """Quarters of a flows table that keep the load and GEN ports whatever the grid columns hold.

    The merged datasets replace the grid, battery and SoC columns, so placeholders stand in for
    them here: a quarter is then missing only for lack of time coverage, and a NaN load or GEN
    cell stays NaN. Rows of 3.600 s split flat (no plant shape), shorter rows aggregate.
    """
    t = {k: np.asarray(v).copy() for k, v in flows.items()}
    n = len(t["ts"])
    for col, value in _PORT_PLACEHOLDERS.items():
        t[col] = np.full(n, value)
    step = np.asarray(t["step_s"], dtype=np.int64)
    if n and (step == HOUR_S).all():
        return hourly_to_quarters(t)
    return to_quarters(t)


def blind_config(site: dict, lat: float, lon: float) -> SiteConfig:
    """The site config with the given location, blind PV (no planes, not none), no EV.

    A site dict without a pack takes a neutral 10 kWh pack: the merged datasets carry no
    battery flows, so the pack never enters the residual.
    """
    d = {k: v for k, v in site.items() if k != "reference"}
    d["latitude"], d["longitude"] = float(lat), float(lon)
    d.setdefault(
        "pack",
        {
            "capacity_kwh": 10.0,
            "charge_kw": 5.0,
            "discharge_kw": 5.0,
            "soc_floor": 0.1,
            "soc_ceiling": 1.0,
        },
    )
    d["pv"] = {"none": False, "planes": [], "correction": 1.0}
    d["ev"] = {"present": False}
    return C.from_dict(d)


def check(name: str, kind: str, value, target: str, passed: bool | None, shown: str) -> dict:
    """One acceptance check: plain value for the JSON, `shown` (European) for the report."""
    return {
        "name": name,
        "kind": kind,
        "value": value,
        "target": target,
        "pass": passed,
        "shown": shown,
    }


def write_section(path: Path, key: str, body: str, title: str) -> None:
    """Put body between the section markers of key; other text in the file stays.

    A new file starts with the title. A file without the markers gets the section appended.
    """
    start, end = f"<!-- section:{key} -->", f"<!-- /section:{key} -->"
    block = f"{start}\n{body.rstrip()}\n{end}\n"
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {title}\n\n{block}")
        return
    text = path.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        tail = rest.split(end, 1)[1].lstrip("\n")
        path.write_text(head + block + ("\n" + tail if tail else ""))
    else:
        path.write_text(text.rstrip() + "\n\n" + block)


def _eu(x: float | None, dec: int = 2) -> str:
    """eu with "not applicable" for None and NaN."""
    return NA_TEXT if x is None or not np.isfinite(x) else eu(x, dec)


def _pct(x: float | None, dec: int = 1) -> str:
    return NA_TEXT if x is None or not np.isfinite(x) else f"{eu(100.0 * x, dec)} %"


def _deg(x: float | None) -> str:
    return NA_TEXT if x is None or not np.isfinite(x) else f"{eu(x, 0)} degrees"


def _scale_label(s: float) -> str:
    return str(int(s)) if float(s).is_integer() else eu(s, 1)


def _num(x) -> float | None:
    """A float for JSON, None for NaN or None."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


_MONTH_ABBR = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def window_months(ts: np.ndarray, rows: np.ndarray) -> list[int]:
    """The local months (1..12) of the given rows, sorted."""
    _, _, day = local_fields(np.asarray(ts, dtype=np.int64)[np.asarray(rows, dtype=bool)])
    months = day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1
    return sorted({int(m) for m in months})


def factors_shown(month_factor: list[float], months: list[int]) -> str:
    """The month factors of the given months, European numbers ("Sep 0,95, Oct 1,02")."""
    if not months:
        return NA_TEXT
    return ", ".join(f"{_MONTH_ABBR[m - 1]} {eu(month_factor[m - 1], 2)}" for m in months)


# --- merged-micro building blocks -----------------------------------------------------------


def merged_residual(quarters: Table, scale: float = 1.0) -> tuple[Table, np.ndarray]:
    """A blind flows table of the merged house, and the truth PV in kW.

    The house is load_kwh (the load register) and the unmetered PV is gen_kwh * scale:
    grid import max(load - pv, 0), grid export max(pv - load, 0), battery zero, SoC 50 %
    constant, load, gen and pv NaN (blind). A quarter is missing where the input is missing or
    either port is NaN; its grid cells are NaN. An `hourly_origin` column carries over. The
    truth is gen_kwh * scale * 4 kW.
    """
    ts = np.asarray(quarters["ts"], dtype=np.int64)
    n = len(ts)
    load = np.asarray(quarters["load_kwh"], dtype=float)
    pv = np.asarray(quarters["gen_kwh"], dtype=float) * float(scale)
    missing = np.asarray(quarters["missing"], dtype=bool) | np.isnan(load) | np.isnan(pv)
    net = load - pv
    blind = np.full(n, np.nan)
    merged: Table = {
        "ts": ts,
        "step_s": np.asarray(quarters["step_s"], dtype=np.int64),
        "grid_import_kwh": np.where(missing, np.nan, np.maximum(net, 0.0)),
        "grid_export_kwh": np.where(missing, np.nan, np.maximum(-net, 0.0)),
        "batt_charge_kwh": np.zeros(n),
        "batt_discharge_kwh": np.zeros(n),
        "soc_pct": np.full(n, 50.0),
        "soc_min_pct": np.full(n, 50.0),
        "soc_max_pct": np.full(n, 50.0),
        "load_kwh": blind,
        "gen_kwh": blind.copy(),
        "pv_kwh": blind.copy(),
        "missing": missing,
    }
    if "hourly_origin" in quarters:  # flat splits of hourly data stay marked for the fit
        merged["hourly_origin"] = np.asarray(quarters["hourly_origin"], dtype=bool)
    return merged, pv * 4.0


def effective_kwps(
    truth_kw: np.ndarray, irr: Table, planes: list[Plane], lat: float, lon: float, ts: np.ndarray
) -> np.ndarray:
    """Huber-fitted joint scales of the metered series on the planes' shapes, clear quarters.

    Clear: daylight and clearness at or above 0,8. Delta is a third of the 90th percentile of
    the metered power on those quarters. NaN scales without clear quarters.
    """
    ts = np.asarray(ts, dtype=np.int64)
    truth = np.asarray(truth_kw, dtype=float)
    aligned = align(irr, ts, list(IRRADIANCE_COLS))
    shapes = np.column_stack([plane_shape(aligned, p, lat, lon) for p in planes])
    zenith, _ = solar_position(ts, lat, lon)
    kt = np.nan_to_num(clearness(aligned["ghi_w_m2"], clear_sky_ghi(zenith)), nan=-1.0)
    use = daylight(zenith) & (kt >= CLEAR_KT) & np.isfinite(truth)
    use &= np.isfinite(shapes).all(axis=1)
    if not use.any():
        return np.full(len(planes), np.nan)
    delta = max(float(np.percentile(truth[use], 90)) / 3.0, 1e-3)
    return huber_fit(shapes[use], truth[use], delta)


def effective_kwp(
    truth_kw: np.ndarray, irr: Table, plane: Plane, lat: float, lon: float, ts: np.ndarray
) -> float:
    """Huber-fitted scale of the metered series on the plane's shape, clear quarters only.

    Clear: daylight and clearness at or above 0,8. Delta is a third of the 90th percentile of
    the metered power on those quarters. NaN without clear quarters.
    """
    return float(effective_kwps(truth_kw, irr, [plane], lat, lon, ts)[0])


def _normal(tilt: float, azimuth: float) -> np.ndarray:
    t, a = np.radians(tilt), np.radians(azimuth)
    return np.array([np.sin(t) * np.sin(a), np.sin(t) * np.cos(a), np.cos(t)])


def _circular(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def compare_geometry(model: PvModel, truth_planes: list[tuple[float, float, float]]) -> dict:
    """Per truth plane (index -> dict): the nearest learned plane and the deltas.

    Truth planes are (tilt, azimuth, kWp). Nearest is the smallest angle between the plane
    normals; learned_index is its index in model.planes. Deltas are absolute degrees (azimuth
    on the circle); scale_ratio is the learned scale over the truth kWp. Without learned planes
    the index, the deltas and the ratio are None.
    """
    out = {}
    for i, (tilt, az, kwp) in enumerate(truth_planes):
        row = {
            "truth_tilt": float(tilt),
            "truth_azimuth": float(az),
            "truth_kwp": _num(kwp),
            "learned_index": None,
            "learned_tilt": None,
            "learned_azimuth": None,
            "learned_kwp": None,
            "tilt_delta": None,
            "azimuth_delta": None,
            "scale_ratio": None,
        }
        if model.planes:
            n = _normal(tilt, az)
            angles = [
                float(np.arccos(np.clip(n @ _normal(p.tilt_deg, p.azimuth_deg), -1.0, 1.0)))
                for p in model.planes
            ]
            j = int(np.argmin(angles))
            p, s = model.planes[j], float(model.scales_kwp[j])
            row.update(
                learned_index=j,
                learned_tilt=float(p.tilt_deg),
                learned_azimuth=float(p.azimuth_deg),
                learned_kwp=s,
                tilt_delta=abs(float(p.tilt_deg) - float(tilt)),
                azimuth_delta=_circular(float(p.azimuth_deg), float(az)),
                scale_ratio=_num(s / float(kwp)) if _num(kwp) else None,
            )
        out[i] = row
    return out


def hourly_error(
    est_kw: np.ndarray, truth_kw: np.ndarray, ts: np.ndarray, mask: np.ndarray
) -> dict:
    """Hourly error of an estimate against the metered truth over the masked quarters.

    Per clock hour the mean of the masked quarters of each series. mae_share is the hourly MAE
    over the mean hourly production, bias_share the mean hourly error over the same.
    """
    use = np.asarray(mask, dtype=bool) & np.isfinite(est_kw) & np.isfinite(truth_kw)
    if not use.any():
        return {"mae_share": None, "bias_share": None, "mae_kw": None, "mean_kw": None, "hours": 0}
    hour = np.asarray(ts, dtype=np.int64)[use] // HOUR_S
    _, inv = np.unique(hour, return_inverse=True)
    count = np.bincount(inv)
    est_h = np.bincount(inv, weights=np.asarray(est_kw, dtype=float)[use]) / count
    tru_h = np.bincount(inv, weights=np.asarray(truth_kw, dtype=float)[use]) / count
    mean = float(tru_h.mean())
    mae = float(np.abs(est_h - tru_h).mean())
    bias = float((est_h - tru_h).mean())
    return {
        "mae_share": mae / mean if mean > 0 else None,
        "bias_share": bias / mean if mean > 0 else None,
        "mae_kw": mae,
        "mean_kw": mean,
        "hours": int(len(count)),
    }


def learning_pass_day(
    quarters: Table,
    irr: Table,
    cfg: SiteConfig,
    first_day: str,
    max_days: int = 60,
    trace: list | None = None,
    run_on: bool = False,
) -> str | None:
    """The first local day after whose end the registry champion passes the holdout rule.

    The registry starts cold on first_day and runs weekly_update at every local midnight that
    follows (daily cadence, data from first_day on). The day returned is the last day the
    passing fit saw. None when no champion passes within max_days or before the data ends.
    A trace list, when given, gets one row per fit: the day, the challenger's holdout ratio
    (holdout error over null error), its stability pair, and whether challenger and champion
    pass the rule. With run_on the registry keeps refitting after the pass day up to max_days
    or the end of the data, so the trace shows whether the champion stays out of learning; the
    day returned is still the first pass.
    """
    q = window(quarters, local_midnight(first_day))
    if len(q["ts"]) == 0:
        return None
    end = int(np.max(q["ts"])) + 900
    state = initial_state(cfg)
    first: str | None = None
    for k in range(1, max_days + 1):
        now = local_midnight(_plus_days(first_day, k))
        if now > end:
            break
        state = weekly_update(state, q, irr, cfg, now)
        if trace is not None and state.challenger is not None and state.last_fit_ts == now:
            ch = state.challenger
            ratio = None
            if ch.holdout_error is not None and ch.null_error:
                ratio = ch.holdout_error / ch.null_error
            trace.append(
                {
                    "day": _plus_days(first_day, k - 1),
                    "holdout_ratio": _num(ratio),
                    "scale_ratio": _num(ch.stability[0]),
                    "azimuth_delta": _num(ch.stability[1]),
                    "challenger_passes": holdout_passes(ch),
                    "champion_passes": holdout_passes(state.champion),
                }
            )
        if first is None and state.champion is not None and holdout_passes(state.champion):
            first = _plus_days(first_day, k - 1)
            if not run_on:
                return first
    return first


# --- merged-micro run -----------------------------------------------------------------------


def _run(
    ports: Table,
    irr: Table,
    cfg: SiteConfig,
    scale: float,
    micro: Plane,
    learn_from: str | None,
    log,
) -> dict:
    t0 = time.monotonic()
    lat, lon = cfg.latitude, cfg.longitude
    merged, truth = merged_residual(ports, scale)
    ts = np.asarray(merged["ts"], dtype=np.int64)
    missing = np.asarray(merged["missing"], dtype=bool)
    now = int(ts.max()) + 900
    model, _, _ = fit_pv(merged, irr, cfg, now)
    aligned = align(irr, ts, list(IRRADIANCE_COLS))
    hourly_rows = hourly_origin_rows(merged)
    est = predict_pv_kw(model, aligned, lat, lon, hour_mean_rows=hourly_rows)
    zenith, _ = solar_position(ts, lat, lon)
    lit = daylight(zenith)
    valid = lit & ~missing & np.isfinite(truth)
    eff = effective_kwp(np.where(missing, np.nan, truth), irr, micro, lat, lon, ts)
    geo = compare_geometry(model, [(micro.tilt_deg, micro.azimuth_deg, eff)])[0]
    err = hourly_error(est, truth, ts, valid)
    # the floor the weather data sets: the reference plane at its effective kWp
    ref_est = (eff if np.isfinite(eff) else 0.0) * plane_shape(aligned, micro, lat, lon)
    ref_est = hour_mean(ref_est, ts, hourly_rows)
    err_ref = hourly_error(ref_est, truth, ts, valid)
    curtailed = int((curtailment_mask(merged) & lit & ~missing).sum())
    months = window_months(ts, lit & ~missing)
    load_kw = np.asarray(ports["load_kwh"], dtype=float) * 4.0
    load_sum = float(load_kw[~missing].sum())
    pv_to_load = float(truth[~missing].sum()) / load_sum if load_sum > 0.0 else None
    fit_s = time.monotonic() - t0
    log(
        f"  fit {eu(fit_s, 1)} s: {len(model.planes)} plane(s), scale ratio "
        f"{eu(geo['scale_ratio'] or float('nan'), 3)}, hourly MAE {_pct(err['mae_share'])}"
    )
    learn_day = None
    trace: list = []
    if learn_from is not None:
        learn_day = learning_pass_day(merged, irr, cfg, learn_from, LEARN_MAX_DAYS, trace)
        log(f"  learning pass day: {learn_day or 'none'}")
    return {
        "scale": float(scale),
        "first_day": local_day(int(ts.min())),
        "last_day": local_day(int(ts.max())),
        "quarters": int(len(ts)),
        "missing_quarters": int(missing.sum()),
        "planes": len(model.planes),
        "learned": [
            {"tilt_deg": float(p.tilt_deg), "azimuth_deg": float(p.azimuth_deg), "kwp": float(s)}
            for p, s in zip(model.planes, model.scales_kwp, strict=True)
        ],
        "scale_kwp": float(np.sum(model.scales_kwp)),
        "effective_kwp": _num(eff),
        "geometry": geo,
        "months": months,
        "month_factor": [float(x) for x in model.month_factor],
        "ac_max_kw": _num(model.ac_max_kw),
        "holdout_error": _num(model.holdout_error),
        "null_error": _num(model.null_error),
        "stability": [_num(x) for x in model.stability],
        "holdout_passes": holdout_passes(model),
        "hourly": err,
        "hourly_reference": err_ref,
        "curtailed_quarters": curtailed,
        "pv_to_load": pv_to_load,
        "learning_day": learn_day,
        "learning_trace": trace,
        "wall_s": time.monotonic() - t0,
    }


def _run_key(window_name: str, scale: float) -> str:
    return f"{window_name}_{scale:g}"


def _checks(runs: dict, learn_by: str) -> list[dict]:
    out = []
    for name in ("minute", "hourly"):
        r = runs[_run_key(name, 1.0)]
        g = r["geometry"]
        out.append(
            check(
                f"plane count ({name})",
                "plane count",
                r["planes"],
                "1",
                r["planes"] == 1,
                str(r["planes"]),
            )
        )
        for kind, key in (("azimuth", "azimuth_delta"), ("tilt", "tilt_delta")):
            v = g[key]
            out.append(
                check(
                    f"{kind} ({name})",
                    kind,
                    v,
                    f"delta at most {eu(GEOMETRY_MAX_DEG, 0)} degrees",
                    v is not None and v <= GEOMETRY_MAX_DEG,
                    f"delta {_deg(v)}",
                )
            )
        ratio = g["scale_ratio"]
        out.append(
            check(
                f"scale ({name})",
                "scale",
                ratio,
                f"learned / effective kWp within {_pct(SCALE_TOL, 0)} of 1",
                ratio is not None and abs(ratio - 1.0) <= SCALE_TOL,
                f"ratio {_eu(ratio, 3)} (month factors "
                f"{factors_shown(r['month_factor'], r.get('months', []))})",
            )
        )
        mae = r["hourly"]["mae_share"]
        out.append(
            check(
                f"hourly error ({name})",
                "hourly error",
                mae,
                f"hourly MAE at most {_pct(HOURLY_MAE_MAX, 0)} of the mean production",
                mae is not None and mae <= HOURLY_MAE_MAX,
                f"{_pct(mae)} (bias {_pct(r['hourly']['bias_share'])})",
            )
        )
    curtailed = sum(r["curtailed_quarters"] for r in runs.values())
    out.append(
        check(
            "curtailment",
            "curtailment",
            curtailed,
            "0 masked quarters",
            curtailed == 0,
            f"{curtailed} quarters",
        )
    )
    clipped = [k for k, r in runs.items() if r["ac_max_kw"] is not None]
    out.append(
        check(
            "clipping",
            "clipping",
            len(clipped),
            "no AC maximum learned",
            not clipped,
            f"{len(clipped)} of {len(runs)} runs",
        )
    )
    base = runs[_run_key("minute", 1.0)]["scale_kwp"]
    for s in SCALES[1:]:
        got = runs[_run_key("minute", s)]["scale_kwp"]
        rel = got / (s * base) if base > 0 else None
        out.append(
            check(
                f"proportional scale {_scale_label(s)}",
                "proportional scale",
                rel,
                f"kWp ratio within {_pct(PROPORTION_TOL, 0)} of {_scale_label(s)}",
                rel is not None and abs(rel - 1.0) <= PROPORTION_TOL,
                f"ratio {_eu(got / base, 3) if base > 0 else NA_TEXT}",
            )
        )
    day = runs[_run_key("minute", 1.0)]["learning_day"]
    out.append(
        check(
            f"learning by {learn_by}",
            "learning",
            day,
            f"pass day at most {learn_by}",
            day is not None and day <= learn_by,
            day or "no pass",
        )
    )
    return out


def _detection_floor(runs: dict) -> dict | None:
    """Smallest PV-to-load energy ratio among the minute scales that pass the learning check."""
    passing = [
        (r["pv_to_load"], r["scale"])
        for k, r in runs.items()
        if k.startswith("minute_") and r["learning_day"] is not None and r["pv_to_load"] is not None
    ]
    if not passing:
        return None
    ratio, scale = min(passing)
    return {"pv_to_load": ratio, "scale": scale}


def merged_micro(
    flows_minute_path: Path,
    flows_hourly_path: Path,
    irradiance_path: Path,
    site_planes_json: Path,
    lat: float,
    lon: float,
    out_json: Path,
    out_md: Path,
    minute_from: str = MINUTE_FROM,
    hourly_from: str = HOURLY_FROM,
    hourly_to: str = HOURLY_TO,
    learn_by: str = LEARN_BY,
    log=print,
) -> dict:
    """The merged-micro validation of design 6.9: fits, checks, a JSON and a report section."""
    t0 = time.monotonic()
    site = json.loads(Path(site_planes_json).read_text())
    ref = site["reference"]["micro"]
    micro = Plane(float(ref["tilt_deg"]), float(ref["azimuth_deg"]))
    cfg = blind_config(site, lat, lon)
    irr = read_irradiance_csv(Path(irradiance_path))
    minute = port_quarters(window(read_flows(Path(flows_minute_path)), local_midnight(minute_from)))
    hourly = port_quarters(
        window(
            read_flows(Path(flows_hourly_path)),
            local_midnight(hourly_from),
            local_midnight(hourly_to),
        )
    )
    runs = {}
    for s in SCALES:
        log(f"minute window, scale {_scale_label(s)}")
        runs[_run_key("minute", s)] = _run(minute, irr, cfg, s, micro, minute_from, log)
    log("hourly window, scale 1")
    runs[_run_key("hourly", 1.0)] = _run(hourly, irr, cfg, 1.0, micro, None, log)
    result = {
        "windows": {
            "minute": {"from": minute_from, "last_day": runs["minute_1"]["last_day"]},
            "hourly": {"from": hourly_from, "to": hourly_to},
        },
        "learn_by": learn_by,
        "runs": runs,
        "checks": _checks(runs, learn_by),
        "detection_floor": _detection_floor(runs),
        "wall_s": time.monotonic() - t0,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, allow_nan=False))
    write_section(Path(out_md), "merged-micro", render_merged_micro(result), REPORT_TITLE)
    return result


# --- blind fit ------------------------------------------------------------------------------


def learns_all_pv(quarters: Table) -> tuple[Table, np.ndarray]:
    """The blind flows of the full house and the truth PV in kW.

    The replay learns all PV: the fit sees the grid, the battery and the SoC, with the GEN port
    at 0 and the load and PV registers NaN. The truth is (pv_kwh + gen_kwh) * 4 kW, NaN where
    the quarter is missing or either register is NaN.
    """
    n = len(quarters["ts"])
    blind = {k: np.asarray(v).copy() for k, v in quarters.items()}
    blind["gen_kwh"] = np.zeros(n)
    blind["load_kwh"] = np.full(n, np.nan)
    blind["pv_kwh"] = np.full(n, np.nan)
    missing = np.asarray(quarters["missing"], dtype=bool)
    pv = np.asarray(quarters["pv_kwh"], dtype=float)
    gen = np.asarray(quarters["gen_kwh"], dtype=float)
    return blind, np.where(missing, np.nan, (pv + gen) * 4.0)


def clean_hour_mask(ts: np.ndarray, valid: np.ndarray, dirty: np.ndarray) -> np.ndarray:
    """The valid quarters of the clock hours that hold no dirty quarter."""
    hour = np.asarray(ts, dtype=np.int64) // HOUR_S
    _, inv = np.unique(hour, return_inverse=True)
    bad_hour = np.bincount(inv, weights=np.asarray(dirty, dtype=float)) > 0
    return np.asarray(valid, dtype=bool) & ~bad_hour[inv]


def curtailment_by_month(quarters: Table, lat: float, lon: float, months: tuple) -> dict:
    """Curtailment-masked daylight quarters per month ("YYYY-MM" -> count) of a quarters table."""
    ts = np.asarray(quarters["ts"], dtype=np.int64)
    zenith, _ = solar_position(ts, lat, lon)
    rows = curtailment_mask(quarters) & daylight(zenith)
    rows &= ~np.asarray(quarters["missing"], dtype=bool)
    out = {}
    for ym in months:
        y, m = (int(x) for x in ym.split("-"))
        lo = local_midnight(f"{y:04d}-{m:02d}-01")
        hi = local_midnight(f"{y + m // 12:04d}-{m % 12 + 1:02d}-01")
        out[ym] = int((rows & (ts >= lo) & (ts < hi)).sum())
    return out


def _angle(a: Plane, b: Plane) -> float:
    c = float(
        np.clip(_normal(a.tilt_deg, a.azimuth_deg) @ _normal(b.tilt_deg, b.azimuth_deg), -1, 1)
    )
    return float(np.degrees(np.arccos(c)))


def _blind_run(
    q: Table,
    irr: Table,
    cfg: SiteConfig,
    mains: list[Plane],
    micro: Plane,
    learn_from: str | None,
    log,
) -> dict:
    """One blind fit on a quarters window, scored against the metered strings and the micro."""
    t0 = time.monotonic()
    lat, lon = cfg.latitude, cfg.longitude
    blind, truth = learns_all_pv(q)
    ts = np.asarray(blind["ts"], dtype=np.int64)
    missing = np.asarray(blind["missing"], dtype=bool)
    hourly_rows = hourly_origin_rows(blind)
    now = int(ts.max()) + 900
    model, _, _ = fit_pv(blind, irr, cfg, now)
    aligned = align(irr, ts, list(IRRADIANCE_COLS))
    est = predict_pv_kw(model, aligned, lat, lon, hour_mean_rows=hourly_rows)
    zenith, _ = solar_position(ts, lat, lon)
    lit = daylight(zenith)
    valid = lit & ~missing & np.isfinite(truth) & np.isfinite(est)
    curtailed = curtailment_mask(blind) & lit & ~missing
    clipped = np.zeros(len(ts), dtype=bool)
    if model.ac_max_kw is not None:
        free = replace(model, ac_max_kw=None)
        unclipped = predict_pv_kw(free, aligned, lat, lon, hour_mean_rows=hourly_rows)
        clipped = lit & (unclipped >= (1.0 - CLIP_TOL) * model.ac_max_kw)
    clean = clean_hour_mask(ts, valid, lit & (curtailed | clipped | ~valid))
    err = hourly_error(est, truth, ts, clean)
    # effective kWp: the Deye strings jointly on the main planes, the micro on its own plane
    pv_kw = np.where(missing, np.nan, np.asarray(q["pv_kwh"], dtype=float) * 4.0)
    gen_kw = np.where(missing, np.nan, np.asarray(q["gen_kwh"], dtype=float) * 4.0)
    eff_main = [_num(x) for x in effective_kwps(pv_kw, irr, mains, lat, lon, ts)]
    eff_micro = _num(effective_kwp(gen_kw, irr, micro, lat, lon, ts))
    refs = [(p.tilt_deg, p.azimuth_deg, k) for p, k in zip(mains, eff_main, strict=True)]
    refs.append((micro.tilt_deg, micro.azimuth_deg, eff_micro))
    geo = compare_geometry(model, refs)
    # a learned plane carries the effective kWp of every reference plane nearest to it
    carried: dict[int, float] = {}
    for g in geo.values():
        if g["learned_index"] is not None and g["truth_kwp"] is not None:
            carried[g["learned_index"]] = carried.get(g["learned_index"], 0.0) + g["truth_kwp"]
    rows = []
    for i, p in enumerate(mains):
        g = dict(geo[i])
        load = carried.get(g["learned_index"]) if g["learned_index"] is not None else None
        g["carried_kwp"] = _num(load)
        g["carried_ratio"] = (
            _num(g["learned_kwp"] / load) if load and g["learned_kwp"] is not None else None
        )
        g["effective_share"] = (
            _num(eff_main[i] / float(p.kwp)) if eff_main[i] is not None and p.kwp else None
        )
        rows.append(g)
    eff_total = sum(x for x in (*eff_main, eff_micro) if x is not None)
    total_ratio = _num(float(np.sum(model.scales_kwp)) / eff_total) if eff_total > 0 else None
    # the floor of the weather data: the reference planes at their effective kWp
    ref_kw = [x or 0.0 for x in (*eff_main, eff_micro)]
    ref_est = sum(
        k * plane_shape(aligned, p, lat, lon) for k, p in zip(ref_kw, (*mains, micro), strict=True)
    )
    ref_est = hour_mean(ref_est, ts, hourly_rows)
    err_ref = hourly_error(ref_est, truth, ts, clean)
    # the planes-given variant: the main reference planes with their kWp, scales fitted
    given_planes = tuple(Plane(p.tilt_deg, p.azimuth_deg, p.kwp) for p in mains)
    given_cfg = replace(cfg, pv=replace(cfg.pv, planes=given_planes, correction=1.0))
    given, _, _ = fit_pv(blind, irr, given_cfg, now)
    est_g = predict_pv_kw(given, aligned, lat, lon, hour_mean_rows=hourly_rows)
    near = int(np.argmin([_angle(micro, p) for p in mains]))
    given_rows = []
    for i in range(len(mains)):
        load = (eff_main[i] or 0.0) + ((eff_micro or 0.0) if i == near else 0.0)
        s = float(given.scales_kwp[i])
        given_rows.append(
            {
                "kwp": s,
                "carried_kwp": _num(load),
                "carried_ratio": _num(s / load) if load > 0 else None,
            }
        )
    err_g = hourly_error(est_g, truth, ts, clean)
    months = window_months(ts, lit & ~missing)
    shown = ", ".join(_eu(g["carried_ratio"], 3) for g in rows)
    log(
        f"  fit {eu(time.monotonic() - t0, 1)} s: {len(model.planes)} plane(s), scale ratios "
        f"{shown}, hourly MAE {_pct(err['mae_share'])} (bias {_pct(err['bias_share'])})"
    )
    learn_day = None
    trace: list = []
    if learn_from is not None:
        learn_day = learning_pass_day(
            blind, irr, cfg, learn_from, LEARN_MAX_DAYS, trace, run_on=True
        )
        log(f"  learning pass day: {learn_day or 'none'}")
    return {
        "first_day": local_day(int(ts.min())),
        "last_day": local_day(int(ts.max())),
        "quarters": int(len(ts)),
        "missing_quarters": int(missing.sum()),
        "hourly_origin": bool(hourly_rows.any()),
        "planes": len(model.planes),
        "learned": [
            {"tilt_deg": float(p.tilt_deg), "azimuth_deg": float(p.azimuth_deg), "kwp": float(s)}
            for p, s in zip(model.planes, model.scales_kwp, strict=True)
        ],
        "reference": rows,
        "micro": dict(geo[len(mains)]),
        "total_ratio": total_ratio,
        "months": months,
        "month_factor": [float(x) for x in model.month_factor],
        "ac_max_kw": _num(model.ac_max_kw),
        "holdout_error": _num(model.holdout_error),
        "null_error": _num(model.null_error),
        "stability": [_num(x) for x in model.stability],
        "holdout_passes": holdout_passes(model),
        "hourly": err,
        "hourly_reference": err_ref,
        "clean_quarters": int(clean.sum()),
        "curtailed_quarters": int(curtailed.sum()),
        "clipped_quarters": int(clipped.sum()),
        "given": {
            "planes": given_rows,
            "month_factor": [float(x) for x in given.month_factor],
            "hourly": err_g,
        },
        "learning_day": learn_day,
        "learning_trace": trace,
        "wall_s": time.monotonic() - t0,
    }


def _blind_checks(runs: dict, learn_by: str) -> list[dict]:
    """The design 6.9 checks of the blind fit per window, then the learning check."""
    out = []
    for name, r in runs.items():
        out.append(
            check(
                f"plane count ({name})",
                "plane count",
                r["planes"],
                str(BLIND_PLANES),
                r["planes"] == BLIND_PLANES,
                str(r["planes"]),
            )
        )
        for i, g in enumerate(r["reference"], start=1):
            for kind, key in (("tilt", "tilt_delta"), ("azimuth", "azimuth_delta")):
                v = g[key]
                out.append(
                    check(
                        f"{kind}, reference plane {i} ({name})",
                        kind,
                        v,
                        f"delta at most {eu(GEOMETRY_MAX_DEG, 0)} degrees",
                        v is not None and v <= GEOMETRY_MAX_DEG,
                        f"delta {_deg(v)}",
                    )
                )
            ratio = g["carried_ratio"]
            out.append(
                check(
                    f"scale, reference plane {i} ({name})",
                    "scale",
                    ratio,
                    f"learned / effective kWp within {_pct(SCALE_TOL, 0)} of 1",
                    ratio is not None and abs(ratio - 1.0) <= SCALE_TOL,
                    f"ratio {_eu(ratio, 3)} (month factors "
                    f"{factors_shown(r['month_factor'], r['months'])})",
                )
            )
        mae, bias = r["hourly"]["mae_share"], r["hourly"]["bias_share"]
        out.append(
            check(
                f"hourly error ({name})",
                "hourly error",
                mae,
                f"hourly MAE at most {_pct(HOURLY_MAE_MAX, 0)} of the mean production",
                mae is not None and mae <= HOURLY_MAE_MAX,
                _pct(mae),
            )
        )
        out.append(
            check(
                f"hourly bias ({name})",
                "hourly bias",
                bias,
                f"within {_pct(BIAS_MAX, 0)} of the mean production",
                bias is not None and abs(bias) <= BIAS_MAX,
                _pct(bias),
            )
        )
    day = runs["minute"]["learning_day"]
    out.append(
        check(
            f"learning by {learn_by}",
            "learning",
            day,
            f"pass day at most {learn_by}",
            day is not None and day <= learn_by,
            day or "no pass",
        )
    )
    return out


def blind_fit(
    flows_minute_path: Path,
    flows_hourly_path: Path,
    irradiance_path: Path,
    site_json: Path,
    site_planes_json: Path,
    out_json: Path,
    out_md: Path,
    minute_from: str = MINUTE_FROM,
    hourly_from: str = HOURLY_FROM,
    hourly_to: str = HOURLY_TO,
    curtail_months: tuple[str, ...] = CURTAIL_MONTHS,
    learn_by: str = LEARN_BY,
    log=print,
) -> dict:
    """The blind fit of design 6.9 on the full house: fits, checks, a JSON and a report section.

    The site file gives the location and the pack (with its `dc` flag). The git-ignored
    site-planes file gives `reference.planes` (the main planes with kWp) and `reference.micro`;
    only the scoring reads them, the fit never sees them. The minute window runs the learning
    check from its first day; the summer hourly window is split flat and marked hourly-origin.
    """
    t0 = time.monotonic()
    site = json.loads(Path(site_json).read_text())
    ref = json.loads(Path(site_planes_json).read_text())["reference"]
    mains = [
        Plane(float(p["tilt_deg"]), float(p["azimuth_deg"]), p.get("kwp")) for p in ref["planes"]
    ]
    micro = Plane(float(ref["micro"]["tilt_deg"]), float(ref["micro"]["azimuth_deg"]))
    cfg = blind_config(site, site["latitude"], site["longitude"])
    irr = read_irradiance_csv(Path(irradiance_path))
    minute = to_quarters(window(read_flows(Path(flows_minute_path)), local_midnight(minute_from)))
    hourly_all = read_flows(Path(flows_hourly_path))
    hourly = hourly_to_quarters(
        window(hourly_all, local_midnight(hourly_from), local_midnight(hourly_to))
    )
    curtail = hourly_to_quarters(window(hourly_all, local_midnight(f"{curtail_months[0]}-01")))
    runs = {}
    log("minute window")
    runs["minute"] = _blind_run(minute, irr, cfg, mains, micro, minute_from, log)
    log("summer hourly window")
    runs["hourly"] = _blind_run(hourly, irr, cfg, mains, micro, None, log)
    m = runs["minute"]
    result = {
        "planes": m["learned"],
        "scale_ratio": [g["carried_ratio"] for g in m["reference"]],
        "hourly_error": m["hourly"]["mae_share"],
        "learning_pass_day": m["learning_day"],
        "windows": {
            "minute": {"from": minute_from, "last_day": m["last_day"]},
            "hourly": {"from": hourly_from, "to": hourly_to},
        },
        "learn_by": learn_by,
        "runs": runs,
        "checks": _blind_checks(runs, learn_by),
        "curtailment_by_month": curtailment_by_month(
            curtail, cfg.latitude, cfg.longitude, curtail_months
        ),
        "factor_kind": "month",
        "wall_s": time.monotonic() - t0,
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, allow_nan=False))
    write_section(Path(out_md), "blind-fit", render_blind_fit(result), REPORT_TITLE)
    return result


# --- rendering ------------------------------------------------------------------------------

REPORT_TITLE = "Plan 2 site models: acceptance on the owner's site"


def _checks_table(checks: list[dict], diagnostic_kinds: frozenset = frozenset()) -> list[str]:
    """Checks table. Rows of a diagnostic kind are labelled: they are not design 6.9 checks."""
    lines = ["| Check | Kind | Measured | Target | Result |", "|---|---|---|---|---|"]
    for c in checks:
        result = "pass" if c["pass"] else "miss"
        kind = "diagnostic" if c["kind"] in diagnostic_kinds else "design 6.9"
        lines.append(f"| {c['name']} | {kind} | {c['shown']} | {c['target']} | {result} |")
    return lines


def _misses(
    checks: list[dict],
    diagnostic_kinds: frozenset = frozenset(),
    causes: dict[str, str] | None = None,
) -> list[str]:
    """One line per miss. causes (per check kind) gives the measured cause from the section's
    reading; a kind without one names the model piece of _MISS_PIECE."""
    lines = []
    for c in checks:
        if causes and c["kind"] in causes:
            if not c["pass"]:
                label = "Diagnostic miss" if c["kind"] in diagnostic_kinds else "Miss"
                lines.append(
                    f"- {label}, {c['name']}: measured {c['shown']} against {c['target']}. "
                    f"Cause, from the reading below: {causes[c['kind']]}."
                )
            continue
        if not c["pass"]:
            label = "Diagnostic miss" if c["kind"] in diagnostic_kinds else "Miss"
            lines.append(
                f"- {label}, {c['name']}: measured {c['shown']} against {c['target']}. "
                f"The piece to change: {_MISS_PIECE[c['kind']]}."
            )
    return lines


def render_merged_micro(result: dict) -> str:
    runs = result["runs"]
    m1 = runs["minute_1"]
    h1 = runs["hourly_1"]
    w = result["windows"]
    out = [
        "## Merged-micro validation (design 6.9)",
        "",
        "The dataset is the owner's site with the house taken from the load register and the "
        "microinverter on the GEN port as the only PV, merged as if it fed the load port. The "
        "dataset has no battery. The fit is blind: it gets no plane hints and sees only the "
        "merged grid flows and the irradiance. The micro's metered output is the truth. The "
        "geometry and scale checks compare with the micro's reference plane from the git-ignored "
        "site file, so this section gives deltas and ratios only.",
        "",
        f"- Minute window: {w['minute']['from']} to {w['minute']['last_day']}, "
        f"{eu(m1['quarters'], 0)} quarters, {eu(m1['missing_quarters'], 0)} missing.",
        f"- Hourly window: {w['hourly']['from']} up to {w['hourly']['to']} (exclusive), hourly "
        f"means split flat into quarters, {eu(h1['quarters'], 0)} quarters, "
        f"{eu(h1['missing_quarters'], 0)} missing.",
        "- The effective kWp is the Huber-fitted scale of the metered micro on the reference "
        "plane's plant-model shape over the clear quarters (clearness at or above 0,8).",
        f"- {ESTIMATOR_NOTE}",
        f"- {IRRADIANCE_NOTE}",
        "- Hourly errors cover every daylight quarter of the window (fit and holdout days). The "
        "reference-plane MAE is the hourly error of the reference plane at the effective kWp: "
        "the floor that the irradiance data and the plant model set for any fit.",
        "",
        "### Checks",
        "",
        "Rows of kind diagnostic are not checks of the merged-micro table of design 6.9: the "
        "proportional scales feed the detection floor, the learning date belongs to the blind "
        "fit's table, and the clipping row checks the 6.4 mask.",
        "",
        *_checks_table(result["checks"], MERGED_DIAGNOSTICS),
        "",
    ]
    misses = _misses(result["checks"], MERGED_DIAGNOSTICS, MERGED_CAUSES)
    if misses:
        out += ["### Misses", "", *misses, ""]
    out += [
        "### Runs",
        "",
        "| Run | Planes | Tilt delta | Azimuth delta | Learned / effective kWp | Month factors | "
        "Hourly MAE | Hourly bias | Reference-plane MAE | PV / load energy | Holdout | "
        "Learning pass day |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, r in runs.items():
        g = r["geometry"]
        name = f"{key.split('_')[0]}, scale {_scale_label(r['scale'])}"
        ratio = g["scale_ratio"]
        hold = "pass" if r["holdout_passes"] else "fail"
        shown = _eu(ratio, 3)
        out.append(
            f"| {name} | {r['planes']} | {_deg(g['tilt_delta'])} | {_deg(g['azimuth_delta'])} | "
            f"{shown} | {factors_shown(r['month_factor'], r.get('months', []))} | "
            f"{_pct(r['hourly']['mae_share'])} | "
            f"{_pct(r['hourly']['bias_share'])} | {_pct(r['hourly_reference']['mae_share'])} | "
            f"{_pct(r['pv_to_load'])} | {hold} | "
            f"{r['learning_day'] or ('none' if key.startswith('minute') else 'not run')} |"
        )
    out += [
        "",
        "Learning trace of the minute runs (the registry refits at every local midnight from "
        "the 14th daylight day on; the holdout rule needs a holdout ratio of at most 0,7, a "
        "scale ratio between the holdout halves of 0,8 to 1,25 and an azimuth delta of at most "
        "30 degrees):",
        "",
        "| Run | Fits | Challenger passes | First challenger pass | Lowest holdout ratio "
        "| Fits with ratio at most 0,7 | Fits with stable halves |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, r in runs.items():
        tr = r.get("learning_trace") or []
        if not key.startswith("minute") or not tr:
            continue
        ratios = [t["holdout_ratio"] for t in tr if t["holdout_ratio"] is not None]
        low = min(ratios) if ratios else None
        good = sum(1 for x in ratios if x <= 0.7)
        stable = sum(
            1
            for t in tr
            if t["scale_ratio"] is not None
            and 0.8 <= t["scale_ratio"] <= 1.25
            and t["azimuth_delta"] is not None
            and t["azimuth_delta"] <= 30.0
        )
        passing = [t["day"] for t in tr if t["challenger_passes"]]
        out.append(
            f"| scale {_scale_label(r['scale'])} | {len(tr)} | {len(passing)} | "
            f"{passing[0] if passing else 'none'} | "
            f"{_eu(low, 2)} | {good} | {stable} |"
        )
    out.append("")
    floor = result["detection_floor"]
    criterion = (
        "The criterion: a scale passes when the registry, started cold on the first day of "
        f"the minute window and refitted at every local midnight (at most {LEARN_MAX_DAYS} "
        "daily fits), leaves the learning state, so its champion passes the holdout rule of "
        "6.8. The PV-to-load ratio is the metered micro energy over the metered house energy "
        "of the window."
    )
    if floor is None:
        out.append(
            "Detection floor: none of the tested scales passes the learning check, so the "
            f"floor lies above the largest tested PV-to-load energy ratio. {criterion}"
        )
    else:
        out.append(
            f"Detection floor: the smallest tested PV-to-load energy ratio that passes the "
            f"learning check is {_pct(floor['pv_to_load'])} (micro scale "
            f"{_scale_label(floor['scale'])}). The true floor can lie lower; the test ladder "
            f"stops at scale 0,5. {criterion}"
        )
    out += ["", ATTRIBUTION, ""]
    return "\n".join(out)


def _learned_name(g: dict) -> str:
    return NA_TEXT if g.get("learned_index") is None else str(int(g["learned_index"]) + 1)


def _shares(learned: list[dict]) -> str:
    total = sum(p["kwp"] for p in learned)
    if not learned or total <= 0:
        return NA_TEXT
    return " / ".join(eu(p["kwp"] / total, 2) for p in learned)


def _holdout_ratio(r: dict) -> float | None:
    if r["holdout_error"] is None or not r["null_error"]:
        return None
    return r["holdout_error"] / r["null_error"]


def _stability_shown(st: list) -> str:
    if st[0] is None or st[1] is None:
        return "not tested"
    return f"{eu(st[0], 2)}, {_deg(st[1])}"


def render_blind_fit(result: dict) -> str:
    """The blind-fit report section: European numbers, deltas and ratios only, the attribution.

    The summary needs only `planes` (the learned planes, counted), `scale_ratio` (learned over
    effective kWp per reference plane), `hourly_error` (the minute window's MAE share) and
    `learning_pass_day`. The full result of blind_fit adds the windows, the checks, the runs,
    the planes-given variant, the curtailment counts and the learning trace.
    """
    ratios = result.get("scale_ratio") or []
    day = result.get("learning_pass_day")
    out = [
        "## Blind fit on the owner's site (design 6.9)",
        "",
        "The fit is blind: it sees the grid flows, the battery flows with the `dc` flag and the "
        "SoC, and gets no plane hints. The replay learns all PV, so the GEN port reads 0 and the "
        "microinverter's production is part of what the fit must learn. The truth is the metered "
        "Deye strings plus the metered micro. The geometry and scale checks compare with the "
        "reference planes from the git-ignored site file, so this section gives deltas and "
        "ratios only.",
        "",
        f"- Learned planes (minute window): {len(result.get('planes') or [])}.",
        "- Learned / effective kWp per reference plane (minute window): "
        + (", ".join(_eu(x, 2) for x in ratios) if ratios else NA_TEXT)
        + ".",
        f"- Hourly MAE on clean hours (minute window): {_pct(result.get('hourly_error'))}.",
        f"- Learning pass day (the registry leaves the learning state): {day or 'no pass'}.",
        "",
    ]
    if result.get("runs"):
        out += _render_blind_detail(result)
    out += [ATTRIBUTION, ""]
    return "\n".join(out)


def _render_blind_detail(result: dict) -> list[str]:
    runs = result["runs"]
    w = result["windows"]
    m, h = runs["minute"], runs["hourly"]
    out = [
        "### Method",
        "",
        f"- Minute window: {w['minute']['from']} to {m['last_day']}, {eu(m['quarters'], 0)} "
        f"quarters, {eu(m['missing_quarters'], 0)} missing, {eu(m['clean_quarters'], 0)} clean "
        "daylight quarters scored.",
        f"- Summer hourly window: {w['hourly']['from']} up to {w['hourly']['to']} (exclusive), "
        "hourly means split flat into quarters and marked hourly-origin, so the fit "
        f"hour-averages the plane shapes; {eu(h['quarters'], 0)} quarters, "
        f"{eu(h['missing_quarters'], 0)} missing, {eu(h['clean_quarters'], 0)} clean daylight "
        "quarters scored.",
        "- Effective kWp: the Huber-fitted scales of the metered series on the reference planes' "
        "plant-model shapes over the clear quarters (clearness at or above 0,8): the Deye "
        "strings jointly on the main reference planes, the micro on its own plane. A learned "
        "plane carries the effective kWp of every reference plane nearest to it (smallest angle "
        "between the plane normals), so the plane nearest the micro carries the micro too.",
        "- Clean hours: clock hours whose daylight quarters are all valid and none is in the "
        "curtailment mask (SoC at or above 95 % with export under 100 W) or at the learned AC "
        "maximum. The hourly MAE and bias are shares of the mean hourly production on them.",
        "- The reference-plane floor is the hourly error of the reference planes at their own "
        "effective kWp: what the irradiance archive and the plant model allow any fit.",
        f"- {ESTIMATOR_NOTE}",
        f"- {IRRADIANCE_NOTE}",
        "",
        "### Checks",
        "",
        *_checks_table(result["checks"]),
        "",
    ]
    misses = _misses(result["checks"], causes=BLIND_CAUSES)
    if misses:
        out += ["### Misses", "", *misses, ""]
    out += [
        "### Planes found",
        "",
        "| Window | Reference plane | Learned plane | Tilt delta | Azimuth delta | "
        "Learned / effective kWp | Effective / reference kWp | Month factors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        factors = factors_shown(r["month_factor"], r["months"])
        for i, g in enumerate(r["reference"], start=1):
            out.append(
                f"| {name} | {i} | {_learned_name(g)} | {_deg(g['tilt_delta'])} | "
                f"{_deg(g['azimuth_delta'])} | {_eu(g['carried_ratio'], 3)} | "
                f"{_eu(g['effective_share'], 3)} | {factors} |"
            )
        g = r["micro"]
        out.append(
            f"| {name} | micro | {_learned_name(g)} | {_deg(g['tilt_delta'])} | "
            f"{_deg(g['azimuth_delta'])} | carried by learned plane {_learned_name(g)} | "
            f"{NA_TEXT} | {factors} |"
        )
    out += [
        "",
        "| Window | Learned planes | Total learned / effective kWp | Plane shares of the "
        "learned kWp | AC maximum learned | Holdout ratio | Stability (scale ratio, azimuth "
        "delta) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        out.append(
            f"| {name} | {r['planes']} | {_eu(r['total_ratio'], 3)} | {_shares(r['learned'])} | "
            f"{'yes' if r['ac_max_kw'] is not None else 'no'} | {_eu(_holdout_ratio(r), 2)} | "
            f"{_stability_shown(r['stability'])} |"
        )
    out += [
        "",
        "### Hourly error",
        "",
        "| Window | Fit MAE | Fit bias | Planes-given MAE | Planes-given bias | "
        "Reference-plane floor MAE | Floor bias | Clean hours |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        e, g, f = r["hourly"], r["given"]["hourly"], r["hourly_reference"]
        out.append(
            f"| {name} | {_pct(e['mae_share'])} | {_pct(e['bias_share'])} | "
            f"{_pct(g['mae_share'])} | {_pct(g['bias_share'])} | {_pct(f['mae_share'])} | "
            f"{_pct(f['bias_share'])} | {eu(e['hours'], 0)} |"
        )
    out += [
        "",
        "Planes-given variant (diagnostic, not a 6.9 check): the main reference planes given "
        "with their reference kWp, scales fitted on the same blind flows. The plane nearest the "
        "micro carries the micro's effective kWp too.",
        "",
        "| Window | Reference plane | Fitted / effective kWp | Month factors |",
        "|---|---|---|---|",
    ]
    for name, r in runs.items():
        factors = factors_shown(r["given"]["month_factor"], r["months"])
        for i, g in enumerate(r["given"]["planes"], start=1):
            out.append(f"| {name} | {i} | {_eu(g['carried_ratio'], 3)} | {factors} |")
    cur = result["curtailment_by_month"]
    out += [
        "",
        "### Curtailment mask, April to June (hourly statistics)",
        "",
        "| Month | Masked daylight quarters |",
        "|---|---|",
        *(f"| {ym} | {eu(n, 0)} |" for ym, n in cur.items()),
        "",
        f"In the fit windows the mask removes {eu(m['curtailed_quarters'], 0)} daylight "
        f"quarters (minute) and {eu(h['curtailed_quarters'], 0)} (summer hourly).",
        "",
        "### Learning check",
        "",
        "The registry starts cold on the first day of the minute window and refits at every "
        "local midnight from the 14th daylight day on (90-day window, at most "
        f"{LEARN_MAX_DAYS} fits). The learning state ends on the first day the champion passes "
        "the holdout rule of 6.8: a holdout error at most 0,7 of the null error and, when each "
        "holdout half has at least 5 lit days, a scale ratio between the halves of 0,8 to 1,25 "
        "and an azimuth delta of at most 30 degrees.",
        "",
        "The registry keeps refitting after the pass day to the end of the window, so the table "
        "also shows whether the champion stays out of learning.",
        "",
        "| Window | Fits | Pass day | Challenger passes | Champion passes after the pass day | "
        "Lowest holdout ratio | Fits with ratio at most 0,7 | Fits with the stability test run |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in runs.items():
        tr = r.get("learning_trace") or []
        if not tr:
            continue
        ratios = [t["holdout_ratio"] for t in tr if t["holdout_ratio"] is not None]
        low = min(ratios) if ratios else None
        after = [t for t in tr if r["learning_day"] and t["day"] >= r["learning_day"]]
        kept = f"{sum(1 for t in after if t['champion_passes'])} of {len(after)}"
        out.append(
            f"| {name} | {len(tr)} | {r['learning_day'] or 'no pass'} | "
            f"{sum(1 for t in tr if t['challenger_passes'])} | {kept} | {_eu(low, 2)} | "
            f"{sum(1 for x in ratios if x <= HOLDOUT_RATIO)} | "
            f"{sum(1 for t in tr if t['scale_ratio'] is not None)} |"
        )
    out.append("")
    return out


# --- replay-windows: the replay on the owner's two windows (Task 25) -------------------------


@dataclass(frozen=True)
class ReplayWindow:
    """One replay window: inclusive local days, the band-cache tiers and the flows resolution."""

    name: str
    label: str
    first: str
    last: str | None  # None: the last local day with settled prices in the master
    tiers: tuple[str, ...]
    resolution: str  # "minute" or "hourly"
    last_source: str = "fixed"  # "fixed", "master" or "override"

    @property
    def noun(self) -> str:
        """The label inside a sentence: "live window", "A/B window"."""
        return self.label if self.label[1:2] == "/" else self.label.lower()


REPLAY_WINDOWS = (
    ReplayWindow(
        "live", "Live window", "2026-09-02", None, ("A", "B", "C", "D"), "minute", "master"
    ),
    ReplayWindow("ab", "A/B window", "2026-02-01", "2026-08-31", ("A", "B"), "hourly"),
)
SUMMER_GREED_MONTH = 7
ANCHOR_SEARCH_DAYS = 7
GREED_EXAMPLES = 5
CACHE_DIR = "cache"
STATE_FILE = "state.pkl"
REPLAY_SECTION = "replay-windows"
REPLAY_TITLE = "Plan 2 replay on the owner's site"
SITE_MODELS_REPORT = "2026-10-31-plan2-site-models.md"
OPEN_ITEMS = (
    "Two open items from the site-model report carry into these numbers. The blind PV fit "
    "misses the roof geometry (one learned plane is 49 degrees off in azimuth, the other 19 "
    "degrees off in tilt) and makes up for it in the scale, so the hourly PV error stays at "
    "about the weather data's own floor but the planes are not the roofs. The PV scale is "
    "anchored on the night load: a house whose daytime load sits above its night mean gets "
    "a PV scale that is low by that excess, and an evening peak inside the night rows pulls "
    f"it high. {DRIFT_NOTE}"
)


def last_settled_day(master: Table) -> str:
    """The last local day whose quarters are all in the master with a finite settled buy
    price (96 quarters, 92 or 100 on the DST days)."""
    ts = np.asarray(master["ts"], dtype=np.int64)
    settled = np.isfinite(np.asarray(master["buy_eur_mwh"], dtype=float))
    days = np.array([local_day(int(t)) for t in ts], dtype=object)
    for day in reversed(list(dict.fromkeys(days))):
        rows = days == day
        expected = (local_midnight(_plus_days(day, 1)) - local_midnight(day)) // 900
        if int(rows.sum()) == expected and bool(settled[rows].all()):
            return str(day)
    raise ValueError("the master has no fully settled local day")


def resolve_window(spec: ReplayWindow, master: Table, live_to: str | None = None) -> ReplayWindow:
    """The window with its last day fixed. A window with fixed dates comes back unchanged
    (``live_to`` does not apply to it). A window without a last day ends on ``live_to`` when
    given (``override``), else on ``last_settled_day(master)`` (``master``)."""
    if spec.last is not None:
        return spec
    if live_to is not None:
        return replace(spec, last=live_to, last_source="override")
    return replace(spec, last=last_settled_day(master), last_source="master")


def cache_dir_for(out_dir: Path, cache_dir: Path | None) -> Path:
    """The band cache directory: ``cache_dir`` when given (shared between runs), else
    ``out_dir/cache``."""
    return Path(cache_dir) if cache_dir is not None else Path(out_dir) / CACHE_DIR


def window_dir(out_dir: Path, name: str) -> Path:
    """The output directory of one window: ``out_dir/<name>``."""
    return Path(out_dir) / name


def anchored_first_day(
    q: Table, minute: Table | None, first: str, last: str, capacity_kwh: float
) -> str:
    """The first local day from ``first`` with a recorded SoC at its start anchor, at most
    ``ANCHOR_SEARCH_DAYS`` days on (``replay.phases.anchored_first_day``, the rule the replay
    CLI applies too). The end anchor raises when the recorded SoC is missing at the end and
    in the final 60 minutes before it (``replay.phases.end_soc_anchor``)."""
    day = P.anchored_first_day(
        q,
        minute,
        dt.date.fromisoformat(first),
        dt.date.fromisoformat(last),
        capacity_kwh,
        max_days=ANCHOR_SEARCH_DAYS,
    )
    return day.isoformat()


def plant_planes(blind_fit_json: Path) -> list[Plane]:
    """The planes of the blind fit's summer hourly window (Task 24), with their learned kWp."""
    run = json.loads(Path(blind_fit_json).read_text())["runs"]["hourly"]
    return [
        Plane(float(p["tilt_deg"]), float(p["azimuth_deg"]), float(p["kwp"]))
        for p in run["learned"]
    ]


def ab_quarters(
    hourly: Table, irr: Table | None, planes: list[Plane], lat: float, lon: float
) -> Table:
    """Hourly flows to quarters: grid, battery and load split flat, the PV and GEN columns
    by the plant shape of ``planes`` (flat without planes). Rows carry ``hourly_origin``.

    The engine never reads the PV and GEN columns (it learns all PV from the grid and the
    battery), so the shape only places the validation columns inside the hour.
    """
    hts = np.asarray(hourly["ts"], dtype=np.int64)
    if not planes or irr is None:
        return hourly_to_quarters(hourly)
    qts = (hts[:, None] + np.arange(HOUR_S // 900, dtype=np.int64) * 900).ravel()
    aligned = align(irr, qts, list(IRRADIANCE_COLS))
    shape = sum(float(p.kwp or 0.0) * plane_shape(aligned, p, lat, lon) for p in planes)
    return hourly_to_quarters(hourly, np.asarray(shape, dtype=float), qts)


def window_cache(
    master: Table, start: int, end: int, cache_dir: Path, tiers: tuple[str, ...], log
) -> tuple[BandCache, dict]:
    """The window's walk-forward band cache through ``replay.__main__.band_cache`` (built or
    reused on a key match) and its record: key, file, folds, build wall time, reused.

    The build wall time is kept in a JSON beside the cache, so a reuse still reports it.
    """
    key, path = M.cache_path(master, start, end, cache_dir, tiers)
    side = path.with_suffix(".json")
    reused = path.exists()
    t0 = time.monotonic()

    def tick(done: int, total: int) -> None:
        log(f"  cache {eu(done, 0)} of {eu(total, 0)} ({eu((time.monotonic() - t0) / 60, 1)} min)")

    cache = M.band_cache(master, start, end, cache_dir, tiers=tiers, progress=tick)
    wall = time.monotonic() - t0
    reused = reused and cache.meta.get("key") == key and side.exists()
    if reused:
        build_s = float(json.loads(side.read_text())["build_s"])
    else:
        build_s = wall
        side.write_text(json.dumps({"key": key, "build_s": build_s}) + "\n")
    info = {
        "key": key[:16],
        "file": path.name,
        "folds": len(cache.meta["folds"]),
        "tiers": list(tiers),
        "build_s": build_s,
        "reused": bool(reused),
    }
    return cache, info


_POOL: dict = {}


def _ladder_task(i: int):
    """One pool task: index 0 is the default replay (the full result), the rest one rung each
    through ``report.run_ladder``."""
    inputs, tasks = _POOL["inputs"], _POOL["tasks"]
    t0 = time.monotonic()
    name, knobs = tasks[i]
    out = replay(inputs) if name is None else R.run_ladder(inputs, rungs=[(name, knobs)])[0]
    return i, out, time.monotonic() - t0


def run_window_ladder(
    inputs: ReplayInputs, ladder: bool, workers: int, log
) -> tuple[ReplayResult, list[dict] | None, dict]:
    """The default replay and, with ``ladder``, the 20 rungs of ``report.ladder_rungs`` on the
    same inputs. ``workers`` > 1 runs the tasks in forked processes (each task is a
    deterministic replay, so the result does not depend on the order). Logs every task."""
    rungs = R.ladder_rungs(inputs.cfg.knobs) if ladder else []
    tasks = [(None, None), *rungs]
    _POOL.update(inputs=inputs, tasks=tasks)
    done: dict[int, tuple] = {}
    t0 = time.monotonic()

    def record(i: int, out, secs: float) -> None:
        done[i] = (out, secs)
        name = (
            "default replay" if i == 0 else f"rung {eu(i, 0)} of {eu(len(rungs), 0)} {tasks[i][0]}"
        )
        diff = out["diff_eur"] if i else R.headline(out)["diff_eur"]
        log(
            f"  {name}: {eu(secs / 60, 1)} min, difference {eu(diff)} EUR ({eu(len(done), 0)} "
            f"of {eu(len(tasks), 0)} done, {eu((time.monotonic() - t0) / 60, 1)} min)"
        )

    try:
        if workers <= 1:
            for i in range(len(tasks)):
                record(*_ladder_task(i))
        else:
            ctx = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
                for fut in as_completed([pool.submit(_ladder_task, i) for i in range(len(tasks))]):
                    record(*fut.result())
    finally:
        _POOL.clear()
    result = done[0][0]
    rows = [done[i][0] for i in range(1, len(tasks))] if ladder else None
    timings = {"replay_s": done[0][1], "rung_s": [done[i][1] for i in range(1, len(tasks))]}
    return result, rows, timings


def window_state(
    spec: ReplayWindow,
    first: str,
    last: str,
    result: ReplayResult,
    ladder: list[dict] | None,
    bias: dict,
    cache_info: dict,
    timings: dict,
) -> dict:
    """Everything the report needs from one window run (pickled under the window directory)."""
    return {
        "spec": spec,
        "first": first,
        "last": last,
        "result": result,
        "ladder": ladder,
        "bias": bias,
        "cache": cache_info,
        "timings": timings,
    }


def run_window(
    spec: ReplayWindow,
    cfg: SiteConfig,
    flows_minute: Path,
    flows_hourly: Path,
    irr: Table,
    master: Table,
    out_dir: Path,
    ladder: bool = True,
    workers: int = 1,
    planes: list[Plane] | None = None,
    log=print,
    cache_dir: Path | None = None,
    live_to: str | None = None,
) -> dict:
    """One window: flows, the start anchor, the band cache, the default replay, the ladder and
    the hourly bias. Saves the state under the window directory and returns it.

    The live window reads the minute flows (the replay's minute loop runs on them); the A/B
    window reads the hourly flows, split by ``ab_quarters`` with the plant shape of
    ``planes``. The hourly bias is measured on minute flows only.
    """
    spec = resolve_window(spec, master, live_to)
    if spec.resolution == "minute":
        minute = read_flows(Path(flows_minute))
        q, _ = M.quarter_flows(minute)
    else:
        minute = None
        q = ab_quarters(
            read_flows(Path(flows_hourly)), irr, planes or [], cfg.latitude, cfg.longitude
        )
    first = anchored_first_day(q, minute, spec.first, spec.last, cfg.pack.capacity_kwh)
    if first != spec.first:
        log(
            f"{spec.label}: no recorded SoC at the start of {spec.first}; the replay starts {first}"
        )
    start, end = M.window_bounds(first, spec.last)
    M.check_start(master, start)
    log(f"{spec.label}: band cache, tiers {', '.join(spec.tiers)}")
    cache, info = window_cache(
        master, start, end, cache_dir_for(out_dir, cache_dir), spec.tiers, log
    )
    log(
        f"  cache {info['file']}: {eu(info['folds'], 0)} folds, build "
        f"{eu(info['build_s'] / 60, 1)} min{' (reused)' if info['reused'] else ''}"
    )
    M.check_start(master, start, cache)
    inputs = ReplayInputs(cfg, q, master, cache, irr, start, end, flows_minute=minute)
    log(f"{spec.label}: replay {first} to {spec.last}{', ladder' if ladder else ''}")
    result, rows, timings = run_window_ladder(inputs, ladder, workers, log)
    bias = R.measure_hourly_bias(inputs, result) if minute is not None else R.HOURLY_INPUT
    state = window_state(spec, first, spec.last, result, rows, bias, info, timings)
    state["last_source"] = spec.last_source
    save_state(out_dir, state)
    return state


class _StateUnpickler(pickle.Unpickler):
    """Classes pickled under ``__main__`` (a ``python -m`` run of this module) load from this
    module."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in globals():
            return globals()[name]
        return super().find_class(module, name)


def save_state(out_dir: Path, state: dict) -> Path:
    """Pickle a window state under its window directory; the window is stored by name."""
    d = window_dir(out_dir, state["spec"].name)
    d.mkdir(parents=True, exist_ok=True)
    path = d / STATE_FILE
    with path.open("wb") as fh:
        pickle.dump(dict(state, spec=state["spec"].name), fh)
    return path


def load_state(out_dir: Path, name: str) -> dict:
    """The saved state of one window, its ``spec`` resolved from ``REPLAY_WINDOWS``."""
    with (window_dir(out_dir, name) / STATE_FILE).open("rb") as fh:
        state = _StateUnpickler(fh).load()
    spec = state["spec"]
    if isinstance(spec, str):
        state["spec"] = {w.name: w for w in REPLAY_WINDOWS}[spec]
    return state


def combine_ladders(*windows: list[dict]) -> list[dict]:
    """One row per rung with the daily differences of every window joined (the windows do
    not share a day), for ``select_default`` with live and A/B days."""
    out = []
    for parts in zip(*windows, strict=True):
        names = {p["name"] for p in parts}
        if len(names) != 1:
            raise ValueError(f"the ladders of the windows differ: {sorted(names)}")
        out.append(
            {
                "name": parts[0]["name"],
                "knobs": parts[0]["knobs"],
                "diff_eur": float(sum(p["diff_eur"] for p in parts)),
                "daily": np.concatenate([np.asarray(p["daily"], dtype=float) for p in parts]),
                "days": np.concatenate([np.asarray(p["days"], dtype=object) for p in parts]),
            }
        )
    return out


def _greed_row(g: Table, i: int, gap: float | None = None) -> dict:
    row = {c: g[c][i].item() if hasattr(g[c][i], "item") else g[c][i] for c in g}
    row["day"] = str(g["day"][i])
    if gap is not None:
        row["gap"] = gap
    return row


def greed_scenarios(g: Table, month: int = SUMMER_GREED_MONTH) -> dict:
    """The greed check per scenario: ``low`` (low SoC, every day), ``high`` (full pack and
    surplus coming, every day) and ``summer`` (the high-SoC rows of local month ``month``).

    Each: days it applies, days it holds. ``low`` and ``high`` list their failing days with
    ``gap`` in EUR/MWh: the night median take minus the charge threshold (``low``), the sale
    threshold minus the night median feed (``high``). ``summer`` lists up to
    ``GREED_EXAMPLES`` example days where it applies.
    """
    check = R.greed_check(g)
    days = np.asarray(g["day"], dtype=object)
    in_month = np.array([int(str(d)[5:7]) == month for d in days], dtype=bool)
    high_ok = np.asarray(g["high_soc_ok"], dtype=bool)
    summer = np.flatnonzero(in_month & np.asarray(g["high_soc_applies"], dtype=bool))

    def col(name: str) -> np.ndarray:
        return np.asarray(g[name], dtype=float)

    gaps = {
        "low": col("night_median_take") - col("charge_threshold"),
        "high": col("sale_threshold") - col("night_median_feed"),
    }
    out = {}
    for side in ("low", "high"):
        fail = np.asarray(g[f"{side}_soc_applies"], bool) & ~np.asarray(g[f"{side}_soc_ok"], bool)
        out[side] = {
            **check[side],
            "fails": [_greed_row(g, int(i), float(gaps[side][i])) for i in np.flatnonzero(fail)],
        }
    out["summer"] = {
        "month": month,
        "days": int(in_month.sum()),
        "applies": int(len(summer)),
        "ok": int(high_ok[summer].sum()),
        "examples": [_greed_row(g, int(i)) for i in summer[:GREED_EXAMPLES]],
        "fail_days": [str(days[i]) for i in summer if not high_ok[i]],
    }
    return out


def _pooled(paths: list[Path], names: list[str]) -> dict:
    from tools.pool_results import pool

    pooled = pool(paths)
    for house, name in zip(pooled["per_house"], names, strict=True):
        house["name"] = name
    return pooled


def _same_window(name: str, st: dict, src: dict) -> None:
    """Raise when a ladder state covers other days or another band cache than the state it
    is reported next to."""
    for key in ("first", "last"):
        if st.get(key) != src.get(key):
            raise ValueError(
                f"{name}: the ladder states cover another window ({key} {src.get(key)} against "
                f"{st.get(key)}); re-run the ladder on this window"
            )
    if st.get("cache", {}).get("key") != src.get("cache", {}).get("key"):
        raise ValueError(f"{name}: the ladder states used another band cache key")


def with_ladders(states: dict[str, dict], ladder_states: dict[str, dict] | None) -> dict:
    """The window states with the ladder of ``ladder_states`` when given: its rows
    (``ladder``), its timings (``ladder_timings``), the knobs its default replay ran with
    (``baseline_knobs``) and its days (``baseline_days``). Without ``ladder_states`` the
    states carry their own. Raises ``ValueError`` when a ladder state's first or last day or
    its band cache key differs from the state's."""
    out = {}
    for name, st in states.items():
        src = st if ladder_states is None else ladder_states[name]
        if ladder_states is not None:
            _same_window(name, st, src)
        out[name] = dict(
            st,
            ladder=src["ladder"],
            ladder_timings=src["timings"],
            baseline_knobs=src["result"].knobs,
            baseline_days={str(d) for d in src["result"].pairs["day"]},
        )
    return out


def assemble_windows(
    states: dict[str, dict],
    cfg: SiteConfig,
    out_dir: Path,
    report_md: Path,
    log=print,
    ladder_states: dict[str, dict] | None = None,
    settled_last: str | None = None,
) -> dict:
    """The selection on the joined ladders, the files per window, the pooled summary and the
    report section. Returns the summary (also written as ``out_dir/summary.json``).

    The selection runs only with both windows' ladders: ``live_days`` are the live window's
    days and ``ab_days`` the A/B window's days, against the rung of the knobs the default
    replay ran with (``report.default_rung_name``), so saved states keep their baseline when
    the defaults change. With ``ladder_states`` (``with_ladders``) the ladder, the selection
    and its baseline come from those states, and everything else from ``states``.
    """
    states = with_ladders(states, ladder_states)
    live, ab = states.get("live"), states.get("ab")
    selection = None
    if live and ab and live["ladder"] and ab["ladder"]:
        joined = combine_ladders(live["ladder"], ab["ladder"])
        selection = R.select_default(
            joined,
            R.default_rung_name(live["baseline_knobs"]),
            live["baseline_days"],
            ab["baseline_days"],
        )
    paths, names = [], []
    for name, st in states.items():
        meta = {"site": "the owner's site", "window": f"{st['first']} to {st['last']}"}
        meta["flows"] = st["spec"].resolution
        R.write_outputs(
            st["result"],
            window_dir(out_dir, name),
            meta,
            st["ladder"],
            selection,
            st["bias"],
            cfg=cfg,
        )
        paths.append(window_dir(out_dir, name) / "results.json")
        names.append(st["spec"].label)
    pooled = _pooled(paths, names)
    greed = {name: greed_scenarios(st["result"].greed) for name, st in states.items()}
    write_section(
        Path(report_md),
        REPLAY_SECTION,
        render_replay_windows(states, selection, greed, pooled, settled_last),
        REPLAY_TITLE,
    )
    summary = {
        "windows": {
            name: {
                "first": st["first"],
                "last": st["last"],
                "headline": R.headline(st["result"]),
                "cache": st["cache"],
                "timings": st["timings"],
                "bias": st["bias"],
            }
            for name, st in states.items()
        },
        "selection": None
        if selection is None
        else {
            "chosen": selection["chosen"],
            "reason": selection["reason"],
            "table": selection["table"],
        },
        "greed": greed,
        "pooled": {k: pooled[k] for k in ("houses", "days", "mean_eur_day", "interval")},
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(_plain(summary), indent=1, sort_keys=True)
    )
    return summary


def _plain(x):
    """JSON-safe: non-finite floats to None, tuples to lists."""
    if isinstance(x, float | np.floating):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    if isinstance(x, np.integer | np.bool_):
        return x.item()
    return x


# --- replay-windows rendering ---------------------------------------------------------------


def _demote(lines: list[str]) -> list[str]:
    return [("#" + ln) if ln.startswith("#") else ln for ln in lines]


def _window_detail(st: dict) -> list[str]:
    """The per-window replay report (``report.render_report``) one heading level down, without
    its title, its data terms (the report ends with them) and the absolute plane geometry
    (the site-model report states the geometry as deltas)."""
    meta = {"site": "the owner's site", "window": f"{st['first']} to {st['last']}"}
    text = R.render_report(st["result"], meta, st["ladder"], None, None)
    text = text.split("\n## Data terms", 1)[0]
    out, planes = [], 0
    for ln in text.splitlines():
        if ln.startswith("# Replay report"):
            continue
        if ln.startswith("- Plane "):
            planes += 1
            continue
        if ln.startswith("- Holdout error:") and planes:
            out.append(
                f"- Planes learned at the end: {eu(planes, 0)} "
                f"(geometry as deltas in {SITE_MODELS_REPORT})"
            )
        out.append(ln)
    while out and not out[0].strip():
        out = out[1:]
    return [f"## {st['spec'].label} in detail", "", *_demote(out), ""]


def _live_end_lines(states: dict[str, dict], settled_last: str | None) -> list[str]:
    """Where the live window's last day comes from, against the master's last settled day."""
    st = states.get("live")
    if st is None or settled_last is None:
        return []
    if st["last"] == settled_last:
        text = (
            f"The live window ends on {st['last']}, the last local day with settled prices in "
            "the master."
        )
    else:
        how = "was set by hand" if st.get("last_source") == "override" else "was fixed at the run"
        text = (
            f"The live window ends on {st['last']} (the end {how}); the last local day with "
            f"settled prices in the master is {settled_last}."
        )
    return [text, ""]


def _method_section(states: dict[str, dict], settled_last: str | None = None) -> list[str]:
    """The windows table and the method. No wall-clock time: the run times differ between runs
    and machines, so they go in the hand-written reading (M8 of the final review)."""
    rows = []
    for st in states.values():
        spec, c = st["spec"], st["cache"]
        rows.append(
            [
                spec.label,
                f"{st['first']} to {st['last']}",
                "minute" if spec.resolution == "minute" else "hourly, disaggregated",
                ", ".join(spec.tiers),
                eu(c["folds"], 0),
            ]
        )
    lines = [
        "## Windows and method",
        "",
        *_table(["window", "days", "flows", "tiers", "band folds"], rows),
        "",
        "The replay outputs under the window directories are deterministic: the same inputs give "
        "byte-identical files. The run times are informational and stay in the reading.",
        "",
        "Every quarter is priced with bands from the walk-forward cache: one price model per "
        "weekly fold, trained on all history before the fold (up to 365 days), and a quarter "
        "takes its band only from a fold whose training ended at least 4 days plus the longest "
        "horizon (6 h) before it. The site models walk forward on the recorded flows before "
        "each quarter. So the prices and the bands are walk-forward and embargoed: no band in "
        "this report comes from a model that saw the quarter it pays. "
        f"{PV_EXPECTATION_NOTE}",
        "",
        *_live_end_lines(states, settled_last),
    ]
    for st in states.values():
        spec = st["spec"]
        if st["first"] != spec.first:
            lines += [
                f"The {spec.noun} starts on {st['first']}, not {spec.first}: the "
                f"recorded SoC at {spec.first} 00:00 is missing, and the replay needs it as "
                "the start anchor of both paths.",
                "",
            ]
    if "ab" in states:
        lines += [
            "The A/B window has hourly flows only. They split flat into quarters for the grid, "
            "the battery and the load; the PV columns follow the plant shape of the blind fit "
            "of the site-model report, but the replay never reads them (it learns all PV from "
            "the grid and the battery). The A/B window has the tiers A and B only: the live "
            "price feed of tiers C and D starts in September.",
            "",
        ]
    return lines


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _knobs_lines(states: dict[str, dict]) -> list[str]:
    """Which knobs the headline, the greed check and the detail ran with, and which the
    ladder used as its default rung."""
    st = next(iter(states.values()))
    ran = R.default_rung_name(st["result"].knobs)
    base = R.default_rung_name(st["baseline_knobs"])
    text = (
        f"The headline, the greed check and the window detail come from replays on the knobs {ran}."
    )
    if base != ran:
        text += (
            f" The ladder and the selection come from the ladder run, whose default rung was "
            f"{base}."
        )
    return [text, ""]


def _headline_section(states: dict[str, dict]) -> list[str]:
    rows = []
    for st in states.values():
        h = R.headline(st["result"])
        rows.append(
            [
                st["spec"].label,
                eu(h["days"], 0),
                eu(h["diff_eur"]),
                eu(h["mean_eur_day"]),
                f"{eu(h['lo_eur_day'])} to {eu(h['hi_eur_day'])}",
                eu(h["terminal_eur"]),
                eu(h["missing_quarters"], 0),
                eu(st["result"].learning_days, 0),
            ]
        )
    lines = [
        "## Headline",
        "",
        *_knobs_lines(states),
        *_table(
            [
                "window",
                "days",
                "difference EUR",
                "mean EUR/day",
                "90 % interval EUR/day",
                "terminal valuation EUR",
                "missing quarters",
                "PV learning days",
            ],
            rows,
        ),
        "",
        "The difference is the actual path's cost minus the virtual path's cost, summed over "
        "the days: a positive number means the shadow trader's battery would have cost the "
        "house less than what the house did. The interval is the 90 % bootstrap interval of the "
        f"daily mean. {TERMINAL_NOTE}",
        "",
    ]
    live = states.get("live")
    if "ab" in states:
        if live is not None and not live["bias"].get("hourly_input"):
            b = live["bias"]
            text = (
                "Next to the A/B headline: booked on hourly means instead of quarters, the live "
                f"window's recorded flows cost {eu(b['bias_eur'])} EUR more (hourly minus "
                f"quarter), {eu(b['bias_share'] * 100, 1)} % of the live headline. The A/B "
                "window books disaggregated hourly flows, so its difference carries a bias of "
                "this kind."
            )
            if b.get("indicative_only"):
                text += " The A/B headline is indicative only."
            lines += [text, ""]
        else:
            lines += [R.HOURLY_INPUT_SENTENCE, ""]
    return lines


def _ladder_section(states: dict[str, dict], selection: dict | None) -> list[str]:
    lines = ["## Ladder and the chosen defaults", ""]
    if selection is None:
        return lines + ["The ladder ran on fewer than two windows; no selection.", ""]
    by_window = {n: {r["name"]: r for r in (st["ladder"] or [])} for n, st in states.items()}
    rows = []
    for t in selection["table"]:
        live, ab = by_window.get("live", {}).get(t["name"]), by_window.get("ab", {}).get(t["name"])
        rows.append(
            [
                t["name"],
                eu(live["diff_eur"]) if live else NA_TEXT,
                eu(ab["diff_eur"]) if ab else NA_TEXT,
                eu(t["mean_gain"], 3),
                f"{eu(t['lo'], 3)} to {eu(t['hi'], 3)}",
                eu(t["ab_mean"], 3),
                "yes" if t["candidate"] else "no",
            ]
        )
    lines += _table(
        [
            "rung",
            "live difference EUR",
            "A/B difference EUR",
            "live gain EUR/day",
            "90 % interval",
            "A/B gain EUR/day",
            "candidate",
        ],
        rows,
    )
    lines += [
        "",
        "The gain is the rung's daily difference minus the default rung's on the same day. A "
        "rung is a candidate when its mean live gain is above zero, the low end of its 90 % "
        "bootstrap interval is above zero, and its mean gain on the A/B days is at least zero. "
        "The candidate with the largest live gain wins; without one the default stays.",
        "",
        f"Chosen: {selection['chosen']}.",
        "",
        selection["reason"],
        "",
    ]
    lines += _selection_caveats(states, selection, by_window)
    return lines


def _who_leads(mean: float, lo: float, hi: float) -> str:
    """Who is ahead on a daily difference (actual minus virtual) and its 90 % interval."""
    if hi < 0:
        return "the house's actual dispatch beat the shadow trader"
    if lo > 0:
        return "the shadow trader beat the house's actual dispatch"
    return "the interval crosses zero, so neither side is ahead beyond the noise"


def _rung_on_its_own(row: dict) -> tuple[float, float, float]:
    return bootstrap_interval(np.asarray(row["daily"], dtype=float), seed=0)


def _selection_caveats(states: dict[str, dict], selection: dict, by_window: dict) -> list[str]:
    """The limits of the selection: in-sample knob choice, the A/B margin, the chosen rung
    and the default rung of the ladder run on their own against the house."""
    chosen = next(t for t in selection["table"] if t["name"] == selection["chosen"])
    baseline = R.default_rung_name(next(iter(states.values()))["baseline_knobs"])
    lines = [
        "The knobs were chosen on the same live days that score them, so the live gain "
        "carries selection optimism. The A/B window is the only out-of-sample check, and the "
        f"chosen rung's margin there is {eu(chosen['ab_mean'])} EUR per day.",
        "",
    ]
    for label, name in (("the chosen rung", selection["chosen"]), (None, baseline)):
        parts = []
        for wname, st in states.items():
            row = by_window.get(wname, {}).get(name)
            if row is None:
                continue
            mean, lo, hi = _rung_on_its_own(row)
            parts.append(
                f"on the {st['spec'].noun} {eu(mean)} EUR per day (90 % interval {eu(lo)} to "
                f"{eu(hi)} EUR): {_who_leads(mean, lo, hi)}"
            )
        if not parts:
            continue
        head = (
            f"On its own, {label}'s daily difference (actual minus virtual) is "
            if label
            else f"With the default rung of the ladder run ({baseline}), the daily difference is "
        )
        lines += [head + "; ".join(parts) + ".", ""]
    return lines


def _greed_section(states: dict[str, dict], greed: dict) -> list[str]:
    lines = ["## Greed check", ""]
    rows = []
    for name, st in states.items():
        g = greed[name]
        for key, label in (("low", "low SoC at 18:00"), ("high", "full pack, surplus coming")):
            c = g[key]
            rows.append(
                [
                    st["spec"].label,
                    label,
                    eu(c["applies"], 0),
                    eu(c["ok"], 0),
                    f"first {c['first_fail']}" if c["first_fail"] else "none",
                ]
            )
        if name == "ab":
            s = g["summer"]
            rows.append(
                [
                    st["spec"].label,
                    "July evenings, full pack, surplus coming",
                    eu(s["applies"], 0),
                    eu(s["ok"], 0),
                    ", ".join(s["fail_days"]) or "none",
                ]
            )
    lines += _table(
        ["window", "scenario", "days it applies", "days it holds", "failing days"], rows
    )
    lines.append("")
    lines += [
        "Low SoC: at 18:00 with the pack within 20 % of capacity above the floor, the charge "
        "threshold must be at or above the coming night's median take price. Full pack: with "
        "the pack at 80 % or more and at least half a pack of surplus coming tomorrow, the sale "
        "threshold must be at or below the night's median feed price.",
        "",
    ]
    ab = greed.get("ab")
    if ab is not None:
        s = ab["summer"]
        if s["applies"]:
            verdict = (
                "holds on every one"
                if s["ok"] == s["applies"]
                else (f"holds on {eu(s['ok'], 0)} of them")
            )
            lines += [
                f"Summer scenario: the full-pack check applies on {eu(s['applies'], 0)} of the "
                f"{eu(s['days'], 0)} July evenings of the A/B window and {verdict}. Example days:",
                "",
            ]
            lines += _table(
                [
                    "day",
                    "SoC kWh",
                    "surplus tomorrow kWh",
                    "sale threshold EUR/MWh",
                    "night feed EUR/MWh",
                    "holds",
                ],
                [
                    [
                        r["day"],
                        eu(r["soc_kwh"], 1),
                        eu(r["surplus_tomorrow_kwh"], 1),
                        eu(r["sale_threshold"], 1),
                        eu(r["night_median_feed"], 1),
                        "yes" if r["high_soc_ok"] else "no",
                    ]
                    for r in s["examples"]
                ],
            )
            lines.append("")
        else:
            lines += [
                f"Summer scenario: no July evening of the A/B window had a full pack and half a "
                f"pack of surplus coming ({eu(s['days'], 0)} July greed rows), so the scenario "
                "did not occur on this window.",
                "",
            ]
    for name, st in states.items():
        for side in ("low", "high"):
            lines += _greed_fails(st["spec"], side, greed[name][side], st["result"].knobs.impl)
    return lines


_FAIL_TEXT = {
    "low": (
        "Low-SoC miss",
        "charge threshold",
        "below the night median take",
        ("charge_threshold", "night_median_take"),
        ("charge threshold EUR/MWh", "night take EUR/MWh"),
    ),
    "high": (
        "Full-pack miss",
        "sale threshold",
        "above the night median feed",
        ("sale_threshold", "night_median_feed"),
        ("sale threshold EUR/MWh", "night feed EUR/MWh"),
    ),
}
_FAIL_READING = {
    ("low", "ranking"): (
        "The ranking sets the charge threshold at the price of the last slot the need takes; "
        "when the need is a few slots out of the roughly 190 slots of the window, that price "
        "can sit below the night median by construction, as Task 18 predicted."
    ),
    ("high", "ranking"): (
        "The ranking's sale threshold is the minimum over the ranked sale slots, which exceeds "
        "the night median feed whenever only the best slots are sold."
    ),
    ("low", "dp"): (
        "Under dp the charge threshold of a slot is the marginal value of stored energy (the "
        "slope of the value function) less the charge losses, the fees and half the hurdle; "
        "the check compares its night median with the night median take."
    ),
    ("high", "dp"): (
        "Under dp the sale threshold of a slot is the marginal value of stored energy (the "
        "slope of the value function) plus the wear, the discharge losses, the fees and half "
        "the hurdle; the check compares its night median with the night median feed."
    ),
}
_FAIL_OPEN = (
    "The definitions of both greed checks are the open question; the rule is not changed here."
)


def _greed_fails(spec: ReplayWindow, side: str, c: dict, impl: str) -> list[str]:
    """The failing days of one greed check: the count, the gap range, the reading for the
    ``impl`` that ran and example days."""
    fails = c["fails"]
    if not fails:
        return []
    title, what, where, cols, heads = _FAIL_TEXT[side]
    reading = _FAIL_READING.get((side, impl), "")
    gaps = [f["gap"] for f in fails if np.isfinite(f["gap"])]
    span = f" {eu(min(gaps), 1)} to {eu(max(gaps), 1)} EUR/MWh" if gaps else ""
    lines = [
        f"{title} on the {spec.noun}: the check fails on {eu(len(fails), 0)} of "
        f"{eu(c['applies'], 0)} days, with the {what}{span} {where}. {reading} {_FAIL_OPEN} "
        "Example days:",
        "",
    ]
    rows = [
        [f["day"], eu(f["soc_kwh"], 1), eu(f[cols[0]], 1), eu(f[cols[1]], 1), eu(f["gap"], 1)]
        for f in fails[:GREED_EXAMPLES]
    ]
    return lines + _table(["day", "SoC kWh", *heads, "gap EUR/MWh"], rows) + [""]


def _bias_section(states: dict[str, dict]) -> list[str]:
    lines = ["## Hourly bias", ""]
    for st in states.values():
        # a measured bias belongs to a window booked on quarters, whose own headline is exact:
        # the indicative-only sentence goes with the A/B headline (the headline section)
        bias = (
            st["bias"]
            if st["bias"].get("hourly_input")
            else dict(st["bias"], indicative_only=False)
        )
        body = R.bias_paragraph(bias)[2:]
        if body:
            lines += [f"{st['spec'].label}: {body[0]}", ""]
    return lines


# tools.pool_results counts houses; here the entries are the two windows of one house
_POOLED_LABELS = {
    "- Houses:": "- Windows:",
    "### Houses per capacity class (kWh)": "### Windows per capacity class (kWh)",
    "| class | houses |": "| class | windows |",
    "### Per house": "### Per window",
    "| house | days | mean EUR/day |": "| window | days | mean EUR/day |",
}


def _pooled_section(pooled: dict) -> list[str]:
    """``tools.pool_results.render`` with its entries labelled windows, not houses: the
    acceptance pools the two windows of one house. Raises when a label is not found."""
    from tools.pool_results import render

    text = render(pooled).splitlines()
    if text and text[0].startswith("# "):
        text = _demote(text[1:])
    while text and not text[0].strip():
        text = text[1:]
    found = set()
    for i, ln in enumerate(text):
        for old, new in _POOLED_LABELS.items():
            if ln.startswith(old):
                text[i] = new + ln[len(old) :]
                found.add(old)
    missing = set(_POOLED_LABELS) - found
    if missing:
        raise ValueError(f"pool_results.render changed; labels not found: {sorted(missing)}")
    return [
        "## Pooled summary",
        "",
        "The two windows are one house, so the pool counts windows, not houses.",
        "",
        *text,
        "",
    ]


def _site_section(states: dict[str, dict]) -> list[str]:
    return [
        "## Site models",
        "",
        f"The blind fit, the merged-micro validation and the learning check are in "
        f"{SITE_MODELS_REPORT}. The per-window site lines below (in the window detail) are the "
        "state of the walk-forward registry at the end of each window.",
        "",
        OPEN_ITEMS,
        "",
    ]


def render_replay_windows(
    states: dict[str, dict],
    selection: dict | None,
    greed: dict,
    pooled: dict,
    settled_last: str | None = None,
) -> str:
    """The combined report section: method, headline, ladder and selection, greed, bias,
    pooled summary, site models, the window detail and the data terms. Numbers through
    ``eu``; no coordinates, ids or paths."""
    lines = _method_section(states, settled_last)
    lines += _headline_section(states)
    lines += _ladder_section(states, selection)
    lines += _greed_section(states, greed)
    lines += _bias_section(states)
    lines += _pooled_section(pooled)
    lines += _site_section(states)
    for st in states.values():
        lines += _window_detail(st)
    lines += ["## Data terms", "", *(f"- {t}" for t in R.DATA_TERMS), "", ATTRIBUTION, ""]
    return _english("\n".join(lines))


def _english(text: str) -> str:
    """The report is English: ``core.fmt``'s missing-value mark becomes "not applicable"."""
    return text.replace(FMT_NA, NA_TEXT)


def replay_windows(
    site_json: Path,
    flows_minute: Path,
    flows_hourly: Path,
    irradiance: Path,
    data_dir: Path,
    out_dir: Path,
    report_md: Path,
    ladder: bool = True,
    windows: tuple[str, ...] = ("live", "ab"),
    workers: int = 1,
    plant_json: Path | None = None,
    assemble: bool = True,
    log=print,
    cache_dir: Path | None = None,
    ladder_dir: Path | None = None,
    live_to: str | None = None,
) -> dict:
    """The replay acceptance of plan 2 on the owner's windows (``REPLAY_WINDOWS``).

    Runs every window in ``windows`` (``run_window``; its state is saved, so a window can run
    in its own process), then, with ``assemble``, loads the saved state of every window and
    writes the files and the report (``assemble_windows``). ``plant_json`` is the blind-fit
    JSON whose summer planes shape the A/B window's PV columns. ``cache_dir`` shares the band
    caches between output directories. ``ladder_dir`` holds the saved states whose ladders
    (and the knobs their default replay ran with) give the ladder and the selection, when the
    states under ``out_dir`` come from a run without the ladder. The live window ends on the
    last local day with settled prices in the master, or on ``live_to``.
    """
    cfg = C.from_dict(json.loads(Path(site_json).read_text()))
    if ladder and windows:
        R.check_ladder_default(cfg.knobs)  # before hours of replays, not after them
    master = load_inputs(Path(data_dir))
    if windows:
        irr = read_irradiance_csv(Path(irradiance))
        planes = plant_planes(plant_json) if plant_json is not None else []
        by_name = {w.name: w for w in REPLAY_WINDOWS}
        for name in windows:
            run_window(
                by_name[name],
                cfg,
                flows_minute,
                flows_hourly,
                irr,
                master,
                out_dir,
                ladder,
                workers,
                planes,
                log,
                cache_dir,
                live_to,
            )
    if not assemble:
        return {}
    states = {w.name: load_state(out_dir, w.name) for w in REPLAY_WINDOWS}
    ladder_states = (
        None
        if ladder_dir is None
        else {w.name: load_state(ladder_dir, w.name) for w in REPLAY_WINDOWS}
    )
    return assemble_windows(
        states,
        cfg,
        out_dir,
        report_md,
        log,
        ladder_states=ladder_states,
        settled_last=last_settled_day(master),
    )


# --- CLI ------------------------------------------------------------------------------------


def _log(msg: str) -> None:
    print(msg, flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    mm = sub.add_parser("merged-micro", help="merged-micro validation of the blind PV fit")
    mm.add_argument("--minute", type=Path, required=True)
    mm.add_argument("--hourly", type=Path, required=True)
    mm.add_argument("--irradiance", type=Path, required=True)
    mm.add_argument("--site-planes", type=Path, required=True)
    mm.add_argument("--lat", type=float, required=True)
    mm.add_argument("--lon", type=float, required=True)
    mm.add_argument("--out-json", type=Path, required=True)
    mm.add_argument("--out-md", type=Path, required=True)
    mm.add_argument("--minute-from", default=MINUTE_FROM)
    mm.add_argument("--hourly-from", default=HOURLY_FROM)
    mm.add_argument("--hourly-to", default=HOURLY_TO)
    mm.add_argument("--learn-by", default=LEARN_BY)
    mm.add_argument("--render-only", action="store_true", help="re-render --out-json only")
    bf = sub.add_parser("blind-fit", help="blind fit of the full house with the learning check")
    bf.add_argument("--minute", type=Path, required=True)
    bf.add_argument("--hourly", type=Path, required=True)
    bf.add_argument("--irradiance", type=Path, required=True)
    bf.add_argument("--site", type=Path, required=True)
    bf.add_argument("--site-planes", type=Path, required=True)
    bf.add_argument("--out-json", type=Path, required=True)
    bf.add_argument("--out-md", type=Path, required=True)
    bf.add_argument("--minute-from", default=MINUTE_FROM)
    bf.add_argument("--hourly-from", default=HOURLY_FROM)
    bf.add_argument("--hourly-to", default=HOURLY_TO)
    bf.add_argument("--learn-by", default=LEARN_BY)
    bf.add_argument("--render-only", action="store_true", help="re-render --out-json only")
    rw = sub.add_parser("replay-windows", help="the replay on the live and the A/B window")
    rw.add_argument("--site", type=Path, required=True)
    rw.add_argument("--minute", type=Path, required=True)
    rw.add_argument("--hourly", type=Path, required=True)
    rw.add_argument("--irradiance", type=Path, required=True)
    rw.add_argument("--data", type=Path, required=True, help="directory with master_inputs.npz")
    rw.add_argument("--out", type=Path, required=True, help="output directory (per window)")
    rw.add_argument("--report", type=Path, required=True)
    rw.add_argument("--plant", type=Path, help="blind-fit JSON: its summer planes shape the A/B PV")
    rw.add_argument("--no-ladder", action="store_true", help="the default replay only")
    rw.add_argument(
        "--windows", default="live,ab", help="windows to run, comma separated; empty for none"
    )
    rw.add_argument("--workers", type=int, default=1, help="processes for the ladder")
    rw.add_argument("--no-assemble", action="store_true", help="run windows, write no report")
    rw.add_argument("--cache", type=Path, help="band cache directory (default OUT/cache)")
    rw.add_argument(
        "--live-to", help="last day of the live window (default: last settled day in the master)"
    )
    rw.add_argument(
        "--ladder-states", type=Path, help="output directory of the run whose ladders to report"
    )
    args = p.parse_args(argv)
    if args.command == "replay-windows":
        summary = replay_windows(
            args.site,
            args.minute,
            args.hourly,
            args.irradiance,
            args.data,
            args.out,
            args.report,
            ladder=not args.no_ladder,
            windows=tuple(w for w in args.windows.split(",") if w),
            workers=args.workers,
            plant_json=args.plant,
            assemble=not args.no_assemble,
            log=_log,
            cache_dir=args.cache,
            ladder_dir=args.ladder_states,
            live_to=args.live_to,
        )
        for name, w in summary.get("windows", {}).items():
            h = w["headline"]
            _log(
                f"{name}: difference {eu(h['diff_eur'])} EUR over {eu(h['days'], 0)} days, "
                f"90 % interval {eu(h['lo_eur_day'])} to {eu(h['hi_eur_day'])} EUR per day"
            )
        if summary.get("selection"):
            _log(f"chosen: {summary['selection']['chosen']}")
        return 0
    if args.command == "blind-fit":
        if args.render_only:
            result = json.loads(args.out_json.read_text())
            write_section(args.out_md, "blind-fit", render_blind_fit(result), REPORT_TITLE)
            _log(f"rendered {args.out_json} to {args.out_md}")
            return 0
        result = blind_fit(
            args.minute,
            args.hourly,
            args.irradiance,
            args.site,
            args.site_planes,
            args.out_json,
            args.out_md,
            args.minute_from,
            args.hourly_from,
            args.hourly_to,
            learn_by=args.learn_by,
            log=_log,
        )
        for c in result["checks"]:
            _log(f"{c['name']}: {c['shown']} ({'pass' if c['pass'] else 'miss'})")
        _log(f"wall time {eu(result['wall_s'] / 60.0, 1)} min; report {args.out_md}")
        return 0
    if args.command == "merged-micro":
        if args.render_only:
            result = json.loads(args.out_json.read_text())
            write_section(args.out_md, "merged-micro", render_merged_micro(result), REPORT_TITLE)
            _log(f"rendered {args.out_json} to {args.out_md}")
            return 0
        result = merged_micro(
            args.minute,
            args.hourly,
            args.irradiance,
            args.site_planes,
            args.lat,
            args.lon,
            args.out_json,
            args.out_md,
            args.minute_from,
            args.hourly_from,
            args.hourly_to,
            args.learn_by,
            log=_log,
        )
        for c in result["checks"]:
            _log(f"{c['name']}: {c['shown']} ({'pass' if c['pass'] else 'miss'})")
        _log(f"wall time {eu(result['wall_s'] / 60.0, 1)} min; report {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
