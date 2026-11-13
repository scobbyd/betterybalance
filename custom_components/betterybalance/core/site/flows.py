"""Flows interchange table: minute or quarter energy flows of a site, and the AC-side residual."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..timegrid import ISP_S, Table, isp_start
from .contract import Pack

COLUMNS = (
    "ts",
    "step_s",
    "grid_import_kwh",
    "grid_export_kwh",
    "batt_charge_kwh",
    "batt_discharge_kwh",
    "soc_pct",
    "soc_min_pct",
    "soc_max_pct",
    "load_kwh",
    "gen_kwh",
    "pv_kwh",
)
# optional per-string PV energy: read, written and summed only where a table carries them
STRING_COLUMNS = tuple(f"pv{i}_kwh" for i in range(1, 9))
_INT_COLUMNS = ("ts", "step_s")
_FLOAT_COLUMNS = tuple(c for c in COLUMNS if c not in _INT_COLUMNS)
_REQUIRED = (
    "grid_import_kwh",
    "grid_export_kwh",
    "batt_charge_kwh",
    "batt_discharge_kwh",
    "soc_pct",
)


def _strings(present) -> tuple[str, ...]:
    return tuple(c for c in STRING_COLUMNS if c in present)


def read_flows(path: Path) -> Table:
    """Read a flows CSV. An empty cell is NaN, ts and step_s are int64, the rest float64.

    The string columns pv1_kwh .. pv8_kwh are read when the header has them.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or ()
        rows = list(reader)
    out: Table = {}
    for col in (*COLUMNS, *_strings(header)):
        if col in _INT_COLUMNS:
            out[col] = np.asarray([int(r[col]) for r in rows], dtype=np.int64)
        else:
            out[col] = np.asarray([float(r[col]) if r[col] != "" else np.nan for r in rows])
    return out


def write_flows(table: Table, path: Path) -> None:
    """Write a flows CSV. NaN becomes an empty cell. String columns the table has follow COLUMNS."""
    n = len(table["ts"])
    columns = (*COLUMNS, *_strings(table))
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(n):
            row = []
            for col in columns:
                v = table[col][i]
                if col in _INT_COLUMNS:
                    row.append(int(v))
                else:
                    row.append("" if np.isnan(v) else repr(float(v)))
            writer.writerow(row)


def to_quarters(flows: Table) -> Table:
    """Aggregate rows of at most 900 s into ISP quarters.

    Energy sums (the string columns the table has too), soc_pct is the mean, soc_min_pct the
    min, soc_max_pct the max. A quarter that lacks time coverage, or a cell of a required
    column, is NaN in every float column and flagged in `missing`.
    """
    step = np.asarray(flows["step_s"], dtype=np.int64)
    if (step > ISP_S).any():
        raise ValueError("hourly rows need disaggregation first")
    order = np.argsort(np.asarray(flows["ts"], dtype=np.int64), kind="stable")
    ts = np.asarray(flows["ts"], dtype=np.int64)[order]
    step = step[order]
    float_columns = (*_FLOAT_COLUMNS, *_strings(flows))
    cols = {c: np.asarray(flows[c], dtype=float)[order] for c in float_columns}
    quarter = isp_start(ts)
    n = len(ts)
    if n == 0:
        starts = np.zeros(0, dtype=np.int64)
    else:
        starts = np.flatnonzero(np.r_[True, quarter[1:] != quarter[:-1]])
    out: Table = {"ts": quarter[starts], "step_s": np.full(len(starts), ISP_S, dtype=np.int64)}
    covered = np.add.reduceat(step, starts) if n else np.zeros(0, dtype=np.int64)
    bad = covered < ISP_S
    for col in _REQUIRED:
        bad |= np.add.reduceat(np.isnan(cols[col]).astype(np.int64), starts) > 0 if n else bad
    for col in float_columns:
        if col == "soc_min_pct":
            agg = np.minimum.reduceat(cols[col], starts) if n else np.zeros(0)
        elif col == "soc_max_pct":
            agg = np.maximum.reduceat(cols[col], starts) if n else np.zeros(0)
        elif col == "soc_pct":
            counts = np.diff(np.r_[starts, n])
            agg = np.add.reduceat(cols[col], starts) / counts if n else np.zeros(0)
        else:
            agg = np.add.reduceat(cols[col], starts) if n else np.zeros(0)
        agg = np.where(bad, np.nan, agg)
        out[col] = agg
    out["missing"] = bad
    return out


def ac_legs(pack: Pack) -> tuple[float, float]:
    """Charge and discharge leg efficiencies (ec, ed): the round trip split evenly."""
    leg = float(np.sqrt(pack.eta_rt))
    return leg, leg


def residual_kw(quarters: Table, pack: Pack) -> np.ndarray:
    """AC-side residual mean power per quarter in kW: the load the battery does not serve.

    (grid_import - grid_export - charge_ac + discharge_ac + gen) * 4. For a DC-side pack the
    battery energy converts to the AC side through the legs. NaN where the quarter is missing.
    """
    charge = np.asarray(quarters["batt_charge_kwh"], dtype=float)
    discharge = np.asarray(quarters["batt_discharge_kwh"], dtype=float)
    if pack.side == "dc":
        ec, ed = ac_legs(pack)
        charge = charge / ec
        discharge = discharge * ed
    gen = np.nan_to_num(np.asarray(quarters["gen_kwh"], dtype=float), nan=0.0)
    res = (
        np.asarray(quarters["grid_import_kwh"], dtype=float)
        - np.asarray(quarters["grid_export_kwh"], dtype=float)
        - charge
        + discharge
        + gen
    ) * 4.0
    return np.where(np.asarray(quarters["missing"], dtype=bool), np.nan, res)
