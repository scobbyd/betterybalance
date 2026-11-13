"""Inputs: the price master of a data directory and hourly site flows from HA statistics.

The price master is the npz from tools/fetch_history.py plus live minutes, moved from
``tools.study.october_2026`` so the replay CLI loads the master without ``tools/``. The
statistics helpers turn Home Assistant long-term statistics rows (``recorder``
``statistics_during_period``) into hourly flows; they moved from ``tools/export_site.py`` so the
integration builds the same table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .features import build_master
from .feeds.ida import AUCTIONS
from .feeds.live import load_live_minutes, minute_to_isp
from .timegrid import Table

_LOGGER = logging.getLogger(__name__)

STAT_ROLES = (
    "grid_import_kwh",
    "grid_export_kwh",
    "grid_power_w",
    "battery_power_w",
    "battery_charge_kwh",
    "battery_discharge_kwh",
    "soc_pct",
    "load_power_w",
    "gen_power_w",
    "pv_power_w",
    *(f"pv{i}_power_w" for i in range(1, 9)),
)
HOUR_S = 3600

Rows = list[dict]

NPZ_FILE = "master_inputs.npz"
LIVE_FILE = "live_minutes.csv"


def load_inputs(data_dir: Path) -> Table:
    """Master table from the npz plus the live archive (stale minutes masked by minute_to_isp)."""
    data_dir = Path(data_dir)
    z = np.load(data_dir / NPZ_FILE)
    grid = np.asarray(z["ts"], dtype=np.int64)
    da = {"ts": grid, "da_eur_mwh": np.asarray(z["da_eur_mwh"], dtype=float)}
    settled = {
        "ts": grid,
        **{c: np.asarray(z[c], dtype=float) for c in ("buy_eur_mwh", "sell_eur_mwh")},
        "regulation_state": np.asarray(z["regulation_state"], dtype=float),
    }
    ida = {
        a: {"ts": grid, "ida_eur_mwh": np.asarray(z[a.lower() + "_eur_mwh"], dtype=float)}
        for a in AUCTIONS
    }
    live_path = data_dir / LIVE_FILE
    live_isp = minute_to_isp(load_live_minutes(live_path), grid) if live_path.exists() else None
    return build_master(da, settled, ida, live_isp, grid)


def _nan_if_none(value) -> float:
    return np.nan if value is None else float(value)


def counter_deltas(rows: Rows) -> tuple[np.ndarray, np.ndarray]:
    """Hourly differences of the `sum` field of a statistics series.

    `sum` is the counter at the end of the hour that starts at `start`, so the difference of two
    consecutive rows is the energy of the later row's hour and is labelled with that hour start.
    A negative difference (counter reset), a missing value or a gap longer than one hour is NaN.
    The first row yields no value.
    """
    rows = sorted(rows, key=lambda r: r["start"])
    ts = np.asarray([int(r["start"]) // 1000 for r in rows], dtype=np.int64)
    total = np.asarray([_nan_if_none(r.get("sum")) for r in rows], dtype=float)
    if len(rows) < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    delta = np.diff(total)
    delta = np.where((delta < 0) | (np.diff(ts) > HOUR_S), np.nan, delta)
    return ts[1:], delta


def mean_power_to_kwh(rows: Rows) -> tuple[np.ndarray, np.ndarray]:
    """Hourly energy from the mean power: kwh = mean_w / 1000 per hour."""
    rows = sorted(rows, key=lambda r: r["start"])
    ts = np.asarray([int(r["start"]) // 1000 for r in rows], dtype=np.int64)
    kwh = np.asarray([_nan_if_none(r.get("mean")) for r in rows], dtype=float) / 1000.0
    return ts, kwh


def _ids(spec) -> list[str]:
    return [spec] if isinstance(spec, str) else list(spec)


def _sum_role(stats: dict, role: str, spec, convert) -> dict[int, float]:
    """Sum the converted series of all ids of a role into {hour start: value}.

    An id with no rows makes the whole role empty (NaN) and prints a warning that names the role,
    not the id. An hour that one id lacks is NaN in the sum.
    """
    per_id = []
    for sid in _ids(spec):
        rows = stats.get(sid) or []
        if not rows:
            _LOGGER.warning("no statistics for role %s; its column stays empty", role)
            return {}
        per_id.append(dict(zip(*(a.tolist() for a in convert(rows)))))
    hours = set().union(*per_id)
    return {h: sum(d.get(h, np.nan) for d in per_id) for h in hours}


def _stat_field(field: str):
    def convert(rows: Rows) -> tuple[np.ndarray, np.ndarray]:
        rows = sorted(rows, key=lambda r: r["start"])
        ts = np.asarray([int(r["start"]) // 1000 for r in rows], dtype=np.int64)
        return ts, np.asarray([_nan_if_none(r.get(field)) for r in rows], dtype=float)

    return convert


def _column(grid: np.ndarray, series: dict[int, float]) -> np.ndarray:
    return np.asarray([series.get(int(t), np.nan) for t in grid], dtype=float)


def _split(signed_kwh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positive part and negative part (as a positive number) of a signed energy; NaN stays NaN."""
    return np.maximum(signed_kwh, 0.0), np.maximum(-signed_kwh, 0.0)


def flows_from_statistics(
    stats: dict[str, Rows], roles: dict, battery_sign: int = 1, grid_sign: int = 1
) -> Table:
    """Hourly flows (step_s 3600) on the union of the hours of all mapped series.

    Grid and battery come from counters when their counter roles are mapped, else they split
    from the signed power mean. soc_pct, soc_min_pct and soc_max_pct come from mean, min and max.
    Load, gen and PV come from means. Unmapped roles are NaN. A mapped string role
    pv{i}_power_w adds the column pv{i}_kwh; an unmapped string adds no column.
    """
    series: dict[str, dict[int, float]] = {}
    for role, spec in roles.items():
        if role not in STAT_ROLES:
            raise ValueError(f"unknown role: {role}")
        if role.endswith("_kwh"):
            series[role] = _sum_role(stats, role, spec, counter_deltas)
        else:
            series[role] = _sum_role(stats, role, spec, mean_power_to_kwh)
    soc = {}
    if "soc_pct" in roles:
        for key, field in (("soc_pct", "mean"), ("soc_min_pct", "min"), ("soc_max_pct", "max")):
            ids = _ids(roles["soc_pct"])
            per_id = [_sum_role(stats, "soc_pct", i, _stat_field(field)) for i in ids]
            soc[key] = {
                h: float(np.mean([d.get(h, np.nan) for d in per_id])) for h in set().union(*per_id)
            }
    hours: set[int] = set()
    for d in (*series.values(), *soc.values()):
        hours.update(d)
    grid = np.asarray(sorted(hours), dtype=np.int64)
    out: Table = {"ts": grid, "step_s": np.full(len(grid), HOUR_S, dtype=np.int64)}
    nan = np.full(len(grid), np.nan)

    def col(role: str) -> np.ndarray | None:
        return _column(grid, series[role]) if series.get(role) else None

    gi, ge, gp = col("grid_import_kwh"), col("grid_export_kwh"), col("grid_power_w")
    if gi is None and ge is None and gp is not None:
        gi, ge = _split(gp * grid_sign)
    out["grid_import_kwh"] = gi if gi is not None else nan.copy()
    out["grid_export_kwh"] = ge if ge is not None else nan.copy()
    bc, bd, bp = col("battery_charge_kwh"), col("battery_discharge_kwh"), col("battery_power_w")
    if bc is None and bd is None and bp is not None:
        bc, bd = _split(bp * battery_sign)
    out["batt_charge_kwh"] = bc if bc is not None else nan.copy()
    out["batt_discharge_kwh"] = bd if bd is not None else nan.copy()
    for key in ("soc_pct", "soc_min_pct", "soc_max_pct"):
        out[key] = _column(grid, soc[key]) if key in soc else nan.copy()
    for key, role in (
        ("load_kwh", "load_power_w"),
        ("gen_kwh", "gen_power_w"),
        ("pv_kwh", "pv_power_w"),
    ):
        c = col(role)
        out[key] = c if c is not None else nan.copy()
    for i in range(1, 9):
        if f"pv{i}_power_w" in roles:
            c = col(f"pv{i}_power_w")
            out[f"pv{i}_kwh"] = c if c is not None else nan.copy()
    return out
