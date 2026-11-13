"""Hourly flows to quarters, quarters back to hourly means, and the hourly bias measure."""

from __future__ import annotations

import numpy as np

from ..core.ledger import LedgerRow
from ..core.site.flows import COLUMNS, STRING_COLUMNS
from ..core.timegrid import ISP_S, Table

HOUR_S = 3600
_PER_HOUR = HOUR_S // ISP_S
_FLOAT_COLUMNS = tuple(c for c in COLUMNS if c not in ("ts", "step_s"))
_REQUIRED = (
    "grid_import_kwh",
    "grid_export_kwh",
    "batt_charge_kwh",
    "batt_discharge_kwh",
    "soc_pct",
)
_SHAPED = ("gen_kwh", "pv_kwh", *STRING_COLUMNS)
_CARRIED = ("soc_pct", "soc_min_pct", "soc_max_pct")


def _float_columns(table: Table) -> tuple[str, ...]:
    """The flow float columns, then the string columns pv1_kwh .. pv8_kwh the table carries."""
    return (*_FLOAT_COLUMNS, *(c for c in STRING_COLUMNS if c in table))


def _shape_weights(
    hour_ts: np.ndarray, pv_shape_kw: np.ndarray | None, shape_ts: np.ndarray | None
) -> np.ndarray:
    """Weights (n_hours, 4) summing to 1.

    Flat 0,25 where the shape is absent, incomplete or sums to 0.
    """
    n = len(hour_ts)
    weights = np.full((n, _PER_HOUR), 1.0 / _PER_HOUR)
    if pv_shape_kw is None or shape_ts is None or n == 0:
        return weights
    shape = np.asarray(pv_shape_kw, dtype=float)
    sts = np.asarray(shape_ts, dtype=np.int64)
    order = np.argsort(sts, kind="stable")
    sts, shape = sts[order], shape[order]
    want = hour_ts[:, None] + np.arange(_PER_HOUR, dtype=np.int64) * ISP_S
    pos = np.clip(np.searchsorted(sts, want), 0, max(len(sts) - 1, 0))
    found = (sts[pos] == want) if len(sts) else np.zeros(want.shape, dtype=bool)
    vals = np.where(found, shape[pos], np.nan) if len(sts) else np.full(want.shape, np.nan)
    total = vals.sum(axis=1)
    usable = found.all(axis=1) & np.isfinite(total) & (total > 0.0)
    weights[usable] = vals[usable] / total[usable, None]
    return weights


def hourly_to_quarters(
    flows_hourly: Table,
    pv_shape_kw: np.ndarray | None = None,
    shape_ts: np.ndarray | None = None,
) -> Table:
    """Split every 3.600 s row into four 900 s rows.

    Grid, battery and load energies split flat. ``gen_kwh``, ``pv_kwh`` and the string columns
    ``pv1_kwh`` .. ``pv8_kwh`` the table carries split by the plant shape's four values inside
    the hour (flat without a complete or positive shape). ``soc_pct``,
    ``soc_min_pct`` and ``soc_max_pct`` carry to all four. A NaN required value flags the four
    quarters ``missing`` with NaN in every float column. Optional NaN stays NaN. Every output
    row carries ``hourly_origin`` True: its energies are flat splits of an hourly mean, so the
    PV fit hour-averages its plane shapes before it compares them with these rows.
    """
    step = np.asarray(flows_hourly["step_s"], dtype=np.int64)
    if (step != HOUR_S).any():
        raise ValueError("hourly_to_quarters needs rows of 3600 s")
    ts_h = np.asarray(flows_hourly["ts"], dtype=np.int64)
    n = len(ts_h)
    weights = _shape_weights(ts_h, pv_shape_kw, shape_ts)
    bad_h = np.zeros(n, dtype=bool)
    for col in _REQUIRED:
        bad_h |= np.isnan(np.asarray(flows_hourly[col], dtype=float))
    bad = np.repeat(bad_h, _PER_HOUR)
    out: Table = {
        "ts": (ts_h[:, None] + np.arange(_PER_HOUR, dtype=np.int64) * ISP_S).ravel(),
        "step_s": np.full(n * _PER_HOUR, ISP_S, dtype=np.int64),
    }
    for col in _float_columns(flows_hourly):
        hourly = np.asarray(flows_hourly[col], dtype=float)
        if col in _CARRIED:
            split = np.repeat(hourly, _PER_HOUR)
        elif col in _SHAPED:
            split = (hourly[:, None] * weights).ravel()
        else:
            split = np.repeat(hourly / _PER_HOUR, _PER_HOUR)
        out[col] = np.where(bad, np.nan, split)
    out["missing"] = bad
    out["hourly_origin"] = np.ones(n * _PER_HOUR, dtype=bool)
    return out


def flatten_to_hours(quarters: Table) -> Table:
    """Give every quarter the mean of its hour's quarters, on the same ``ts`` grid.

    NaN propagates through the mean. ``missing`` is True for the whole hour when any quarter is.
    A trailing partial hour averages over the quarters it has. The string columns the table
    carries flatten too.
    """
    ts = np.asarray(quarters["ts"], dtype=np.int64)
    _, inv = np.unique(ts // HOUR_S, return_inverse=True)
    counts = np.bincount(inv)
    out: Table = {"ts": ts.copy(), "step_s": np.asarray(quarters["step_s"], dtype=np.int64).copy()}
    for col in _float_columns(quarters):
        mean = np.bincount(inv, weights=np.asarray(quarters[col], dtype=float)) / counts
        out[col] = mean[inv]
    any_missing = np.bincount(inv, weights=np.asarray(quarters["missing"], dtype=float)) > 0
    out["missing"] = any_missing[inv]
    return out


def _booked_eur(rows: list[LedgerRow]) -> float:
    return float(sum(r.money_eur for r in rows if not r.missing and not np.isnan(r.money_eur)))


def hourly_bias(
    actual_quarters: list[LedgerRow], actual_flattened: list[LedgerRow], headline_eur: float
) -> dict:
    """Money difference caused by hourly resolution: ``hourly - quarter`` over non-missing rows.

    ``bias_share`` is ``|bias| / |headline|`` (inf with a zero headline unless the bias is 0
    too). ``indicative_only`` is set above 10 %.
    """
    quarter = _booked_eur(actual_quarters)
    hourly = _booked_eur(actual_flattened)
    bias = hourly - quarter
    if headline_eur == 0:
        share = 0.0 if bias == 0 else float("inf")
    else:
        share = abs(bias) / abs(headline_eur)
    return {
        "quarter_eur": quarter,
        "hourly_eur": hourly,
        "bias_eur": bias,
        "bias_share": share,
        "indicative_only": share > 0.1,
    }
