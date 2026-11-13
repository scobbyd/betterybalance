"""The anonymised results export: what a house sends back and what gets pooled.

The export carries binned site classes, tiers by month, the daily differences and the
interval. It carries no coordinates, prices, entity ids or paths. Machine file: plain
decimals, every EUR value rounded to cents (so no pair of values reads as a coordinate pair),
and non-finite numbers become ``null``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .site.contract import SiteConfig, tax_mode_at

RESULTS_FORMAT = 1
EUR_DECIMALS = 2  # every EUR value in cents


def _cents(x: float) -> float:
    return round(float(x), EUR_DECIMALS)


def capacity_bin(kwh: float) -> str:
    """Capacity class: ``<10``, ``10-20``, ``20-40`` or ``>=40`` kWh (lower bound inclusive)."""
    if kwh < 10:
        return "<10"
    if kwh < 20:
        return "10-20"
    if kwh < 40:
        return "20-40"
    return ">=40"


def power_bin(kw: float) -> str:
    """Power class: ``<5``, ``5-10`` or ``>=10`` kW (lower bound inclusive)."""
    if kw < 5:
        return "<5"
    if kw < 10:
        return "5-10"
    return ">=10"


def export_results(result, cfg: SiteConfig, version: str, source: str = "replay") -> dict:
    """The anonymised results of one replay as a plain dict (see the module docstring)."""
    days = [str(d) for d in result.pairs["day"]]
    diffs = [_cents(x) for x in result.pairs["diff_eur"]]
    start_ts = int(result.virtual[0].ts) if result.virtual else 0
    pack = cfg.pack
    pv = "none" if cfg.pv.none else ("planes_given" if cfg.pv.planes else "learned")
    return {
        "format": RESULTS_FORMAT,
        "version": str(version),
        "source": str(source),
        "window": {"from": days[0] if days else None, "to": days[-1] if days else None},
        "contract_mode": str(tax_mode_at(cfg.terms, start_ts)),
        "site_class": {
            "capacity_bin": capacity_bin(pack.capacity_kwh),
            "power_bin": power_bin(max(pack.charge_kw, pack.discharge_kw)),
            "pv": pv,
            "ev": bool(cfg.ev.present),
        },
        "tiers_by_month": {str(m): str(t) for m, t in sorted(result.tiers_by_month.items())},
        "learning_days": int(result.learning_days),
        "terminal_eur": _cents(result.terminal_eur),
        "interval": [_cents(result.interval[1]), _cents(result.interval[2])],
        "daily": [{"day": d, "diff_eur": x} for d, x in zip(days, diffs, strict=True)],
    }


def _clean(x):
    """Non-finite floats to None, recursively."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_clean(v) for v in x]
    return x


def results_bytes(d: dict) -> bytes:
    """The JSON text of ``d``: sorted keys, 2-space indent, no NaN, final newline."""
    text = json.dumps(_clean(d), sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def write_results(d: dict, path: Path) -> None:
    """Write ``d`` as strict JSON (sorted keys, 2-space indent) to ``path``."""
    Path(path).write_bytes(results_bytes(d))
