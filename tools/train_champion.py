"""Train and ship the NL market champion price model.

Run: python -m tools.train_champion --data data/ --to 2026-10-28 \
--out custom_components/betterybalance/models/market_nl.json

Per tier two fits run on the trailing train_days local days. Fit 1 stops where the holdout
starts and scores the last holdout_days days (buy q50 MAE per cell). Fit 2 runs to the end of
--to and gives the shipped weights. Tiers B to D train only where their feeds have rows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from betterybalance import __version__  # noqa: E402
from betterybalance.core.evaluate import mae  # noqa: E402
from betterybalance.core.features import ELAPSED_MIN, HORIZONS, TIERS, build_matrix  # noqa: E402
from betterybalance.core.fmt import eu  # noqa: E402
from betterybalance.core.pricemodel import (  # noqa: E402
    FORMAT,
    MIN_ROWS,
    PriceModel,
    coverage_factor,
    dual_tables,
    fit_model,
    has_cell,
    predict_rows,
    to_json,
)
from betterybalance.core.timegrid import ZONE, Table  # noqa: E402

from tools.study.october_2026 import load_inputs, local_midnight  # noqa: E402

SOURCE_REPORT = "2026-10-29-october-price-study.md"
COVERAGE_TARGET = 0.8


def _day(day: str, delta: int) -> int:
    return local_midnight((dt.date.fromisoformat(day) + dt.timedelta(days=delta)).isoformat())


def windows(to: str, holdout_days: int = 14, train_days: int = 365) -> dict:
    """Epoch-s bounds, local midnights. end is exclusive (midnight after --to)."""
    end = _day(to, 1)
    holdout_start = _day(to, 1 - holdout_days)
    return {
        "end": end,
        "holdout_start": holdout_start,
        "start": _day(to, 1 - train_days),
        "holdout_fit_start": _day(to, 1 - train_days - holdout_days),
    }


def mask(ts: np.ndarray, lo: int, hi: int) -> np.ndarray:
    ts = np.asarray(ts, dtype=np.int64)
    return (ts >= lo) & (ts < hi)


def _cells():
    return [(h, e) for h in HORIZONS for e in (ELAPSED_MIN if h == 0 else (None,))]


def holdout_scores(master: Table, model: PriceModel, tier: str, lo: int, hi: int) -> dict:
    """Buy q50 MAE and n per cell of one tier on target rows in [lo, hi)."""
    out = {}
    for h, e in _cells():
        if not has_cell(model, tier, h, e):
            continue
        mx = build_matrix(master, tier, h, e)
        sel = (mx.ts_target >= lo) & (mx.ts_target < hi)
        if not sel.any():
            out["%s:%d:%s" % (tier, h, "-" if e is None else e)] = {"mae": None, "n": 0}
            continue
        p = predict_rows(model, tier, h, mx.X[sel], mx.names, e)
        key = "%s:%d:%s" % (tier, h, "-" if e is None else e)
        out[key] = {"mae": round(mae(mx.y_buy[sel], p["buy_q50"]), 4), "n": int(sel.sum())}
    return out


def holdout_cov(master: Table, model: PriceModel, tier: str, lo: int, hi: int) -> dict:
    """Coverage factor per (h, e) cell from the pooled buy and sell needs on holdout rows."""
    out = {}
    for h, e in _cells():
        if not has_cell(model, tier, h, e):
            continue
        mx = build_matrix(master, tier, h, e)
        sel = (mx.ts_target >= lo) & (mx.ts_target < hi)
        if not sel.any():
            continue
        p = predict_rows(model, tier, h, mx.X[sel], mx.names, e)
        out[(h, e)] = coverage_factor(
            np.concatenate([mx.y_buy[sel], mx.y_sell[sel]]),
            np.concatenate([p["buy_q10"], p["sell_q10"]]),
            np.concatenate([p["buy_q50"], p["sell_q50"]]),
            np.concatenate([p["buy_q90"], p["sell_q90"]]),
            COVERAGE_TARGET,
        )
    return out


def train(
    master: Table,
    to: str,
    holdout_days: int = 14,
    train_days: int = 365,
    tiers=TIERS,
    horizons=HORIZONS,
    holdout_fits: dict | None = None,
) -> PriceModel:
    """Fit the shipped model. A dict passed as holdout_fits receives the holdout fit per tier."""
    w = windows(to, holdout_days, train_days)
    ts = np.asarray(master["ts"], dtype=np.int64)
    ship_mask = mask(ts, w["start"], w["end"])
    hold_mask = mask(ts, w["holdout_fit_start"], w["holdout_start"])
    final = PriceModel()
    holdout: dict = {}
    by_tier: dict = {}
    cov_mean: dict = {}
    ends: list = []
    for tier in tiers:
        trial = fit_model(master, tiers=(tier,), horizons=horizons, train_mask=hold_mask)
        if holdout_fits is not None:
            holdout_fits[tier] = trial
        holdout.update(holdout_scores(master, trial, tier, w["holdout_start"], w["end"]))
        factors = holdout_cov(master, trial, tier, w["holdout_start"], w["end"])
        cov_mean[tier] = round(float(np.mean(list(factors.values()))), 4) if factors else None
        part = fit_model(master, tiers=(tier,), horizons=horizons, train_mask=ship_mask)
        for (h, e), f in factors.items():
            if (tier, h, e) in part.models:
                part.models[(tier, h, e)].cov_factor = f
        final.models.update(part.models)
        by_tier[tier] = part.meta["train_start"]
        if part.meta["train_end"]:
            ends.append(part.meta["train_end"])
        print("tier %s: %d cells" % (tier, len(part.models)), flush=True)
    starts = [s for s in by_tier.values() if s]
    final.tables = dual_tables(master, ship_mask)
    final.meta = {
        "format": FORMAT,
        "coverage_target": COVERAGE_TARGET,
        "holdout_cov_mean": cov_mean,
        "fitted_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": __version__,
        "min_rows": MIN_ROWS,
        "cells": len(final.models),
        "train_start": min(starts) if starts else None,
        "train_end": max(ends) if ends else None,
        "train_start_by_tier": by_tier,
        "train_days": train_days,
        "holdout_days": holdout_days,
        "holdout_start": _iso(w["holdout_start"]),
        "holdout_end": to,
        "holdout_mae": holdout,
        "holdout_protocol": (
            "The holdout fit excludes the last %d days before the end of the training window and "
            "the shipped weights come from a second fit on every labelled row up to the end day."
            % holdout_days
        ),
        "source_report": SOURCE_REPORT,
    }
    return final


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, ZONE).date().isoformat()


def holdout_table(model: PriceModel) -> str:
    rows = ["tier  " + "  ".join("h%-6d" % h for h in HORIZONS)]
    hm = model.meta["holdout_mae"]
    for tier in TIERS:
        cells = []
        for h in HORIZONS:
            e = ELAPSED_MIN[0] if h == 0 else None
            v = hm.get("%s:%d:%s" % (tier, h, "-" if e is None else e), {}).get("mae")
            cells.append("%-7s" % ("-" if v is None else eu(v, 1)))
        mean = model.meta.get("holdout_cov_mean", {}).get(tier)
        tail = "  cov %s" % ("-" if mean is None else eu(mean, 2))
        rows.append("%-4s  " % tier + "  ".join(cells) + tail)
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--to", required=True, help="last local day with settled labels, ISO")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-days", type=int, default=14)
    ap.add_argument("--train-days", type=int, default=365)
    a = ap.parse_args(argv)
    model = train(load_inputs(a.data), a.to, a.holdout_days, a.train_days)
    text = to_json(model)
    a.out.write_text(text, encoding="utf-8")
    print(holdout_table(model))
    print("wrote %s (%s bytes)" % (a.out, len(text.encode())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
