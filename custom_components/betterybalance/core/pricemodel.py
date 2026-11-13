"""Tiered settled-price model, its JSON registry format and the day-ahead table baseline.

One HorizonModel per (tier, horizon, elapsed minute) cell. Each holds five quantiles of the buy
and sell price, a coverage factor and three logistic regime classifiers on the standardised
features of that cell.
The JSON form is the shipped champion and the on-box challenger format.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from .features import ELAPSED_MIN, HORIZONS, TIERS, build_matrix
from .quantile import (
    apply_standardise,
    fit_logistic,
    fit_quantile,
    predict_linear,
    predict_proba,
    standardise,
)
from .timegrid import ZONE, Table, local_calendar

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
FORMAT = 2
MIN_ROWS = 50
HOUR_BLOCKS = ((0, 6), (6, 10), (10, 16), (16, 24))
_HOUR_BLOCKS = HOUR_BLOCKS
DUAL_RATE_FALLBACK = 0.4
_COV_MIN_ROWS = 20
_COV_CLIP = (0.5, 5.0)
_CLASSIFIERS = ("dual", "negsell", "spike")
_Key = tuple[str, int, "int | None"]


class MissingCellError(KeyError):
    """The model has no fitted cell for the requested (tier, h, elapsed)."""

    def __str__(self) -> str:
        return str(self.args[0])


@dataclass
class HorizonModel:
    names: list[str]
    center: np.ndarray
    scale: np.ndarray
    w_buy: dict[float, np.ndarray]
    w_sell: dict[float, np.ndarray]
    w_dual: np.ndarray | None
    w_negsell: np.ndarray | None
    w_spike: np.ndarray | None
    n_train: int
    cov_factor: float = 1.0


@dataclass
class PriceModel:
    models: dict[_Key, HorizonModel] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)

    def cov_factor(
        self, tier: str, horizon: int, q: float, elapsed_min: int | None = None
    ) -> float:
        """Coverage factor of a cell: one per cell, so the same for every q. 1,0 if missing."""
        hm = self.models.get((tier, horizon, elapsed_min if horizon == 0 else None))
        return 1.0 if hm is None else float(hm.cov_factor)


@dataclass
class Prediction:
    buy: dict[float, float]
    sell: dict[float, float]
    p_dual: float
    p_negsell: float
    p_spike: float
    cov_factor: float = 1.0


def _fit_cell(mx) -> HorizonModel | None:
    n = len(mx.y_buy)
    if n < MIN_ROWS:
        return None
    Xs, center, scale = standardise(mx.X)
    w_buy = {q: fit_quantile(Xs, mx.y_buy, q) for q in QUANTILES}
    w_sell = {q: fit_quantile(Xs, mx.y_sell, q) for q in QUANTILES}
    cls: dict[str, np.ndarray | None] = {}
    for name in _CLASSIFIERS:
        y = np.asarray(getattr(mx, "y_" + name), dtype=bool)
        cls[name] = fit_logistic(Xs, y) if 0 < int(y.sum()) < n else None
    return HorizonModel(
        names=list(mx.names),
        center=center,
        scale=scale,
        w_buy=w_buy,
        w_sell=w_sell,
        w_dual=cls["dual"],
        w_negsell=cls["negsell"],
        w_spike=cls["spike"],
        n_train=n,
    )


def fit_model(
    master: Table,
    tiers=TIERS,
    horizons=HORIZONS,
    train_mask: np.ndarray | None = None,
    meta: dict | None = None,
) -> PriceModel:
    """Fit every (tier, horizon, elapsed) cell.

    A cell with fewer than MIN_ROWS training rows (after train_mask) is omitted from
    PriceModel.models. Check has_cell before predicting. predict_rows and predict_one raise
    MissingCellError (a KeyError) for an omitted cell. The meta of the result holds train_start,
    train_end (ISO local Europe/Amsterdam calendar dates of the fitted rows), fitted_at, version,
    cells and min_rows.
    Keys in the meta argument win over these.
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    out = PriceModel()
    lo = hi = None
    for tier in tiers:
        for h in horizons:
            for e in ELAPSED_MIN if h == 0 else (None,):
                mx = build_matrix(master, tier, h, e)
                if train_mask is not None:
                    keep = np.asarray(train_mask, dtype=bool)[np.searchsorted(ts, mx.ts_target)]
                    mx.X, mx.y_buy, mx.y_sell = mx.X[keep], mx.y_buy[keep], mx.y_sell[keep]
                    mx.y_dual, mx.y_negsell = mx.y_dual[keep], mx.y_negsell[keep]
                    mx.y_spike = mx.y_spike[keep]
                    mx.ts_target = mx.ts_target[keep]
                cell = _fit_cell(mx)
                if cell is not None:
                    out.models[(tier, h, e)] = cell
                    lo = int(mx.ts_target.min()) if lo is None else min(lo, int(mx.ts_target.min()))
                    hi = int(mx.ts_target.max()) if hi is None else max(hi, int(mx.ts_target.max()))
    rows = np.ones(len(ts), dtype=bool)
    if train_mask is not None:
        rows = np.asarray(train_mask, dtype=bool)
    out.tables = dual_tables(master, rows)
    now = dt.datetime.now(dt.UTC)
    out.meta = {
        "format": FORMAT,
        "train_start": None if lo is None else _iso_date(lo),
        "train_end": None if hi is None else _iso_date(hi),
        "fitted_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": __version__,
        "cells": len(out.models),
        "min_rows": MIN_ROWS,
        **(meta or {}),
    }
    return out


def dual_tables(master: Table, train_mask: np.ndarray) -> dict:
    """Regulation-state-2 rate by (hour block, month) and median buy - sell spread by hour block."""
    ts = np.asarray(master["ts"], dtype=np.int64)
    m = np.asarray(train_mask, dtype=bool)
    cal = local_calendar(ts)
    block = _block(cal["hour"])
    month = np.asarray(cal["month"], dtype=np.int64)
    state = np.asarray(master["regulation_state"], dtype=float)
    has = m & np.isfinite(state)
    is2 = state == 2
    overall = float(is2[has].mean()) if has.any() else DUAL_RATE_FALLBACK
    rate = [[overall] * 12 for _ in HOUR_BLOCKS]
    for b in range(len(HOUR_BLOCKS)):
        for mo in range(1, 13):
            sel = has & (block == b) & (month == mo)
            if sel.any():
                rate[b][mo - 1] = float(is2[sel].mean())
    buy = np.asarray(master["buy_eur_mwh"], dtype=float)
    diff = buy - np.asarray(master["sell_eur_mwh"], dtype=float)
    spread = []
    for b in range(len(HOUR_BLOCKS)):
        sel = has & is2 & (block == b) & np.isfinite(diff)
        spread.append(float(np.median(diff[sel])) if sel.any() else 0.0)
    return {"dual_rate": rate, "dual_spread": spread}


def dual_rate_at(tables: dict, hour: int, month: int) -> float:
    return float(tables["dual_rate"][int(_block(np.array([hour]))[0])][int(month) - 1])


def dual_spread_at(tables: dict, hour: int) -> float:
    return float(tables["dual_spread"][int(_block(np.array([hour]))[0])])


def coverage_factor(
    y: np.ndarray, q_lo: np.ndarray, q_mid: np.ndarray, q_hi: np.ndarray, target: float = 0.8
) -> float:
    """Widening about q_mid that would cover the target share of y; 1,0 on fewer than 20 rows."""
    y, lo, mid, hi = (np.asarray(a, dtype=float) for a in (y, q_lo, q_mid, q_hi))
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(mid) & np.isfinite(hi)
    y, lo, mid, hi = y[ok], lo[ok], mid[ok], hi[ok]
    if len(y) < _COV_MIN_ROWS:
        return 1.0
    down, up = mid - lo, hi - mid
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(down > 0, (mid - y) / down, 0.0)
        u = np.where(up > 0, (y - mid) / up, 0.0)
    need = np.maximum(np.maximum(d, u), 0.0)
    return float(np.clip(np.quantile(need, target), *_COV_CLIP))


def apply_coverage(pred: dict[str, np.ndarray], side: str, factor: float) -> None:
    """Widen the side's outer quantiles about its median in place."""
    mid = pred[side + "_q50"]
    for name in ("q10", "q25", "q75", "q90"):
        key = "%s_%s" % (side, name)
        pred[key] = mid + (pred[key] - mid) * factor


def _iso_date(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, ZONE).date().isoformat()


def has_cell(model: PriceModel, tier: str, h: int, elapsed_min: int | None = None) -> bool:
    return (tier, h, elapsed_min if h == 0 else None) in model.models


def _cell(model: PriceModel, tier: str, h: int, elapsed_min: int | None) -> HorizonModel:
    key = (tier, h, elapsed_min if h == 0 else None)
    try:
        return model.models[key]
    except KeyError:
        raise MissingCellError(
            f"no cell for tier {tier}, h {h}, elapsed {key[2]}: fewer than {MIN_ROWS} training rows"
        ) from None


def predict_rows(
    model: PriceModel,
    tier: str,
    h: int,
    X: np.ndarray,
    names: list[str],
    elapsed_min: int | None = None,
) -> dict[str, np.ndarray]:
    hm = _cell(model, tier, h, elapsed_min)
    if list(names) != hm.names:
        raise ValueError("feature names do not match the model")
    Xs = apply_standardise(X, hm.center, hm.scale)
    out: dict[str, np.ndarray] = {}
    for side, weights in (("buy", hm.w_buy), ("sell", hm.w_sell)):
        q = np.sort(np.column_stack([predict_linear(Xs, weights[v]) for v in QUANTILES]), axis=1)
        for i, v in enumerate(QUANTILES):
            out["%s_q%d" % (side, round(v * 100))] = q[:, i]
    for name in _CLASSIFIERS:
        w = getattr(hm, "w_" + name)
        p = np.full(Xs.shape[0], 0.5) if w is None else np.clip(predict_proba(Xs, w), 0.0, 1.0)
        out["p_" + name] = p
    return out


def predict_one(
    model: PriceModel, tier: str, h: int, row: dict[str, float], elapsed_min: int | None = None
) -> Prediction:
    hm = _cell(model, tier, h, elapsed_min)
    X = np.array([[row[n] for n in hm.names]], dtype=np.float64)
    p = predict_rows(model, tier, h, X, hm.names, elapsed_min)
    return Prediction(
        buy={v: float(p["buy_q%d" % round(v * 100)][0]) for v in QUANTILES},
        sell={v: float(p["sell_q%d" % round(v * 100)][0]) for v in QUANTILES},
        p_dual=float(p["p_dual"][0]),
        p_negsell=float(p["p_negsell"][0]),
        p_spike=float(p["p_spike"][0]),
        cov_factor=model.cov_factor(tier, h, 0.5, elapsed_min),
    )


def _enc(a: np.ndarray | None):
    return None if a is None else [float(x) for x in a]


def _dec(a) -> np.ndarray | None:
    return None if a is None else np.array(a, dtype=np.float64)


def to_json(model: PriceModel) -> str:
    cells = {}
    for (tier, h, e), hm in model.models.items():
        cells["%s:%d:%s" % (tier, h, "-" if e is None else e)] = {
            "names": hm.names,
            "center": _enc(hm.center),
            "scale": _enc(hm.scale),
            "w_buy": {repr(q): _enc(w) for q, w in hm.w_buy.items()},
            "w_sell": {repr(q): _enc(w) for q, w in hm.w_sell.items()},
            "w_dual": _enc(hm.w_dual),
            "w_negsell": _enc(hm.w_negsell),
            "w_spike": _enc(hm.w_spike),
            "n_train": hm.n_train,
            "cov_factor": float(hm.cov_factor),
        }
    doc = {"meta": model.meta, "models": cells, "tables": model.tables}
    return json.dumps(doc, allow_nan=False)


def from_json(s: str) -> PriceModel:
    raw = json.loads(s)
    out = PriceModel(meta=raw["meta"])
    for key, c in raw["models"].items():
        tier, h, e = key.split(":")
        w_buy = {float(q): _dec(w) for q, w in c["w_buy"].items()}
        w_sell = {float(q): _dec(w) for q, w in c["w_sell"].items()}
        for w in (w_buy, w_sell):
            _fill_quantiles(w)
        out.models[(tier, int(h), None if e == "-" else int(e))] = HorizonModel(
            names=list(c["names"]),
            center=_dec(c["center"]),
            scale=_dec(c["scale"]),
            w_buy=w_buy,
            w_sell=w_sell,
            w_dual=_dec(c["w_dual"]),
            w_negsell=_dec(c["w_negsell"]),
            w_spike=_dec(c["w_spike"]),
            n_train=int(c["n_train"]),
            cov_factor=float(c.get("cov_factor", 1.0)),
        )
    if "tables" in raw:
        out.tables = raw["tables"]
    else:
        flat = float(out.meta.get("dual_rate_flat", DUAL_RATE_FALLBACK))
        out.tables = {
            "dual_rate": [[flat] * 12 for _ in HOUR_BLOCKS],
            "dual_spread": [0.0] * len(HOUR_BLOCKS),
        }
        out.meta["tables_fallback"] = True
    return out


def _fill_quantiles(w: dict[float, np.ndarray]) -> None:
    """Interpolate q25 and q75 weights of a format 1 cell."""
    if 0.25 not in w:
        w[0.25] = (w[0.1] + w[0.5]) / 2
    if 0.75 not in w:
        w[0.75] = (w[0.5] + w[0.9]) / 2


def _block(hours: np.ndarray) -> np.ndarray:
    blk = np.zeros(len(hours), dtype=np.int64)
    for i, (lo, hi) in enumerate(_HOUR_BLOCKS):
        blk[(hours >= lo) & (hours < hi)] = i
    return blk


def _qs(v: np.ndarray) -> np.ndarray:
    return np.quantile(v, QUANTILES)


class TableBaseline:
    """Day-ahead decile x local hour block quantile table (the owner's earlier baseline)."""

    def fit(self, master: Table, train_mask: np.ndarray) -> TableBaseline:
        da = np.asarray(master["da_eur_mwh"], dtype=float)
        block = _block(local_calendar(np.asarray(master["ts"]))["hour"])
        m = np.asarray(train_mask, dtype=bool) & np.isfinite(da)
        self.edges = np.unique(np.quantile(da[m], np.linspace(0.0, 1.0, 11)[1:-1]))
        dec = np.searchsorted(self.edges, da[m], side="left")
        self.cells: dict[str, dict[tuple[int, int], np.ndarray]] = {}
        self.fallback: dict[str, np.ndarray] = {}
        for side in ("buy", "sell"):
            y = np.asarray(master[side + "_eur_mwh"], dtype=float)[m]
            ok = np.isfinite(y)
            self.fallback[side] = _qs(y[ok])
            cells = {}
            for d in np.unique(dec[ok]):
                for b in range(len(_HOUR_BLOCKS)):
                    sel = ok & (dec == d) & (block[m] == b)
                    if sel.any():
                        cells[(int(d), b)] = _qs(y[sel])
            self.cells[side] = cells
        return self

    def predict_rows(self, master: Table, idx: np.ndarray) -> dict[str, np.ndarray]:
        idx = np.asarray(idx)
        da = np.asarray(master["da_eur_mwh"], dtype=float)[idx]
        block = _block(local_calendar(np.asarray(master["ts"])[idx])["hour"])
        dec = np.searchsorted(self.edges, da, side="left")
        out: dict[str, np.ndarray] = {}
        for side in ("buy", "sell"):
            q = np.tile(self.fallback[side], (len(idx), 1))
            for i in range(len(idx)):
                if np.isfinite(da[i]):
                    q[i] = self.cells[side].get((int(dec[i]), int(block[i])), q[i])
            for j, v in enumerate(QUANTILES):
                out["%s_q%d" % (side, round(v * 100))] = q[:, j]
        return out


def shipped_model() -> PriceModel:
    """The packaged NL market champion (models/market_nl.json).

    Resolved from this file's path, not importlib.resources: under Home Assistant the package
    imports as custom_components.betterybalance, so a resource name would differ.
    """
    path = Path(__file__).resolve().parent.parent / "models" / "market_nl.json"
    return from_json(path.read_text(encoding="utf-8"))
