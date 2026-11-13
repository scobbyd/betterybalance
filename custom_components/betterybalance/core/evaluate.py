"""Walk-forward evaluation of the tiered price model, decay curve and markdown report."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from .features import ELAPSED_MIN, HORIZONS, TIERS, build_matrix
from .fmt import eu
from .pricemodel import QUANTILES, TableBaseline, fit_model, has_cell, predict_rows
from .quantile import pinball
from .timegrid import ZONE, Table

_TARGETS = ("buy", "sell")
_AUCS = ("dual", "negsell", "spike")
_QNAMES = tuple("q%d" % round(q * 100) for q in QUANTILES)
_NAIVES = ("naive_da", "naive_persist")


@dataclass
class Fold:
    train: np.ndarray
    test: np.ndarray


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat))) if len(y) else float("nan")


def auc(y_bool: np.ndarray, score: np.ndarray) -> float:
    """Rank-based area under the ROC curve, ties averaged. NaN when a class is absent."""
    y = np.asarray(y_bool, dtype=bool)
    s = np.asarray(score, dtype=float)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    ranks = (start + (counts + 1) / 2.0)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def wrong_side_rate(y: np.ndarray, yhat: np.ndarray, da: np.ndarray) -> float:
    """Share of rows where the prediction and the outcome lie on opposite sides of da."""
    y = np.asarray(y, dtype=float)
    keep = y != np.asarray(da, dtype=float)
    if not keep.any():
        return float("nan")
    da = np.asarray(da, dtype=float)[keep]
    wrong = np.sign(np.asarray(yhat, dtype=float)[keep] - da) != np.sign(y[keep] - da)
    return float(wrong.sum() / len(wrong))


def _plus_local_days(ts: int, days: int) -> int:
    """Epoch s of the same local wall-clock time `days` calendar days later (Europe/Amsterdam)."""
    local = dt.datetime.fromtimestamp(int(ts), ZONE).replace(tzinfo=None)
    return int((local + dt.timedelta(days=days)).replace(tzinfo=ZONE).timestamp())


def walk_forward(
    ts: np.ndarray,
    test_start: int,
    test_end: int,
    refit_every_days: int = 7,
    train_days: int = 365,
) -> list[Fold]:
    """Test blocks of refit_every_days local calendar days from test_start, clipped at test_end.

    Boundaries keep the local wall-clock time of test_start, so a local midnight start gives local
    midnight boundaries and a DST day holds 92 or 100 ISPs. Each block trains on the ISPs from
    train_days local days before its start up to its start.
    """
    ts = np.asarray(ts, dtype=np.int64)
    folds: list[Fold] = []
    i = 0
    start = int(test_start)
    while start < test_end:
        nxt = _plus_local_days(int(test_start), (i + 1) * refit_every_days)
        end = min(nxt, int(test_end))
        lo = _plus_local_days(start, -train_days)
        train = np.flatnonzero((ts >= lo) & (ts < start))
        test = np.flatnonzero((ts >= start) & (ts < end))
        if len(train) and len(test):
            folds.append(Fold(train=train, test=test))
        i += 1
        start = nxt
    return folds


def _shift(arr: np.ndarray, s: int) -> np.ndarray:
    """out[k] = arr[k - s], NaN outside the array."""
    out = np.full(len(arr), np.nan)
    if s >= 0:
        if s < len(arr):
            out[s:] = arr[: len(arr) - s]
    elif -s < len(arr):
        out[:s] = arr[-s:]
    return out


class _Pool:
    """Predictions of one cell, pooled over folds."""

    def __init__(self) -> None:
        self.parts: dict[str, list[np.ndarray]] = {}

    def add(self, **arrays: np.ndarray) -> None:
        for k, v in arrays.items():
            self.parts.setdefault(k, []).append(np.asarray(v))

    def get(self, key: str) -> np.ndarray:
        return np.concatenate(self.parts[key])


def _persist(master: Table, h: int, e: int | None, k: np.ndarray) -> dict[str, np.ndarray]:
    """Running price known at forecast time.

    For h >= 1 this is the minute-1 partial mean of the forecast ISP k - h. The end-of-ISP
    value of that ISP is not public yet. For h == 0 it is the partial mean of ISP k at the
    elapsed minute.
    """
    out = {}
    for side in _TARGETS:
        if h >= 1:
            src = _shift(np.asarray(master[side + "_live_e1"], dtype=float), h)
        else:
            src = np.asarray(master["%s_live_e%d" % (side, e)], dtype=float)
        out[side] = src[k]
    return out


def _fold_maes(y, yhat, fold) -> list[float]:
    return [mae(y[fold == f], yhat[fold == f]) for f in np.unique(fold)]


def _target_scores(y, preds, da, fold) -> dict:
    out = {
        "mae_q50": mae(y, preds["q50"]),
        "n": int(len(y)),
        "wrong_side": wrong_side_rate(y, preds["q50"], da),
        "folds": _fold_maes(y, preds["q50"], fold),
    }
    for name, q in zip(_QNAMES, QUANTILES, strict=True):
        out["pinball_" + name] = pinball(y, preds[name], q)
    return out


def _naive_scores(y, yhat, da, fold, wrong_side=True) -> dict:
    return {
        "mae": mae(y, yhat),
        "n": int(len(y)),
        "wrong_side": wrong_side_rate(y, yhat, da) if wrong_side else float("nan"),
        "folds": _fold_maes(y, yhat, fold),
    }


def _put(results: dict, group: str, h: int, e: int | None, value: dict) -> None:
    node = results.setdefault(group, {}).setdefault(h, {})
    if h == 0:
        node = node.setdefault(e, {})
    node.update(value)


def _preds(pool: _Pool, side: str, sel: np.ndarray) -> dict[str, np.ndarray]:
    return {n: pool.get("%s_%s" % (side, n))[sel] for n in _QNAMES}


def evaluate(master: Table, folds: list[Fold], tiers=TIERS, horizons=HORIZONS) -> dict:
    """Walk-forward scores per tier, horizon and target, plus naive and table baselines.

    Per horizon (and elapsed minute at h == 0) every column is scored on one common row set:
    the rows every evaluated tier predicts, where both naive inputs are finite. Each tier also
    keeps its own-row MAE under "own_rows".
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    da_all = np.asarray(master["da_eur_mwh"], dtype=float)
    y_all = {s: np.asarray(master[s + "_eur_mwh"], dtype=float) for s in _TARGETS}
    cells = [(h, e) for h in horizons for e in (ELAPSED_MIN if h == 0 else (None,))]
    cache: dict[tuple, object] = {}

    def matrix(tier: str, h: int, e: int | None):
        key = (tier, h, e)
        if key not in cache:
            cache[key] = build_matrix(master, tier, h, e)
        return cache[key]

    model_pool: dict[tuple, _Pool] = {}
    base_pool: dict[int, _Pool] = {}
    skipped = 0
    for fi, fold in enumerate(folds):
        train_mask = np.zeros(len(ts), dtype=bool)
        train_mask[fold.train] = True
        test_mask = np.zeros(len(ts), dtype=bool)
        test_mask[fold.test] = True
        model = fit_model(master, tiers, horizons, train_mask=train_mask)
        base = TableBaseline().fit(master, train_mask)
        for h, e in cells:
            for tier in tiers:
                mx = matrix(tier, h, e)
                k_all = np.searchsorted(ts, mx.ts_target)
                sel = test_mask[k_all]
                if not sel.any():
                    continue
                if not has_cell(model, tier, h, e):
                    skipped += 1
                    continue
                p = predict_rows(model, tier, h, mx.X[sel], mx.names, e)
                pool = model_pool.setdefault((tier, h, e), _Pool())
                pool.add(
                    k=k_all[sel],
                    fold=np.full(int(sel.sum()), fi),
                    l_dual=mx.y_dual[sel],
                    l_negsell=mx.y_negsell[sel],
                    l_spike=mx.y_spike[sel],
                    p_dual=p["p_dual"],
                    p_negsell=p["p_negsell"],
                    p_spike=p["p_spike"],
                )
                for side in _TARGETS:
                    for name in _QNAMES:
                        pool.add(**{"%s_%s" % (side, name): p["%s_%s" % (side, name)]})
            if h >= 1:
                # tier A rows are a superset of every other tier's rows
                mx = matrix("A", h, e)
                k_all = np.searchsorted(ts, mx.ts_target)
                k = k_all[test_mask[k_all]]
                if len(k):
                    bp = base.predict_rows(master, k)
                    bpool = base_pool.setdefault(h, _Pool())
                    bpool.add(k=k)
                    for side in _TARGETS:
                        for name in _QNAMES:
                            bpool.add(**{"%s_%s" % (side, name): bp["%s_%s" % (side, name)]})

    results: dict = {"skipped_cells": skipped}
    for h, e in cells:
        pools = {t: model_pool[(t, h, e)] for t in tiers if (t, h, e) in model_pool}
        if not pools:
            continue
        common = None
        for pool in pools.values():
            kk = pool.get("k")
            common = kk if common is None else np.intersect1d(common, kk)
        per = _persist(master, h, e, common)
        common = common[np.isfinite(per["buy"]) & np.isfinite(per["sell"])]
        da = da_all[common]
        for tier, pool in pools.items():
            sel = np.isin(pool.get("k"), common)
            fold = pool.get("fold")[sel]
            value: dict = {}
            for side in _TARGETS:
                kk = pool.get("k")[sel]
                y = y_all[side][kk]
                value[side] = _target_scores(y, _preds(pool, side, sel), da_all[kk], fold)
                own = pool.get("k")
                value[side]["own_rows"] = {
                    "mae_q50": mae(y_all[side][own], pool.get("%s_q50" % side)),
                    "n": int(len(own)),
                }
            for a in _AUCS:
                value["auc_" + a] = auc(pool.get("l_" + a)[sel], pool.get("p_" + a)[sel])
            _put(results, tier, h, e, value)
        first = next(iter(pools.values()))
        fold_of = np.full(len(ts), -1, dtype=np.int64)
        fold_of[first.get("k")] = first.get("fold")
        fold = fold_of[common]
        per = _persist(master, h, e, common)
        da_val, per_val = {}, {}
        for side in _TARGETS:
            y = y_all[side][common]
            da_val[side] = _naive_scores(y, da, da, fold, wrong_side=False)
            per_val[side] = _naive_scores(y, per[side], da, fold)
        _put(results, "naive_da", h, e, da_val)
        _put(results, "naive_persist", h, e, per_val)
        if h >= 1 and h in base_pool:
            bpool = base_pool[h]
            sel = np.isin(bpool.get("k"), common)
            bk = bpool.get("k")[sel]
            value = {}
            for side in _TARGETS:
                value[side] = _target_scores(
                    y_all[side][bk], _preds(bpool, side, sel), da_all[bk], fold_of[bk]
                )
            _put(results, "baseline", h, None, value)
    return results


def _tiers_in(results: dict) -> list[str]:
    return [t for t in TIERS if t in results]


def _cell(results: dict, group: str, h: int, e: int | None = None) -> dict:
    node = results.get(group, {}).get(h, {})
    return node.get(e, {}) if h == 0 else node


def _skill(results: dict, tier: str, h: int, e: int | None) -> float:
    m = _cell(results, tier, h, e).get("buy", {}).get("mae_q50", float("nan"))
    ref = _cell(results, "A", h, e).get("buy", {}).get("mae_q50", float("nan"))
    return 1.0 - m / ref if ref and np.isfinite(ref) else float("nan")


def decay_curve(results: dict) -> dict[str, dict]:
    """skill = 1 - buy q50 MAE of the tier / buy q50 MAE of tier A.

    skill[tier][h] for h >= 1 and skill[tier][0][e] per elapsed minute e at h == 0.
    """
    out: dict[str, dict] = {}
    for tier in _tiers_in(results):
        for h in sorted(results[tier]):
            if h == 0:
                out.setdefault(tier, {})[0] = {
                    e: _skill(results, tier, 0, e) for e in sorted(results[tier][0])
                }
            else:
                out.setdefault(tier, {})[h] = _skill(results, tier, h, None)
    return out


def _decay_at(decay: dict, tier: str, h: int, e: int | None) -> float:
    v = decay.get(tier, {}).get(h, float("nan"))
    if h == 0:
        v = v.get(e, float("nan")) if isinstance(v, dict) else float("nan")
    return v


def provisional_gap(master: Table) -> dict:
    out: dict = {}
    for side in _TARGETS:
        live = np.asarray(master[side + "_live_end"], dtype=float)
        settled = np.asarray(master[side + "_eur_mwh"], dtype=float)
        ok = np.isfinite(live) & np.isfinite(settled)
        d = (live - settled)[ok]
        out[side] = {
            "mean": float(d.mean()) if len(d) else float("nan"),
            "median": float(np.median(d)) if len(d) else float("nan"),
            "n": int(len(d)),
        }
    return out


def _row_labels(results: dict) -> list[tuple[str, int, int | None]]:
    hs = sorted({h for g in results if g != "skipped_cells" for h in results[g]})
    rows = []
    for h in hs:
        if h == 0:
            rows.extend((f"0 ({e} min)", 0, e) for e in ELAPSED_MIN)
        else:
            rows.append((str(h), h, None))
    return rows


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(r) + " |" for r in rows)
    return lines


def _n(results: dict, h: int, e: int | None) -> str:
    n = _cell(results, "naive_da", h, e).get("buy", {}).get("n")
    return eu(float("nan") if n is None else n, 0)


def _compare_table(results: dict, target: str, metric: str, naive_metric: str, dec: int):
    tiers = _tiers_in(results)
    header = ["h", *tiers, "naive_da", "naive_persist", "baseline", "n"]
    rows = []
    for label, h, e in _row_labels(results):
        r = [label]
        for g in (*tiers, *_NAIVES, "baseline"):
            m = naive_metric if g in _NAIVES else metric
            v = _cell(results, g, h, e).get(target, {}).get(m, float("nan"))
            r.append(eu(v, dec))
        r.append(_n(results, h, e))
        rows.append(r)
    return _table(header, rows)


def render_markdown(results: dict, decay: dict, gap: dict, meta: dict) -> str:
    out = ["# Price model evaluation", ""]
    for k, v in meta.items():
        out.append(f"- {k}: {v}")
    out += [
        "",
        "Every column of a horizon row is scored on the same rows. These are the rows that all "
        "tiers predict and where the live running price exists. The column n gives their count.",
        "",
    ]
    for target in _TARGETS:
        out += [
            f"Mean absolute error of the {target} price median in EUR/MWh per horizon h in ISPs. "
            "Lower is better.",
            "",
            *_compare_table(results, target, "mae_q50", "mae", 2),
            "",
        ]
    for target in _TARGETS:
        out += [
            f"Share of rows where the {target} forecast lies on the wrong side of the day-ahead "
            "price. Lower is better, and the naive day-ahead column has no side.",
            "",
            *_compare_table(results, target, "wrong_side", "wrong_side", 3),
            "",
        ]
    for tier in _tiers_in(results):
        rows = []
        for label, h, e in _row_labels(results):
            c = _cell(results, tier, h, e)
            rows.append(
                [label, *(eu(c.get("auc_" + a, float("nan")), 3) for a in _AUCS), _n(results, h, e)]
            )
        out += [
            f"Area under the ROC curve of tier {tier} for three events. 0,5 is chance and 1 is "
            "perfect.",
            "",
            *_table(["h", *_AUCS, "n"], rows),
            "",
        ]
    tiers = list(decay)
    rows = []
    for label, h, e in _row_labels(results):
        rows.append([label, *(eu(_decay_at(decay, t, h, e), 3) for t in tiers), _n(results, h, e)])
    out += [
        "Skill is one minus the buy MAE of the tier divided by the buy MAE of tier A. "
        "Positive means the tier beats the day-ahead-only tier.",
        "",
        *_table(["h", *tiers, "n"], rows),
        "",
    ]
    b, s = gap["buy"], gap["sell"]
    out += [
        "Live feed value minus settled value in EUR/MWh over rows where both exist.",
        "",
        f"Provisional gap buy: mean {eu(b['mean'])}, median {eu(b['median'])}, n {eu(b['n'], 0)}. "
        f"Sell: mean {eu(s['mean'])}, median {eu(s['median'])}, n {eu(s['n'], 0)}.",
        "",
        f"Skipped cells (too few training rows): {eu(results.get('skipped_cells', 0), 0)}.",
        "",
    ]
    return "\n".join(out)
