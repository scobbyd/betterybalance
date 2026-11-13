"""The October 2026 price study (spec section 12, drop 1).

Question: at the horizons a battery trader acts on, does each fresh information source lower the
error of the settled imbalance price prediction over the day-ahead plus calendar baseline?

Inputs (data_dir):
- master_inputs.npz from tools/fetch_history.py: day-ahead, settled imbalance, IDA1 to IDA3.
- live_minutes.csv from tools/vm_live_archive.py: the owner's live minute archive.

Minute convention of the live archive (tools/vm_live_archive.py): a minute labelled t holds the
samples taken in [t, t + 60) and is known at t + 60.

Run: python -m tools.study.october_2026 --data data/ --report docs/reports/<date>-....md
Re-render only: python -m tools.study.october_2026 --render-only data/results/october_2026.json
--report docs/reports/<date>-....md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from betterybalance.core.evaluate import (
    Fold,
    decay_curve,
    evaluate,
    mae,
    provisional_gap,
    walk_forward,
)
from betterybalance.core.features import ELAPSED_MIN, HORIZONS, TIERS, build_matrix
from betterybalance.core.feeds.ida import AUCTIONS
from betterybalance.core.feeds.live import load_live_minutes, stale_mask
from betterybalance.core.fmt import eu
from betterybalance.core.inputs import LIVE_FILE, NPZ_FILE, load_inputs  # noqa: F401 (re-export)
from betterybalance.core.pricemodel import MIN_ROWS
from betterybalance.core.timegrid import ZONE, Table, local_calendar

from tools.study.report import load_results, pct, render_report, restore_keys, rule_sentence

HEADROOM_HORIZONS = (0, 1, 2, 3, 4)
READING_HORIZONS = (1, 2, 3, 4)
LOAD_BEARING_GAIN = 0.05  # reading rule: minimum mean buy MAE gain over h1 to h4
ADOPT_MIN_GAIN = 0.05  # spec 8.3: minimum mean MAE gain over h0 to h4 to adopt boosting
# The owner's earlier unpublished baseline study (q50 pinball 27,4 EUR/MWh, 2023 to 2026)
# implies an MAE near 55 EUR/MWh. "Same order of magnitude" is read as a factor of 3 either way.
ANCHOR_MAE = 55.0
ANCHOR_FACTOR = 3.0
_TARGETS = ("buy", "sell")


# --- inputs ---------------------------------------------------------------------------------


def local_midnight(day: str) -> int:
    """Epoch s of 00:00 Europe/Amsterdam on an ISO day."""
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=ZONE).timestamp())


def _local_day(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date().isoformat()


def live_archive_stats(data_dir: Path) -> dict:
    path = Path(data_dir) / LIVE_FILE
    if not path.exists():
        return {"rows": 0}
    t = load_live_minutes(path)
    valid = ~(np.isnan(t["buy_live"]) | np.isnan(t["sell_live"]))
    stale = stale_mask(t)
    return {
        "rows": int(len(t["ts"])),
        "first": _iso_utc(int(t["ts"].min())),
        "last": _iso_utc(int(t["ts"].max())),
        "valid_share": float(valid.mean()),
        "stale_share": float(stale.sum() / max(int(valid.sum()), 1)),
    }


def _iso_utc(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def coverage(master: Table, start: int, mid: int, end: int) -> dict:
    """Share of finite values per feed in the training span [start, mid) and the test [mid, end).

    IDA3 covers the afternoon only, so its share counts slots from 12:00 local.
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    hour = local_calendar(ts)["hour"]
    cols = {
        "day-ahead": "da_eur_mwh",
        "settled buy": "buy_eur_mwh",
        "IDA1": "ida1_eur_mwh",
        "IDA2": "ida2_eur_mwh",
        "IDA3": "ida3_eur_mwh",
        "live end-of-ISP buy": "buy_live_end",
    }
    out = {}
    for label, col in cols.items():
        vals = np.isfinite(np.asarray(master[col], dtype=float))
        scope = hour >= 12 if label == "IDA3" else np.ones(len(ts), dtype=bool)
        row = {}
        for name, lo, hi in (("train", start, mid), ("test", mid, end)):
            sel = scope & (ts >= lo) & (ts < hi)
            row[name] = float(vals[sel].mean()) if sel.any() else float("nan")
        first = np.flatnonzero(vals & (ts >= start))
        row["first"] = _local_day(int(ts[first[0]])) if len(first) else None
        out[label] = row
    return out


# --- cells, rows and the boosted headroom ---------------------------------------------------


def _cells(horizons) -> list[tuple[int, int | None]]:
    return [(h, e) for h in horizons for e in (ELAPSED_MIN if h == 0 else (None,))]


def _masks(n: int, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(n, dtype=bool)
    train[fold.train] = True
    test = np.zeros(n, dtype=bool)
    test[fold.test] = True
    return train, test


def cell_info(master: Table, folds: list[Fold], tiers, horizons) -> dict:
    """Per (tier, h, e): training rows per fold, predicted test rows and Matrix.dropped.

    Mirrors fit_model and evaluate: a fold predicts a cell only if it has MIN_ROWS training rows.
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    masks = [_masks(len(ts), f) for f in folds]
    out = {}
    for tier in tiers:
        for h, e in _cells(horizons):
            mx = build_matrix(master, tier, h, e)
            k = np.searchsorted(ts, mx.ts_target)
            n_train, test_k = [], []
            for train, test in masks:
                n = int(train[k].sum())
                n_train.append(n)
                if n >= MIN_ROWS:
                    test_k.append(k[test[k]])
            out[(tier, h, e)] = {
                "n_train": n_train,
                "test_k": np.concatenate(test_k) if test_k else np.zeros(0, dtype=np.int64),
                "dropped": dict(mx.dropped),
                "kept": int(len(k)),
            }
    return out


def _shift(arr: np.ndarray, s: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if s < len(arr):
        out[s:] = arr[: len(arr) - s]
    return out


def common_rows(master: Table, info: dict, tiers, h: int, e: int | None) -> np.ndarray:
    """The common row set evaluate scores a horizon on: every tier's rows, finite persistence."""
    common = None
    for tier in tiers:
        k = info[(tier, h, e)]["test_k"]
        if not len(k):
            continue
        common = k if common is None else np.intersect1d(common, k)
    if common is None:
        return np.zeros(0, dtype=np.int64)
    common = np.unique(common)
    ok = np.ones(len(common), dtype=bool)
    for side in _TARGETS:
        if h >= 1:
            src = _shift(np.asarray(master[side + "_live_e1"], dtype=float), h)
        else:
            src = np.asarray(master["%s_live_e%d" % (side, e)], dtype=float)
        ok &= np.isfinite(src[common])
    return common[ok]


def boosted_headroom(
    master: Table,
    folds: list[Fold],
    tier: str = "D",
    horizons=HEADROOM_HORIZONS,
    rows: dict[int, np.ndarray] | None = None,
) -> dict[int, float]:
    """Buy MAE of a gradient-boosted median model on the tier's matrices, per horizon.

    One HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05)
    per fold, trained on the fold's training rows. h0 uses elapsed minute 1. rows[h] (master
    indices) limits the scored test rows to the linear tier's common row set.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    ts = np.asarray(master["ts"], dtype=np.int64)
    out: dict[int, float] = {}
    for h in horizons:
        e = ELAPSED_MIN[0] if h == 0 else None
        mx = build_matrix(master, tier, h, e)
        k = np.searchsorted(ts, mx.ts_target)
        scope = np.ones(len(ts), dtype=bool)
        if rows is not None:
            scope = np.zeros(len(ts), dtype=bool)
            scope[rows[h]] = True
        y_true, y_hat = [], []
        for fold in folds:
            train, test = _masks(len(ts), fold)
            sel_tr = train[k]
            sel_te = test[k] & scope[k]
            if int(sel_tr.sum()) < MIN_ROWS or not sel_te.any():
                continue
            model = HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=300, learning_rate=0.05, random_state=0
            )
            model.fit(mx.X[sel_tr], mx.y_buy[sel_tr])
            y_hat.append(model.predict(mx.X[sel_te]))
            y_true.append(mx.y_buy[sel_te])
        out[h] = mae(np.concatenate(y_true), np.concatenate(y_hat)) if y_true else float("nan")
    return out


def adopt_boosting(
    linear_mae: dict[int, float], boosted_mae: dict[int, float], min_gain: float = ADOPT_MIN_GAIN
) -> bool:
    """True only if the mean over h0 to h4 of 1 - boosted / linear is at least min_gain.

    A missing or non-finite horizon means no adoption.
    """
    gains = []
    for h in HEADROOM_HORIZONS:
        lin, boo = linear_mae.get(h), boosted_mae.get(h)
        if lin is None or boo is None or not (math.isfinite(lin) and math.isfinite(boo)):
            return False
        if lin <= 0:
            return False
        gains.append(1.0 - boo / lin)
    return bool(float(np.mean(gains)) >= min_gain)


# --- the study ------------------------------------------------------------------------------


def _buy_mae(results: dict, tier: str, h: int) -> float:
    node = results.get(tier, {}).get(h, {})
    if h == 0:
        node = node.get(ELAPSED_MIN[0], {})
    return float(node.get("buy", {}).get("mae_q50", float("nan")))


def _buy_n(results: dict, tier: str, h: int) -> int | None:
    node = results.get(tier, {}).get(h, {})
    if h == 0:
        node = node.get(ELAPSED_MIN[0], {})
    return node.get("buy", {}).get("n")


def relative_gain(results: dict, tier: str, ref: str, horizons=READING_HORIZONS) -> dict:
    """Mean over the horizons of 1 - buy q50 MAE(tier) / buy q50 MAE(ref), on common rows.

    An incomplete horizon set gives mean NaN and load_bearing None (no verdict).
    """
    per = {}
    for h in horizons:
        m, r = _buy_mae(results, tier, h), _buy_mae(results, ref, h)
        per[h] = 1.0 - m / r if math.isfinite(m) and math.isfinite(r) and r > 0 else float("nan")
    vals = [v for v in per.values() if math.isfinite(v)]
    complete = len(vals) == len(per) and len(per) > 0
    mean = float(np.mean(vals)) if complete else float("nan")
    verdict = bool(mean >= LOAD_BEARING_GAIN) if complete else None
    return {"per_h": per, "mean": mean, "load_bearing": verdict}


def clip_to_first(folds: list[Fold], ts: np.ndarray, first_ts: int) -> list[Fold]:
    """Folds whose training rows start no earlier than first_ts (test rows unchanged)."""
    ts = np.asarray(ts, dtype=np.int64)
    return [Fold(train=f.train[ts[f.train] >= first_ts], test=f.test) for f in folds]


def _first_finite_ts(master: Table, cols) -> int | None:
    """Latest over the columns of each column's first finite timestamp."""
    ts = np.asarray(master["ts"], dtype=np.int64)
    firsts = []
    for c in cols:
        idx = np.flatnonzero(np.isfinite(np.asarray(master[c], dtype=float)))
        if not len(idx):
            return None
        firsts.append(int(ts[idx[0]]))
    return max(firsts)


def ida_window(master: Table, train_lo: int) -> tuple[dict, int | None]:
    """IDA start against the fold-0 train start, at local-day granularity.

    The IDA start is the local midnight of the day on which the last of the three auctions has its
    first value (IDA3 clears afternoon slots only, so its first value is at 12:00). Returns the
    report facts and the timestamp to clip training rows to, or None when no clip is needed.
    """
    first = _first_finite_ts(master, [a.lower() + "_eur_mwh" for a in AUCTIONS])
    day = None if first is None else _local_day(first)
    clip = None if day is None else local_midnight(day)
    clipped = bool(clip is not None and clip > train_lo)
    info = {"first_day": day, "train_start_day": _local_day(train_lo), "clipped": clipped}
    return info, clip if clipped else None


def data_facts(master: Table, folds: list[Fold], training: dict) -> dict:
    """Dates and spans the report text states, all read from the data."""
    ts = np.asarray(master["ts"], dtype=np.int64)
    buy = np.isfinite(np.asarray(master["buy_eur_mwh"], dtype=float))
    live = np.isfinite(np.asarray(master["buy_live_end"], dtype=float))
    last_label = _local_day(int(ts[np.flatnonzero(buy)[-1]])) if buy.any() else None
    master_last = _local_day(int(ts[-1]))
    missing_from = None
    if last_label is not None and last_label < master_last:
        missing_from = (dt.date.fromisoformat(last_label) + dt.timedelta(days=1)).isoformat()
    days = {}
    for tier, t in training.items():
        per = [n / 96.0 for n in t["per_fold"]]
        days[tier] = {"min": min(per), "max": max(per)} if per else None
    return {
        "master_first_day": _local_day(int(ts[0])),
        "master_last_day": master_last,
        "label_last_day": last_label,
        "label_missing_from": missing_from,
        "live_first_day": _local_day(int(ts[np.flatnonzero(live)[0]])) if live.any() else None,
        "train_start_day": _local_day(int(ts[folds[0].train[0]])) if folds else None,
        "train_days_per_tier": days,
    }


def _slice(master: Table, lo: int, hi: int) -> Table:
    ts = np.asarray(master["ts"], dtype=np.int64)
    sel = (ts >= lo) & (ts < hi)
    return {k: np.asarray(v)[sel] for k, v in master.items()}


def run_study(
    master: Table,
    test_start: str = "2026-10-01",
    test_end: str = "2026-11-01",
    train_days: int = 365,
    refit_every_days: int = 7,
    tiers=TIERS,
    horizons=HORIZONS,
    headroom: bool = True,
) -> dict:
    """Walk-forward evaluation over the test window plus every derived table of the report."""
    tiers = [t for t in TIERS if t in tiers]
    horizons = [h for h in HORIZONS if h in horizons]
    t0, t1 = local_midnight(test_start), local_midnight(test_end)
    ts = np.asarray(master["ts"], dtype=np.int64)
    folds = walk_forward(ts, t0, t1, refit_every_days, train_days)
    train_lo = int(ts[folds[0].train[0]]) if folds else t0
    ida, clip = ida_window(master, train_lo)
    if clip is not None:
        # evaluate shares one fold set across tiers: clip every tier, so B against A stays like
        # for like on the same training span
        folds = clip_to_first(folds, ts, clip)
    timing = {}
    clock = time.monotonic()
    results = evaluate(master, folds, tiers, horizons)
    timing["evaluate_s"] = time.monotonic() - clock
    clock = time.monotonic()
    info = cell_info(master, folds, tiers, horizons)
    common = {(h, e): common_rows(master, info, tiers, h, e) for h, e in _cells(horizons)}
    timing["cells_s"] = time.monotonic() - clock

    fold_rows = []
    for f in folds:
        fold_rows.append(
            {
                "train_first": _local_day(int(ts[f.train[0]])),
                "train_last": _local_day(int(ts[f.train[-1]])),
                "test_first": _local_day(int(ts[f.test[0]])),
                "test_last": _local_day(int(ts[f.test[-1]])),
                "test_slots": int(len(f.test)),
            }
        )
    n_check = {}
    for h, e in _cells(horizons):
        n_eval = results.get("naive_da", {}).get(h, {})
        n_eval = (n_eval.get(e, {}) if h == 0 else n_eval).get("buy", {}).get("n")
        n_check["%d:%s" % (h, "-" if e is None else e)] = {
            "evaluate": n_eval,
            "study": int(len(common[(h, e)])),
        }

    training = {}
    for tier in tiers:
        ns = [n for (t, _, _), c in info.items() if t == tier for n in c["n_train"]]
        ref_h = 1 if 1 in horizons else horizons[0]
        ref_e = ELAPSED_MIN[0] if ref_h == 0 else None
        training[tier] = {
            "ref_h": ref_h,
            "per_fold": info[(tier, ref_h, ref_e)]["n_train"],
            "min": int(min(ns)),
            "max": int(max(ns)),
        }
    dropped = {}
    test_slots = int(((ts >= t0) & (ts < t1)).sum())
    for tier in tiers:
        for h, e in _cells(horizons):
            c = info[(tier, h, e)]
            dropped["%s:%d:%s" % (tier, h, "-" if e is None else e)] = {
                **c["dropped"],
                "kept": c["kept"],
                "test_predicted": int(len(c["test_k"])),
                "test_slots": test_slots,
            }

    reading = {}
    for i, tier in enumerate(tiers):
        if tier == "A":
            continue
        prev = tiers[i - 1]
        reading[tier] = {
            "ref": prev,
            **relative_gain(results, tier, prev),
            "over_A": relative_gain(results, tier, "A") if "A" in tiers else None,
        }
    if "D" in tiers and "C" in tiers:
        reading["D_h0_h2"] = relative_gain(results, "D", "C", horizons=(0, 1, 2))
    if "D" in tiers and "B" in tiers:
        reading["D_over_B_h0_h2"] = relative_gain(results, "D", "B", horizons=(0, 1, 2))
    anchor_mae = _buy_mae(results, "A", 1) if "A" in tiers and 1 in horizons else float("nan")
    anchor = {
        "tier_a_buy_mae_h1": anchor_mae,
        "expected": ANCHOR_MAE,
        "ratio": anchor_mae / ANCHOR_MAE,
        "in_range": bool(
            math.isfinite(anchor_mae)
            and ANCHOR_MAE / ANCHOR_FACTOR <= anchor_mae <= ANCHOR_MAE * ANCHOR_FACTOR
        ),
    }

    head = None
    hh = [h for h in HEADROOM_HORIZONS if h in horizons]
    if headroom and "D" in tiers and hh:
        clock = time.monotonic()
        rows = {h: common[(h, ELAPSED_MIN[0] if h == 0 else None)] for h in hh}
        boosted = boosted_headroom(master, folds, "D", hh, rows=rows)
        linear = {h: _buy_mae(results, "D", h) for h in hh}
        gains = {h: 1.0 - boosted[h] / linear[h] if linear[h] > 0 else float("nan") for h in hh}
        finite = [g for g in gains.values() if math.isfinite(g)]
        head = {
            "linear": linear,
            "boosted": boosted,
            "n": {h: int(len(rows[h])) for h in hh},
            "n_linear": {h: _buy_n(results, "D", h) for h in hh},
            "gain": gains,
            "mean_gain": float(np.mean(finite)) if len(finite) == len(hh) else float("nan"),
            "complete": hh == list(HEADROOM_HORIZONS),
            "adopt": adopt_boosting(linear, boosted, ADOPT_MIN_GAIN),
        }
        timing["headroom_s"] = time.monotonic() - clock

    train_start = int(ts[folds[0].train[0]]) if folds else t0
    return {
        "test_start": test_start,
        "test_end": test_end,
        "master_rows": int(len(ts)),
        "test_last_day": _local_day(t1 - 1),
        "train_days": train_days,
        "refit_every_days": refit_every_days,
        "tiers": tiers,
        "horizons": horizons,
        "folds": fold_rows,
        "results": results,
        "decay": decay_curve(results),
        "gap_test": provisional_gap(_slice(master, t0, t1)),
        "gap_all": provisional_gap(master),
        "coverage": coverage(master, train_start, t0, t1),
        "training": training,
        "dropped": dropped,
        "n_check": n_check,
        "reading": reading,
        "ida": ida,
        "rules": {
            "load_bearing_gain": LOAD_BEARING_GAIN,
            "adopt_min_gain": ADOPT_MIN_GAIN,
            "min_rows": MIN_ROWS,
        },
        "facts": data_facts(master, folds, training),
        "anchor": anchor,
        "headroom": head,
        "timing": timing,
    }


# --- JSON and CLI ---------------------------------------------------------------------------


def jsonable(x):
    """Nested dicts and lists of plain JSON values: numpy to lists, NaN to None, keys to str."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    return x


def _ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--tiers", default=",".join(TIERS))
    p.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    p.add_argument("--test-start", default="2026-10-01")
    p.add_argument("--test-end", default="2026-10-29")
    p.add_argument("--smoke", action="store_true", help="one weekly fold only")
    p.add_argument("--render-only", type=Path, metavar="RESULTS_JSON")
    args = p.parse_args(argv)
    if args.render_only is not None:
        study = load_results(args.render_only)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_report(study))
        print(f"rendered {args.render_only} to {args.report}")
        return 0
    wall = time.monotonic()
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    horizons = _ints(args.horizons)
    test_end = args.test_end
    if args.smoke:
        end = dt.date.fromisoformat(args.test_start) + dt.timedelta(days=7)
        test_end = min(end.isoformat(), args.test_end)
    master = load_inputs(args.data)
    study = run_study(master, args.test_start, test_end, tiers=tiers, horizons=horizons)
    bad = [k for k, v in study["n_check"].items() if v["evaluate"] != v["study"]]
    if bad:
        print("row-set mismatch against evaluate at: " + ", ".join(bad), file=sys.stderr)
        print("no report written", file=sys.stderr)
        return 1
    study["archive"] = live_archive_stats(args.data)
    study["wall_s"] = time.monotonic() - wall
    month_end = dt.date.fromisoformat(args.test_start).replace(day=28) + dt.timedelta(days=4)
    study["month_end"] = month_end.replace(day=1).isoformat()
    study["partial"] = test_end < study["month_end"]
    out_dir = args.data / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "october_2026_smoke.json" if args.smoke else "october_2026.json"
    text = json.dumps(jsonable(study), allow_nan=False, indent=1)
    (out_dir / name).write_text(text)
    study = restore_keys(json.loads(text))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(study))
    an = study["anchor"]
    print(
        f"tier A buy MAE h1: {eu(an['tier_a_buy_mae_h1'])} EUR/MWh "
        f"(anchor about {eu(ANCHOR_MAE, 0)}, ratio {eu(an['ratio'])}, "
        f"{'in range' if an['in_range'] else 'OUT OF RANGE'})"
    )
    for tier in ("B", "C", "D"):
        if tier in study["reading"]:
            print(rule_sentence(tier, study["reading"][tier]))
    if study["headroom"] is not None:
        hd = study["headroom"]
        print(f"boosted headroom mean gain h0-h4: {pct(hd['mean_gain'])}, adopt {hd['adopt']}")
    print(f"wall time {eu(study['wall_s'] / 60.0, 1)} min; report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
