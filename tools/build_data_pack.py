"""Build a data pack: the quarter table and the weekly market model series.

Run (background, log under data/results/plan3/):
python -m tools.build_data_pack --data data --from 2024-01-01 --to 2026-10-31 --month 2026-11 \
--out data/releases/2026-11

The master comes from ``core.inputs.load_inputs`` (the npz of tools/fetch_history.py plus the
live archive), cut to the local days --from .. --to. The IDA columns are NaN in the master
and in the pack unless --with-ida (the Nord Pool release gate), so no model reads them.

The series: one model per refit step, trained on the train_days local days before its
training end. The training ends step back by refit_days local days from the midnight after
--to while a full training window fits after --from. So the newest model trains up to the
end of the data. The folds before it are the folds of ``evaluate.walk_forward``, each one
fitted on its training rows only (``fit_model``, format 2).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# One BLAS thread per process, set before numpy loads: on the owner's box a weekly fold takes
# 1.464 s with the default thread pool and 328 s with one thread (the IRLS solves are small).
# Parallel speed comes from --workers. An explicit environment value wins.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from betterybalance.core.evaluate import walk_forward  # noqa: E402
from betterybalance.core.features import LOOKBACK_ISP, cut_master  # noqa: E402
from betterybalance.core.fmt import eu  # noqa: E402
from betterybalance.core.inputs import load_inputs  # noqa: E402
from betterybalance.core.pack import IDA_COLUMNS, QUARTER_COLUMNS, write_pack  # noqa: E402
from betterybalance.core.pricemodel import PriceModel, fit_model  # noqa: E402
from betterybalance.core.timegrid import ISP_S, ZONE, Table  # noqa: E402


def local_midnight(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=ZONE).timestamp())


def plus_local_days(ts: int, days: int) -> int:
    """Epoch s of the same local wall-clock time ``days`` calendar days later."""
    local = dt.datetime.fromtimestamp(int(ts), ZONE).replace(tzinfo=None)
    return int((local + dt.timedelta(days=days)).replace(tzinfo=ZONE).timestamp())


def _iso(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date().isoformat()


def series_folds(
    ts: np.ndarray, start: int, end: int, refit_days: int = 7, train_days: int = 365
) -> list[tuple[int, np.ndarray]]:
    """(train_end, training row indices) per model, ascending; train_end is exclusive.

    The ends are end, end - refit_days, ... (local days), back to the earliest end whose
    train_days window starts at or after start. Every end but the last is a walk-forward fold
    start; the last one has no test block, so it is added here.
    """
    ts = np.asarray(ts, dtype=np.int64)
    k = 0
    while plus_local_days(plus_local_days(end, -(k + 1) * refit_days), -train_days) >= start:
        k += 1
    first = plus_local_days(end, -k * refit_days)
    if plus_local_days(first, -train_days) < start:
        raise ValueError(
            "the span %s to %s holds less than one training window" % (_iso(start), _iso(end - 1))
        )
    out = []
    for fold in walk_forward(ts, first, end, refit_days, train_days):
        out.append((int(ts[fold.test[0]]), fold.train))
    lo = plus_local_days(end, -train_days)
    out.append((int(end), np.flatnonzero((ts >= lo) & (ts < end))))
    return out


# features.LOOKBACK_ISP: the quarters build_matrix reads before a target row.
LOOKBACK_S = LOOKBACK_ISP * ISP_S


def _fit(master: Table, rows: np.ndarray, meta: dict, **fit_kw) -> PriceModel:
    """``fit_model`` on the training rows, over a master cut to the rows it can read.

    The cut keeps LOOKBACK_S before the first training row, so every feature of a training
    row is the one the full master gives and the weights are identical; it only makes the
    feature build independent of the master's length.
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    part = cut_master(master, int(ts[rows[0]]) - LOOKBACK_S, int(ts[rows[-1]]) + ISP_S)
    pts = np.asarray(part["ts"], dtype=np.int64)
    mask = np.isin(pts, ts[rows])
    return fit_model(part, train_mask=mask, meta=meta, **fit_kw)


_WORKER_MASTER: Table | None = None


def _init_worker(master: Table) -> None:
    global _WORKER_MASTER
    _WORKER_MASTER = master


def _fit_in_worker(args: tuple[np.ndarray, dict]) -> PriceModel:
    rows, meta = args
    assert _WORKER_MASTER is not None
    return _fit(_WORKER_MASTER, rows, meta)


def fit_series(
    master: Table,
    folds: list[tuple[int, np.ndarray]],
    month: str,
    workers: int = 1,
    log=print,
) -> list[tuple[int, PriceModel]]:
    """One model per fold, in fold order. Workers change the speed, never the weights."""
    jobs = [(rows, {"pack": month}) for _, rows in folds]
    t0 = time.monotonic()
    models: list[PriceModel] = []

    def note(i: int) -> None:
        mins = (time.monotonic() - t0) / 60
        log(
            "model %d of %d fitted, %s min elapsed" % (i + 1, len(jobs), eu(mins, 1)),
        )

    if workers <= 1:
        for i, job in enumerate(jobs):
            models.append(_fit(master, *job))
            note(i)
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(master,)) as ex:
            for i, model in enumerate(ex.map(_fit_in_worker, jobs)):
                models.append(model)
                note(i)
    return [(end, m) for (end, _), m in zip(folds, models, strict=True)]


def quarter_table(master: Table) -> Table:
    return {c: np.asarray(master[c]) for c in QUARTER_COLUMNS}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True, help="data directory of load_inputs")
    ap.add_argument("--from", dest="start", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--to", dest="end", type=dt.date.fromisoformat, required=True)
    ap.add_argument("--month", required=True, help="pack version, YYYY-MM")
    ap.add_argument("--out", type=Path, required=True, help="release folder")
    ap.add_argument("--with-ida", action="store_true", help="keep the IDA columns")
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--train-days", type=int, default=365)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)

    def log(text: str) -> None:
        print(text, flush=True)

    lo, hi = local_midnight(a.start), local_midnight(a.end + dt.timedelta(days=1))
    master = cut_master(load_inputs(a.data), lo, hi)
    ts = np.asarray(master["ts"], dtype=np.int64)
    if len(ts) == 0 or int(ts[0]) > lo or int(ts[-1]) + ISP_S < hi:
        ap.error("the master inputs do not cover %s to %s" % (a.start, a.end))
    if not a.with_ida:
        for c in IDA_COLUMNS:
            master[c] = np.full(len(ts), np.nan)
    try:
        folds = series_folds(ts, lo, hi, a.refit_days, a.train_days)
    except ValueError as err:
        ap.error(str(err))
    log("quarters: %s rows, %s to %s" % (eu(len(ts), 0), a.start, a.end))
    log("models: %d, refit %d days, train %d days" % (len(folds), a.refit_days, a.train_days))
    series = fit_series(master, folds, a.month, a.workers, log)
    built = dt.datetime.now(ZONE).date().isoformat()
    man = write_pack(a.out, a.month, quarter_table(master), series, built=built)
    size = (a.out / ("betterybalance-pack-%s.npz" % a.month)).stat().st_size / 1e6
    log("wrote %s: %d models, %s MB" % (a.out, man["n_models"], eu(size, 1)))
    log("coverage %s to %s" % (man["coverage"]["from"], man["coverage"]["to"]))
    log("ida_included %s" % str(man["ida_included"]).lower())
    log("sha256 %s" % man["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
