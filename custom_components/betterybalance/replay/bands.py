"""Walk-forward band cache for the replay and the BandSource that reads it.

The replay prices every historical quarter with bands from a model that never saw that
quarter's settled price. ``build_band_cache`` fits one model per weekly fold of
``evaluate.walk_forward`` on the fold's training rows and predicts every cell on the fold's
test rows, exactly as ``evaluate.evaluate`` does. The cache stores the raw quantiles once,
so every ladder rung of the replay reuses them.

Invariants:

- Embargo. Let ``lag = max(horizons) * 900 + EMBARGO_S``. Row ``k`` of the test range is
  priced by the latest fold ``f`` with ``ts[k] - lag >= fold_train_end[f]``. A row that no
  fold satisfies has ``fold_of = -1`` and no band, so the first ``lag`` seconds of the test
  range (4 days and 6 h at h24) are empty. Every cell of row ``k`` was forecast at
  ``ts[k] - h * 900 + 60 > ts[k] - max(horizons) * 900 >= fold_train_end[f] + EMBARGO_S``,
  so at least 96 h after the training end.
- The bound ``EMBARGO_S`` satisfies. A decision at ``now`` that reads row ``k`` at lead
  ``L = ts[k] - now <= 72 h`` (the window reach) has ``now >= fold_train_end[f] + lag - L``,
  so ``fold_train_end[f] <= now - (max(horizons) * 900 + 96 h - 72 h)``. At h24 that is
  ``now - 30 h`` for the farthest slot, which covers the D+1 publication of the settled
  prices, and more for every nearer slot (``now - 102 h`` for the running quarter). So
  every training label was settled before the decision that reads the model.
- The coverage factor ``cov`` of the rows priced by fold ``f`` uses only rows priced by folds
  ``0 .. f - 1`` with ``ts <= fold_train_end[f]``. Those settled (D+1) before fold ``f``
  prices its first row at ``fold_train_end[f] + lag``. Fold 0 has 1,0.
- The cache has no shipped-model path: every plan-2 replay window precedes the shipped
  champion's training end.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.evaluate import Fold, walk_forward
from ..core.features import ELAPSED_MIN, HORIZONS, LOOKBACK_ISP, TIERS, build_matrix, cut_master
from ..core.pack import DataPack
from ..core.pricemodel import (
    QUANTILES,
    PriceModel,
    coverage_factor,
    dual_rate_at,
    fit_model,
    has_cell,
    predict_rows,
)
from ..core.timegrid import ISP_S, Table, local_calendar

COVERAGE_TARGET = 0.8
# The window reach (72 h) plus one day for the D+1 publication of the settled prices.
EMBARGO_S = 4 * 86400
_QNAMES = tuple("q%d" % round(q * 100) for q in QUANTILES)
_LO, _MID, _HI = QUANTILES.index(0.1), QUANTILES.index(0.5), QUANTILES.index(0.9)
_GROUPS = ("take", "feed", "p_dual", "cov")
_KEY_COLUMNS = ("buy_eur_mwh", "sell_eur_mwh", "regulation_state", "da_eur_mwh")
_NAN_STAND_IN = -1e9
_Key = tuple[str, int, "int | None"]


@dataclass
class BandCache:
    """Walk-forward bands per cell key ``(tier, h, e)`` over the master rows.

    ``take[key]`` and ``feed[key]`` have shape ``(5, n)`` in ``QUANTILES`` order (buy and sell
    price), NaN where no prediction. ``p_dual[key]`` and ``cov[key]`` have shape ``(n,)``.
    ``fold_of[k]`` is the fold that priced master row ``k`` under the embargo, -1 for a row
    outside the test range or inside its first ``lag`` seconds.
    ``fold_train_end[f]`` is the end (epoch s) of fold ``f``'s last training quarter.
    ``tables[f]`` holds the dual tables of fold ``f``'s model.
    """

    ts: np.ndarray
    take: dict[tuple, np.ndarray]
    feed: dict[tuple, np.ndarray]
    p_dual: dict[tuple, np.ndarray]
    cov: dict[tuple, np.ndarray]
    fold_of: np.ndarray
    fold_train_end: np.ndarray
    tables: list[dict]
    meta: dict


def _elapsed(h: int) -> tuple:
    return ELAPSED_MIN if h == 0 else (None,)


def _cells(tiers, horizons) -> list[_Key]:
    return [(t, h, e) for t in tiers for h in horizons for e in _elapsed(h)]


def cache_key(
    master: Table,
    test_start: int,
    test_end: int,
    refit_every_days: int,
    train_days: int,
    tiers=TIERS,
    horizons=HORIZONS,
) -> str:
    """SHA-256 over the master ISP grid, the price columns, the bounds, the folds and the cells.

    The price, regulation-state and day-ahead columns are hashed with NaN as -1e9, so a
    re-exported master with corrected prices never matches a stale cache. The tiers, horizons
    and ``EMBARGO_S`` are in the checksum, so a partial cache never matches a full one, and a
    cache built under another embargo never matches.
    """
    digest = hashlib.sha256(np.ascontiguousarray(master["ts"], dtype=np.int64).tobytes())
    for name in _KEY_COLUMNS:
        col = np.nan_to_num(np.asarray(master[name], dtype=np.float64), nan=_NAN_STAND_IN)
        digest.update(np.ascontiguousarray(col).tobytes())
    params = [int(test_start), int(test_end), int(refit_every_days), int(train_days)]
    cells = [list(tiers), [int(h) for h in horizons]]
    digest.update(json.dumps([params, cells, EMBARGO_S]).encode())
    return digest.hexdigest()


def _fit_folds(
    master: Table, folds: list[Fold], tiers, horizons, tick: Callable[[], None]
) -> list[PriceModel]:
    """One model per fold, fitted on the fold's training rows only."""
    n = len(master["ts"])
    models = []
    for fold in folds:
        train_mask = np.zeros(n, dtype=bool)
        train_mask[fold.train] = True
        models.append(fit_model(master, tiers, horizons, train_mask=train_mask))
        tick()
    return models


def embargo_lag(horizons) -> int:
    """Seconds between a fold's training end and the first row it may price."""
    return max(int(h) for h in horizons) * ISP_S + EMBARGO_S


def _assign_models(
    ts: np.ndarray, test_start: int, test_end: int, train_end: np.ndarray, lag: int
) -> np.ndarray:
    """Latest model whose training end lies at least ``lag`` before the row, test range only.

    On a contiguous master grid the rows of ``[test_start, test_end)`` are exactly the union
    of the walk-forward test blocks, so this is the fold assignment of ``build_band_cache``.
    """
    in_test = (ts >= int(test_start)) & (ts < int(test_end))
    latest = np.searchsorted(train_end, ts - lag, side="right") - 1
    return np.where(in_test, latest, -1).astype(np.int64)


def _predict_cell(
    master: Table,
    key: _Key,
    models: list[PriceModel],
    fold_of: np.ndarray,
    fold_train_end: np.ndarray,
    lag: int,
    hour: np.ndarray,
    month: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Take, feed and p_dual of one cell, each fold's model on the rows it prices only.

    One feature matrix per cell serves every fold, and each fold predicts all its rows in one
    call. ``p_dual`` is the cell's state-2 classifier output, or the fold's dual rate table
    at the row's local hour and month when the cell has no fitted classifier.
    """
    tier, h, e = key
    ts = np.asarray(master["ts"], dtype=np.int64)
    n = len(ts)
    take = np.full((len(QUANTILES), n), np.nan)
    feed = np.full((len(QUANTILES), n), np.nan)
    p_dual = np.full(n, np.nan)
    mx = build_matrix(master, tier, h, e)
    k_all = np.searchsorted(ts, mx.ts_target)
    row_fold = fold_of[k_all]
    for f, model in enumerate(models):
        sel = row_fold == f
        if not sel.any() or not has_cell(model, tier, h, e):
            continue
        k = k_all[sel]
        if (ts[k] - lag < fold_train_end[f]).any():
            raise RuntimeError("a priced row lies inside its fold's embargo")
        p = predict_rows(model, tier, h, mx.X[sel], mx.names, e)
        take[:, k] = np.vstack([p["buy_" + q] for q in _QNAMES])
        feed[:, k] = np.vstack([p["sell_" + q] for q in _QNAMES])
        if model.models[key].w_dual is not None:
            p_dual[k] = p["p_dual"]
        else:
            p_dual[k] = [dual_rate_at(model.tables, hour[i], month[i]) for i in k]
    return take, feed, p_dual


def _pooled_cov(
    take: np.ndarray,
    feed: np.ndarray,
    y_buy: np.ndarray,
    y_sell: np.ndarray,
    ts: np.ndarray,
    fold_of: np.ndarray,
    fold_train_end: np.ndarray,
) -> np.ndarray:
    """Per-row coverage factor from earlier folds and settled rows only.

    Rows of fold ``f`` get the q10-q90 coverage factor (target 0,8) of the pooled buy and
    sell outcomes over the rows predicted by folds ``0 .. f - 1`` with
    ``ts <= fold_train_end[f]``. Fold 0 gets 1,0, and so does a fold with fewer than 20
    pooled rows (``coverage_factor``). NaN on rows no fold prices.
    """
    cov = np.full(len(fold_of), np.nan)
    predicted = np.isfinite(take[_MID]) & np.isfinite(feed[_MID])
    for f in range(len(fold_train_end)):
        rows = fold_of == f
        if f == 0:
            cov[rows] = 1.0
            continue
        prior = predicted & (fold_of >= 0) & (fold_of < f) & (ts <= fold_train_end[f])
        cov[rows] = coverage_factor(
            np.concatenate([y_buy[prior], y_sell[prior]]),
            np.concatenate([take[_LO, prior], feed[_LO, prior]]),
            np.concatenate([take[_MID, prior], feed[_MID, prior]]),
            np.concatenate([take[_HI, prior], feed[_HI, prior]]),
            COVERAGE_TARGET,
        )
    return cov


def cache_from_models(
    master: Table,
    models: list[PriceModel],
    fold_train_end: np.ndarray,
    test_start: int,
    test_end: int,
    tiers=TIERS,
    horizons=HORIZONS,
    lag: int | None = None,
    progress=None,
) -> BandCache:
    """Bands of fitted models on the rows of ``[test_start, test_end)``.

    ``models[f]`` was trained on rows that end at ``fold_train_end[f]`` (ascending). Row ``k``
    of the test range is priced by the latest model ``f`` with
    ``ts[k] - lag >= fold_train_end[f]``, -1 when no model satisfies that. ``lag`` None is
    ``embargo_lag(horizons)`` (the replay embargo of the module docstring). ``lag`` 0 removes
    the embargo: every row after a model's training end may read it.

    ``cov`` is the pooled coverage factor of ``_pooled_cov`` (1,0 for model 0).
    ``meta`` holds the tiers, the horizons and the lag.

    ``progress``, when given, is called as ``progress(done, total)`` after each cell
    prediction, with ``total = cells``.
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    fold_train_end = np.asarray(fold_train_end, dtype=np.int64)
    if len(models) != len(fold_train_end):
        raise ValueError("one training end per model")
    lag = embargo_lag(horizons) if lag is None else int(lag)
    fold_of = _assign_models(ts, test_start, test_end, fold_train_end, lag)
    cells = _cells(tiers, horizons)
    cal = local_calendar(ts)
    hour = np.asarray(cal["hour"], dtype=np.int64)
    month = np.asarray(cal["month"], dtype=np.int64)
    y_buy = np.asarray(master["buy_eur_mwh"], dtype=float)
    y_sell = np.asarray(master["sell_eur_mwh"], dtype=float)
    take, feed, p_dual, cov = {}, {}, {}, {}
    for i, key in enumerate(cells):
        take[key], feed[key], p_dual[key] = _predict_cell(
            master, key, models, fold_of, fold_train_end, lag, hour, month
        )
        cov[key] = _pooled_cov(take[key], feed[key], y_buy, y_sell, ts, fold_of, fold_train_end)
        if progress is not None:
            progress(i + 1, len(cells))
    return BandCache(
        ts=ts,
        take=take,
        feed=feed,
        p_dual=p_dual,
        cov=cov,
        fold_of=fold_of,
        fold_train_end=fold_train_end,
        tables=[m.tables for m in models],
        meta={"tiers": list(tiers), "horizons": [int(h) for h in horizons], "lag": lag},
    )


def cache_from_series(
    pack: DataPack,
    start: int,
    end: int,
    tiers=TIERS,
    horizons=HORIZONS,
    progress=None,
) -> BandCache:
    """Bands from a data pack's model series on the rows of ``[start, end)``, no training.

    ``cache_from_models`` over ``pack.master()`` with every series model and its training end
    under the replay embargo ``embargo_lag(horizons)``: row ``k`` reads the latest model with
    ``ts[k] - lag >= train_end``, so a band never reads a model trained on labels unsettled
    at decision time. ``meta`` adds the pack version and the model versions.

    The master is cut to ``[start - LOOKBACK_ISP quarters, end)`` first, so memory and time
    follow the requested span, not the pack's length. The bands on ``[start, end)`` are the
    ones the whole master gives: the features of those rows read at most ``LOOKBACK_ISP``
    quarters back, ``fold_of`` is -1 outside the span and the pooled coverage reads priced
    rows only. ``cache.ts`` is the cut grid; ``CachedBandSource`` looks rows up by ``ts``.
    """
    models = [pack.model(i) for i in range(len(pack.train_ends))]
    master = cut_master(pack.master(), int(start) - LOOKBACK_ISP * ISP_S, int(end))
    cache = cache_from_models(
        master,
        models,
        pack.train_ends,
        start,
        end,
        tiers,
        horizons,
        lag=embargo_lag(horizons),
        progress=progress,
    )
    cache.meta["pack"] = pack.version
    cache.meta["model_versions"] = [pack.model_version(i) for i in range(len(models))]
    return cache


def build_band_cache(
    master: Table,
    test_start: int,
    test_end: int,
    tiers=TIERS,
    horizons=HORIZONS,
    refit_every_days: int = 7,
    train_days: int = 365,
    progress=None,
) -> BandCache:
    """Walk-forward bands for every cell on the test rows (invariants in the module docstring).

    The folds of ``walk_forward``, one model per fold (``_fit_folds``), then
    ``cache_from_models`` under the embargo ``embargo_lag(horizons)``.
    ``meta["folds"]`` holds the walk-forward test blocks ``[start, end)``. Fold ``f`` prices
    the rows ``fold_of == f``, which start ``lag`` after ``fold_train_end[f]``.

    ``progress``, when given, is called as ``progress(done, total)`` after each fold fit and
    after each cell prediction, with ``total = folds + cells``.
    """
    ts = np.asarray(master["ts"], dtype=np.int64)
    folds = walk_forward(ts, test_start, test_end, refit_every_days, train_days)
    if not folds:
        raise ValueError("no fold has both training and test rows")
    total = len(folds) + len(_cells(tiers, horizons))
    done = [0]

    def tick() -> None:
        done[0] += 1
        if progress is not None:
            progress(done[0], total)

    fold_train_end = np.array([int(ts[fold.train[-1]]) + ISP_S for fold in folds], dtype=np.int64)
    models = _fit_folds(master, folds, tiers, horizons, tick)
    cache = cache_from_models(
        master,
        models,
        fold_train_end,
        test_start,
        test_end,
        tiers,
        horizons,
        progress=lambda d, n: tick(),
    )
    cache.meta = {
        "key": cache_key(
            master, test_start, test_end, refit_every_days, train_days, tiers, horizons
        ),
        "train_days": int(train_days),
        "refit_every_days": int(refit_every_days),
        "tiers": list(tiers),
        "horizons": [int(h) for h in horizons],
        "folds": [[int(ts[fold.test[0]]), int(ts[fold.test[-1]]) + ISP_S] for fold in folds],
    }
    return cache


def _key_str(key: _Key) -> str:
    tier, h, e = key
    return "%s:%d:%s" % (tier, h, "-" if e is None else e)


def _parse_key(s: str) -> _Key:
    tier, h, e = s.split(":")
    return tier, int(h), None if e == "-" else int(e)


def save_cache(cache: BandCache, path: Path) -> None:
    """Compressed npz: arrays ``take/A:1:-`` style per group, ``meta`` and ``tables`` as JSON.

    A real cache is mostly NaN (about 190 MB uncompressed), hence ``savez_compressed``. The
    file is an intermediate artefact and is exempt from byte identity (the zip members carry
    timestamps). Its arrays load back identical, and the replay's own outputs stay
    byte-identical.
    """
    arrays: dict[str, np.ndarray] = {
        "ts": cache.ts,
        "fold_of": cache.fold_of,
        "fold_train_end": cache.fold_train_end,
        "meta": np.array(json.dumps(cache.meta)),
        "tables": np.array(json.dumps(cache.tables)),
    }
    for group in _GROUPS:
        for key, arr in getattr(cache, group).items():
            arrays["%s/%s" % (group, _key_str(key))] = arr
    with open(Path(path), "wb") as fh:
        np.savez_compressed(fh, **arrays)


def load_cache(path: Path) -> BandCache:
    with np.load(Path(path), allow_pickle=False) as z:
        groups: dict[str, dict] = {g: {} for g in _GROUPS}
        for name in z.files:
            group, sep, key = name.partition("/")
            if sep:
                groups[group][_parse_key(key)] = z[name]
        return BandCache(
            ts=z["ts"],
            fold_of=z["fold_of"],
            fold_train_end=z["fold_train_end"],
            tables=json.loads(str(z["tables"])),
            meta=json.loads(str(z["meta"])),
            **groups,
        )


class CachedBandSource:
    """``BandSource`` over a ``BandCache``: raw quantiles plus the row's ``cov``, or None."""

    def __init__(self, cache: BandCache):
        self.cache = cache
        meta = cache.meta
        self._lag = int(meta["lag"]) if "lag" in meta else embargo_lag(meta["horizons"])

    def _row(self, ts: int) -> int | None:
        """Master row of ``ts`` when it is on the cache grid and inside a fold, else None."""
        grid = self.cache.ts
        k = int(np.searchsorted(grid, int(ts)))
        if k >= len(grid) or int(grid[k]) != int(ts) or self.cache.fold_of[k] < 0:
            return None
        return k

    def bands(self, tier: str, h: int, elapsed_min: int | None, target_ts: int) -> dict | None:
        """None for an unknown cell, a target off the grid or outside the folds, or a NaN band."""
        key = (tier, int(h), elapsed_min if h == 0 else None)
        if key not in self.cache.take:
            return None
        k = self._row(target_ts)
        if k is None:
            return None
        take = self.cache.take[key][:, k]
        feed = self.cache.feed[key][:, k]
        if not (np.isfinite(take).all() and np.isfinite(feed).all()):
            return None
        return {
            "take": {q: float(v) for q, v in zip(QUANTILES, take, strict=True)},
            "feed": {q: float(v) for q, v in zip(QUANTILES, feed, strict=True)},
            "p_dual": float(self.cache.p_dual[key][k]),
            "cov": float(self.cache.cov[key][k]),
        }

    def tables_at(self, ts: int) -> dict:
        """Tables of the fold that prices ``ts`` (as ``fold_of``), the first fold's before them.

        Rows after the test range get the last fold's tables.
        """
        ends = self.cache.fold_train_end
        f = int(np.searchsorted(ends, int(ts) - self._lag, side="right")) - 1
        return self.cache.tables[max(f, 0)]

    def healthy_tiers(self, ts: int) -> tuple[str, ...]:
        """Tiers (in ``TIERS`` order) with a finite h1 band at row ``ts``."""
        return tuple(t for t in TIERS if self.bands(t, 1, None, ts) is not None)
