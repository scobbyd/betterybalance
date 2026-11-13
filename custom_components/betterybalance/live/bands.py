"""Band sources for the live loop: one price model over the live master, and the store.

``ModelBandSource`` prices window cells with the current champion. The champion was trained
before now, so the live bands carry no embargo, and ``cov`` is the model's own coverage
factor of the cell. The live master holds only data known at the master's build time, so a
cell whose features are not known yet has no band. Live targets are not settled yet, so the
features come from ``build_matrix`` in its label-free mode. Every band the source serves
goes to ``record(row, now, epoch)`` as a ``bands`` table row (``band_record``), recorded at
the ``now`` and run ``epoch`` of the last ``set_now`` call; a cell without a band goes as a
row of NaN.

``StoredBandSource`` serves those rows back for re-execution, as of the ``now`` and run
``epoch`` of its own ``set_now`` call (required before the first band). It reads the
``bands`` table only, so a re-executed window sees the bands the live window saw at that
minute, also when a late feed changed a cell afterwards.

The ``BandSource`` protocol of ``core.rule.window`` passes no ``now``, so the caller calls
``set_now(now, epoch)`` on either source before it builds a window.

Both satisfy the ``BandSource`` protocol of ``core.rule.window`` plus ``tables_at`` and
``healthy_tiers``: None for an unknown cell, a target off the master grid or a non-finite
band.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.features import TIERS, build_matrix
from ..core.pricemodel import QUANTILES, PriceModel, dual_rate_at, has_cell, predict_rows
from ..core.store import Store
from ..core.timegrid import Table, local_calendar

_QNAMES = tuple("q%d" % round(q * 100) for q in QUANTILES)
_Cell = tuple[np.ndarray, np.ndarray, np.ndarray]
_NO_BAND = {
    "take": dict.fromkeys(QUANTILES, float("nan")),
    "feed": dict.fromkeys(QUANTILES, float("nan")),
    "p_dual": float("nan"),
    "cov": float("nan"),
}


def _key(tier: str, h: int, elapsed_min: int | None) -> tuple[str, int, int | None]:
    return tier, int(h), elapsed_min if int(h) == 0 else None


def _finite(got: dict) -> bool:
    vals = [*got["take"].values(), *got["feed"].values(), got["cov"]]
    return bool(np.isfinite(np.asarray(vals, dtype=float)).all())


def band_record(
    target_ts: int, tier: str, h: int, e: int | None, model_version: str, got: dict
) -> dict:
    """The ``bands`` table row of one served band (``e`` -1 for no elapsed minute)."""
    row: dict = {
        "target_ts": int(target_ts),
        "tier": tier,
        "h": int(h),
        "e": -1 if e is None else int(e),
        "model_version": model_version,
    }
    for side in ("take", "feed"):
        for q, name in zip(QUANTILES, _QNAMES, strict=True):
            row[f"{side}_{name}"] = float(got[side][q])
    row["p_dual"] = float(got["p_dual"])
    row["cov"] = float(got["cov"])
    return row


class ModelBandSource:
    """Bands of one ``PriceModel`` over the rows of a live master.

    Each cell is predicted once per ``refresh``, over every master row with finite features,
    on the first request of that cell. ``p_dual`` is the cell's state-2 classifier output, or
    the model's dual rate table at the row's local hour and month when the cell has none
    (as ``replay.bands``).
    """

    def __init__(
        self,
        model: PriceModel,
        version: str,
        master: Table,
        record: Callable[[dict, int, int], object] | None = None,
        skip: tuple[str, ...] = (),
    ):
        self.model = model
        self.version = version
        self.record = record
        self.skip = tuple(skip)  # tiers never healthy (tier B without the IDA feed)
        self.now: int | None = None
        self.epoch = 0
        self.refresh(master)

    def set_now(self, now: int, epoch: int = 0) -> None:
        """The decision time and run epoch the next served bands are recorded at."""
        self.now, self.epoch = int(now), int(epoch)

    def refresh(self, master: Table) -> None:
        """Serve from ``master`` from now on; drops every cell predicted before."""
        self._master = master
        self._ts = np.asarray(master["ts"], dtype=np.int64)
        self._calendar: tuple[np.ndarray, np.ndarray] | None = None
        self._cells: dict[tuple, _Cell | None] = {}

    def _hour_month(self) -> tuple[np.ndarray, np.ndarray]:
        if self._calendar is None:
            cal = local_calendar(self._ts)
            self._calendar = (
                np.asarray(cal["hour"], dtype=np.int64),
                np.asarray(cal["month"], dtype=np.int64),
            )
        return self._calendar

    def _cell(self, key: tuple) -> _Cell | None:
        if key not in self._cells:
            self._cells[key] = self._predict(key)
        return self._cells[key]

    def _predict(self, key: tuple) -> _Cell | None:
        tier, h, e = key
        if not has_cell(self.model, tier, h, e):
            return None
        n = len(self._ts)
        take = np.full((len(QUANTILES), n), np.nan)
        feed = np.full((len(QUANTILES), n), np.nan)
        p_dual = np.full(n, np.nan)
        mx = build_matrix(self._master, tier, h, e, labels=False)
        if len(mx.ts_target):
            k = np.searchsorted(self._ts, mx.ts_target)
            p = predict_rows(self.model, tier, h, mx.X, mx.names, e)
            take[:, k] = np.vstack([p["buy_" + q] for q in _QNAMES])
            feed[:, k] = np.vstack([p["sell_" + q] for q in _QNAMES])
            if self.model.models[key].w_dual is not None:
                p_dual[k] = p["p_dual"]
            else:
                hour, month = self._hour_month()
                p_dual[k] = [dual_rate_at(self.model.tables, hour[i], month[i]) for i in k]
        return take, feed, p_dual

    def _band(self, tier: str, h: int, elapsed_min: int | None, target_ts: int) -> dict | None:
        key = _key(tier, h, elapsed_min)
        k = int(np.searchsorted(self._ts, int(target_ts)))
        if k >= len(self._ts) or int(self._ts[k]) != int(target_ts):
            return None
        cell = self._cell(key)
        if cell is None:
            return None
        take, feed, p_dual = cell[0][:, k], cell[1][:, k], cell[2][k]
        got = {
            "take": {q: float(v) for q, v in zip(QUANTILES, take, strict=True)},
            "feed": {q: float(v) for q, v in zip(QUANTILES, feed, strict=True)},
            "p_dual": float(p_dual),
            "cov": self.model.cov_factor(key[0], key[1], 0.5, key[2]),
        }
        return got if _finite(got) else None

    def bands(self, tier: str, h: int, elapsed_min: int | None, target_ts: int) -> dict | None:
        """Raw quantiles, ``p_dual`` and ``cov`` of one cell, or None; ``record`` gets each
        serve, a None as a band with every value NaN (the store keeps it only after a band
        of the cell, so a re-executed window sees the cell turn unavailable as live).

        Raises ``ValueError`` when ``record`` is set and no ``set_now`` came before.
        """
        got = self._band(tier, h, elapsed_min, target_ts)
        if self.record is not None:
            if self.now is None:
                raise ValueError("set_now before serving recorded bands")
            _, h_, e = _key(tier, h, elapsed_min)
            row = band_record(target_ts, tier, h_, e, self.version, got or _NO_BAND)
            self.record(row, self.now, self.epoch)
        return got

    def tables_at(self, ts: int) -> dict:
        return self.model.tables

    def healthy_tiers(self, ts: int) -> tuple[str, ...]:
        """Tiers (in ``TIERS`` order, without ``skip``) with a finite h1 band at ``ts``. Not
        recorded."""
        return tuple(
            t for t in TIERS if t not in self.skip and self._band(t, 1, None, ts) is not None
        )


class StoredBandSource:
    """Bands of one model version from the store's ``bands`` table, as they were served.

    ``healthy`` and ``tables`` are the ones of the live window (its ``windows`` row and the
    model), since the store keeps no unserved band. Bands are read as of the ``now`` and run
    ``epoch`` of the last ``set_now`` call (``Store.band``); a band asked for before any
    ``set_now`` raises ``ValueError``.
    """

    def __init__(self, store: Store, version: str, healthy: tuple[str, ...], tables: dict):
        self.store = store
        self.version = version
        self.healthy = tuple(healthy)
        self.tables = tables
        self.now: int | None = None
        self.epoch: int | None = None

    def set_now(self, now: int, epoch: int | None = None) -> None:
        """The decision time and run epoch whose bands the source serves (None: any run)."""
        self.now = int(now)
        self.epoch = None if epoch is None else int(epoch)

    def bands(self, tier: str, h: int, elapsed_min: int | None, target_ts: int) -> dict | None:
        """The stored band of the cell, or None when it was never served or is non-finite.

        Raises ``ValueError`` before any ``set_now``.
        """
        if self.now is None:
            raise ValueError("set_now before reading stored bands")
        tier, h, e = _key(tier, h, elapsed_min)
        row = self.store.band(
            int(target_ts), tier, h, e, self.version, as_of=self.now, epoch=self.epoch
        )
        if row is None:
            return None
        got = {
            side: {q: float(row[f"{side}_{n}"]) for q, n in zip(QUANTILES, _QNAMES, strict=True)}
            for side in ("take", "feed")
        }
        got["p_dual"] = float(row["p_dual"])
        got["cov"] = float(row["cov"])
        return got if _finite(got) else None

    def tables_at(self, ts: int) -> dict:
        return self.tables

    def healthy_tiers(self, ts: int) -> tuple[str, ...]:
        return self.healthy
