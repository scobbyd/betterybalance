"""Decision window: one calibrated take and feed band per ISP up to the horizon.

The window runs from the running quarter to local midnight of the day after tomorrow.
Every slot reads a band from the best available tier at an honest horizon: the h-cell
stored at a target row was forecast at ``target - h * 900 + 60``, so a slot ``j``
quarters ahead only reads horizons ``h >= j``. A slot whose day-ahead price is not yet
known, or for which no model answers, takes a predicted band: the same-clock band of
an earlier day, with the half-widths widened by ``Knobs.predicted_widen``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..features import ELAPSED_MIN, HORIZONS, TIERS
from ..pricemodel import QUANTILES, dual_rate_at, dual_spread_at
from ..site.contract import Knobs
from ..timegrid import ISP_S, ZONE, isp_start, local_calendar

DAY_SLOTS = 96
FAR_TIER = "A"
FAR_HORIZON = HORIZONS[-1]
MIN_INTO_QUARTER_S = 60
# Tier A reads the day-ahead prices da_p1..da_p4 after the target (features._A_BASE).
DA_NEIGHBOUR_QUARTERS = 4
_MID = QUANTILES.index(0.5)
_Q25 = QUANTILES.index(0.25)
_Q75 = QUANTILES.index(0.75)


class BandSource(Protocol):
    def bands(self, tier: str, h: int, elapsed_min: int | None, target_ts: int) -> dict | None:
        """Raw take and feed quantiles, ``p_dual`` and ``cov`` of one cell, or None.

        None when the tier has no cell, the features are not available for that target
        at that horizon, or the target lies outside the source's data.
        """
        ...


@dataclass
class Window:
    ts: np.ndarray
    take: np.ndarray
    feed: np.ndarray
    p_dual: np.ndarray
    dual_spread: np.ndarray
    predicted: np.ndarray
    tier: list[str]
    elapsed_min: int


@dataclass
class Nowcast:
    take_point: float
    feed_point: float
    w_take: float
    w_feed: float
    p_dual: float
    take_q50: float
    feed_q50: float


def window_slots(now: int) -> np.ndarray:
    """ISP starts from ``isp_start(now)`` up to (excluding) local midnight of day + 3."""
    today = dt.datetime.fromtimestamp(int(now), ZONE).date()
    end_day = today + dt.timedelta(days=3)
    end = int(dt.datetime(end_day.year, end_day.month, end_day.day, tzinfo=ZONE).timestamp())
    first = int(isp_start(int(now)))
    return np.arange(first, end, ISP_S, dtype=np.int64)


def fitted_horizon(j: int) -> int:
    """The smallest fitted horizon at or above ``j`` (never a lower one: honesty)."""
    for h in HORIZONS:
        if h >= j:
            return h
    raise ValueError(f"no fitted horizon at or above {j}")


def _elapsed_min(now: int) -> int:
    """Largest of ``ELAPSED_MIN`` at or below the minutes into the running quarter, at least 1."""
    minutes = (int(now) - int(isp_start(int(now)))) // 60
    below = [e for e in ELAPSED_MIN if e <= minutes]
    return below[-1] if below else ELAPSED_MIN[0]


def _tier_order(healthy: tuple[str, ...]) -> list[str]:
    """Healthy tiers from the highest to the lowest (D, C, B, A)."""
    return [t for t in reversed(TIERS) if t in healthy]


def _candidates(j: int, order: list[str], elapsed_min: int) -> list[tuple[str, int, int | None]]:
    """The (tier, h, elapsed) cells slot ``j`` may read, in order of preference.

    Slot 0 reads only the h0 cell at ``elapsed_min``. Slots 1 to 24 read
    ``fitted_horizon(j)`` and then every higher fitted horizon up to 24, highest tier
    first at each horizon, then tier A at 24. Far slots read tier A at 24 only.
    """
    if j == 0:
        return [(t, 0, elapsed_min) for t in order]
    far = (FAR_TIER, FAR_HORIZON, None)
    if j > FAR_HORIZON:
        return [far]
    out = [(t, h, None) for h in HORIZONS if h >= fitted_horizon(j) for t in order]
    if far not in out:
        out.append(far)
    return out


def _widen(band: np.ndarray, factor: float) -> np.ndarray:
    """Multiply the half-widths about q50 by ``factor`` and keep the quantiles sorted."""
    mid = band[_MID]
    return np.sort(mid + (band - mid) * factor)


def _same_clock(j: int, own: np.ndarray) -> int | None:
    """Slot whose own band stands in for slot ``j``.

    The latest same-clock slot (``j - 96``, ``j - 192``, ...) that has its own band, then
    the latest earlier slot that has its own band. None when no earlier slot has one.
    """
    k = j - DAY_SLOTS
    while k >= 0:
        if own[k]:
            return k
        k -= DAY_SLOTS
    earlier = np.flatnonzero(own[:j])
    return int(earlier[-1]) if len(earlier) else None


def _da_ready(da_known: np.ndarray, n: int) -> np.ndarray:
    """Slot ``j`` is day-ahead-known only when ``da_known`` holds for ``j .. j + 4``.

    The lookahead is clamped to the window's last slot. ``da_known`` is indexed per
    slot and may be longer than the window.
    """
    known = np.asarray(da_known, dtype=bool)[np.arange(n)]
    ahead = np.minimum(np.arange(n)[:, None] + np.arange(DA_NEIGHBOUR_QUARTERS + 1), n - 1)
    return known[ahead].all(axis=1)


def build_window(
    now: int,
    source: BandSource,
    healthy: tuple[str, ...],
    da_known: np.ndarray,
    tables: dict,
    knobs: Knobs,
) -> Window:
    """Price every slot of the window (rules in the module and helper docstrings).

    Own bands are calibrated once: half-widths about q50 times the source's ``cov``.
    A cell with a non-finite quantile or ``cov`` counts as no band.
    A slot that is not day-ahead-known (``_da_ready``: ``da_known`` False at the slot or
    at one of the next ``DA_NEIGHBOUR_QUARTERS`` slots), or with no band from any
    candidate cell, takes the calibrated band of ``_same_clock`` with half-widths times
    ``knobs.predicted_widen`` and ``predicted = True``. ``da_known`` is indexed per
    slot, never shape-checked.
    Slot 0 has no earlier slot, so it always reads its own h0 cell and raises
    ``ValueError`` when no tier answers. ``p_dual`` of slot 0 comes from that cell, of
    the other slots from the dual-pricing tables at the slot's local hour and month.
    Raises ``ValueError`` when ``now`` lies in the first 60 s of the running quarter:
    the h0@1 and the ``h == j`` cells are forecast at ``isp_start + 60``, so such a call
    would read cells made after ``now``. The ranking runs at minute 1 or later, and the
    minute-0 pre-commit uses the previous quarter's window.
    """
    if int(now) - int(isp_start(int(now))) < MIN_INTO_QUARTER_S:
        raise ValueError("window needs at least 60 s of the running quarter")
    ts = window_slots(now)
    n = len(ts)
    elapsed = _elapsed_min(now)
    order = _tier_order(healthy)
    take = np.full((len(QUANTILES), n), np.nan)
    feed = np.full((len(QUANTILES), n), np.nan)
    tier = [""] * n
    own = np.zeros(n, dtype=bool)
    cell_p_dual = np.nan
    ready = _da_ready(da_known, n)
    for j in range(n):
        if j > 0 and not ready[j]:
            continue
        for t, h, e in _candidates(j, order, elapsed):
            got = source.bands(t, h, e, int(ts[j]))
            if got is None:
                continue
            cov = float(got["cov"])
            raw_take = np.array([got["take"][q] for q in QUANTILES], float)
            raw_feed = np.array([got["feed"][q] for q in QUANTILES], float)
            if not (
                np.isfinite(cov) and np.isfinite(raw_take).all() and np.isfinite(raw_feed).all()
            ):
                continue  # a non-finite cell counts as no band
            take[:, j] = _widen(raw_take, cov)
            feed[:, j] = _widen(raw_feed, cov)
            tier[j] = t
            own[j] = True
            if j == 0:
                cell_p_dual = float(got["p_dual"])
            break
    if not own[0]:
        raise ValueError("no band for slot 0")
    predicted = ~own
    for j in np.flatnonzero(predicted).tolist():
        k = _same_clock(j, own)
        take[:, j] = _widen(take[:, k], knobs.predicted_widen)
        feed[:, j] = _widen(feed[:, k], knobs.predicted_widen)
        tier[j] = tier[k]
    cal = local_calendar(ts)
    hours, months = cal["hour"].tolist(), cal["month"].tolist()
    p_dual = np.array([dual_rate_at(tables, h, m) for h, m in zip(hours, months)], float)
    p_dual[0] = cell_p_dual
    dual_spread = np.array([dual_spread_at(tables, h) for h in hours], float)
    return Window(ts, take, feed, p_dual, dual_spread, predicted, tier, elapsed)


def _at(band: np.ndarray, q: float) -> float:
    """Band value at quantile ``q``, linear between the fitted quantiles."""
    return float(np.interp(q, QUANTILES, band))


def nowcast(window: Window, knobs: Knobs) -> Nowcast:
    """Slot 0 points at ``q_take`` and ``q_feed`` and the q25-q75 half-widths."""
    take, feed = window.take[:, 0], window.feed[:, 0]
    return Nowcast(
        _at(take, knobs.q_take),
        _at(feed, knobs.q_feed),
        float(take[_Q75] - take[_Q25]) / 2,
        float(feed[_Q75] - feed[_Q25]) / 2,
        float(window.p_dual[0]),
        float(take[_MID]),
        float(feed[_MID]),
    )


def band_for_slot(window: Window, j: int) -> dict:
    """Take and feed quantiles of slot ``j`` as ``{q: value}`` dicts."""
    return {
        "take": {q: float(window.take[i, j]) for i, q in enumerate(QUANTILES)},
        "feed": {q: float(window.feed[i, j]) for i, q in enumerate(QUANTILES)},
    }
