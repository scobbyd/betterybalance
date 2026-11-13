"""House load: a per-wall-slot median profile keyed on local clock time."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from ..timegrid import ZONE, nl_holidays

_DAY_S = 86400
_SLOTS = 96


def _utc_offset_s(ts: np.ndarray) -> np.ndarray:
    """Local UTC offset in seconds per timestamp. Offsets change on whole UTC hours."""
    hours = ts // 3600
    uniq, inv = np.unique(hours, return_inverse=True)
    offs = np.array(
        [
            int(dt.datetime.fromtimestamp(int(h) * 3600, ZONE).utcoffset().total_seconds())
            for h in uniq
        ],
        dtype=np.int64,
    )
    return offs[inv]


def local_fields(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wall_slot, weekday, day_number) in local Europe/Amsterdam time.

    day_number counts local days since 1970-01-01, weekday is 0 Monday to 6 Sunday.
    """
    ts = np.asarray(ts, dtype=np.int64)
    local = ts + _utc_offset_s(ts)
    day = local // _DAY_S
    sec = local - day * _DAY_S
    return (sec // 900).astype(np.int64), ((day + 3) % 7).astype(np.int64), day


def wall_slot(ts: np.ndarray) -> np.ndarray:
    """hour * 4 + minute // 15 in local time, 0..95 (computed from wall time, not elapsed time)."""
    return local_fields(ts)[0]


def _rest_class(weekday: np.ndarray, day: np.ndarray) -> np.ndarray:
    """True on weekends and Dutch public holidays."""
    rest = weekday >= 5
    if len(day) == 0:
        return rest
    holidays: set[int] = set()
    first = dt.date(1970, 1, 1) + dt.timedelta(days=int(day.min()))
    last = dt.date(1970, 1, 1) + dt.timedelta(days=int(day.max()))
    for year in range(first.year, last.year + 1):
        holidays |= {(h - dt.date(1970, 1, 1)).days for h in nl_holidays(year)}
    if holidays:
        rest = rest | np.isin(day, np.fromiter(holidays, dtype=np.int64))
    return rest


@dataclass
class LoadModel:
    profile_kw: np.ndarray  # (96,) or (2, 96): row 0 weekdays, row 1 weekend and holidays
    trusted: np.ndarray
    mean_kw: float
    weekday_split: bool


def load_median(
    load_kw: np.ndarray,
    ts: np.ndarray,
    now: int,
    mask: np.ndarray | None = None,
    days: int = 7,
    min_values: int = 4,
    weekday_split: bool = False,
) -> LoadModel:
    """Median load per wall slot over [now - days * 86400, now), skipping masked and NaN rows.

    A slot with fewer than min_values values is untrusted and takes the window mean.
    """
    load = np.asarray(load_kw, dtype=float)
    ts = np.asarray(ts, dtype=np.int64)
    keep = (ts >= now - days * _DAY_S) & (ts < now) & ~np.isnan(load)
    if mask is not None:
        keep &= ~np.asarray(mask, dtype=bool)
    vals = load[keep]
    mean_kw = float(vals.mean()) if len(vals) else 0.0
    slot, weekday, day = local_fields(ts[keep])
    rows = 2 if weekday_split else 1
    cls = (
        _rest_class(weekday, day).astype(np.int64)
        if weekday_split
        else np.zeros(len(vals), np.int64)
    )
    profile = np.full((rows, _SLOTS), mean_kw)
    trusted = np.zeros((rows, _SLOTS), dtype=bool)
    for r in range(rows):
        in_row = cls == r
        for s in range(_SLOTS):
            v = vals[in_row & (slot == s)]
            if len(v) >= max(min_values, 1):
                profile[r, s] = float(np.median(v))
                trusted[r, s] = True
    if not weekday_split:
        return LoadModel(profile[0], trusted[0], mean_kw, False)
    return LoadModel(profile, trusted, mean_kw, True)


def forecast_load_kw(model: LoadModel, slots_ts: np.ndarray) -> np.ndarray:
    """The profile at each timestamp's wall slot (and weekday class)."""
    slot, weekday, day = local_fields(slots_ts)
    if not model.weekday_split:
        return model.profile_kw[slot]
    return model.profile_kw[_rest_class(weekday, day).astype(np.int64), slot]
