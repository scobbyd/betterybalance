"""Tiered feature matrices with knowable-at-forecast-time rules.

Forecast time (see forecast_time): for h >= 1 the forecast is made 60 s into the ISP k-h, where k
is the index of the target ISP. For h == 0 the forecast is made elapsed_min minutes into the
target ISP. Every feature below reads only data public at that time:

- A: day-ahead prices (auction at D-1 12:00) and the calendar. Known days ahead.
- B: IDA spreads, zero plus a 0/1 flag unless available_at(auction, target, forecast time).
- C: every lag and rolling feature reads the live feed's end-of-ISP values (buy_live_end,
  sell_live_end): lag1 from ISP k-h-1 (ended 60 s or more before the forecast), lags 2 to 4 and
  the rolling statistics from ISPs k-h-2 and older. Settled prices (buy_eur_mwh, sell_eur_mwh)
  are labels only: TenneT publishes settled values in bulk the next day, never during the
  running day, so they are not knowable at forecast time.
- D: live partial means of the forecast ISP k-h (minute 1 for h >= 1, minute elapsed for h == 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .feeds.ida import AUCTIONS, available_mask
from .timegrid import ISP_S, Table, align, local_calendar

HORIZONS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24)
TIERS = ("A", "B", "C", "D")
ELAPSED_MIN = (1, 5, 10)
FORECAST_LEAD_S = 60
SPIKE_EUR_MWH = 100.0

_LIVE_COLS = (
    "buy_live_end",
    "sell_live_end",
    "buy_live_e1",
    "sell_live_e1",
    "buy_live_e5",
    "sell_live_e5",
    "buy_live_e10",
    "sell_live_e10",
)
_SETTLED_COLS = ("buy_eur_mwh", "sell_eur_mwh", "regulation_state")
_NEIGHBOURS = tuple(range(-4, 5))
_SLOTS_PER_DAY = 96
_LAGS = (1, 2, 3, 4)
_ROLL_WINDOWS = (("1h", 4), ("3h", 12))
_ROLL_SHIFT = 2
# Quarters before a target row that build_matrix reads beyond the master's own columns: the
# tier C lags (h + 4) and the rolling statistics (h + 2 + window - 1); tier D reads h back,
# tiers A and B read the target row only. A master cut LOOKBACK_ISP rows before its first
# target row gives the same features on every target row (the day-ahead neighbours and day
# statistics are master columns, computed before the cut).
LOOKBACK_ISP = max(HORIZONS) + max(max(_LAGS), _ROLL_SHIFT + max(w for _, w in _ROLL_WINDOWS) - 1)


def _nan_like(grid: np.ndarray) -> np.ndarray:
    return np.full(len(grid), np.nan)


def _shift(arr: np.ndarray, s: int) -> np.ndarray:
    """Return out[k] = arr[k - s], NaN where k - s falls outside the array."""
    out = np.full(len(arr), np.nan)
    if s >= 0:
        if s < len(arr):
            out[s:] = arr[: len(arr) - s]
    elif -s < len(arr):
        out[:s] = arr[-s:]
    return out


def _rolling(arr: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and population SD of arr[j-w+1..j]. NaN if any value is NaN or j < w - 1."""
    mean = np.full(len(arr), np.nan)
    sd = np.full(len(arr), np.nan)
    if len(arr) >= w:
        win = np.lib.stride_tricks.sliding_window_view(arr, w)
        mean[w - 1 :] = win.mean(axis=1)
        sd[w - 1 :] = win.std(axis=1)
    return mean, sd


def _day_stats(da: np.ndarray, day_start: np.ndarray) -> Table:
    n = len(da)
    keys = ("da_day_mean", "da_day_min", "da_day_max", "da_rank_day")
    out = {k: np.full(n, np.nan) for k in keys}
    _, inv = np.unique(day_start, return_inverse=True)
    for g in range(int(inv.max()) + 1 if n else 0):
        idx = np.flatnonzero(inv == g)
        vals = da[idx]
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        out["da_day_mean"][idx] = vals[ok].mean()
        out["da_day_min"][idx] = vals[ok].min()
        out["da_day_max"][idx] = vals[ok].max()
        m = int(ok.sum())
        ranks = np.empty(m)
        ranks[np.argsort(vals[ok], kind="stable")] = np.arange(m)
        rk = np.full(len(idx), np.nan)
        rk[ok] = ranks / (m - 1) if m > 1 else 0.0
        out["da_rank_day"][idx] = rk
    return out


def build_master(
    da: Table,
    settled: Table,
    ida: dict[str, Table],
    live_isp: Table | None,
    grid: np.ndarray,
) -> Table:
    """Align every feed onto grid and add day-ahead neighbours, day statistics and calendar."""
    grid = np.asarray(grid, dtype=np.int64)
    if len(grid) > 1:
        bad = np.flatnonzero(np.diff(grid) != ISP_S)
        if len(bad):
            raise ValueError(
                f"grid must be sorted and contiguous at {ISP_S} s: gap after index {bad[0]}"
            )
    m: Table = {"ts": grid}
    m.update(align(da, grid, ["da_eur_mwh"]))
    m["ts"] = grid
    dav = m["da_eur_mwh"]
    for j in _NEIGHBOURS:
        name = "da_m%d" % -j if j < 0 else "da_p%d" % j
        if j == 0:
            continue
        m[name] = _shift(dav, -j)  # da_pK reads slot k+K, da_mK reads slot k-K
    cal = local_calendar(grid)
    day_start = grid - cal["slot_of_day"] * ISP_S
    m.update(_day_stats(dav, day_start))
    m.update({k: v for k, v in cal.items() if k != "ts"})
    m.update({k: v for k, v in align(settled, grid, list(_SETTLED_COLS)).items() if k != "ts"})
    for key in AUCTIONS:
        col = key.lower() + "_eur_mwh"
        if key in ida:
            m[col] = align(ida[key], grid, ["ida_eur_mwh"])["ida_eur_mwh"]
        else:
            m[col] = _nan_like(grid)
    if live_isp is None:
        for c in _LIVE_COLS:
            m[c] = _nan_like(grid)
    else:
        m.update({k: v for k, v in align(live_isp, grid, list(_LIVE_COLS)).items() if k != "ts"})
    return m


def cut_master(master: Table, lo: int, hi: int) -> Table:
    """Rows with lo <= ts < hi of every column (a contiguous slice of the grid)."""
    ts = np.asarray(master["ts"], dtype=np.int64)
    a, b = int(np.searchsorted(ts, int(lo))), int(np.searchsorted(ts, int(hi)))
    return {k: np.asarray(v)[a:b] for k, v in master.items()}


def forecast_time(ts_target: int, h: int, elapsed_min: int | None) -> int:
    """Epoch s at which the forecast is made. All availability decisions anchor here."""
    if h >= 1:
        return int(ts_target) - h * ISP_S + FORECAST_LEAD_S
    if elapsed_min is None:
        raise ValueError("elapsed_min is required for h == 0")
    return int(ts_target) + int(elapsed_min) * 60


@dataclass
class Matrix:
    X: np.ndarray
    names: list[str]
    y_buy: np.ndarray
    y_sell: np.ndarray
    y_dual: np.ndarray
    y_negsell: np.ndarray
    y_spike: np.ndarray
    ts_target: np.ndarray
    dropped: dict[str, int]


_A_BASE = (
    ["da_eur_mwh"]
    + ["da_m%d" % j for j in (4, 3, 2, 1)]
    + ["da_p%d" % j for j in (1, 2, 3, 4)]
    + ["da_day_mean", "da_day_min", "da_day_max", "da_rank_day"]
)


def _tier_a(m: Table) -> dict[str, np.ndarray]:
    f = {name: np.asarray(m[name], dtype=float) for name in _A_BASE}
    slot = np.asarray(m["slot_of_day"], dtype=float)
    hod = 2 * np.pi * slot / _SLOTS_PER_DAY
    dow = 2 * np.pi * np.asarray(m["dow"], dtype=float) / 7
    mon = 2 * np.pi * (np.asarray(m["month"], dtype=float) - 1) / 12
    f["hod_sin"], f["hod_cos"] = np.sin(hod), np.cos(hod)
    f["dow_sin"], f["dow_cos"] = np.sin(dow), np.cos(dow)
    f["is_weekend"] = np.asarray(m["is_weekend"], dtype=float)
    f["is_holiday"] = np.asarray(m["is_holiday"], dtype=float)
    f["month_sin"], f["month_cos"] = np.sin(mon), np.cos(mon)
    f["da_x_hod_sin"] = f["da_eur_mwh"] * f["hod_sin"]
    f["da_x_hod_cos"] = f["da_eur_mwh"] * f["hod_cos"]
    return f


def _tier_b(m: Table, h: int, elapsed_min: int | None) -> dict[str, np.ndarray]:
    ts = np.asarray(m["ts"], dtype=np.int64)
    da = np.asarray(m["da_eur_mwh"], dtype=float)
    f: dict[str, np.ndarray] = {}
    ft = np.array([forecast_time(int(t), h, elapsed_min) for t in ts], dtype=np.int64)
    for auction in AUCTIONS:
        val = np.asarray(m[auction.lower() + "_eur_mwh"], dtype=float)
        avail = available_mask(auction, ts, ft)
        spread = val - da
        known = avail & np.isfinite(spread)
        f[auction.lower() + "_spread"] = np.where(known, spread, 0.0)
        f[auction.lower() + "_known"] = known.astype(float)
    return f


def _tier_c(m: Table, h: int) -> dict[str, np.ndarray]:
    # Live end-of-ISP values only. Settled columns are labels and must not reach a feature.
    buy = np.asarray(m["buy_live_end"], dtype=float)
    sell = np.asarray(m["sell_live_end"], dtype=float)
    f: dict[str, np.ndarray] = {}
    for j in _LAGS:
        f["lag%d_buy" % j] = _shift(buy, h + j)
        f["lag%d_sell" % j] = _shift(sell, h + j)
    for label, w in _ROLL_WINDOWS:
        for side, arr in (("buy", buy), ("sell", sell)):
            mean, sd = _rolling(arr, w)
            f["roll%s_%s_mean" % (label, side)] = _shift(mean, h + _ROLL_SHIFT)
            f["roll%s_%s_sd" % (label, side)] = _shift(sd, h + _ROLL_SHIFT)
    l1b, l1s = f["lag1_buy"], f["lag1_sell"]
    nan = np.isnan(l1b) | np.isnan(l1s)
    f["lag1_neg"] = np.where(np.isnan(l1s), np.nan, (l1s < 0).astype(float))
    f["lag1_dual_live"] = np.where(nan, np.nan, (l1b > l1s).astype(float))
    return f


def _tier_d(m: Table, h: int, elapsed_min: int | None) -> dict[str, np.ndarray]:
    f: dict[str, np.ndarray] = {}
    if h >= 1:
        bname, sname = "buy_live_e1", "sell_live_e1"
        b = _shift(np.asarray(m[bname], dtype=float), h)
        s = _shift(np.asarray(m[sname], dtype=float), h)
    else:
        bname, sname = "buy_live_e%d" % elapsed_min, "sell_live_e%d" % elapsed_min
        b = np.asarray(m[bname], dtype=float)
        s = np.asarray(m[sname], dtype=float)
    f[bname], f[sname] = b, s
    if h == 0:
        f["elapsed_min"] = np.full(len(b), float(elapsed_min))
    f["live_dual_now"] = np.where(np.isnan(b) | np.isnan(s), np.nan, (b > s).astype(float))
    return f


def build_matrix(
    master: Table, tier: str, h: int, elapsed_min: int | None = None, labels: bool = True
) -> Matrix:
    """Feature matrix of one tier at horizon h. Rows with a NaN feature or label are dropped.

    ``labels`` False is the label-free mode for prediction on rows not settled yet: only rows
    with a NaN feature are dropped, and every ``y_*`` array is NaN (no label is read).
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier: {tier}")
    if h not in HORIZONS:
        raise ValueError(f"unknown horizon: {h}")
    if h == 0:
        if elapsed_min is None:
            raise ValueError("elapsed_min is required for h == 0")
        if elapsed_min not in ELAPSED_MIN:
            raise ValueError(f"elapsed_min must be one of {ELAPSED_MIN}")
    feats = _tier_a(master)
    level = TIERS.index(tier)
    if level >= 1:
        feats.update(_tier_b(master, h, elapsed_min))
    if level >= 2:
        feats.update(_tier_c(master, h))
    if level >= 3:
        feats.update(_tier_d(master, h, elapsed_min))
    names = list(feats)
    X = np.column_stack([feats[n] for n in names]).astype(np.float64)
    bad_feat = np.isnan(X).any(axis=1)
    if not labels:
        keep = ~bad_feat
        none = np.full(int(keep.sum()), np.nan)
        return Matrix(
            X=X[keep],
            names=names,
            y_buy=none,
            y_sell=none.copy(),
            y_dual=none.copy(),
            y_negsell=none.copy(),
            y_spike=none.copy(),
            ts_target=np.asarray(master["ts"], dtype=np.int64)[keep],
            dropped={"features": int(bad_feat.sum()), "labels": 0},
        )
    buy = np.asarray(master["buy_eur_mwh"], dtype=float)
    sell = np.asarray(master["sell_eur_mwh"], dtype=float)
    state = np.asarray(master["regulation_state"], dtype=float)
    da = np.asarray(master["da_eur_mwh"], dtype=float)
    bad_label = np.isnan(buy) | np.isnan(sell) | np.isnan(state) | np.isnan(da)
    keep = ~(bad_feat | bad_label)
    dropped = {
        "features": int(bad_feat.sum()),
        "labels": int((bad_label & ~bad_feat).sum()),
    }
    return Matrix(
        X=X[keep],
        names=names,
        y_buy=buy[keep],
        y_sell=sell[keep],
        y_dual=state[keep] == 2,
        y_negsell=sell[keep] < 0,
        y_spike=buy[keep] > da[keep] + SPIKE_EUR_MWH,
        ts_target=np.asarray(master["ts"], dtype=np.int64)[keep],
        dropped=dropped,
    )
