"""The site's live master: the store's features, live minutes and settlements as a master.

``LiveSite`` (``live.site``) serves its bands from ``live_master``. Re-execution
(``replay.reexec``) never builds a master: it reads the bands the live box recorded.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np

from ..core.features import build_master
from ..core.feeds.ida import AUCTIONS
from ..core.feeds.live import minute_to_isp
from ..core.store import Store
from ..core.timegrid import ISP_S, ZONE, Table, align, isp_grid, isp_start
from ..replay.phases import local_midnight

MINUTE_S = 60
LIVE_OVERLAY_S = 4 * 3600  # live minutes fill the live columns of this many recent hours
LIVE_COLS = (
    "buy_live_end", "sell_live_end", "buy_live_e1", "sell_live_e1",
    "buy_live_e5", "sell_live_e5", "buy_live_e10", "sell_live_e10",
)


def _midnight(ts: int, days: int) -> int:
    """Local midnight of the Amsterdam day of ``ts`` plus ``days`` days."""
    d = dt.datetime.fromtimestamp(int(ts), ZONE).date() + dt.timedelta(days=days)
    return local_midnight(d)


def _known_at(col: str, ts: np.ndarray) -> np.ndarray:
    """When a live ISP column is public: the ISP end for ``*_end``, minute m for ``*_em``."""
    if col.endswith("_end"):
        return ts + ISP_S
    return ts + int(col.rsplit("_e", 1)[1]) * MINUTE_S


def live_master(store: Store, now: int, days_back: int = 30) -> Table:
    """``build_master`` over ``[local midnight of now - days_back, local midnight of day + 3)``.

    Day-ahead, IDA and live end-of-ISP columns come from ``features``; live minutes labelled
    before ``now`` fill the live columns of the last ``LIVE_OVERLAY_S`` where the features
    have none (the running quarter, a quarter not booked after a restart). A live value is
    kept only when it is public at ``now`` (``_known_at``). The settled columns are the
    newest settlements (labels only; the live bands are label-free).
    """
    now = int(now)
    lo, hi = _midnight(now, -days_back), _midnight(now, 3)
    grid = isp_grid(lo, hi)
    f = store.features(lo, hi)
    da = {"ts": f["ts"], "da_eur_mwh": f["da_eur_mwh"]}
    ida = {a: {"ts": f["ts"], "ida_eur_mwh": f[a.lower() + "_eur_mwh"]} for a in AUCTIONS}
    live = align(f, grid, list(LIVE_COLS))
    qnow = int(isp_start(now))
    minutes = store.live_minutes(qnow - LIVE_OVERLAY_S, now)
    if len(minutes["ts"]):
        fresh = align(
            minute_to_isp(minutes, isp_grid(qnow - LIVE_OVERLAY_S, qnow + ISP_S)),
            grid,
            list(LIVE_COLS),
        )
        for c in LIVE_COLS:
            live[c] = np.where(np.isnan(live[c]), fresh[c], live[c])
    for c in LIVE_COLS:
        live[c] = np.where(_known_at(c, grid) <= now, live[c], np.nan)
    rows = store.settlements(lo, hi)
    settled = {
        "ts": np.array([r["quarter_ts"] for r in rows], dtype=np.int64),
        "buy_eur_mwh": np.array([r["take"] for r in rows], dtype=float),
        "sell_eur_mwh": np.array([r["feed"] for r in rows], dtype=float),
        "regulation_state": np.array(
            [math.nan if r["regulation_state"] is None else r["regulation_state"] for r in rows],
            dtype=float,
        ),
    }
    return build_master(da, settled, ida, live, grid)
