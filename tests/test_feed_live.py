import pathlib

import numpy as np
from betterybalance.core.feeds import live
from betterybalance.core.timegrid import isp_grid

FIX = (pathlib.Path(__file__).parent / "fixtures" / "tenergy_page.html").read_text()


def test_parse_tenergy_points():
    pts = live.parse_tenergy(FIX)
    assert len(pts) == 20 and pts[-1]["ts"] > pts[0]["ts"] and "buy_live" in pts[0]


def test_stale_mask_flags_frozen_hour():
    ts = np.arange(0, 7200, 60, dtype=np.int64)
    buy = np.where(ts < 3600, ts / 60.0, 59.0)  # frozen from minute 60 on
    sell = buy - 1
    m = live.stale_mask({"ts": ts, "buy_live": buy, "sell_live": sell}, max_unchanged_s=1800)
    assert not m[:60].any() and m[-1] and m[60 + 29] == False and m[60 + 31]  # noqa: E712


def test_minute_to_isp_end_and_elapsed_means():
    ts = np.arange(0, 1800, 60, dtype=np.int64)  # two ISPs
    buy = np.arange(30, dtype=float)
    sell = buy - 1
    t = live.minute_to_isp({"ts": ts, "buy_live": buy, "sell_live": sell}, isp_grid(0, 1800))
    assert t["buy_live_end"][0] == 14.0 and t["buy_live_e1"][1] == 15.0
    assert t["buy_live_e5"][0] == 2.0
