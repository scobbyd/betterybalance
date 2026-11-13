"""Record the Nord Pool continuous intraday statistics window into a day cache.

Run it from a timer every few hours. The route answers for about D-7 to D+1 only, so
history exists only if this tool runs regularly. The tool installs no timer.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
from betterybalance.core.feeds.base import DayCache, FeedError
from betterybalance.core.feeds.sidc_continuous import fetch_continuous_day
from betterybalance.core.timegrid import ZONE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="data directory")
    parser.add_argument("--days", type=int, default=8, help="days back from today")
    args = parser.parse_args(argv)
    cache = DayCache(args.data / "cache" / "idc_NL.jsonl")
    today = dt.datetime.now(ZONE).date()
    for offset in range(-args.days, 2):
        day = today + dt.timedelta(days=offset)
        try:
            table = fetch_continuous_day(day, cache, offline=False)
        except FeedError as exc:
            print(f"{day.isoformat()}  skipped: {exc}")
            continue
        seen = ~np.isnan(table["update_time"])
        count = int(np.sum(seen))
        updated = np.max(table["update_time"][seen]) if count else 0
        stamp = dt.datetime.fromtimestamp(int(updated), tz=dt.UTC).isoformat()
        print(f"{day.isoformat()}  rows={count}  update_time={stamp}")
    print("Run this tool from a timer every few hours. It installs no timer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
