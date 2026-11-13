"""Pull the open-meteo irradiance archive for one location into a quarter-grid CSV.

    python -m tools.irradiance_archive --lat 52.1 --lon 5.2 --from 2026-02-01 --to 2026-10-31 \
        --cache data/cache/openmeteo.jsonl --out irradiance.csv

Writes `ts,ghi_w_m2,dni_w_m2,dhi_w_m2,temp_c` (start-labelled quarters, plain decimals, empty
cell for a missing value) through core.feeds.openmeteo.write_irradiance_csv; read it back with
read_irradiance_csv. Days are fetched through OpenMeteo().archive with a per-day cache, so a
refused run resumes from the cache. `--from` and `--to` are both inclusive. `--to` is clipped
at yesterday (UTC) with a printed note, as the replay CLI's archive route does: the archive has
no future days, and a day cached from forecast values would stay in the cache.

Weather data by Open-Meteo.com (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from betterybalance.core.feeds.base import DayCache
from betterybalance.core.feeds.openmeteo import OpenMeteo, write_irradiance_csv
from betterybalance.core.fmt import eu


def clipped_end(end: dt.date, today: dt.date | None = None) -> dt.date:
    """``end``, at most the day before ``today`` (default: today in UTC)."""
    today = dt.datetime.now(dt.UTC).date() if today is None else today
    return min(end, today - dt.timedelta(days=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lat", type=float, required=True, help="latitude, rounded to 0,1")
    parser.add_argument("--lon", type=float, required=True, help="longitude, rounded to 0,1")
    parser.add_argument("--from", dest="start", required=True, help="first day, inclusive")
    parser.add_argument("--to", dest="end", required=True, help="last day, inclusive")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    start, asked = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    end = clipped_end(asked)
    if end < asked:
        print(
            f"Note: --to {asked.isoformat()} is clipped to {end.isoformat()}, yesterday in UTC: "
            "the archive has no later days."
        )
    if end < start:
        parser.error(f"--from {start.isoformat()} lies after yesterday in UTC ({end.isoformat()})")
    table = OpenMeteo().archive(args.lat, args.lon, start, end, DayCache(args.cache))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_irradiance_csv(table, args.out)
    print(f"Rows: {eu(len(table['ts']), 0)}")
    print(f"Wrote {args.out}")
    print("Weather data by Open-Meteo.com (CC BY 4.0)")


if __name__ == "__main__":
    main()
