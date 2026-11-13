"""Export the VictoriaMetrics minute archive of the live imbalance prices to a CSV.

Writes `ts,buy_live,sell_live` (epoch s, one row per minute, empty where a series has no sample).
The owner supplies the values at run time: VM_URL, BB_BUY_ENTITY, BB_SELL_ENTITY.
Uses urllib and csv only.

Minute convention (shared with core.feeds.live.minute_to_isp and the on-box recorder): the row
labelled t holds the samples of the minute that starts at t and is known at t + 60. VictoriaMetrics
evaluates `avg_over_time(...[1m])` at step T over the window (T - 60, T], so the exporter labels
that value T - 60 (start_labels). The window is then (t, t + 60], the same minute up to the
boundary instant, and known at t + 60.

Days are local Europe/Amsterdam calendar days. `--from` and `--to` are both inclusive: the export
starts at local midnight of `--from` and ends at the local midnight after `--to`, so the last
local day named is complete. The November full-month re-run uses `--to 2026-10-31`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
import zoneinfo

STEP_S = 60
CHUNK_S = 7 * 86400
METRIC = "homeassistant_sensor_unit_eur_per_mwh"


ZONE = zoneinfo.ZoneInfo("Europe/Amsterdam")


def _epoch(day: str) -> int:
    """Epoch s of local midnight at the start of the named day."""
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=ZONE).timestamp())


def window(first_day: str, last_day: str) -> tuple[int, int]:
    """Query bounds: local midnight of first_day to the local midnight after last_day."""
    after = dt.date.fromisoformat(last_day) + dt.timedelta(days=1)
    return _epoch(first_day), _epoch(after.isoformat())


def start_labels(body: dict) -> dict[int, float]:
    """Values of a query_range response, each relabelled from its window end T to T - 60."""
    out: dict[int, float] = {}
    for result in body["data"]["result"]:
        for ts, value in result["values"]:
            out[int(float(ts)) - STEP_S] = float(value)
    return out


def query_range(vm_url: str, entity: str, start: int, end: int) -> dict[int, float]:
    """Start-labelled minute averages of one entity, steps T in [start, end], 7-day chunks."""
    out: dict[int, float] = {}
    query = f'avg_over_time({METRIC}{{entity="{entity}"}}[1m])'
    t = start
    while t <= end:
        stop = min(t + CHUNK_S - STEP_S, end)
        params = urllib.parse.urlencode({"query": query, "start": t, "end": stop, "step": STEP_S})
        url = f"{vm_url.rstrip('/')}/api/v1/query_range?{params}"
        with urllib.request.urlopen(url, timeout=60) as response:
            body = json.load(response)
        out.update(start_labels(body))
        t = stop + STEP_S
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="start", required=True, help="first day, YYYY-MM-DD (local), inclusive"
    )
    parser.add_argument(
        "--to", dest="end", required=True, help="last day, YYYY-MM-DD (local), inclusive"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--vm-url", default=os.environ.get("VM_URL"))
    parser.add_argument("--buy-entity", default=os.environ.get("BB_BUY_ENTITY"))
    parser.add_argument("--sell-entity", default=os.environ.get("BB_SELL_ENTITY"))
    args = parser.parse_args()
    if not (args.vm_url and args.buy_entity and args.sell_entity):
        parser.error("need --vm-url, --buy-entity, --sell-entity (or their env vars)")
    start, end = window(args.start, args.end)
    buy = query_range(args.vm_url, args.buy_entity, start, end)
    sell = query_range(args.vm_url, args.sell_entity, start, end)
    stamps = sorted(buy.keys() | sell.keys())
    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "buy_live", "sell_live"])
        for ts in stamps:
            writer.writerow([ts, buy.get(ts, ""), sell.get(ts, "")])
    print(f"wrote {len(stamps)} rows to {args.out}")


if __name__ == "__main__":
    main()
