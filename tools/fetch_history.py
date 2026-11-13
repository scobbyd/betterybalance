"""Pull the public price history into per-day caches and save one aligned npz.

Feeds: day-ahead (Energy-Charts), settled imbalance (TenneT) and the three intraday
auctions (Nord Pool). The adapters pace their own requests and keep a per-day cache, so
a refused run resumes from the cache on the next call.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
from pathlib import Path

import numpy as np
from betterybalance.core.feeds.base import DayCache, local_days
from betterybalance.core.feeds.energy_charts import fetch_day_ahead
from betterybalance.core.feeds.ida import AUCTIONS, HISTORY_START, fetch_ida
from betterybalance.core.feeds.tennet_settlement import fetch_settlement
from betterybalance.core.fmt import eu
from betterybalance.core.timegrid import ZONE, align, isp_grid

FEEDS = ("da", "settled", "ida")


def plan_requests(
    start: dt.date, end: dt.date, only: tuple[str, ...] = FEEDS
) -> list[tuple[str, dt.date, dt.date]]:
    """List the requests as (feed, first day, last day).

    Day-ahead: one tuple per calendar month that overlaps the range, with the full month.
    Settled: one tuple per local day. IDA: one tuple per auction and day, named ida_<AUCTION>.
    """
    plan: list[tuple[str, dt.date, dt.date]] = []
    if "da" in only:
        year, month = start.year, start.month
        while (year, month) <= (end.year, end.month):
            last = calendar.monthrange(year, month)[1]
            plan.append(("da", dt.date(year, month, 1), dt.date(year, month, last)))
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    if "settled" in only:
        plan.extend(("settled", day, day) for day in local_days(start, end))
    if "ida" in only:
        for auction in AUCTIONS:
            plan.extend((f"ida_{auction}", day, day) for day in local_days(start, end))
    return plan


def _midnight(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=ZONE).timestamp())


def _expected_mask(name: str, ts: np.ndarray, now: int, end_day: dt.date) -> np.ndarray:
    """Slots a feed should hold: published by now, and for IDA after its history start."""
    if name == "da":
        # Day-ahead for D+1 is public from about 13:00 on D.
        today = dt.datetime.fromtimestamp(now, ZONE).date()
        cutoff = _midnight(today + dt.timedelta(days=2))
        return ts < cutoff
    mask = ts < now
    if name.startswith("ida"):
        auction = name.split("_", 1)[1] if "_" in name else name.upper()
        mask &= ts >= _midnight(HISTORY_START[auction])
        if auction == "IDA3":
            hours = np.array([dt.datetime.fromtimestamp(int(t), ZONE).hour for t in ts])
            mask &= hours >= 12
    return mask


def coverage_line(name: str, ts: np.ndarray, values: np.ndarray, now: int, end_day: dt.date) -> str:
    expected = _expected_mask(name, ts, now, end_day)
    n_exp = int(expected.sum())
    present = expected & np.isfinite(values)
    n_pres = int(present.sum())
    pct = 100.0 * n_pres / n_exp if n_exp else 0.0
    gaps = np.flatnonzero(expected & ~np.isfinite(values))
    gap = dt.datetime.fromtimestamp(int(ts[gaps[0]]), ZONE).isoformat() if len(gaps) else "none"
    return f"{name}: expected {n_exp} rows, present {n_pres} ({eu(pct, 1)} %), first gap {gap}"


def _parse_day(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", type=_parse_day, required=True)
    parser.add_argument("--to", dest="end", type=_parse_day, required=True)
    parser.add_argument("--data", type=Path, required=True, help="data directory")
    parser.add_argument("--only", default=",".join(FEEDS), help="comma list of da,settled,ida")
    parser.add_argument("--offline", action="store_true", help="read the caches only")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)
    only = tuple(part for part in args.only.split(",") if part)
    unknown = set(only) - set(FEEDS)
    if unknown:
        parser.error(f"unknown feed: {', '.join(sorted(unknown))}")

    plan = plan_requests(args.start, args.end, only)
    counts: dict[str, int] = {}
    for feed, _, _ in plan:
        counts[feed] = counts.get(feed, 0) + 1
    for feed, n in counts.items():
        print(f"plan {feed}: {n} requests")
    if args.dry_run:
        return 0

    cache_dir = args.data / "cache"
    tables: dict[str, dict] = {}
    if "da" in only:
        tables["da"] = fetch_day_ahead(
            args.start, args.end, DayCache(cache_dir / "da.jsonl"), offline=args.offline
        )
    if "settled" in only:
        tables["settled"] = fetch_settlement(
            args.start, args.end, DayCache(cache_dir / "settled.jsonl"), offline=args.offline
        )
    if "ida" in only:
        for auction in AUCTIONS:
            tables[f"ida_{auction}"] = fetch_ida(
                auction,
                args.start,
                args.end,
                DayCache(cache_dir / f"ida_{auction}.jsonl"),
                offline=args.offline,
            )

    grid = isp_grid(_midnight(args.start), _midnight(args.end + dt.timedelta(days=1)))
    out: dict[str, np.ndarray] = {"ts": grid}
    wanted = {
        "da": [("da_eur_mwh", "da_eur_mwh")],
        "settled": [
            ("buy_eur_mwh", "buy_eur_mwh"),
            ("sell_eur_mwh", "sell_eur_mwh"),
            ("regulation_state", "regulation_state"),
        ],
    }
    for auction in AUCTIONS:
        wanted[f"ida_{auction}"] = [(f"ida{auction[-1]}_eur_mwh", "ida_eur_mwh")]
    for name in ("da", "settled", *(f"ida_{a}" for a in AUCTIONS)):
        if name not in tables:
            for key, _ in wanted[name]:
                out[key] = np.full(len(grid), np.nan)
            continue
        for key, col in wanted[name]:
            out[key] = align(tables[name], grid, [col])[col]
    args.data.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.data / "master_inputs.npz", **out)

    now = int(dt.datetime.now(ZONE).timestamp())
    for name, key in (
        ("da", "da_eur_mwh"),
        ("settled", "buy_eur_mwh"),
        ("ida_IDA1", "ida1_eur_mwh"),
        ("ida_IDA2", "ida2_eur_mwh"),
        ("ida_IDA3", "ida3_eur_mwh"),
    ):
        if name in tables:
            print(coverage_line(name, grid, out[key], now, args.end))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
