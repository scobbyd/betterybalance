"""The install-time backtest (design 11): the shadow battery over the recorder's hourly history.

``plan_backtest`` picks the span: the whole Amsterdam days of finite grid import hours, the last
``MAX_DAYS`` before today, from the first quarter the pack can price (its first training end
plus the replay embargo, ``replay.bands.embargo_lag``) to the last whole day the pack covers.
``run_backtest`` splits the hourly flows into quarters (``hourly_to_quarters``), builds the band
cache from the pack's model series (``cache_from_series``, the 4-day embargo, honest horizons)
and runs ``replay.engine.replay`` with the Previous Runs irradiance (``IrrByLead``) for the
window's PV regressors and the archive for the site-model fits. The PV fit is the replay's
first step: the walk-forward registry fits at the start and every Monday on the rows before
(the strings path when the flows carry measured strings pv1 .. pv8, else the blind fit;
``core.site.registry.weekly_update``).

The backtest writes the ``runs`` tables only: ``run_quarters`` (three paths per quarter) and
``run_decisions`` (the decision rows without the repeat holds, as the live chain writes them).
It never writes ``decisions`` or ``bookings``. The headline is the shadow's own result (house
minus shadow, design 10.4); the terminal valuation of the shadow's SoC change is printed
beside it, never added. Deterministic: the bootstrap draws with seed 0.

Nothing here imports Home Assistant. ``python -m betterybalance.live.backtest`` runs the same
pipeline on a flows CSV outside Home Assistant (the owner-data check of Task 23).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..core.features import HORIZONS, LOOKBACK_ISP, cut_master
from ..core.fmt import eu
from ..core.ledger import (
    bootstrap_interval,
    decision_row,
    is_repeat_hold,
    terminal_valuation,
    trailing_mean_buy,
)
from ..core.pack import DataPack
from ..core.site.flows import ac_legs
from ..core.store import Store
from ..core.timegrid import ISP_S, ZONE, Table
from ..replay import report as R
from ..replay.bands import cache_from_series, embargo_lag
from ..replay.disaggregate import hourly_to_quarters
from ..replay.engine import ReplayInputs, ReplayResult, replay
from ..replay.irradiance import IrrByLead
from ..replay.phases import actual_soc_kwh
from .site import LiveConfig

MAX_DAYS = 365
MIN_DAYS = 14
INDICATIVE_HOURLY = (
    "Indicative: replayed on hourly history, which flattens the quarters; on the owner's house "
    "the hourly result was 93 % of the minute result."
)
REASONS = ("no_pack", "short_history", "no_end_soc")
DAY_S = 86400
HOUR_S = 3600
FIT_DAYS = 90  # the PV registry's trailing window: hours this far before the start feed it
WINDOW_DAYS = 3  # the window reach after the last quarter: bands and irradiance cover it
TRAILING_BUY_DAYS = 7
BOOTSTRAP_SEED = 0
EST_S_PER_QUARTER = 0.035  # plan 2 on a desktop core; the first estimate before a day ends
# run rows per write transaction: the backtest's own connection holds SQLite's one write lock
# for milliseconds at a time, so the live minute's writes never wait long for it
WRITE_CHUNK = 2000
PATHS = (("virtual", "shadow"), ("house", "house"), ("actual", "actual"))
_NAN_BAND = {
    "take": dict.fromkeys((0.1, 0.5, 0.9), float("nan")),
    "feed": dict.fromkeys((0.1, 0.5, 0.9), float("nan")),
}


class Stopped(Exception):
    """``should_stop`` turned true: the run ends before its next step and writes nothing."""


@dataclass
class BacktestPlan:
    start: int
    end: int
    days: int
    reason: str | None  # None, "no_pack", "short_history" or "no_end_soc"


def _date(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date()


def _midnight(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=ZONE).timestamp())


def _ceil_day(ts: int) -> dt.date:
    """The Amsterdam day whose midnight is the first at or after ``ts``."""
    d = _date(ts)
    return d if _midnight(d) >= int(ts) else d + dt.timedelta(days=1)


def plan_backtest(hours: Table, pack: DataPack | None, now: int) -> BacktestPlan:
    """The backtest span (module docstring); ``reason`` says why it cannot run.

    ``no_pack`` without a pack; ``short_history`` when the span holds fewer than ``MIN_DAYS``
    whole days (also without a finite grid import hour); ``no_end_soc`` when the hour before
    the end has no finite ``soc_pct`` (the replay's terminal valuation reads the recorded SoC
    in the final hour, ``replay.phases.end_soc_anchor``). The end is the earliest of today's
    Amsterdam midnight, the midnight that closes the last finite hour's day and the last
    midnight the pack's quarters reach; the start the latest of the first whole day of finite
    hours, the first day the pack can price and ``MAX_DAYS`` before the end.
    """
    if pack is None:
        return BacktestPlan(0, 0, 0, "no_pack")
    ts = np.asarray(hours["ts"], dtype=np.int64)
    ok = np.isfinite(np.asarray(hours["grid_import_kwh"], dtype=float))
    if not ok.any():
        return BacktestPlan(0, 0, 0, "short_history")
    first, last_end = int(ts[ok][0]), int(ts[ok][-1]) + HOUR_S
    pack_end = int(np.asarray(pack.quarters["ts"], dtype=np.int64)[-1]) + ISP_S
    end_day = _date(min(last_end, _midnight(_date(now)), pack_end))
    priced = int(pack.train_ends[0]) + embargo_lag(HORIZONS)
    start_day = max(_ceil_day(first), _ceil_day(priced), end_day - dt.timedelta(days=MAX_DAYS))
    days = max((end_day - start_day).days, 0)
    start, end = _midnight(start_day), _midnight(end_day)
    if days < MIN_DAYS:
        return BacktestPlan(start, max(end, start), days, "short_history")
    last = ts == end - HOUR_S
    soc = np.asarray(hours["soc_pct"], dtype=float)[last]
    if not np.isfinite(soc).any():
        return BacktestPlan(start, end, days, "no_end_soc")
    return BacktestPlan(start, end, days, None)


def _finite(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _trailing_settled_buy(result: ReplayResult, end: int) -> float:
    """``ledger.trailing_mean_buy`` over the shadow's quarters: the settled buy (EUR/MWh) of
    the last 7 days, a quarter booked at a live price counted as NaN; NaN with none."""
    rows = result.virtual
    ts = np.array([r.ts for r in rows], dtype=np.int64)
    buy = np.array([r.buy_eur_mwh if r.settled else np.nan for r in rows], dtype=float)
    return trailing_mean_buy(buy, ts, end, TRAILING_BUY_DAYS)


def result_headline(result: ReplayResult) -> dict:
    """The headline of a backtest: the shadow's result (house minus shadow, the sum of
    ``result_pairs``), the mean per day with its 90 % bootstrap interval (seed 0), the days,
    the missing quarters (either path), the PV learning days, the irradiance note, the
    indicative sentence and the terminal valuation.

    ``terminal_eur`` values the shadow's end SoC minus its start SoC
    (``site_report["soc_start_kwh"]``, set by ``run_backtest``) at the trailing 7-day settled
    buy and the discharge leg ``site_report["ed"]`` (``ledger.terminal_valuation``). It is
    shown beside the result, never added (design 10.4). Non-finite values are None, so the
    dict is JSON for the store and the sensor.
    """
    vals = np.asarray(result.result_pairs["result_eur"], dtype=float)
    mean, lo, hi = bootstrap_interval(vals, seed=BOOTSTRAP_SEED)
    rep = result.site_report
    end = result.virtual[-1].ts + ISP_S if result.virtual else 0
    soc0 = rep.get("soc_start_kwh")
    soc_end = float(result.soc_virtual_kwh[-1]) if len(result.soc_virtual_kwh) else float("nan")
    terminal = float("nan")
    if soc0 is not None and rep.get("ed") is not None:
        trailing = _trailing_settled_buy(result, end)
        terminal = terminal_valuation(soc_end, float(soc0), trailing, float(rep["ed"]))
    missing = sum(v.missing or h.missing for v, h in zip(result.virtual, result.house))
    return {
        "result_eur": float(vals.sum()) if len(vals) else 0.0,
        "mean_eur_day": _finite(mean),
        "lo_eur_day": _finite(lo),
        "hi_eur_day": _finite(hi),
        "days": int(len(vals)),
        "missing_quarters": int(missing),
        "learning_days": int(result.learning_days),
        "irradiance": str(rep.get("irradiance", "")),
        "indicative": INDICATIVE_HOURLY,
        "terminal_eur": _finite(terminal),
    }


def _day_ends(start: int, end: int) -> dict[int, int]:
    """Quarters done (from ``start``) at which an Amsterdam day is complete -> day number."""
    out, day, d = {}, 0, _date(start)
    while _midnight(d) < end:
        d += dt.timedelta(days=1)
        day += 1
        out[(min(_midnight(d), end) - start) // ISP_S] = day
    return out


def _quarter_rows(result: ReplayResult) -> list[dict]:
    rows = []
    for attr, path in PATHS:
        for r in getattr(result, attr):
            rows.append(
                {
                    "quarter_ts": int(r.ts), "path": path, "import_kwh": float(r.import_kwh),
                    "export_kwh": float(r.export_kwh), "money_eur": float(r.money_eur),
                    "missing": bool(r.missing),
                }
            )
    return rows


def _model_version(cache, qts: int) -> str:
    """The pack model version that priced quarter ``qts`` (``fold_of``), "" off the folds."""
    k = int(np.searchsorted(cache.ts, qts))
    if k >= len(cache.ts) or int(cache.ts[k]) != qts or int(cache.fold_of[k]) < 0:
        return ""
    return str(cache.meta["model_versions"][int(cache.fold_of[k])])


def _write_runs(store: Store, run_id: int, quarters: list[dict], decisions: list[dict]) -> None:
    """The run's rows in chunks of ``WRITE_CHUNK``; on a raise the run's rows go again."""
    try:
        for i in range(0, len(quarters), WRITE_CHUNK):
            store.put_run_quarters(run_id, quarters[i : i + WRITE_CHUNK])
        for i in range(0, len(decisions), WRITE_CHUNK):
            store.put_run_decisions(run_id, decisions[i : i + WRITE_CHUNK])
    except BaseException:
        store.delete_run_rows(run_id)
        raise


class _Nowcast:
    def __init__(self, take: float, feed: float):
        self.take_point, self.feed_point = take, feed


def _decision_rows(result: ReplayResult, cache, capacity_kwh: float) -> list[dict]:
    """The decision rows of the run without the repeat holds (``ledger.is_repeat_hold``), as
    the live chain writes them: slot 0's band of the quarter's ranking, the nowcast points
    the trader saw, the SoC before the minute and the pack model that priced the quarter."""
    marks = result.minute_marks
    rows, prev = [], None
    versions: dict[int, str] = {}
    for i, d in enumerate(result.decisions):
        if not is_repeat_hold(prev, d):
            qts = int(d.ts) - int(d.ts) % ISP_S
            soc, take, feed = (float(x) for x in marks[i])
            nc = None if not (math.isfinite(take) and math.isfinite(feed)) else _Nowcast(take, feed)
            if qts not in versions:
                versions[qts] = _model_version(cache, qts)
            band = result.quarter_bands.get(qts, _NAN_BAND)
            pct = soc / capacity_kwh * 100.0 if capacity_kwh > 0 else float("nan")
            rows.append(decision_row(d, band, nc, pct, versions[qts]))
        prev = d
    return rows


def run_backtest(
    store: Store,
    lc: LiveConfig,
    pack: DataPack,
    plan: BacktestPlan,
    irr_archive: Table,
    irr_by_lead: IrrByLead,
    progress: Callable[[int, int], None],
    run_id: int,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[dict, str]:
    """Replay the shadow over ``plan`` (module docstring) and fill run ``run_id``.

    Blocking (the coordinator runs it in the executor on a ``Store`` connection of its own,
    never the live one). ``progress(day, days)`` follows every Amsterdam day. ``should_stop``
    is read after the hourly split, after every band cell and after every quarter; when it
    turns true the run raises ``Stopped`` before it writes a row. The run rows go in chunks
    of ``WRITE_CHUNK``; a failure while writing deletes the run's rows written so far
    (``Store.delete_run_rows``), so a failed run leaves none and no other row is touched.
    Returns the headline (``result_headline``) and the markdown report. Raises
    ``ValueError`` for a plan with a reason; the caller marks the run failed on any raise.
    """
    if plan.reason is not None:
        raise ValueError(f"the backtest plan is refused: {plan.reason}")

    def check(*_args) -> None:
        if should_stop is not None and should_stop():
            raise Stopped("the backtest was stopped")

    cfg = lc.cfg
    hours = store.hours(plan.start - FIT_DAYS * DAY_S, plan.end)
    q = hourly_to_quarters(hours)
    check()
    pack_end = int(np.asarray(pack.quarters["ts"], dtype=np.int64)[-1]) + ISP_S
    reach = min(plan.end + WINDOW_DAYS * DAY_S, pack_end)
    cache = cache_from_series(pack, plan.start, reach, progress=check)
    master = cut_master(pack.master(), plan.start - LOOKBACK_ISP * ISP_S, reach)
    pk = cfg.pack
    floor, ceiling = pk.soc_floor * pk.capacity_kwh, pk.soc_ceiling * pk.capacity_kwh
    soc0 = actual_soc_kwh(q, None, plan.start, pk.capacity_kwh, at_end=False)
    soc0 = min(max(floor if not np.isfinite(soc0) else float(soc0), floor), ceiling)
    inputs = ReplayInputs(
        cfg, q, master, cache, irr_archive, plan.start, plan.end,
        seed=BOOTSTRAP_SEED, soc0_kwh=soc0, irr_forecast=irr_by_lead,
    )
    ends = _day_ends(plan.start, plan.end)

    def tick(done: int, total: int) -> None:
        check()
        day = ends.get(done)
        if day is not None:
            progress(day, plan.days)

    result = replay(inputs, progress=tick)
    _, ed = ac_legs(pk)
    result.site_report["soc_start_kwh"] = soc0
    result.site_report["ed"] = float(ed)
    decisions = _decision_rows(result, cache, pk.capacity_kwh)
    _write_runs(store, run_id, _quarter_rows(result), decisions)
    head = result_headline(result)
    meta = {
        "window": f"{_date(plan.start).isoformat()} to "
        f"{(_date(plan.end) - dt.timedelta(days=1)).isoformat()}",
        "flows": "hourly",
        "pack": pack.version,
    }
    return head, R.render_backtest_report(result, head, meta)


# -- the command line ------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """``python -m betterybalance.live.backtest``: the pipeline on a flows CSV (hourly rows)
    in a scratch store under ``--out``, with the open-meteo caches beside it. Prints the
    headline and the wall time; writes ``report.md`` and ``headline.json`` under ``--out``."""
    import argparse
    import json
    import sys
    import time
    from pathlib import Path

    from ..core.feeds.base import DayCache, FeedError
    from ..core.feeds.openmeteo import OpenMeteo, previous_runs
    from ..core.pack import load_pack
    from ..core.pricemodel import shipped_model
    from ..core.site.contract import from_dict, round_location
    from ..core.site.flows import read_flows

    p = argparse.ArgumentParser(prog="python -m betterybalance.live.backtest")
    p.add_argument("--flows", type=Path, required=True, help="hourly flows CSV")
    p.add_argument("--site", type=Path, required=True, help="site.json (core.site.contract)")
    p.add_argument("--pack", type=Path, required=True, help="the data pack folder")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--now", help="the run's local date, YYYY-MM-DD (default: today)")
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    cfg = from_dict(json.loads(a.site.read_text(encoding="utf-8")))
    pack = load_pack(a.pack)
    flows = read_flows(a.flows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "backtest.sqlite3"
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    store = Store(db)
    try:
        store.migrate()
        store.put_hours(flows)
        if a.now is None:
            now = int(time.time())
        else:
            now = _midnight(dt.date.fromisoformat(a.now)) + 12 * HOUR_S
        plan = plan_backtest(store.hours(0, 2**40), pack, now)
        if plan.reason is not None:
            print(f"refused: {plan.reason}", file=sys.stderr)
            return 2
        lat, lon = round_location(cfg.latitude, cfg.longitude)
        yesterday = _date(now) - dt.timedelta(days=1)
        first = _date(plan.start - FIT_DAYS * DAY_S)
        last = min(_date(plan.end) + dt.timedelta(days=WINDOW_DAYS), yesterday)
        print(f"plan: {plan.days} days from {_date(plan.start)}; fetching irradiance")
        try:
            archive = OpenMeteo().archive(
                lat, lon, first, last, DayCache(out / "openmeteo_archive.jsonl")
            )
            by_lead = previous_runs(
                lat, lon, _date(plan.start) - dt.timedelta(days=1),
                _date(plan.end) + dt.timedelta(days=WINDOW_DAYS),
                cache=DayCache(out / "openmeteo_previous_runs.jsonl"),
            )
        except FeedError as exc:
            print(f"error: {type(exc).__name__}: the irradiance could not be fetched",
                  file=sys.stderr)
            return 2
        t1 = time.perf_counter()
        lc = LiveConfig(cfg, "backtest", shipped_model(), "shipped", pack.version)
        run_id = store.new_run(now, plan.start, plan.end)

        def progress(day: int, days: int) -> None:
            if day % 10 == 0 or day == days:
                print(f"day {day} of {days} ({eu(time.perf_counter() - t1, 0)} s)", flush=True)

        head, report = run_backtest(
            store, lc, pack, plan, archive, IrrByLead(by_lead, archive), progress, run_id
        )
        store.finish_run(run_id, now, "done", head, out / "report.md")
    finally:
        store.close()
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "headline.json").write_text(json.dumps(head, indent=2, sort_keys=True) + "\n")
    wall = time.perf_counter() - t0
    for line in R.backtest_headline_lines(head):
        print(line)
    print(
        f"wall time {eu(wall / 60, 1)} min for {eu(plan.days, 0)} days "
        f"(replay {eu((time.perf_counter() - t1) / 60, 1)} min); the x86 budget is one year "
        "under 30 min"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
