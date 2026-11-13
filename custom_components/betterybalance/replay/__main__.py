"""Replay CLI: ``python -m betterybalance.replay``.

Argument parsing and the pipeline only: the master (``core.inputs.load_inputs``), the band
cache (built from ``from - 5 days``, the embargo plus the longest horizon, so the first
replay quarter has a band; saved under ``--cache`` with its ``cache_key`` in the file name
and reloaded when the key matches), the flows (hourly rows disaggregated flat), the
irradiance (``--irradiance`` CSV, else the open-meteo archive at ``--lat --lon`` through a
``DayCache`` under ``--cache``), the start anchor (the first day from ``--from`` with a
recorded SoC at its first instant, ``anchored_start``), the replay, the ladder with
``--ladder``, the hourly bias (quarter or minute flows only: hourly flows have no finer
booking to compare), the outputs and the headline. With ``--live-store`` the report states
the mean absolute difference of the daily result from the live days.

``--reexecute STORE`` re-executes the live site over ``[from, to + 1 day)`` from the inputs
the store holds (``replay.reexec``) into ``--out/reexec.sqlite3`` and prints
``identical: N decisions, M bookings`` or the first difference (exit 1). The window tables
come from the shipped model, or from ``--model`` (a ``core.pricemodel.to_json`` file).
``--pack-dir`` (the install folder ``<config>/betterybalance/pack``) maps every model of the
installed version folders (the active and the previous pack) by its model version
(``live.pack_update.models_from_dir``), so a span with a ``model_switch`` re-executes on the
tables of the model each quarter ran.

Every printed number goes through ``core.fmt.eu``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ..core.features import TIERS
from ..core.feeds.base import DayCache, FeedError
from ..core.feeds.openmeteo import OpenMeteo, read_irradiance_csv
from ..core.fmt import eu
from ..core.inputs import load_inputs
from ..core.pricemodel import from_json, shipped_model
from ..core.site.contract import SiteConfig, from_dict, round_location
from ..core.site.flows import read_flows, to_quarters
from ..core.store import Store
from ..core.timegrid import ISP_S, Table, isp_start
from ..live.pack_update import models_from_dir
from ..live.site import LiveConfig
from . import report as R
from .bands import BandCache, build_band_cache, cache_key, load_cache, save_cache
from .disaggregate import HOUR_S, hourly_to_quarters
from .engine import ReplayInputs, replay
from .phases import DAY_S, anchored_first_day, local_date, local_midnight
from .reexec import reexecute

CACHE_LEAD_DAYS = 5  # the embargo (4 days) plus the longest horizon (6 h), rounded up
WINDOW_DAYS = 3  # the window reach ahead of the last replay quarter
REFIT_EVERY_DAYS = 7
TRAIN_DAYS = 365
FEED_ERROR_SENTENCE = (
    "the irradiance archive could not be fetched or was incomplete; check the network and retry"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m betterybalance.replay",
        description="Replay the shadow trader on a site's recorded flows and write the report.",
        epilog=(
            "--from and --to are inclusive local dates: the replay covers [from 00:00, "
            "to + 1 day 00:00) Europe/Amsterdam. The band cache covers 5 days before --from "
            "(the embargo plus the longest horizon) to 3 days after the end (the window "
            "reach). The hourly bias is measured for quarter or minute flows only."
        ),
    )
    p.add_argument("--site", type=Path, required=True, help="site.json (core.site.contract)")
    p.add_argument("--flows", type=Path, help="flows CSV (minute, quarter or hour)")
    p.add_argument("--flows-minute", type=Path, help="minute flows CSV for the minute loop")
    p.add_argument("--data", type=Path, help="directory with master_inputs.npz")
    p.add_argument(
        "--from",
        dest="date_from",
        metavar="DATE",
        required=True,
        help="first local day, YYYY-MM-DD (inclusive)",
    )
    p.add_argument(
        "--to",
        dest="date_to",
        metavar="DATE",
        required=True,
        help="last local day, YYYY-MM-DD (inclusive)",
    )
    p.add_argument("--out", type=Path, required=True, help="output directory")
    p.add_argument("--ladder", action="store_true", help="also run the 20-rung ladder")
    p.add_argument("--cache", type=Path, help="directory for the band and irradiance caches")
    p.add_argument("--irradiance", type=Path, help="irradiance CSV (ts plus the four columns)")
    p.add_argument("--lat", type=float, help="latitude for the open-meteo archive")
    p.add_argument("--lon", type=float, help="longitude for the open-meteo archive")
    p.add_argument("--live-store", type=Path, help="the live store, for the live-day difference")
    p.add_argument("--reexecute", metavar="STORE", type=Path, help="re-execute a live store")
    p.add_argument("--install-id-file", type=Path, help="the install id (with --reexecute)")
    p.add_argument("--model", type=Path, help="model JSON for --reexecute (default: shipped)")
    p.add_argument(
        "--pack-dir", type=Path, help="the installed pack folders, for --reexecute across switches"
    )
    a = p.parse_args(argv)
    if a.reexecute is not None:
        if a.install_id_file is None:
            p.error("--reexecute needs --install-id-file")
        return a
    if a.flows is None or a.data is None:
        p.error("give --flows and --data (or --reexecute STORE)")
    if a.irradiance is None and (a.lat is None or a.lon is None):
        p.error("give --irradiance, or --lat and --lon for the open-meteo archive")
    if a.irradiance is not None and (a.lat is not None or a.lon is not None):
        p.error("give --irradiance or --lat --lon, not both")
    if a.irradiance is None and a.cache is None:
        p.error("--lat --lon need --cache for the open-meteo day cache")
    return a


def window_bounds(date_from: str, date_to: str) -> tuple[int, int]:
    """``[from 00:00, to + 1 day 00:00)`` local; ``to`` is inclusive."""
    first, last = dt.date.fromisoformat(date_from), dt.date.fromisoformat(date_to)
    if last < first:
        raise ValueError(f"--to {date_to} lies before --from {date_from}")
    return local_midnight(first), local_midnight(last + dt.timedelta(days=1))


def cache_bounds(master: Table, start: int, end: int) -> tuple[int, int]:
    """The band cache test range: ``CACHE_LEAD_DAYS`` before ``start`` to ``WINDOW_DAYS``
    after ``end``, clipped at the end of the master."""
    last = int(np.asarray(master["ts"], dtype=np.int64)[-1]) + ISP_S
    return start - CACHE_LEAD_DAYS * DAY_S, min(end + WINDOW_DAYS * DAY_S, last)


def quarter_flows(flows: Table) -> tuple[Table, str]:
    """Quarter flows and their source resolution (``hourly`` or ``quarter``).

    Hourly rows split flat into quarters (``hourly_to_quarters`` without a plant shape);
    rows of at most 900 s aggregate (``to_quarters``). Mixed hourly and finer rows raise.
    """
    step = np.asarray(flows["step_s"], dtype=np.int64)
    if len(step) and (step == HOUR_S).all():
        return hourly_to_quarters(flows), "hourly"
    if (step > ISP_S).any():
        raise ValueError("flows mix hourly rows with finer rows")
    return to_quarters(flows), "quarter"


def cache_path(
    master: Table, start: int, end: int, cache_dir: Path, tiers=TIERS
) -> tuple[str, Path]:
    """The band cache key of the window and its file under ``cache_dir``
    (``bands_`` plus the first 16 hex digits of the key)."""
    lo, hi = cache_bounds(master, start, end)
    key = cache_key(master, lo, hi, REFIT_EVERY_DAYS, TRAIN_DAYS, tiers=tiers)
    return key, _cache_file(cache_dir, key)


def _cache_file(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"bands_{key[:16]}.npz"


def band_cache(
    master: Table, start: int, end: int, cache_dir: Path | None, tiers=TIERS, progress=None
) -> BandCache:
    """Load the band cache whose key matches, else build it (and save it under ``cache_dir``).

    ``tiers`` limits the cells (the key covers them); ``progress`` goes to
    ``build_band_cache``.
    """
    lo, hi = cache_bounds(master, start, end)
    key = cache_key(master, lo, hi, REFIT_EVERY_DAYS, TRAIN_DAYS, tiers=tiers)
    path = None if cache_dir is None else _cache_file(cache_dir, key)
    if path is not None and path.exists():
        cache = load_cache(path)
        if cache.meta.get("key") == key:
            return cache
    cache = build_band_cache(
        master,
        lo,
        hi,
        tiers=tiers,
        refit_every_days=REFIT_EVERY_DAYS,
        train_days=TRAIN_DAYS,
        progress=progress,
    )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_cache(cache, path)
    return cache


def irradiance(a: argparse.Namespace, q: Table, end: int) -> Table:
    """The irradiance CSV, else the open-meteo archive from the first flows day to the
    window reach after ``end``, cached per day. The last day is clipped at yesterday by the
    machine's clock (``date.today()``): the archive has no future days, so a window that
    ends today or later gets its last irradiance days from the CSV route only."""
    if a.irradiance is not None:
        return read_irradiance_csv(a.irradiance)
    lat, lon = round_location(a.lat, a.lon)
    tag = hashlib.sha256(json.dumps([lat, lon]).encode()).hexdigest()[:12]
    cache = DayCache(Path(a.cache) / f"openmeteo_{tag}.jsonl")
    first = local_date(int(np.asarray(q["ts"], dtype=np.int64)[0]))
    last = min(local_date(end + WINDOW_DAYS * DAY_S), dt.date.today() - dt.timedelta(days=1))
    return OpenMeteo().archive(lat, lon, first, last, cache)


def _midnight_on_or_after(t: int) -> dt.date:
    day = local_date(t)
    return day if local_midnight(day) >= t else day + dt.timedelta(days=1)


def earliest_from(master: Table) -> dt.date:
    """The first local day the CLI can start at: its band cache starts ``CACHE_LEAD_DAYS``
    earlier, and the first fold needs a training quarter of the master before that."""
    first = int(np.asarray(master["ts"], dtype=np.int64)[0])
    return _midnight_on_or_after(first + CACHE_LEAD_DAYS * DAY_S + ISP_S)


def _embargo_error(day: dt.date) -> ValueError:
    return ValueError(
        f"--from lies inside the band embargo; the earliest valid start is {day.isoformat()}"
    )


def check_start(master: Table, start: int, cache: BandCache | None = None) -> None:
    """Raise a plain ``ValueError`` naming the earliest valid start when ``start`` has no band.

    Without ``cache``: against the master (before the cache is built). With it: against the
    cache's first priced quarter (``report.earliest_start``).
    """
    if start < local_midnight(earliest_from(master)):
        raise _embargo_error(earliest_from(master))
    if cache is None:
        return
    earliest = R.earliest_start(cache)
    if earliest is None:
        raise ValueError("the band cache prices no quarter; the master is too short")
    if start < earliest:
        raise _embargo_error(_midnight_on_or_after(earliest))


def anchored_start(
    q: Table, minute: Table | None, date_from: str, date_to: str, capacity_kwh: float
) -> str:
    """The first local day from ``--from`` with a recorded SoC at its start anchor
    (``phases.anchored_first_day``), printed when it is not ``--from``.

    ``tools.export_site`` writes hourly flows from ``--from`` 00:00, so the quarter that ends
    at that midnight is absent and a replay from the export's first day would start the
    virtual pack at the floor. The end anchor raises when the recorded SoC is missing at the
    end and in the final 60 minutes before it.
    """
    first = anchored_first_day(
        q, minute, dt.date.fromisoformat(date_from), dt.date.fromisoformat(date_to), capacity_kwh
    ).isoformat()
    if first != date_from:
        print(f"no recorded SoC at the start of {date_from} 00:00; the replay starts {first}")
    return first


def _progress(label: str):
    def tick(done: int, total: int) -> None:
        print(f"{label} {eu(done, 0)} of {eu(total, 0)}", file=sys.stderr)

    return tick


def _first_version(store: Store, start: int, end: int) -> str | None:
    """The model version of the first ``windows`` row in ``[start, end)``."""
    for qts in range(int(isp_start(start)), end, ISP_S):
        w = store.window(qts)
        if w is not None and w["model_version"] is not None:
            return str(w["model_version"])
    return None


def run_reexecute(a: argparse.Namespace) -> int:
    """``--reexecute``: 0 when identical, 1 at the first difference."""
    if not Path(a.reexecute).is_file():
        raise ValueError(f"no store at {a.reexecute}")
    cfg: SiteConfig = from_dict(json.loads(Path(a.site).read_text(encoding="utf-8")))
    install_id = Path(a.install_id_file).read_text(encoding="utf-8").strip()
    if a.model is not None:
        model = from_json(Path(a.model).read_text(encoding="utf-8"))
    else:
        model = shipped_model()
    start, end = window_bounds(a.date_from, a.date_to)
    src = Store(a.reexecute)
    out_path = Path(a.out) / "reexec.sqlite3"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        Path(str(out_path) + suffix).unlink(missing_ok=True)
    out = Store(out_path)
    try:
        out.migrate()
        version = _first_version(src, start, end) or "unknown"
        lc = LiveConfig(cfg, install_id, model, version)
        models = None
        if a.pack_dir is not None:
            models = models_from_dir(a.pack_dir, shipped_model())
        r = reexecute(src, lc, start, end, out, models=models)
    finally:
        out.close()
        src.close()
    if r.differences:
        print(f"first difference: {r.differences[0]}")
        return 1
    print(f"identical: {eu(r.n_decisions, 0)} decisions, {eu(r.n_bookings, 0)} bookings")
    return 0


def run(a: argparse.Namespace) -> None:
    cfg: SiteConfig = from_dict(json.loads(Path(a.site).read_text(encoding="utf-8")))
    if a.ladder:
        R.check_ladder_default(cfg.knobs)  # before hours of replays, not after them
    window_bounds(a.date_from, a.date_to)  # --to before --from raises before any read
    master = load_inputs(a.data)
    q, resolution = quarter_flows(read_flows(a.flows))
    minute = read_flows(a.flows_minute) if a.flows_minute is not None else None
    first = anchored_start(q, minute, a.date_from, a.date_to, cfg.pack.capacity_kwh)
    start, end = window_bounds(first, a.date_to)
    check_start(master, start)
    cache = band_cache(master, start, end, a.cache)
    check_start(master, start, cache)
    irr = irradiance(a, q, end)
    inputs = ReplayInputs(cfg, q, master, cache, irr, start, end, flows_minute=minute)
    result = replay(inputs)
    ladder = selection = None
    if a.ladder:
        ladder = R.run_ladder(inputs, progress=_progress("rung"))
        selection = R.select_default(
            ladder, R.default_rung_name(cfg.knobs), set(map(str, result.pairs["day"])), set()
        )
    bias = R.HOURLY_INPUT if resolution == "hourly" else R.measure_hourly_bias(inputs, result)
    meta = {
        "window": f"{first} to {a.date_to}",
        "flows": resolution if minute is None else f"{resolution}, minute loop",
    }
    live = Store(a.live_store) if a.live_store is not None else None
    try:
        R.write_outputs(result, a.out, meta, ladder, selection, bias, cfg=cfg, live_store=live)
        lines = R.live_difference_lines(result, live) if live is not None else []
    finally:
        if live is not None:
            live.close()
    for line in R.headline_lines(result):
        print(line)
    if resolution == "hourly":
        for line in R.hourly_input_lines():
            print(line)
    if selection is not None:
        print(f"Chosen rung: {selection['chosen']}")
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        if a.reexecute is not None:
            return run_reexecute(a)
        run(a)
    except FeedError as exc:
        # a FeedError message carries the request URL and its parameters (the location):
        # print the class and a fixed sentence only
        print(f"error: {type(exc).__name__}: {FEED_ERROR_SENTENCE}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
