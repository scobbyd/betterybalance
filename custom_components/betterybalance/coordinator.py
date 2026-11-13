"""The coordinator: drives ``live.site.LiveSite`` from HA's minute clock and state events.

It is the only module that touches HA's clock and state events. Four kinds of work run
apart, so the minutes never wait for a feed or the nightly work:

- The minute loop: a tick at second 2 of every minute (the 2 s are the recorder's grace),
  under the minute lock. Ticks coalesce: a tick that finds a newer one queued returns at
  once, and the one that runs reads the wall clock after it holds the lock (never the fire
  time). It closes every minute from the next one to the previous wall minute, in order,
  through the ``MinuteRecorder`` (on the loop: the state callbacks write the recorder on the
  loop too, so no thread shares it). Then one executor job: a gap above ``GAP_S`` (or a
  failed job before) restores the site instead (a ``restart`` event) and feeds the minutes
  from ``isp_start(now)`` to the previous wall minute as missing; else each closed minute
  goes to ``on_minute``. A failed job is retried with a back-off (``retry_delay``: the next
  tick, then 2, 4, 8 and at most 15 minutes after the failure); the ticks in between process
  no minute and write nothing. The first failure of an episode logs its traceback, a retry
  that fails again one line; from the second failure in a row the repair issue
  ``minute_loop_failed`` is up, and the first job that succeeds removes it. Minute 1 of a
  quarter is set up right before it
  (``setup_quarter(ts)``): at one label instant the events first, then the setup, then the
  minute, the order re-execution replays. Then the settlement rows pulled since the last tick
  are booked at the next minute (``LiveSite.settle``, rows published after that minute wait
  for a later tick), so their ``settle`` event applies before that minute, as re-execution
  reads it. A failed booking keeps its rows for the next tick; a second failure in a row
  drops the waiting rows and logs one line (the stored settlements come back through the
  catch-up at the next start). Then ``async_set_updated_data(snapshot)``; the snapshot's
  ``totals`` are ``live.results`` totals (a ``TotalsCache``).
- The result statistic: after a settlement booked quarters, one job (``stats``) computes the
  cumulative settled result per hour from the first changed hour to the last settled hour and
  imports it as ``betterybalance:result`` (``stats.import_result``). It reads beside the
  minutes. At start the stored settlements of the last ``SETTLE_CATCHUP_DAYS`` days are booked
  again where a booking lacks them (a pull whose booking a restart lost) and those days'
  hours are imported again.
- The feeds (``live.feedrun``): one run at a time, started by a tick; a tick that finds the
  run still going skips it and the catch-up window of ``live.feeds`` retries the slot.
  ``last_try`` is the run's own wall time; the live poll labels its row by the wall time
  after its response.
- The nightly work in ``[03:30, 04:15)`` Amsterdam, once per local day (held in memory, so
  an HA restart inside the window runs it again; prune, vacuum and a refit are safe to
  repeat): the settlement pull of older gaps, the refit on Mondays or when the drift alarm
  turned on, ``prune`` and ``vacuum``. The refit fits beside the minutes (``LiveSite.fit``
  writes nothing) and is adopted under the minute lock at the next minute the loop
  processes (``adopt_refit``), so its event applies before that minute, as re-execution
  reads it.
- The pack update at 04:10 Amsterdam (the slot and catch-up window of ``live.feeds``) and
  once at start: one run at a time, beside the minutes. It reads the release at the
  ``pack_source`` option (``live.pack_update.check_release``), installs a newer one into
  ``<config>/betterybalance/pack/`` (``install``, atomic) and records it in the store's
  ``pack`` table. Then, under the minute lock, the site takes the pack version and
  ``choose_model`` (never an older model than the one in use); a newer model is set at the
  next minute the loop processes (``LiveSite.set_model``), so its ``model_switch`` event
  applies before that minute and the switch lands at the next quarter boundary, as
  re-execution reads it. A failure counts on ``FeedState("pack")``. An empty ``pack_source``
  stops the updates: an installed pack keeps serving, and without one the shipped champion
  serves.
- At start the site begins on the model in effect by the ``model_switch`` events
  (``pack_update.model_state``), looked up in the active pack, then in the kept previous
  version folder, then in the folder of the version's own month (a model version names its
  pack's month, and an install never prunes the pack of the model in use). Only when none of
  them holds it (a pack that does not load, a new shipped champion) the start puts a
  ``model_switch`` to the shipped champion. A switch still pending from the run before is
  cancelled the same way (``choose_model`` puts it again when it still holds). Both go under
  the minute lock right before the ``restart`` event, at the restart's wall time (after every
  minute of the run before): the shipped switch is effective at ``isp_start(now)``, the
  quarter boundary where the restored run starts deciding, so the whole re-run quarter is on
  one model, and the cancel at the pending boundary; re-execution applies both at the
  ``restart`` event.

Every store and model call runs in the executor. The store is one SQLite connection
(``check_same_thread=False``) used from several executor threads, so every store call goes
through one threading lock (``_LockedStore``); each ``Store`` method is its own transaction,
so calls of different jobs never interleave inside one. Gates are Europe/Amsterdam wall time
whatever HA's time zone is. Shadow only: the coordinator reads states and calls no service.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import Event, HomeAssistant, State, callback
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_state_report_event,
    async_track_time_change,
)
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

from .config_flow import effective_options
from .const import DOMAIN, PACK_DIRNAME, STORE_DIRNAME, STORE_FILE
from .core.feeds import base
from .core.feeds import openmeteo as OM
from .core.pack import DataPack, PackError, load_pack
from .core.pricemodel import shipped_model
from .core.site import contract as C
from .core.store import Store
from .core.timegrid import ZONE, isp_start
from .history import backfill_hours
from .live import backtest as BT
from .live import feedrun as FR
from .live import feeds as FD
from .live import fetch as FX
from .live import pack_update as PU
from .live import results as R
from .live import views as V
from .live.recorder import OPTIONAL_SERIES, REQUIRED_SERIES, MinuteRecorder
from .live.site import REFIT_DAYS, LiveConfig, LiveSite
from .replay.irradiance import IrrByLead
from .stats import import_result

_LOGGER = logging.getLogger(__name__)

MINUTE_S = 60
DAY_S = 86400
TICK_SECOND = 2  # the recorder's grace after the minute boundary
GAP_S = 900  # a longer gap since the next minute restores instead of looping minutes
RETRY_MAX_S = 900  # the longest back-off between two tries of a failing minute job
TIMING_N = 1440
BACKFILL_DAYS = 365
NIGHTLY = dt.time(3, 30)
NIGHTLY_WINDOW_S = 45 * 60
SETTLE_GAP_DAYS = 7  # the nightly pull looks this many days back past yesterday
SETTLE_CATCHUP_DAYS = SETTLE_GAP_DAYS + 2  # the start books and imports these days again
ARCHIVE_CACHE = "openmeteo_archive.jsonl"
# the backtest's Previous Runs days: a file of their own, never shared with the archive cache
PREVIOUS_RUNS_CACHE = "openmeteo_previous_runs.jsonl"
RUNS_DIRNAME = "runs"
INTERRUPTED = "interrupted by a restart of Home Assistant"
STOPPED = "stopped because the integration was unloaded or Home Assistant stopped"
BT_BUSY_S = 60.0  # the backtest connection waits this long for the live writer's lock
PAUSE_MAX_S = 30 * 60  # a backup pause ends by itself after this long
PAUSE_WAIT_S = 120.0  # a backup waits this long for the running minute job, then goes on
FEED_ERROR = "the irradiance could not be fetched from open-meteo; check the network and retry"
NO_PACK_SOURCE = "no pack source is set"
# unit_of_measurement -> factor to the unit the recorder reads (energy kWh, power W)
_TO = {"kWh": {"Wh": 0.001, "kWh": 1.0, "MWh": 1000.0}, "W": {"W": 1.0, "kW": 1000.0, "MW": 1e6}}


def unit_factor(unit: str | None, target: str | None) -> float:
    """Factor from ``unit`` to ``target`` (``kWh`` or ``W``); 1 for an unknown pair."""
    return _TO.get(target or "", {}).get(unit or "", 1.0)


def _irradiance_archive(cache_dir: Path, lat: float, lon: float, now: int) -> dict:
    """The open-meteo archive of the ``REFIT_DAYS`` local days before today (blocking)."""
    lat, lon = C.round_location(lat, lon)
    today = dt.datetime.fromtimestamp(int(now), ZONE).date()
    cache = base.DayCache(cache_dir / ARCHIVE_CACHE)
    return OM.OpenMeteo().archive(
        lat, lon, today - dt.timedelta(days=REFIT_DAYS), today - dt.timedelta(days=1), cache
    )


def _backtest_irradiance(
    cache_dir: Path, lat: float, lon: float, plan: BT.BacktestPlan, now: int
) -> tuple[dict, IrrByLead]:
    """The backtest's irradiance (blocking): the archive from ``FIT_DAYS`` before the start to
    the window reach (clipped at yesterday, the archive has no later day) for the site-model
    fits and as fallback, and the Previous Runs leads 1 to 3 over the UTC days start - 1 to
    end + ``WINDOW_DAYS`` (``IrrByLead``, day 0 from lead 1, its own cache file)."""
    lat, lon = C.round_location(lat, lon)
    day = lambda ts: dt.datetime.fromtimestamp(int(ts), ZONE).date()  # noqa: E731
    yesterday = day(now) - dt.timedelta(days=1)
    reach = day(plan.end) + dt.timedelta(days=BT.WINDOW_DAYS)
    archive = OM.OpenMeteo().archive(
        lat, lon, day(plan.start - BT.FIT_DAYS * DAY_S), min(reach, yesterday),
        base.DayCache(cache_dir / ARCHIVE_CACHE),
    )
    by_lead = OM.previous_runs(
        lat, lon, day(plan.start) - dt.timedelta(days=1), reach,
        cache=base.DayCache(cache_dir / PREVIOUS_RUNS_CACHE),
    )
    return archive, IrrByLead(by_lead, archive)


class BacktestRunning(Exception):
    """A backtest request while one runs (the service refuses it with ``backtest_running``)."""


class BacktestStopping(Exception):
    """A backtest request while the entry stops (refused with ``backtest_stopping``)."""


def _local(now: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(int(now), ZONE)


def _wall() -> int:
    return int(dt_util.utcnow().timestamp())


def _percentiles(values: deque) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    p50, p95 = np.percentile(np.asarray(values, dtype=float), [50, 95])
    return float(p50), float(p95)


class _Gate:
    """The store calls of every wrapped connection pass only while it is open (closed during
    a Home Assistant backup, ``BetteryCoordinator.async_pause_for_backup``)."""

    def __init__(self) -> None:
        self._open = threading.Event()
        self._open.set()
        self._locks: list = []  # the locks of the wrapped connections

    @property
    def is_open(self) -> bool:
        return self._open.is_set()

    def wait(self) -> None:
        self._open.wait()

    def register(self, lock) -> None:
        self._locks.append(lock)

    def unregister(self, lock) -> None:
        if lock in self._locks:
            self._locks.remove(lock)

    def close(self) -> None:
        """Close, then wait until the calls in flight on every connection ended (blocking)."""
        self._open.clear()
        for lock in list(self._locks):
            with lock:
                pass

    def open(self) -> None:
        self._open.set()


class _LockedStore:
    """A ``Store`` whose every call holds one threading lock (module docstring) and waits
    while the gate is closed."""

    def __init__(self, store: Store, gate: _Gate | None = None) -> None:
        self._store = store
        self._lock = threading.RLock()
        self._gate = gate if gate is not None else _Gate()
        self._gate.register(self._lock)

    def __getattr__(self, name: str):
        attr = getattr(self._store, name)
        if not callable(attr):
            return attr
        lock, gate = self._lock, self._gate

        def call(*args, **kwargs):
            while True:
                gate.wait()
                with lock:
                    if gate.is_open:  # a close between the wait and the lock: wait again
                        return attr(*args, **kwargs)

        return call


def scale_issue_id(entry_id: str) -> str:
    """The repair-issue id of the fill-in scale check of one entry."""
    return f"pv_scale_off_{entry_id}"


def loop_issue_id(entry_id: str) -> str:
    """The repair-issue id of a minute loop that keeps failing, of one entry."""
    return f"minute_loop_failed_{entry_id}"


def retry_delay(failures: int) -> int:
    """Seconds from the ``failures``-th failure in a row to the next try: 60 (the next tick),
    then doubled per failure up to ``RETRY_MAX_S``."""
    return min(MINUTE_S * 2 ** max(int(failures) - 1, 0), RETRY_MAX_S)


class BetteryCoordinator(DataUpdateCoordinator[dict]):
    """One site's live loop (module docstring)."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(hass, _LOGGER, config_entry=entry, name=DOMAIN, update_interval=None)
        self.entry = entry
        # the effective options this run started on (``__init__._async_entry_updated``)
        self.options = effective_options(entry.options)
        data = entry.data
        entities = dict(data["entities"])
        self.mapping = {k: v for k, v in entities.items() if isinstance(v, str)}
        self.signs = dict(data["signs"])
        self.counters: dict[str, list[str]] | None = None
        if data.get("grid_source", "inverter") != "inverter":
            self.counters = {
                "import": list(entities["grid_import_kwh"]),
                "export": list(entities["grid_export_kwh"]),
            }
        # the entities the recorder reads, each with the unit it reads them in
        self._target: dict[str, str | None] = {
            eid: ("W" if role.endswith("_power") else None)
            for role, eid in self.mapping.items()
            if role in (*REQUIRED_SERIES, *OPTIONAL_SERIES)
        }
        for ids in (self.counters or {}).values():
            self._target.update(dict.fromkeys(ids, "kWh"))
        self.site: LiveSite | None = None
        self.runner: FR.FeedRunner | None = None
        self.recorder: MinuteRecorder | None = None
        self.feeds = FR.new_states()
        self.fetch_ida = bool(self.options["fetch_ida"])
        self.feeds["ida"].enabled = self.fetch_ida  # off: Nord Pool is never called
        self.drift_pending = False
        self._drift_seen = False
        self._store: Store | None = None
        self._lock = asyncio.Lock()  # the minute lock
        self._jobs: dict[str, asyncio.Task] = {}  # feeds, nightly, backfill: one run each
        self._unsubs: list[Callable[[], None]] = []
        self._stopped = False
        self._ticks = 0  # fired ticks; a queued tick runs only when it is the newest
        self._next: int | None = None  # the next minute the recorder closes
        self._broken = False  # a failed site job: the next try restores
        self._failures = 0  # failed site jobs in a row
        self._retry_at = 0  # a broken loop tries again from this wall time
        self._settle_failures = 0  # failed settlement bookings in a row
        self._nightly_day: dt.date | None = None
        self._dir = Path(hass.config.path(STORE_DIRNAME))
        self._timing = {k: deque(maxlen=TIMING_N) for k in ("minute", "setup")}
        self._settle_wait: list[dict] = []  # pulled rows published after the last settle minute
        self._stats_from: int | None = None  # the first hour the stats job imports again
        self._totals = R.TotalsCache()
        self._plan_ranked = None  # the ranking the cached plan belongs to
        self._plan_cache: dict | None = None
        self._shipped: PU.Entry | None = None  # the shipped champion (model, version, end)
        self._in_use: PU.Entry | None = None  # the model in use or pending
        self._start_pack: DataPack | None = None  # the installed pack read at start
        self._start_switches: list = []  # (entry, effective ts, from version, reason)
        self._bt: dict[str, Any] = {"running": False, "progress": None, "eta_s": None}
        self._auto_backtest = bool(entry.data.get("backtest", False))  # the onboarding box
        self._gate = _Gate()  # closed while a Home Assistant backup copies the store
        self._paused = False  # the backup pause holds the minute lock
        self._unpause: Callable[[], None] | None = None  # the pause's time-out
        self._stop_unsub: Callable[[], None] | None = None  # the HA stop listener

    # -- start and stop -----------------------------------------------------------------

    def _open(self) -> Store:
        self._dir.mkdir(parents=True, exist_ok=True)
        store = Store(self._dir / STORE_FILE)
        self._store = store
        store.migrate()
        if store.meta("install_id") is None:
            store.set_meta("install_id", self.entry.data["install_id"])
        runs = store.runs()
        for run in runs:  # a run a restart cut off never finishes by itself
            if run["status"] == "running":
                store.delete_run_rows(run["run_id"])  # its partial quarters and decisions
                store.finish_run(run["run_id"], _wall(), "failed", {"error": INTERRUPTED}, None)
        if runs:  # the onboarding backtest ran (or was tried) before
            self._auto_backtest = False
        return store

    def _pack_source(self) -> str | None:
        return (self.entry.options.get("pack_source") or "").strip() or None

    def _installed_pack(self, store) -> DataPack | None:
        """The active installed pack when it loads, else None (whatever the pack source)."""
        row = store.active_pack()
        if row is None:
            return None
        try:
            return load_pack(Path(row["path"]))
        except PackError as err:
            _LOGGER.warning("installed data pack %s not used: %s", row["version"], err)
            return None

    def _previous_pack(self, store) -> DataPack | None:
        """The kept version folder before the active pack's, when it loads, else None."""
        row = store.active_pack()
        if row is None:
            return None
        folder = PU.previous_folder(Path(row["path"]).parent, row["version"])
        if folder is None:
            return None
        try:
            return load_pack(folder)
        except PackError as err:
            _LOGGER.warning("previous data pack %s not used: %s", folder.name, err)
            return None

    def _month_pack(self, version: str) -> DataPack | None:
        """The installed folder of a pack model version's month, when it loads, else None."""
        folder = PU.month_folder(self._dir / PACK_DIRNAME, version)
        if folder is None:
            return None
        try:
            return load_pack(folder)
        except PackError as err:
            _LOGGER.warning("data pack %s not used: %s", folder.name, err)
            return None

    def _put_start_switches(self, now: int) -> None:
        """Under the minute lock, before the restart: the switches ``_live_config`` found
        necessary, put at the restart's wall time ``now`` (module docstring)."""
        for to, at, was, reason in self._start_switches:
            detail = {
                "version": to[1], "effective_ts": int(at), "from_version": was,
                "cov_source": PU.cov_source(to[0]), "reason": reason,
            }
            self.site.store.put_event(now, "model_switch", detail)
        self._start_switches = []

    def _live_config(self, store=None, now: int | None = None) -> LiveConfig:
        """The site's config on the model in effect at ``now`` (module docstring); blocking.

        Writes nothing: the switches the start needs wait in ``_start_switches`` for
        ``_put_start_switches``. Without a store: the shipped champion and no pack.
        """
        site = {**self.entry.data["site"], "knobs": dict(self.options["knobs"])}
        shipped = self._shipped = PU.shipped_entry(shipped_model())
        self._start_switches = []
        pack = None if store is None else self._installed_pack(store)
        in_use = shipped
        if store is not None:
            now = _wall() if now is None else int(now)
            version, pending = PU.model_state(store.events(), now, shipped[1])
            got = PU.resolve(version, shipped, pack)
            if got is None:
                got = PU.resolve(version, shipped, None, self._previous_pack(store))
            if got is None:  # the kept folder of the version's own month
                got = PU.resolve(version, shipped, None, self._month_pack(version))
            if got is None:
                _LOGGER.warning("model %s is not available; the shipped champion serves", version)
                at = int(isp_start(now))
                self._start_switches.append((shipped, at, version, "model not available"))
            else:
                in_use = (got[0], version, got[1])
            if pending is not None:  # set_model puts it again when it still holds
                self._start_switches.append((in_use, pending[0], pending[1], "restart"))
        self._in_use, self._start_pack = in_use, pack
        return LiveConfig(
            cfg=C.from_dict(site),
            install_id=self.entry.data["install_id"],
            model=in_use[0],
            model_version=in_use[1],
            pack_version=None if pack is None else pack.version,
            skip_tiers=() if self.fetch_ida else ("B",),
        )

    def _adopt(self, pack: DataPack | None) -> None:
        """Under the minute lock: the site takes the pack's version and the model of
        ``choose_model``, set at the next minute the loop processes when it is newer."""
        site = self.site
        site.lc = replace(site.lc, pack_version=None if pack is None else pack.version)
        model, version, end = PU.choose(self._shipped, pack, self._in_use)
        if version != self._in_use[1]:
            site.set_model(model, version, site.next_minute)
            self._in_use = (model, version, end)

    def _seed(self, now: int) -> tuple[dict[str, int], bool]:
        """The feeds' ``last_ok`` from the store, and whether ``flows_hour`` is empty."""
        empty = not len(self.site.store.hours(now - BACKFILL_DAYS * DAY_S, now + 1)["ts"])
        return self.runner.seed(self.feeds, now), empty

    async def async_start(self) -> None:
        """Open the store, restore the site, subscribe the states and start the minute clock.

        A failure part-way stops what started (listeners, jobs, the store) and raises.
        """
        hass = self.hass
        try:
            store = await hass.async_add_executor_job(self._open)
            now = _wall()
            lc = await hass.async_add_executor_job(self._live_config, store, now)
            # the site reads its loss profile file: never on the event loop
            self.site = await hass.async_add_executor_job(
                LiveSite, _LockedStore(store, self._gate), lc
            )
            cfg = lc.cfg
            self.runner = FR.FeedRunner(
                self.site.store, cfg.latitude, cfg.longitude,
                lambda: self.entry.options.get("tennet_key"),
            )
            async with self._lock:
                self._new_recorder(now)
                await hass.async_add_executor_job(self._put_start_switches, now)
                await hass.async_add_executor_job(self._restart_job, now, self._soc_now())
                self._next = max(now // MINUTE_S * MINUTE_S, self.site.next_minute)
                pack, self._start_pack = self._start_pack, None
                await hass.async_add_executor_job(self._adopt, pack)
                catchup = now - SETTLE_CATCHUP_DAYS * DAY_S
                try:
                    await hass.async_add_executor_job(self._settle_catchup, catchup, now)
                except Exception:
                    _LOGGER.exception("settlement catch-up failed at %s", now)
                seeds, empty = await hass.async_add_executor_job(self._seed, now)
                for feed, ok in seeds.items():
                    self.feeds[feed].last_ok = ok
                if pack is not None:
                    row = await hass.async_add_executor_job(self.site.store.active_pack)
                    self.feeds["pack"].last_ok = int(row["installed_at"])
                ids = sorted(self._target)  # the mapped series and the grid counters
                self._unsubs.append(async_track_state_change_event(hass, ids, self._on_state))
                self._unsubs.append(async_track_state_report_event(hass, ids, self._on_report))
                self._unsubs.append(
                    async_track_time_change(hass, self._tick, second=TICK_SECOND)
                )
                self._stop_unsub = hass.bus.async_listen_once(
                    EVENT_HOMEASSISTANT_STOP, self._on_hass_stop
                )
                data = await hass.async_add_executor_job(self.snapshot)
            self.async_set_updated_data(data)
            if empty:
                self._spawn("backfill", self._backfill(now))
            else:
                await self._maybe_auto_backtest()
            self._queue_stats(catchup)
        except Exception:
            await self.async_stop()
            raise

    async def async_stop(self) -> None:
        """End a backup pause, cancel the listeners, let running jobs end, wait for a running
        tick, close the store."""
        self._stopped = True
        self.async_resume_after_backup()
        if self._stop_unsub is not None:
            self._stop_unsub()
            self._stop_unsub = None
        while self._unsubs:
            self._unsubs.pop()()
        running = [t for t in self._jobs.values() if not t.done()]
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        async with self._lock:
            if self._store is not None:
                await self.hass.async_add_executor_job(self._store.close)
                self._store = None

    @callback
    def _on_hass_stop(self, _event) -> None:
        """Home Assistant stops: running jobs (a backtest, a fit, the backfill) read
        ``_stopped`` and end promptly; a backup pause ends."""
        self._stop_unsub = None  # a fired once-listener is gone already
        self._stopped = True
        self.async_resume_after_backup()

    # -- Home Assistant backups ----------------------------------------------------------

    async def async_pause_for_backup(self) -> None:
        """Before a Home Assistant backup copies the config folder: hold the minute lock (the
        ticks wait and coalesce), close the store gate (every store call of every job and of
        the backtest waits; the calls in flight end first) and checkpoint the WAL into the
        database file (``wal_checkpoint(TRUNCATE)``), so the copy holds one consistent
        ``store.sqlite3`` and an empty WAL. Never raises: a pause that cannot start logs and
        lets the backup go on. It ends at ``async_resume_after_backup`` or by itself after
        ``PAUSE_MAX_S``; the minutes in between are processed afterwards as after any long
        job (a gap above ``GAP_S`` restores)."""
        if self._paused or self._stopped or self._store is None:
            return
        try:
            await asyncio.wait_for(self._lock.acquire(), PAUSE_WAIT_S)
        except TimeoutError:
            _LOGGER.warning("backup pause not taken: the minute job runs longer than %d s",
                            PAUSE_WAIT_S)
            return
        self._paused = True
        self._unpause = async_call_later(self.hass, PAUSE_MAX_S, self._pause_timed_out)
        try:
            busy = await self.hass.async_add_executor_job(self._close_and_checkpoint)
        except Exception as err:
            _LOGGER.warning("store checkpoint before the backup failed: %r", err)
        else:
            if busy:
                _LOGGER.warning("store checkpoint before the backup was not complete")

    def _close_and_checkpoint(self) -> int:
        """Close the gate, wait for the calls in flight and checkpoint (blocking); SQLite's
        busy flag (0 when the WAL was emptied)."""
        self._gate.close()
        return self._store.checkpoint()[0]

    @callback
    def _pause_timed_out(self, _now) -> None:
        self._unpause = None
        _LOGGER.warning("backup pause ended after %d s without the backup's end", PAUSE_MAX_S)
        self.async_resume_after_backup()

    @callback
    def async_resume_after_backup(self) -> None:
        """After the backup (also when it failed or timed out): open the gate, release the
        minute lock. Does nothing without a pause."""
        if self._unpause is not None:
            self._unpause()
            self._unpause = None
        if not self._paused:
            return
        self._paused = False
        self._gate.open()
        self._lock.release()

    def _spawn(self, kind: str, job: Coroutine) -> bool:
        """Start ``job`` as the one run of ``kind``; False (job dropped) while one runs."""
        task = self._jobs.get(kind)
        if self._stopped or (task is not None and not task.done()):
            job.close()
            return False
        self._jobs[kind] = self.entry.async_create_background_task(
            self.hass, job, f"{DOMAIN}_{kind}"
        )
        return True

    # -- state samples ------------------------------------------------------------------

    def _value(self, st: State | None) -> float | None:
        if st is None or st.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
            return None
        try:
            v = float(st.state)
        except ValueError:
            return None
        unit = st.attributes.get("unit_of_measurement")
        return v * unit_factor(unit, self._target.get(st.entity_id))

    @callback
    def _on_state(self, event: Event) -> None:
        st = event.data["new_state"]
        ts = st.last_reported_timestamp if st is not None else dt_util.utcnow().timestamp()
        if self.recorder is not None:
            self.recorder.sample(event.data["entity_id"], ts, self._value(st))

    @callback
    def _on_report(self, event: Event) -> None:
        st = event.data["new_state"]
        if self.recorder is not None:
            self.recorder.sample(
                event.data["entity_id"], event.data["last_reported"].timestamp(), self._value(st)
            )

    def _new_recorder(self, now: int) -> None:
        """A fresh recorder, each series' held value seeded from ``hass.states``."""
        self.recorder = MinuteRecorder(self.mapping, self.signs, grid_counters=self.counters)
        for eid in self._target:
            st = self.hass.states.get(eid)
            if st is not None:
                self.recorder.sample(eid, min(st.last_reported_timestamp, now), self._value(st))

    # -- the minute loop ----------------------------------------------------------------

    def _timed(self, kind: str, fn, *args):
        t0 = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self._timing[kind].append((time.perf_counter() - t0) * 1000.0)

    def _minute(self, ts: int, row: dict | None) -> int:
        """Process minute ``ts``; 1 when it ran, 0 when it lies before the next minute."""
        site = self.site
        if ts < site.next_minute:  # a clock jump back: restore resumes after the booked quarter
            return 0
        if ts - int(isp_start(ts)) == MINUTE_S:
            self._timed("setup", site.setup_quarter, ts)
        self._timed("minute", site.on_minute, ts, row)
        return 1

    def _soc_now(self) -> float | None:
        """The mapped SoC sensor's current state in %, None without a number (on the loop)."""
        eid = self.mapping.get("soc")
        return None if eid is None else self._value(self.hass.states.get(eid))

    def _restart_job(self, now: int, soc_pct: float | None) -> int:
        """``restore(now, soc_pct)`` (``soc_pct``: the SoC sensor read on the loop), then the
        minutes from ``isp_start(now)`` to the previous wall minute as missing (a minute-1
        setup among them, so the rest of the quarter has a ranking); the number of minutes
        processed."""
        site = self.site
        site.restore(now, soc_pct)
        return sum(
            self._minute(ts, None)
            for ts in range(site.next_minute, now // MINUTE_S * MINUTE_S, MINUTE_S)
        )

    def _minutes_job(self, rows: list[tuple[int, dict | None]]) -> int:
        return sum(self._minute(ts, row) for ts, row in rows)

    async def _tick(self, fire: dt.datetime) -> None:
        """One minute tick (module docstring); ``fire`` is ignored for the wall clock."""
        self._ticks += 1
        mine = self._ticks
        hass = self.hass
        async with self._lock:
            if mine != self._ticks or self._stopped or self.site is None:
                return  # a newer tick is queued, or the clock stopped
            now = _wall()
            w = now // MINUTE_S * MINUTE_S
            if not (self._broken and now < self._retry_at):  # a back-off tick does nothing
                try:
                    if self._broken or w - self._next > GAP_S:
                        self._new_recorder(now)
                        done = await hass.async_add_executor_job(
                            self._restart_job, now, self._soc_now()
                        )
                    else:
                        rows = [
                            (ts, self.recorder.close(ts)) for ts in range(self._next, w, MINUTE_S)
                        ]
                        self._next = max(self._next, w)
                        done = await hass.async_add_executor_job(self._minutes_job, rows)
                    self._next = max(w, self.site.next_minute)
                except Exception as err:
                    self._loop_failed(now, err)
                else:
                    self._loop_recovered(done)
            if not self._broken:
                try:
                    quarters = await hass.async_add_executor_job(self._settle_job)
                except Exception as err:
                    self._settle_failed(now, err)
                else:
                    self._settle_failures = 0
                    if quarters:
                        self._queue_stats(min(quarters))
            try:
                data = await hass.async_add_executor_job(self.snapshot)
            except Exception:
                _LOGGER.exception("snapshot failed at %s", now)
            else:
                drift = bool(data["status"]["drift"])
                if drift and not self._drift_seen:
                    self.drift_pending = True
                self._drift_seen = drift
                self.async_set_updated_data(data)
        self._spawn("feeds", self._feeds())
        st = self.feeds["pack"]
        if st.last_try is None or FD.due("pack", now, st):
            self._spawn("pack", self._pack())
        local = _local(now)
        due = dt.datetime.combine(local.date(), NIGHTLY, tzinfo=ZONE).timestamp()
        if self._nightly_day != local.date() and due <= now < due + NIGHTLY_WINDOW_S:
            if self._spawn("nightly", self._nightly(w)):
                self._nightly_day = local.date()

    def _loop_failed(self, now: int, err: Exception) -> None:
        """A failed site job: the back-off, one traceback per episode, the repair issue from
        the second failure in a row (module docstring)."""
        self._broken = True
        self._failures += 1
        delay = retry_delay(self._failures)
        self._retry_at = now + delay
        if self._failures == 1:
            _LOGGER.exception("minute loop failed at %s; the next tick restores", now)
            return
        _LOGGER.warning(
            "minute loop failed again at %s (%d in a row): %r; the next try is in %d s",
            now, self._failures, err, delay,
        )
        if self._failures == 2:
            ir.async_create_issue(
                self.hass,
                DOMAIN,
                loop_issue_id(self.entry.entry_id),
                is_fixable=False,
                severity=ir.IssueSeverity.ERROR,
                translation_key="minute_loop_failed",
                translation_placeholders={"error": (str(err) or repr(err))[:200]},
            )

    def _loop_recovered(self, minutes: int) -> None:
        """A site job that succeeded: no restore is due. The episode ends and its repair issue
        goes only when the job processed a minute (a restore at a quarter's first minute
        processes none and proves nothing yet)."""
        self._broken = False
        if self._failures and not minutes:
            return
        if self._failures:
            if self._failures > 1:
                ir.async_delete_issue(self.hass, DOMAIN, loop_issue_id(self.entry.entry_id))
            _LOGGER.warning("minute loop runs again after %d failures", self._failures)
        self._broken, self._failures, self._retry_at = False, 0, 0

    def _settle_failed(self, now: int, err: Exception) -> None:
        """A failed settlement booking: the first keeps its rows for the next tick (logged with
        the traceback); the second in a row drops the waiting rows (one line); later ones in
        the same episode drop them silently. The stored settlements come back through the
        catch-up at the next start (``_settle_catchup``)."""
        self._settle_failures += 1
        if self._settle_failures == 1:
            _LOGGER.exception("settlement booking failed at %s", now)
            return
        dropped, self._settle_wait = len(self._settle_wait), []
        if self._settle_failures == 2:
            _LOGGER.warning(
                "settlement booking failed again at %s: %r; %d waiting rows dropped, the next"
                " start books them from the store", now, err, dropped,
            )

    # -- feeds --------------------------------------------------------------------------

    async def _feeds(self) -> None:
        """One feed run: the due feeds at the run's wall time, ``FeedState`` on the loop."""
        hass, now = self.hass, _wall()
        try:
            due = await hass.async_add_executor_job(self.runner.due, self.feeds, now)
            first = {feed for feed in due if self.feeds[feed].last_try is None}
            for feed in due:
                self.feeds[feed].last_try = now
            results = await hass.async_add_executor_job(self.runner.run, due, now, first)
        except Exception:
            _LOGGER.exception("feed run failed at %s", now)
            return
        for feed, (ok, source, error) in results.items():
            st = self.feeds[feed]
            if source is not None:
                st.source = source
            if ok:
                st.last_ok, st.failures, st.last_error = now, 0, None
            else:
                st.failures, st.last_error = st.failures + 1, error
        if results:
            self._push_feeds()

    def _push_feeds(self) -> None:
        if self.data is not None and not self._stopped:
            feeds = {name: replace(st) for name, st in self.feeds.items()}
            self.async_set_updated_data({**self.data, "feeds": feeds})

    # -- the data pack ------------------------------------------------------------------

    def _pack_job(self, source: str, now: int) -> DataPack | None:
        """Install a newer release from ``source`` and record it (blocking); the pack the site
        should take, None when it has it already. Raises what the update raises."""
        store = self.site.store
        row = store.active_pack()
        installed = None
        if row is not None and Path(row["path"]).is_dir():
            installed = row["version"]
        rel = PU.check_release(source, installed)
        if rel is None:  # nothing newer; a pack the site does not use yet still counts
            return self._installed_pack(store) if self.site.lc.pack_version is None else None
        # the packs of the model in effect and of a pending one stay installed
        versions = (self.site.lc.model_version, self._in_use[1])
        keep = {m for m in map(PU.version_month, versions) if m is not None}
        path = PU.install(rel, self._dir / PACK_DIRNAME, keep=keep)
        pack = load_pack(path)
        store.put_pack(pack.version, now, pack.manifest, str(path))
        return pack

    async def _pack(self) -> None:
        """One pack update (module docstring); ``FeedState("pack")`` on the loop."""
        hass, now = self.hass, _wall()
        st = self.feeds["pack"]
        st.last_try = now
        source = self._pack_source()
        try:
            if source is None:  # no updates; an installed pack keeps serving
                st.last_error = NO_PACK_SOURCE
            else:
                pack = await hass.async_add_executor_job(self._pack_job, source, now)
                if pack is not None:
                    async with self._lock:
                        if not self._stopped:
                            await hass.async_add_executor_job(self._adopt, pack)
                st.last_ok, st.failures, st.last_error = now, 0, None
                if pack is not None:
                    await self._maybe_auto_backtest()
        except Exception as err:
            if not isinstance(err, PackError):
                _LOGGER.warning("pack update failed: %r", err)
            st.failures, st.last_error = st.failures + 1, str(err) or repr(err)
        self._push_feeds()

    # -- settlements and the result statistic --------------------------------------------

    def _settle_job(self) -> list[int]:
        """Book the pulled settlement rows at the next minute (under the minute lock).

        Rows published after that minute wait for a later tick: the ``settle`` event at the
        minute must find them published (``LiveSite.settle``). Every row stays waiting until
        ``settle`` returned, so a failed booking is tried again at the next tick.
        """
        rows = self._settle_wait + self.runner.take_pulled()
        self._settle_wait = rows
        if not rows:
            return []
        at = self.site.next_minute
        ready = [r for r in rows if int(r["published_at"]) <= at]
        done = self.site.settle(ready, at) if ready else []
        self._settle_wait = [r for r in rows if int(r["published_at"]) > at]
        return done

    def _settle_catchup(self, lo: int, now: int) -> list[int]:
        """The stored settlements of quarters from ``lo``, booked where a booking lacks them."""
        rows = self.site.store.settlements(lo, now)
        self._settle_wait = [r for r in rows if r["published_at"] is not None]
        return self._settle_job()

    def _result_sums(self, lo: int) -> list[tuple[int, float]]:
        store = self.site.store
        last = R.last_settled_quarter(store)
        if last is None:
            return []
        return R.hourly_sums(store, lo, last - last % 3600 + 3600)

    def _queue_stats(self, lo: int) -> None:
        """Import the result statistic again from the hour of ``lo`` (one job at a time)."""
        self._stats_from = lo if self._stats_from is None else min(self._stats_from, lo)
        self._spawn("stats", self._stats())

    async def _stats(self) -> None:
        """Import the hours from ``_stats_from`` until no import is queued (module docstring)."""
        while self._stats_from is not None and not self._stopped:
            lo, self._stats_from = self._stats_from, None
            try:
                sums = await self.hass.async_add_executor_job(self._result_sums, lo)
                await import_result(self.hass, sums)
            except Exception:
                _LOGGER.exception("result statistic import from %s failed", lo)

    # -- nightly ------------------------------------------------------------------------

    def _settle_gaps(self, now: int) -> None:
        """The settlement pull of the older days with bookings and without a settled day."""
        store, today = self.site.store, _local(now).date()
        for k in range(SETTLE_GAP_DAYS + 1, 1, -1):
            day = today - dt.timedelta(days=k)
            lo, hi = FX._day_bounds(day)
            if store.bookings(lo, hi) and not self.runner.settled(day):
                try:
                    self.runner.pull_settlement(day)
                except Exception as err:
                    _LOGGER.info("settlement of %s not pulled: %s", day, err)

    def _prune(self, now: int) -> None:
        self.site.store.prune(now)
        self.site.store.vacuum()

    async def _nightly(self, now: int) -> None:
        """Settlement gaps, the refit (Mondays or drift), prune and vacuum; each step logs
        its failure and the next one runs."""
        hass = self.hass
        try:
            await hass.async_add_executor_job(self._settle_gaps, now)
        except Exception:
            _LOGGER.exception("nightly settlement pull failed")
        if _local(now).weekday() == 0 or self.drift_pending:
            try:
                await self.async_refit(now)
            except Exception as err:
                _LOGGER.warning("refit skipped: %r", err)
        try:
            await hass.async_add_executor_job(self._prune, now)
        except Exception:
            _LOGGER.exception("nightly prune or vacuum failed")

    async def async_refit(self, now: int | None = None) -> None:
        """The site-model refit at ``now`` (default the wall clock): the irradiance archive
        and ``LiveSite.fit`` beside the minutes, then ``adopt_refit`` under the minute lock at
        the next minute the loop processes, so its event applies before that minute, as
        re-execution reads it. Raises what the archive or the fit raises."""
        hass, site = self.hass, self.site
        now = _wall() if now is None else int(now)
        cfg = site.lc.cfg
        archive = await hass.async_add_executor_job(
            _irradiance_archive, self._dir, cfg.latitude, cfg.longitude, now
        )
        state = await hass.async_add_executor_job(site.fit, now, archive)
        async with self._lock:  # adopted between two minute jobs, at the next minute
            if not self._stopped:
                await hass.async_add_executor_job(
                    site.adopt_refit, state, site.next_minute, archive
                )
                self.drift_pending = False
                self._scale_issue(state)

    def _scale_issue(self, state) -> None:
        """A fill-in house (``pv_mode`` "fill") gets the repair issue ``pv_scale_off`` when the
        refit's fitted kWp of an entered string lies more than 25 % from the entered kWp
        (``LiveSite.scale_off``); a strings-path refit where all agree removes it, a blind
        refit (no verdict) leaves it as it is. A learning house never has it (an issue left
        from a fill-in entry goes). Only a refit raises it, so it persists across restarts;
        removing the entry removes it (``async_remove_entry``)."""
        issue_id = scale_issue_id(self.entry.entry_id)
        if self.entry.data.get("pv_mode") != "fill":
            ir.async_delete_issue(self.hass, DOMAIN, issue_id)
            return
        strings = self.site.scale_off(state)
        if strings is None:
            return
        if not strings:
            ir.async_delete_issue(self.hass, DOMAIN, issue_id)
            return
        ir.async_create_issue(
            self.hass,
            DOMAIN,
            issue_id,
            is_fixable=False,
            is_persistent=True,
            severity=ir.IssueSeverity.WARNING,
            translation_key="pv_scale_off",
            translation_placeholders={"strings": strings},
        )

    async def _backfill(self, now: int) -> None:
        """Hourly flows from the recorder's statistics into ``flows_hour`` (never minutes).

        A backfill that stored hours is followed by one site-model refit (the same fit and
        adopt as the nightly one, ``async_refit``) while the site is still learning and no
        refit ran before, so a fresh install has its first PV model the same day."""
        end = dt.datetime.fromtimestamp(now - now % 3600, dt.UTC)
        start = end - dt.timedelta(days=BACKFILL_DAYS)
        try:
            table = await backfill_hours(
                self.hass, self.mapping, self.signs, start, end, self.counters
            )
            stored = bool(len(table["ts"])) and not self._stopped
            if stored:
                await self.hass.async_add_executor_job(self.site.store.put_hours, table)
        except Exception as err:
            _LOGGER.warning("hourly backfill failed: %r", err)
            return
        if stored and await self.hass.async_add_executor_job(self._first_fit_due):
            try:
                await self.async_refit()
            except Exception as err:
                _LOGGER.warning("first refit after the backfill skipped: %r", err)
        await self._maybe_auto_backtest()

    def _first_fit_due(self) -> bool:
        """The site is learning and has never been refitted (blocking: reads the events)."""
        return (
            not self._stopped
            and bool(self.site.state.learning)
            and self.site.store.last_event("refit") is None
        )

    # -- the install-time backtest --------------------------------------------------------

    @property
    def backtest_running(self) -> bool:
        return bool(self._bt["running"])

    @property
    def stopping(self) -> bool:
        """True once ``async_stop`` began (an unload or an HA stop)."""
        return self._stopped

    def _plan_job(self, now: int) -> tuple[BT.BacktestPlan, DataPack | None]:
        """The backtest plan at ``now`` on the stored hours and the installed pack (blocking)."""
        store = self.site.store
        pack = self._installed_pack(store)
        hours = store.hours(now - (BT.MAX_DAYS + 2) * DAY_S, now + 1)
        return BT.plan_backtest(hours, pack, now), pack

    async def async_plan_backtest(self) -> tuple[BT.BacktestPlan, DataPack | None]:
        return await self.hass.async_add_executor_job(self._plan_job, _wall())

    def start_backtest(self, plan: BT.BacktestPlan, pack: DataPack) -> None:
        """Start the run of ``plan`` in the background (one at a time); raises
        ``BacktestRunning`` while one runs, ``BacktestStopping`` while the entry stops. The
        caller checked ``plan.reason``."""
        if self._stopped:
            raise BacktestStopping
        if self._bt["running"]:
            raise BacktestRunning
        self._auto_backtest = False
        est = plan.days * 96 * BT.EST_S_PER_QUARTER
        self._bt = {"running": True, "progress": 0.0, "eta_s": round(est)}
        if not self._spawn("backtest", self._backtest(plan, pack)):
            self._bt = {"running": False, "progress": None, "eta_s": None}
            raise BacktestStopping if self._stopped else BacktestRunning
        self._push_backtest()

    async def _maybe_auto_backtest(self) -> None:
        """The onboarding backtest (the screen-4 checkbox): once, when the plan has no reason
        (the hours and a pack are there); checked after the backfill, at start and after a
        pack is adopted."""
        if not self._auto_backtest or self._stopped or self._bt["running"]:
            return
        try:
            plan, pack = await self.async_plan_backtest()
        except Exception:
            _LOGGER.exception("backtest plan failed")
            return
        if plan.reason is None and self._auto_backtest and not self._bt["running"]:
            self.start_backtest(plan, pack)

    def _push_backtest(self) -> None:
        if self.data is not None and not self._stopped:
            bt = {**(self.data.get("backtest") or {}), **self._bt}
            self.async_set_updated_data({**self.data, "backtest": bt})

    @callback
    def _backtest_progress(self, day: int, days: int, elapsed: float) -> None:
        if not self._bt["running"]:
            return
        rate = elapsed / day if day else BT.EST_S_PER_QUARTER * 96
        self._bt = {
            "running": True,
            "progress": round(day / days, 4) if days else 1.0,
            "eta_s": round(rate * (days - day)),
        }
        self._push_backtest()

    def _open_backtest_store(self) -> _LockedStore:
        """The backtest's own connection on the store file (WAL, ``BT_BUSY_S`` busy timeout):
        its reads and runs-table writes never share the live connection's transaction. Its
        calls pass the same gate as the live ones (a backup pauses both)."""
        return _LockedStore(Store(self._dir / STORE_FILE, timeout=BT_BUSY_S), self._gate)

    def _close_backtest_store(self, store: _LockedStore) -> None:
        self._gate.unregister(store._lock)
        store._store.close()

    def _backtest_job(
        self, store: Store, plan: BT.BacktestPlan, pack: DataPack, run_id: int, archive,
        by_lead,
    ) -> tuple[dict, Path]:
        """``run_backtest`` on the backtest's connection and the report file (blocking)."""
        t0 = time.perf_counter()
        loop = self.hass.loop

        def progress(day: int, days: int) -> None:
            elapsed = time.perf_counter() - t0
            loop.call_soon_threadsafe(self._backtest_progress, day, days, elapsed)

        head, report = BT.run_backtest(
            store, self.site.lc, pack, plan, archive, by_lead, progress, run_id,
            should_stop=lambda: self._stopped,  # an unload must not wait for the whole year
        )
        day = _local(_wall()).date().isoformat()
        path = self._dir / RUNS_DIRNAME / f"{day}-{run_id}" / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        return head, path

    async def _backtest(self, plan: BT.BacktestPlan, pack: DataPack) -> None:
        """One backtest run (design 11): the irradiance, the replay and the report, all in
        the executor; the run row ends ``done`` with the headline or ``failed`` with the
        error. The minutes never wait for it. Every store call goes through the backtest's own
        connection (``_open_backtest_store``), closed at the end. A stop of the entry ends the
        run after the irradiance fetch or at the next band cell or quarter (``BT.Stopped``)."""
        hass, now = self.hass, _wall()
        store = None
        run_id = None
        try:
            store = await hass.async_add_executor_job(self._open_backtest_store)
            run_id = await hass.async_add_executor_job(store.new_run, now, plan.start, plan.end)
            self._push_backtest()
            cfg = self.site.lc.cfg
            archive, by_lead = await hass.async_add_executor_job(
                _backtest_irradiance, self._dir, cfg.latitude, cfg.longitude, plan, now
            )
            if self._stopped:
                raise BT.Stopped(STOPPED)
            head, path = await hass.async_add_executor_job(
                self._backtest_job, store, plan, pack, run_id, archive, by_lead
            )
            await hass.async_add_executor_job(store.finish_run, run_id, _wall(), "done", head, path)
        except Exception as err:
            if isinstance(err, base.FeedError):  # its message carries the request (the location)
                text = FEED_ERROR
            elif isinstance(err, BT.Stopped):
                text = STOPPED
            else:
                text = str(err) or repr(err)
            _LOGGER.warning("backtest failed: %s", text)
            if run_id is not None:
                error = {"error": text}
                try:
                    await hass.async_add_executor_job(
                        store.finish_run, run_id, _wall(), "failed", error, None
                    )
                except Exception:
                    _LOGGER.exception("backtest run %s not closed", run_id)
        finally:
            if store is not None:
                try:
                    await hass.async_add_executor_job(self._close_backtest_store, store)
                except Exception:
                    _LOGGER.exception("backtest store not closed")
            self._bt = {"running": False, "progress": None, "eta_s": None}
            if not self._stopped:
                try:
                    data = await hass.async_add_executor_job(self.snapshot)
                except Exception:
                    _LOGGER.exception("snapshot after the backtest failed")
                else:
                    self.async_set_updated_data(data)

    # -- the snapshot -------------------------------------------------------------------

    def _facts(self, now: int) -> dict[str, bool]:
        """The store facts the feed problem sensors read (``live.feeds.problem``)."""
        today = _local(now).date()
        return {
            "have_tomorrow_da": self.runner.day_complete(today + dt.timedelta(days=1)),
            "yesterday_settled": self.runner.settled(today - dt.timedelta(days=1)),
            "pack_installed": self.site.lc.pack_version is not None,
        }

    def _plan(self, ranked) -> dict | None:
        """The plan of ``ranked`` (``views.plan_for``), computed once per ranking; None
        without a ranking or when it fails (logged once per ranking)."""
        if ranked is None:
            return None
        if ranked is not self._plan_ranked:
            self._plan_ranked = ranked
            lc = self.site.lc
            try:
                self._plan_cache = V.plan_for(
                    ranked, lc.cfg.pack, self.site.loss, lc.cfg.terms, lc.cfg.knobs
                )
            except Exception:
                _LOGGER.exception("plan view failed for the quarter at %s", ranked.window.ts[0])
                self._plan_cache = None
        return self._plan_cache

    def snapshot(self) -> dict[str, Any]:
        """The state the entities read (blocking: ``status`` reads the chain head, the
        totals the bookings since the last snapshot, ``facts`` a day of features and of
        settlements)."""
        now = _wall()
        status = self.site.status()
        runs = self.site.store.runs()
        m50, m95 = _percentiles(self._timing["minute"])
        s50, s95 = _percentiles(self._timing["setup"])
        return {
            "status": status,
            "plan": self._plan(status["ranked"]),
            "feeds": {name: replace(st) for name, st in self.feeds.items()},
            "totals": self._totals.totals(self.site.store, now),
            "facts": self._facts(now),
            "backtest": {**self._bt, "last": runs[-1] if runs else None},
            "timing": {
                "minute_p50_ms": m50,
                "minute_p95_ms": m95,
                "setup_p50_ms": s50,
                "setup_p95_ms": s95,
            },
        }
