import asyncio
import datetime as dt
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.statistics import statistics_during_period
from homeassistant.config_entries import ConfigEntryState
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_fire_time_changed
from pytest_homeassistant_custom_component.components.recorder.common import (
    async_wait_recording_done,
)

from custom_components.betterybalance import coordinator as CO
from custom_components.betterybalance.const import STAT_RESULT
from custom_components.betterybalance.core.store import Store
from custom_components.betterybalance.core.timegrid import isp_start
from custom_components.betterybalance.live import fetch as FX
from custom_components.betterybalance.live import results as R
from custom_components.betterybalance.live.site import LiveSite
from tests.ha.test_config_flow import CONTRACT, _install_solarman

REAL_POLL_LIVE = FX.poll_live
START = "2026-11-02T10:00:30+00:00"  # a Monday; 11:00:30 in Amsterdam
FEEDS_PATCHED = (
    "fetch_day_ahead", "fetch_ida", "fetch_settlement", "poll_live", "fetch_irradiance",
)
POWER = {"grid_power": 500.0, "battery_power": 0.0, "soc": 50.0, "load_power": 500.0}
UNITS = {"grid_power": "W", "battery_power": "W", "soc": "%", "load_power": "W"}


class Calls:
    """A fetcher stand-in that records its arguments and raises FeedError (the box is offline)."""

    def __init__(self, name: str):
        self.name, self.calls = name, []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise FX.FeedError(f"{self.name}: offline")


@pytest.fixture(autouse=True)
def offline(hass, tmp_path):
    """Every feed and the irradiance archive offline; the store under a temporary config dir."""
    hass.config.config_dir = str(tmp_path)
    fakes = {name: Calls(name) for name in (*FEEDS_PATCHED, "archive")}
    with patch.multiple(FX, **{n: fakes[n] for n in FEEDS_PATCHED}), patch.object(
        CO, "_irradiance_archive", fakes["archive"]
    ):
        yield fakes


async def _setup(hass, freezer, start: str = START, p1: bool = False, loaded: bool = True):
    """Install the Solarman fixture, create the entry through the config flow (Task 10's data
    shape), return the coordinator (with ``loaded`` False: the entry, whatever its state)."""
    freezer.move_to(start)
    await _install_solarman(hass, p1=p1)
    hass.config.latitude, hass.config.longitude = 52.149, 5.231
    flow = hass.config_entries.flow
    r = await flow.async_init("betterybalance", context={"source": "user"})
    r = await flow.async_configure(
        r["flow_id"],
        {"capacity_kwh": 30.72, "charge_kw": 12.0, "discharge_kw": 12.0, "soc_floor_pct": 20},
    )
    r = await flow.async_configure(r["flow_id"], CONTRACT)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "pv_learn"})
    r = await flow.async_configure(r["flow_id"], {"run_backtest": False})
    assert r["type"] == "create_entry"
    await hass.async_block_till_done(wait_background_tasks=True)
    entry = r["result"]
    if not loaded:
        return entry
    assert entry.state is ConfigEntryState.LOADED
    return entry.runtime_data


def _entities(hass) -> dict:
    return hass.config_entries.async_entries("betterybalance")[0].data["entities"]


def _set_states(hass, values: dict | None = None):
    """Set every mapped power and SoC state once (grid 500 W, battery 0 W, SoC 50 %, load 500 W)."""
    entities = _entities(hass)
    for role, value in (values or POWER).items():
        hass.states.async_set(entities[role], str(value), {"unit_of_measurement": UNITS[role]})


async def _step(hass, freezer, seconds: float = 5, states: bool = True):
    freezer.tick(dt.timedelta(seconds=seconds))
    if states:
        _set_states(hass)
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)


def _now() -> int:
    return int(dt_util.utcnow().timestamp())


async def test_twenty_minutes_write_decisions(hass, freezer):
    coord = await _setup(hass, freezer)
    for _ in range(20 * 12):
        await _step(hass, freezer)
    store = coord.site.store
    assert len(store.minutes(0, 2**40)["ts"]) >= 18 and store.head() is not None
    # every processed minute has a nowcast row; the minutes run in order without a gap
    nc = store.nowcasts(0, 2**40)
    assert len(nc) >= 20
    timing = coord.data["timing"]
    assert timing["minute_p50_ms"] is not None and timing["setup_p95_ms"] is not None
    assert set(coord.data) == {
        "status", "plan", "feeds", "totals", "facts", "backtest", "timing",
    }
    totals = coord.data["totals"]
    assert isinstance(totals, R.Totals) and totals.settled_eur == 0.0
    assert totals == R.result_totals(store, _now())


async def test_a_stalled_loop_processes_missed_minutes_as_missing(hass, freezer):
    coord = await _setup(hass, freezer)
    for _ in range(3 * 12):
        await _step(hass, freezer)
    store = coord.site.store
    rows_before = len(store.minutes(0, 2**40)["ts"])
    assert rows_before >= 1
    freezer.tick(dt.timedelta(minutes=3))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert [d["reason"] for d in store.decisions(0, 2**40)][-1] == "missing"
    # the minutes of the stall are processed in order (a nowcast row each); the minute the
    # stall began in holds its last sample (sample and hold), the two after it have no samples
    w = _now() // 60 * 60
    assert coord.site.next_minute == w
    assert len(store.nowcasts(w - 180, w)) == 3 and not len(store.minutes(w - 120, w)["ts"])
    assert len([e for e in store.events() if e["kind"] == "restart"]) == 1


async def test_a_gap_beyond_a_quarter_restores_and_books_missing(hass, freezer):
    coord = await _setup(hass, freezer)
    for _ in range(2 * 12):
        await _step(hass, freezer)
    freezer.tick(dt.timedelta(minutes=40))
    async_fire_time_changed(hass)
    await hass.async_block_till_done(wait_background_tasks=True)
    store = coord.site.store
    now = _now()
    assert len([e for e in store.events() if e["kind"] == "restart"]) == 2
    assert coord.site.next_minute == now // 60 * 60
    q = int(isp_start(now))
    # the whole quarters of the gap are booked missing; the running quarter is not booked yet
    booked = store.bookings(0, 2**40)
    assert booked and all(r["missing"] for r in booked) and max(r["quarter_ts"] for r in booked) < q
    # the minutes of the running quarter up to the previous wall minute are processed missing
    assert len(store.nowcasts(q, now)) == (now // 60 * 60 - q) // 60
    # the live loop continues with a new recorder (samples are recorded again)
    for _ in range(3 * 12):
        await _step(hass, freezer)
    assert len(store.minutes(now // 60 * 60 + 60, 2**40)["ts"]) >= 1


async def test_a_restart_past_minute_one_sets_the_quarter_up_at_once(hass, freezer):
    coord = await _setup(hass, freezer, start="2026-11-02T10:07:30+00:00")
    store = coord.site.store
    q = int(isp_start(_now()))
    w = store.window(q)
    assert w is not None and w["now"] == q + 60 and w["epoch"] == coord.site.epoch
    # minutes 0 to 6 of the quarter were processed missing at start
    assert coord.site.next_minute == q + 7 * 60
    assert len(store.nowcasts(q, q + 7 * 60)) == 7


async def test_gates_use_amsterdam_when_hass_is_utc(hass, freezer, offline):
    await hass.config.async_set_time_zone("UTC")
    coord = await _setup(hass, freezer, start="2026-11-02T11:58:32+00:00")  # 12:58:32 Amsterdam
    tomorrow = dt.date(2026, 11, 3)

    def tomorrow_calls():
        return [c for c in offline["fetch_day_ahead"].calls if c[0][1] == tomorrow]

    while _now() < int(dt.datetime(2026, 11, 2, 12, 4, 55, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, states=False)
    assert tomorrow_calls() == []  # 13:00 to 13:04 Amsterdam: no slot yet
    while _now() < int(dt.datetime(2026, 11, 2, 12, 9, 0, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, states=False)
    assert len(tomorrow_calls()) == 1  # the 13:05 Amsterdam slot, once
    st = coord.data["feeds"]["day_ahead"]
    slot = int(dt.datetime(2026, 11, 2, 12, 5, 2, tzinfo=dt.UTC).timestamp())
    assert st.last_try == slot and st.failures >= 1 and "offline" in st.last_error


async def test_the_nightly_work_runs_at_0330_amsterdam(hass, freezer, offline):
    await hass.config.async_set_time_zone("UTC")
    coord = await _setup(hass, freezer, start="2026-11-02T02:28:30+00:00")  # Monday 03:28:30
    store = coord.site.store
    with patch.object(store, "prune", wraps=store.prune) as prune, patch.object(
        store, "vacuum", wraps=store.vacuum
    ) as vacuum:
        while _now() < int(dt.datetime(2026, 11, 2, 2, 29, 50, tzinfo=dt.UTC).timestamp()):
            await _step(hass, freezer, states=False)
        assert not prune.called and offline["archive"].calls == []
        while _now() < int(dt.datetime(2026, 11, 2, 2, 33, 0, tzinfo=dt.UTC).timestamp()):
            await _step(hass, freezer, states=False)
        assert prune.call_count == 1 and vacuum.call_count == 1
        # Monday: the refit asks for the irradiance archive (offline here, so no refit event)
        assert len(offline["archive"].calls) == 1


async def test_state_reported_counts_as_a_sample(hass, freezer):
    coord = await _setup(hass, freezer, start="2026-11-02T10:00:50+00:00")
    _set_states(hass)
    await _step(hass, freezer, seconds=12, states=False)  # 10:01:02, minute 10:01 starts
    w = _now() // 60 * 60
    for _ in range(5):  # five unchanged writes inside the minute: state_reported events
        freezer.tick(dt.timedelta(seconds=9))
        _set_states(hass)
        await hass.async_block_till_done(wait_background_tasks=True)
    while _now() < w + 63:
        await _step(hass, freezer, seconds=3, states=False)
    m = coord.site.store.minutes(w, w + 60)
    assert list(m["n_samples"]) == [5]
    assert m["grid_import_kwh"][0] == pytest.approx(500.0 * 60 / 3.6e6)


async def test_grid_counters_in_wh_become_kwh(hass, freezer):
    coord = await _setup(hass, freezer, start="2026-11-02T10:00:50+00:00", p1=True)
    entry = hass.config_entries.async_entries("betterybalance")[0]
    assert entry.data["grid_source"] == "p1"
    entities = entry.data["entities"]
    imports, exports = entities["grid_import_kwh"], entities["grid_export_kwh"]
    assert coord.counters == {"import": imports, "export": exports}
    counter = {eid: 1000.0 for eid in (*imports, *exports)}

    def set_all():
        _set_states(hass)
        for eid, value in counter.items():
            hass.states.async_set(eid, str(value), {"unit_of_measurement": "Wh"})

    set_all()
    await _step(hass, freezer, seconds=12, states=False)
    w = _now() // 60 * 60
    for _ in range(6):
        freezer.tick(dt.timedelta(seconds=9))
        counter[imports[0]] += 10.0  # 10 Wh per sample on the first import counter
        set_all()
        await hass.async_block_till_done(wait_background_tasks=True)
    while _now() < w + 63:
        await _step(hass, freezer, seconds=3, states=False)
    m = coord.site.store.minutes(w, w + 60)
    assert m["grid_import_kwh"][0] == pytest.approx(0.06) and m["grid_export_kwh"][0] == 0.0


def test_units_convert_counters_to_kwh_and_power_to_w():
    assert CO.unit_factor("Wh", "kWh") == 0.001 and CO.unit_factor("MWh", "kWh") == 1000.0
    assert CO.unit_factor("kWh", "kWh") == 1.0 and CO.unit_factor("kW", "W") == 1000.0
    assert CO.unit_factor(None, "W") == 1.0 and CO.unit_factor("%", None) == 1.0


async def test_feed_state_is_seeded_from_the_store(hass, freezer, tmp_path):
    freezer.move_to(START)
    now = _now()
    (tmp_path / "betterybalance").mkdir()
    store = Store(tmp_path / "betterybalance" / "store.sqlite3")
    store.migrate()
    minute = now // 60 * 60 - 120
    store.put_live_minutes(
        [{"ts": minute, "buy_live": 100.0, "sell_live": 90.0, "state": 1, "source": "tenergy"}]
    )
    table = {"ts": np.array([now], dtype=np.int64), "ghi_w_m2": np.array([100.0])}
    store.put_fetch(issue_ts=now - 600, source="forecast", fetched_at=now - 600, table=table)
    store.close()
    coord = await _setup(hass, freezer)
    feeds = coord.data["feeds"]
    assert feeds["irradiance"].last_ok == now - 600
    # the live minute is known at its end; the offline first poll does not clear it
    assert feeds["live"].last_ok == minute + 60
    assert feeds["ida"].last_ok is None


async def test_unload_stops_the_clock(hass, freezer):
    coord = await _setup(hass, freezer)
    entry = hass.config_entries.async_entries("betterybalance")[0]
    assert await hass.config_entries.async_unload(entry.entry_id)
    nxt = coord.site.next_minute
    for _ in range(3 * 12):
        await _step(hass, freezer)
    assert coord.site.next_minute == nxt


async def _minute_done(hass, coord, timeout_s: float = 120.0):
    """Wait (real time) until the minute loop served the current wall minute and its lock is
    free, without waiting for the feed or nightly jobs."""
    await hass.async_block_till_done()
    deadline = time.monotonic_ns() + int(timeout_s * 1e9)
    while coord.site.next_minute != _now() // 60 * 60 or coord._lock.locked():
        assert time.monotonic_ns() < deadline, "the minute loop fell behind the wall clock"
        await hass.async_add_executor_job(time.sleep, 0.002)


async def test_a_long_nightly_job_never_holds_up_the_minutes(hass, freezer, offline):
    await hass.config.async_set_time_zone("UTC")
    coord = await _setup(hass, freezer, start="2026-11-02T02:29:32+00:00")  # Monday 03:29:32
    release, started = threading.Event(), threading.Event()

    def slow_archive(*args):  # the refit's archive fetch hangs for 20 minutes of the clock
        started.set()
        assert release.wait(30)
        raise FX.FeedError("archive: offline")

    store = coord.site.store
    with patch.object(CO, "_irradiance_archive", slow_archive), patch.object(
        store, "prune", wraps=store.prune
    ) as prune:
        while not started.is_set():
            freezer.tick(dt.timedelta(seconds=5))
            _set_states(hass)
            async_fire_time_changed(hass)
            await _minute_done(hass, coord)
        held = _now() // 60 * 60
        for _ in range(20 * 12):  # every tick's minutes are processed on time meanwhile
            freezer.tick(dt.timedelta(seconds=5))
            _set_states(hass)
            async_fire_time_changed(hass)
            await _minute_done(hass, coord)
        assert not prune.called
        assert len(store.minutes(held, 2**40)["ts"]) >= 19
        release.set()
        await hass.async_block_till_done(wait_background_tasks=True)
        assert prune.call_count == 1


async def test_queued_ticks_coalesce_and_live_rows_are_labelled_by_their_poll(hass, freezer):
    coord = await _setup(hass, freezer, start="2026-11-02T10:00:32+00:00")
    polls = []

    def rows():
        polls.append(int(time.time()))
        return [{"ts": 0, "buy_live": 100.0, "sell_live": 90.0, "state": None}]

    with patch.object(FX, "poll_live", REAL_POLL_LIVE), patch.object(FX, "_tenergy_rows", rows):
        await _step(hass, freezer, seconds=30)  # 10:01:02, one served tick
        with patch.object(coord, "_minutes_job", wraps=coord._minutes_job) as job:
            await coord._lock.acquire()  # a long minute job holds the lock for three ticks
            for _ in range(3):
                freezer.tick(dt.timedelta(seconds=60))
                async_fire_time_changed(hass)
                await asyncio.sleep(0)
            coord._lock.release()
            await hass.async_block_till_done(wait_background_tasks=True)
            assert job.call_count == 1  # the newest tick served all three minutes at once
        w = _now() // 60 * 60
        assert coord.site.next_minute == w and len(coord.site.store.nowcasts(w - 180, w)) == 3
        live = coord.site.store.live_minutes(0, 2**40)
        assert polls and live["ts"].tolist() == sorted({p - p % 60 for p in polls})
        assert all(int(ts) >= w - 60 for ts in live["ts"][-1:])


async def test_a_failed_restore_closes_the_store(hass, freezer):
    closed = []
    real = Store.close

    def close(self):
        closed.append(self)
        real(self)

    with patch.object(LiveSite, "restore", side_effect=RuntimeError("broken store")), patch.object(
        Store, "close", close
    ):
        entry = await _setup(hass, freezer, loaded=False)
    assert entry.state is ConfigEntryState.SETUP_ERROR and len(closed) == 1


async def test_a_failed_start_after_subscribing_removes_the_listeners(hass, freezer):
    removed = []

    def tracked(fn):
        def track(*args, **kwargs):
            unsub = fn(*args, **kwargs)

            def remove():
                removed.append(fn.__name__)
                unsub()

            return remove

        return track

    names = ("async_track_state_change_event", "async_track_state_report_event",
             "async_track_time_change")
    with patch.multiple(CO, **{n: tracked(getattr(CO, n)) for n in names}), patch.object(
        CO.BetteryCoordinator, "snapshot", side_effect=RuntimeError("no snapshot")
    ):
        entry = await _setup(hass, freezer, loaded=False)
    assert entry.state is ConfigEntryState.SETUP_ERROR and sorted(removed) == sorted(names)


async def test_a_failing_nightly_step_or_snapshot_never_stops_the_clock(hass, freezer):
    await hass.config.async_set_time_zone("UTC")
    coord = await _setup(hass, freezer, start="2026-11-02T02:29:32+00:00")
    store = coord.site.store
    real = coord.snapshot
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("snapshot failed once")
        return real()

    with patch.object(store, "prune", side_effect=RuntimeError("disk full")), patch.object(
        coord, "snapshot", flaky
    ):
        while _now() < int(dt.datetime(2026, 11, 2, 2, 34, 0, tzinfo=dt.UTC).timestamp()):
            await _step(hass, freezer)
    w = _now() // 60 * 60
    assert coord.site.next_minute == w and len(calls) >= 4
    assert coord.data["status"]["head"] == store.head()


async def test_a_settlement_pull_books_at_a_minute_that_knew_it_and_imports_the_result(
    hass, freezer
):
    coord = await _setup(hass, freezer)
    for _ in range(35 * 12):
        await _step(hass, freezer)
    store = coord.site.store
    booked = sorted({b["quarter_ts"] for b in store.bookings(0, 2**40)})
    assert len(booked) == 2

    def fake(store_, day, now=None):  # what the pull appends; known a minute from now
        rows = [
            {
                "quarter_ts": q, "take": 320.0, "feed": -15.0, "regulation_state": 1,
                "published_at": _now() + 60, "source": FX.SETTLEMENT_SOURCE,
            }
            for q in booked
        ]
        for r in rows:
            store_.put_settlement(r)
        return rows

    with patch.object(FX, "fetch_settlement", fake):
        rows = await hass.async_add_executor_job(
            coord.runner.pull_settlement, dt.date(2026, 11, 1)
        )
    published = rows[0]["published_at"]
    await _step(hass, freezer, 60)  # the next minute is before the publication: it waits
    assert coord.site.next_minute < published and store.last_event("settle") is None
    await _step(hass, freezer, 60)
    ev = store.last_event("settle")
    assert ev["ts"] == coord.site.next_minute >= published and ev["detail"]["quarters"] == booked
    newest = store.newest_bookings(booked[0], booked[-1] + 900)
    assert all(b["price_source"] == "settled" and b["buy"] == 320.0 for b in newest.values())
    assert len(newest) == 6
    await async_wait_recording_done(hass)
    hour = booked[0] - booked[0] % 3600
    stats = await get_instance(hass).async_add_executor_job(
        statistics_during_period,
        hass,
        dt.datetime.fromtimestamp(hour, dt.UTC),
        None,
        {STAT_RESULT},
        "hour",
        None,
        {"sum"},
    )
    got = stats[STAT_RESULT]
    assert [int(r["start"]) for r in got] == [hour]
    totals = coord.data["totals"]
    assert totals == R.result_totals(store, _now())
    assert got[0]["sum"] == pytest.approx(totals.settled_eur)
    assert totals.today_settled_share > 0.0


async def test_a_failed_settlement_booking_keeps_its_rows_for_the_next_tick(hass, freezer):
    coord = await _setup(hass, freezer)
    for _ in range(20 * 12):
        await _step(hass, freezer)
    store = coord.site.store
    booked = sorted({b["quarter_ts"] for b in store.bookings(0, 2**40)})
    assert booked

    def fake(store_, day, now=None):
        rows = [
            {
                "quarter_ts": q, "take": 250.0, "feed": 5.0, "regulation_state": 1,
                "published_at": _now(), "source": FX.SETTLEMENT_SOURCE,
            }
            for q in booked
        ]
        for r in rows:
            store_.put_settlement(r)
        return rows

    with patch.object(FX, "fetch_settlement", fake):
        await hass.async_add_executor_job(coord.runner.pull_settlement, dt.date(2026, 11, 1))
    real, calls = coord.site.settle, []

    def flaky(rows, now):
        calls.append(now)
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return real(rows, now)

    with patch.object(coord.site, "settle", flaky):
        await _step(hass, freezer, 60)
        assert len(calls) == 1 and store.last_event("settle") is None
        await _step(hass, freezer, 60)
    assert len(calls) == 2
    assert store.last_event("settle")["detail"]["quarters"] == booked
    newest = store.newest_bookings(booked[0], booked[-1] + 900)
    assert all(b["price_source"] == "settled" for b in newest.values())
    assert coord._settle_wait == []


def test_the_retry_delay_doubles_up_to_fifteen_minutes():
    assert [CO.retry_delay(n) for n in range(1, 8)] == [60, 120, 240, 480, 900, 900, 900]


def _loop_issue(hass, coord):
    from homeassistant.helpers import issue_registry as ir

    issue_id = CO.loop_issue_id(coord.entry.entry_id)
    return ir.async_get(hass).async_get_issue("betterybalance", issue_id)


async def test_a_persistent_minute_failure_backs_off_logs_once_and_raises_an_issue(
    hass, freezer, caplog
):
    coord = await _setup(hass, freezer)
    for _ in range(3):
        await _step(hass, freezer, 60)
    store = coord.site.store
    n_restarts = sum(e["kind"] == "restart" for e in store.events())
    restores = []
    real_restore = LiveSite.restore

    def restore(self, now):
        restores.append(now)
        return real_restore(self, now)

    caplog.clear()
    with patch.object(LiveSite, "on_minute", side_effect=RuntimeError("bad data")), patch.object(
        LiveSite, "restore", restore
    ):
        for _ in range(40):
            await _step(hass, freezer, 60)
        issue = _loop_issue(hass, coord)
        assert issue is not None and issue.translation_key == "minute_loop_failed"
        assert issue.translation_placeholders == {"error": "bad data"}
    # tries at 0, 1, 3, 7, 15 and 30 minutes: five restores in 40 minutes, not 39
    assert 4 <= len(restores) <= 6, restores
    gaps = np.diff(restores).tolist()
    assert gaps == sorted(gaps) and max(gaps) <= CO.RETRY_MAX_S + 60
    restarts = sum(e["kind"] == "restart" for e in store.events()) - n_restarts
    assert restarts == len(restores)  # held ticks write nothing
    failed = [r for r in caplog.records if "minute loop failed" in r.getMessage()]
    assert sum(r.exc_info is not None for r in failed) == 1  # one traceback per episode
    assert len(failed) == len(restores) + 1  # one line per try
    # recovery: the next try processes minutes, the issue goes, the clock is on time
    for _ in range(16):
        await _step(hass, freezer, 60)
    assert coord._failures == 0 and not coord._broken
    assert _loop_issue(hass, coord) is None
    assert coord.site.next_minute == _now() // 60 * 60


async def test_an_entry_reload_removes_the_stopped_trading_issue(hass, freezer):
    coord = await _setup(hass, freezer)
    await _step(hass, freezer, 60)
    with patch.object(LiveSite, "on_minute", side_effect=RuntimeError("bad data")):
        for _ in range(5):
            await _step(hass, freezer, 60)
        assert _loop_issue(hass, coord) is not None
    assert await hass.config_entries.async_reload(coord.entry.entry_id)
    await hass.async_block_till_done()
    assert _loop_issue(hass, coord) is None


async def test_a_failing_restore_writes_no_restart_event(hass, freezer):
    coord = await _setup(hass, freezer)
    await _step(hass, freezer, 60)
    store = coord.site.store
    before = store.events()
    with patch.object(LiveSite, "on_minute", side_effect=OSError("disk full")), patch.object(
        LiveSite, "restore", side_effect=OSError("disk full")
    ):
        for _ in range(20):
            await _step(hass, freezer, 60)
    assert coord._failures >= 3
    assert store.events() == before  # no retry processed a minute, no restart was written


async def test_a_persistent_settlement_failure_drops_the_rows_and_the_start_books_them(
    hass, freezer, caplog
):
    coord = await _setup(hass, freezer)
    for _ in range(20 * 12):
        await _step(hass, freezer)
    store = coord.site.store
    booked = sorted({b["quarter_ts"] for b in store.bookings(0, 2**40)})
    assert booked

    def fake(store_, day, now=None):
        rows = [
            {
                "quarter_ts": q, "take": 250.0, "feed": 5.0, "regulation_state": 1,
                "published_at": _now(), "source": FX.SETTLEMENT_SOURCE,
            }
            for q in booked
        ]
        for r in rows:
            store_.put_settlement(r)
        return rows

    with patch.object(FX, "fetch_settlement", fake):
        await hass.async_add_executor_job(coord.runner.pull_settlement, dt.date(2026, 11, 1))
    calls = []

    def broken(rows, now):
        calls.append(now)
        raise RuntimeError("settle bug")

    caplog.clear()
    with patch.object(coord.site, "settle", broken):
        await _step(hass, freezer, 60)
        assert len(calls) == 1 and len(coord._settle_wait) == len(booked)  # kept for one retry
        await _step(hass, freezer, 60)
        assert len(calls) == 2 and coord._settle_wait == []  # dropped
        for _ in range(5):
            await _step(hass, freezer, 60)
    assert len(calls) == 2  # nothing waits, nothing is tried again
    lines = [r for r in caplog.records if "settlement booking failed" in r.getMessage()]
    assert len(lines) == 2 and lines[0].exc_info is not None and lines[1].exc_info is None
    assert store.last_event("settle") is None
    coord = await _reload(hass)  # the start catch-up books the stored settlements
    store = coord.site.store
    assert store.last_event("settle")["detail"]["quarters"] == booked


def _pack_release(folder, month: str = "2026-11", end_day: int = 8):
    """A synthetic release: one model (the shipped champion's cells without coverage) trained
    to local midnight of 2026-11-<end_day>, later than the shipped champion's end, on one day
    of quarters. Returns the release folder and the model's version."""
    from custom_components.betterybalance.core import pack as PK
    from custom_components.betterybalance.core import pricemodel as PM
    from custom_components.betterybalance.core.timegrid import ZONE

    model = PM.from_json(PM.to_json(PM.shipped_model()))
    for hm in model.models.values():
        hm.cov_factor = 1.0
    end = int(dt.datetime(2026, 11, end_day, tzinfo=ZONE).timestamp())
    ts = np.arange(end - 86400, end, 900, dtype=np.int64)
    quarters = {c: np.full(len(ts), np.nan) for c in PK.QUARTER_COLUMNS}
    quarters["ts"] = ts
    src = folder / month
    PK.write_pack(src, month, quarters, [(end, model)], built="2026-11-01")
    return src, f"pack-{month}-2026-11-{end_day - 1:02d}"


def _set_source(hass, source) -> None:
    """The pack source as if the running entry had started with it (no reload): these tests
    drive the 04:10 slot and the start check. The reload on an options change has its own
    test (``test_an_options_change_reloads_and_the_restart_names_the_new_knobs``)."""
    from custom_components.betterybalance.config_flow import effective_options

    entry = hass.config_entries.async_entries("betterybalance")[0]
    options = {**entry.options, "pack_source": source}
    entry.runtime_data.options = effective_options(options)
    hass.config_entries.async_update_entry(entry, options=options)


async def test_the_pack_update_at_0410_installs_and_switches_at_the_next_quarter(
    hass, freezer, tmp_path
):
    src, version = _pack_release(tmp_path / "releases")
    # Tuesday 04:05:30 Amsterdam; no pack source yet: the start check finds none
    coord = await _setup(hass, freezer, start="2026-11-03T03:05:30+00:00")
    await _step(hass, freezer, seconds=40)  # the first tick, 04:06:02
    st = coord.data["feeds"]["pack"]
    assert st.last_error == CO.NO_PACK_SOURCE and st.failures == 0 and st.last_ok is None
    shipped = coord.site.lc.model_version
    assert shipped.startswith("shipped-") and coord.site.lc.pack_version is None
    assert hass.states.get("binary_sensor.betterybalance_feed_pack").state == "on"
    _set_source(hass, str(src))
    while _now() < int(dt.datetime(2026, 11, 3, 3, 9, 50, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    store = coord.site.store
    assert store.active_pack() is None  # the slot is 04:10
    while _now() < int(dt.datetime(2026, 11, 3, 3, 10, 30, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    installed = tmp_path / "betterybalance" / "pack" / "2026-11"
    row = store.active_pack()
    assert row["version"] == "2026-11" and row["path"] == str(installed)
    assert installed.is_dir() and not list(installed.parent.glob(".tmp-*"))
    st = coord.data["feeds"]["pack"]
    assert st.failures == 0 and st.last_error is None and st.last_ok == row["installed_at"]
    ev = store.last_event("model_switch")
    boundary = int(dt.datetime(2026, 11, 3, 3, 15, tzinfo=dt.UTC).timestamp())
    assert ev["detail"]["version"] == version and ev["detail"]["effective_ts"] == boundary
    assert ev["detail"]["from_version"] == shipped and ev["detail"]["cov_source"] == "shipped"
    assert ev["ts"] <= _now() // 60 * 60  # at a minute the loop processes next
    assert coord.site.lc.pack_version == "2026-11" and coord.site.lc.model_version == shipped
    while _now() < boundary + 2 * 60 + 30:
        await _step(hass, freezer, seconds=20)
    assert coord.site.lc.model_version == version
    assert store.window(boundary)["model_version"] == version
    assert store.window(boundary - 900)["model_version"] == shipped
    assert all(
        d["model_version"] == (version if d["ts"] >= boundary else shipped)
        for d in store.decisions(boundary - 900, boundary + 900)
    )
    assert hass.states.get("binary_sensor.betterybalance_feed_pack").state == "off"
    # a reload starts on the model in effect, without a new switch
    n_switch = sum(e["kind"] == "model_switch" for e in store.events())
    entry = hass.config_entries.async_entries("betterybalance")[0]
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coord = entry.runtime_data
    assert coord.site.lc.model_version == version and coord.site.lc.pack_version == "2026-11"
    store = coord.site.store
    assert sum(e["kind"] == "model_switch" for e in store.events()) == n_switch
    # the source cleared (ruling A): the updates stop, the installed pack keeps serving
    _set_source(hass, None)
    coord.feeds["pack"].last_try = None  # the start check again, without a source
    await _step(hass, freezer, seconds=60)
    assert coord.data["feeds"]["pack"].last_error == CO.NO_PACK_SOURCE
    assert coord.site.lc.model_version == version and coord.site.lc.pack_version == "2026-11"
    assert sum(e["kind"] == "model_switch" for e in store.events()) == n_switch
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    coord = entry.runtime_data
    assert coord.site.lc.model_version == version and coord.site.lc.pack_version == "2026-11"
    assert sum(e["kind"] == "model_switch" for e in coord.site.store.events()) == n_switch


async def test_a_failing_pack_source_counts_on_the_pack_feed(hass, freezer, tmp_path):
    coord = await _setup(hass, freezer, start="2026-11-03T03:08:30+00:00")  # 04:08:30
    _set_source(hass, str(tmp_path / "gone"))
    await _step(hass, freezer, seconds=40)  # the first tick, 04:09:02: the start check
    st = coord.data["feeds"]["pack"]
    assert st.failures == 1 and "no manifest" in st.last_error
    while _now() < int(dt.datetime(2026, 11, 3, 3, 11, 0, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    st = coord.data["feeds"]["pack"]
    assert st.failures == 2 and st.last_ok is None  # the 04:10 slot, once
    assert coord.site.lc.model_version.startswith("shipped-")
    assert coord.site.store.active_pack() is None
    assert not (tmp_path / "betterybalance" / "pack").exists()
    assert hass.states.get("binary_sensor.betterybalance_feed_pack").state == "on"


async def _install_now(hass, freezer, coord, src) -> None:
    """Point the pack source at ``src`` and run the pack update at the next tick."""
    _set_source(hass, str(src))
    coord.feeds["pack"].last_try = None
    await _step(hass, freezer, seconds=60)


async def _reload(hass):
    entry = hass.config_entries.async_entries("betterybalance")[0]
    assert await hass.config_entries.async_reload(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    return entry.runtime_data


def _n_switch(store) -> int:
    return sum(e["kind"] == "model_switch" for e in store.events())


async def test_a_restart_between_a_new_install_and_its_boundary_keeps_the_pack_model(
    hass, freezer, tmp_path
):
    nov, v_nov = _pack_release(tmp_path / "releases", "2026-11", end_day=8)
    dec, v_dec = _pack_release(tmp_path / "releases", "2026-12", end_day=15)
    coord = await _setup(hass, freezer, start="2026-11-03T03:05:30+00:00")  # 04:05:30
    await _install_now(hass, freezer, coord, nov)  # at 04:06:02, effective 04:15
    while _now() < int(dt.datetime(2026, 11, 3, 3, 16, 10, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    assert coord.site.lc.model_version == v_nov
    await _install_now(hass, freezer, coord, dec)  # pending for 04:30
    assert coord.site.pending_model == (
        int(dt.datetime(2026, 11, 3, 3, 30, tzinfo=dt.UTC).timestamp()), v_dec
    )
    assert coord.site.lc.model_version == v_nov
    coord = await _reload(hass)  # v_nov lives only in the previous version folder now
    store = coord.site.store
    assert coord.site.lc.model_version == v_nov and coord.site.lc.pack_version == "2026-12"
    reasons = [e["detail"].get("reason") for e in store.events() if e["kind"] == "model_switch"]
    assert "model not available" not in reasons and reasons[-2] == "restart"
    assert coord.site.pending_model[1] == v_dec  # put again after the restart
    while _now() < int(dt.datetime(2026, 11, 3, 3, 31, 10, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    assert coord.site.lc.model_version == v_dec


async def test_a_release_that_does_not_advance_keeps_the_model_in_use_across_a_restart(
    hass, freezer, tmp_path
):
    nov, v_nov = _pack_release(tmp_path / "releases", "2026-11", end_day=8)
    dec, _ = _pack_release(tmp_path / "releases", "2026-12", end_day=8)  # the same newest end
    coord = await _setup(hass, freezer, start="2026-11-03T03:05:30+00:00")
    await _install_now(hass, freezer, coord, nov)
    while _now() < int(dt.datetime(2026, 11, 3, 3, 16, 10, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    n = _n_switch(coord.site.store)
    await _install_now(hass, freezer, coord, dec)
    assert coord.site.store.active_pack()["version"] == "2026-12"
    assert coord.site.pending_model is None and _n_switch(coord.site.store) == n
    coord = await _reload(hass)
    assert coord.site.lc.model_version == v_nov and coord.site.lc.pack_version == "2026-12"
    assert _n_switch(coord.site.store) == n
    # a second release that does not advance: the 2026-11 folder of the model in use stays
    jan, _ = _pack_release(tmp_path / "releases", "2027-01", end_day=8)
    await _install_now(hass, freezer, coord, jan)
    assert coord.site.store.active_pack()["version"] == "2027-01"
    folders = sorted(p.name for p in (tmp_path / "betterybalance" / "pack").iterdir())
    assert folders == ["2026-11", "2026-12", "2027-01"]
    coord = await _reload(hass)
    store = coord.site.store
    assert coord.site.lc.model_version == v_nov and coord.site.lc.pack_version == "2027-01"
    assert _n_switch(store) == n
    reasons = [e["detail"].get("reason") for e in store.events() if e["kind"] == "model_switch"]
    assert "model not available" not in reasons


async def test_a_model_gone_at_start_switches_at_the_restored_quarter_start(
    hass, freezer, tmp_path
):
    import shutil

    nov, v_nov = _pack_release(tmp_path / "releases", "2026-11", end_day=8)
    coord = await _setup(hass, freezer, start="2026-11-03T03:05:30+00:00")
    await _install_now(hass, freezer, coord, nov)
    while _now() < int(dt.datetime(2026, 11, 3, 3, 21, 10, tzinfo=dt.UTC).timestamp()):
        await _step(hass, freezer, seconds=20)
    assert coord.site.lc.model_version == v_nov
    shutil.rmtree(tmp_path / "betterybalance" / "pack")  # neither pack holds it any more
    coord = await _reload(hass)
    store = coord.site.store
    shipped = coord.site.lc.model_version
    assert shipped.startswith("shipped-") and coord.site.lc.pack_version is None
    ev = store.last_event("model_switch")
    restart = store.last_event("restart")
    assert ev["detail"]["version"] == shipped and ev["detail"]["reason"] == "model not available"
    assert ev["ts"] == restart["ts"] and ev["seq"] < restart["seq"]  # before the restart
    assert ev["detail"]["effective_ts"] == isp_start(ev["ts"])  # the restored quarter start
    q = isp_start(ev["ts"])
    w = store.window(q, epoch=restart["seq"])
    assert w is None or w["model_version"] == shipped


async def test_an_options_change_reloads_and_the_restart_names_the_new_knobs(hass, freezer):
    from dataclasses import asdict

    from custom_components.betterybalance.core.site.contract import Knobs

    coord = await _setup(hass, freezer)
    for _ in range(3):
        await _step(hass, freezer, 60)
    entry = hass.config_entries.async_entries("betterybalance")[0]
    first = coord.site.store.last_event("restart")
    assert first["detail"]["knobs"] == asdict(Knobs())
    # a save that changes nothing (the view step) reloads nothing
    flow = hass.config_entries.options
    r = await flow.async_init(entry.entry_id)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "view"})
    r = await flow.async_configure(r["flow_id"], {})
    assert r["type"] == "create_entry"
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.runtime_data is coord and not coord.stopping
    # the export choices are read at each service call: no reload either
    r = await flow.async_init(entry.entry_id)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "export"})
    r = await flow.async_configure(r["flow_id"], {"export_flows": True, "pool_results": True})
    await hass.async_block_till_done(wait_background_tasks=True)
    assert entry.options["export_flows"] and entry.runtime_data is coord and not coord.stopping
    # a knob change reloads: a new run whose restart event names the new knobs
    r = await flow.async_init(entry.entry_id)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "advanced"})
    knobs = {**asdict(Knobs()), "latency_s": 120, "dwell_s": 600}
    r = await flow.async_configure(r["flow_id"], knobs)
    assert r["type"] == "create_entry"
    await hass.async_block_till_done(wait_background_tasks=True)
    new = entry.runtime_data
    assert new is not coord and coord.stopping
    assert new.site.lc.cfg.knobs.latency_s == 120 and new.site.lc.cfg.knobs.dwell_s == 600
    ev = new.site.store.last_event("restart")
    assert ev["seq"] > first["seq"] and ev["detail"]["knobs"] == knobs


@pytest.mark.parametrize("hours", [3, 0])
async def test_the_first_backfill_is_followed_by_one_refit(hass, freezer, hours):
    from unittest.mock import AsyncMock

    table = {"ts": np.arange(hours, dtype=float) * 3600.0}
    refit = AsyncMock()
    with patch.object(CO, "backfill_hours", AsyncMock(return_value=table)), patch.object(
        Store, "put_hours"
    ), patch.object(CO.BetteryCoordinator, "async_refit", refit):
        coord = await _setup(hass, freezer)
        await hass.async_block_till_done(wait_background_tasks=True)
        assert refit.await_count == (1 if hours else 0)  # no history: nothing to fit yet
        # a later backfill (an empty flows_hour again) after a refit: no second first fit
        coord.site.store.put_event(_now(), "refit", {"learning": True, "drift": False})
        await coord._backfill(_now())
        assert refit.await_count == (1 if hours else 0)


async def test_the_ida_fetch_is_off_by_default_and_on_by_choice(hass, freezer, offline):
    coord = await _setup(hass, freezer)
    for _ in range(5):
        await _step(hass, freezer, 60)
    assert offline["fetch_ida"].calls == []  # Nord Pool is never called
    assert coord.site.lc.skip_tiers == ("B",) and not coord.feeds["ida"].enabled
    assert hass.states.get("binary_sensor.betterybalance_feed_ida").state == "off"
    # the option on (the feeds step): a reload that fetches as before
    entry = coord.entry
    flow = hass.config_entries.options
    r = await flow.async_init(entry.entry_id)
    r = await flow.async_configure(r["flow_id"], {"next_step_id": "feeds"})
    r = await flow.async_configure(r["flow_id"], {"fetch_ida": True})
    assert r["type"] == "create_entry" and entry.options["fetch_ida"] is True
    await hass.async_block_till_done(wait_background_tasks=True)
    coord = entry.runtime_data
    await _step(hass, freezer, 60)
    assert offline["fetch_ida"].calls  # the start fetch of the missing auctions
    assert coord.site.lc.skip_tiers == () and coord.feeds["ida"].enabled
    assert hass.states.get("binary_sensor.betterybalance_feed_ida").state == "on"  # offline


async def test_a_home_assistant_stop_marks_the_coordinator_stopping(hass, freezer, caplog):
    from homeassistant.const import EVENT_HOMEASSISTANT_STOP

    coord = await _setup(hass, freezer)
    assert not coord.stopping
    hass.bus.async_fire(EVENT_HOMEASSISTANT_STOP)
    await hass.async_block_till_done()
    assert coord.stopping  # a running backtest, fit or backfill reads it and ends
    with pytest.raises(CO.BacktestStopping):
        coord.start_backtest(None, None)
    entry = hass.config_entries.async_entries("betterybalance")[0]
    caplog.clear()
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert "Unable to remove unknown job listener" not in caplog.text


async def test_a_run_a_restart_cut_off_is_failed_and_its_partial_rows_go(hass, freezer, tmp_path):
    import math

    from custom_components.betterybalance.const import STORE_DIRNAME, STORE_FILE
    from custom_components.betterybalance.core import ledger as LG

    (tmp_path / STORE_DIRNAME).mkdir()
    store = Store(tmp_path / STORE_DIRNAME / STORE_FILE)
    store.migrate()
    q = 1_793_000_000 - 1_793_000_000 % 900
    rid = store.new_run(q, q - 86400, q)
    store.put_run_quarters(rid, [
        {"quarter_ts": q, "path": p, "import_kwh": 0.1, "export_kwh": 0.0,
         "money_eur": math.nan, "missing": True}
        for p in LG.PATHS
    ])
    store.close()
    coord = await _setup(hass, freezer)
    (run,) = coord.site.store.runs()
    assert run["status"] == "failed" and run["headline"] == {"error": CO.INTERRUPTED}
    db = coord.site.store._store._db
    for table in ("run_quarters", "run_decisions"):
        assert db.execute(f"select count(*) from {table}").fetchone()[0] == 0
