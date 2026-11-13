"""The backup platform: the store is paused and checkpointed while a backup copies it."""

import asyncio
import datetime as dt
import os
import sqlite3

from pytest_homeassistant_custom_component.common import async_fire_time_changed

from custom_components.betterybalance import backup as B
from custom_components.betterybalance import coordinator as CO
from custom_components.betterybalance.const import STORE_DIRNAME, STORE_FILE
from tests.ha.test_coordinator import _now, _set_states, _setup, _step, offline  # noqa: F401


async def _paused_minute(hass, freezer) -> None:
    """One minute of a running backup: the tick waits behind the paused minute lock, so this
    waits for nothing it starts (``async_block_till_done`` would wait for that tick)."""
    freezer.tick(dt.timedelta(seconds=60))
    _set_states(hass)
    async_fire_time_changed(hass)
    for _ in range(20):
        await asyncio.sleep(0)


def _files(hass) -> tuple[str, str]:
    db = hass.config.path(STORE_DIRNAME, STORE_FILE)
    return db, db + "-wal"


def _count(db: str, table: str) -> int:
    """Rows of ``table`` read on a connection of its own (outside the gate)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        con.close()


async def test_a_backup_pauses_the_store_checkpoints_it_and_the_minutes_catch_up(hass, freezer):
    coord = await _setup(hass, freezer)
    for _ in range(5):
        await _step(hass, freezer, 60)
    db, wal = _files(hass)
    assert os.path.getsize(wal) > 0  # the minutes write through the WAL
    await B.async_pre_backup(hass)
    assert coord._paused and not coord._gate.is_open
    assert os.path.getsize(wal) == 0  # every page is in the database file
    copy = sqlite3.connect(f"file:{db}?mode=ro", uri=True)  # what a backup copy reads
    assert copy.execute("PRAGMA quick_check").fetchone() == ("ok",)
    copy.close()
    n = _count(db, "nowcasts")
    for _ in range(4):  # four minutes of a running backup: nothing is written
        await _paused_minute(hass, freezer)
    assert os.path.getsize(wal) == 0 and _count(db, "nowcasts") == n
    await B.async_post_backup(hass)
    assert not coord._paused and coord._gate.is_open
    await _step(hass, freezer, 60)
    w = _now() // 60 * 60
    assert coord.site.next_minute == w  # the paused minutes were processed afterwards
    assert _count(db, "nowcasts") == n + 5
    assert len(coord.site.store.nowcasts(w - 5 * 60, w)) == 5


async def test_a_stuck_backup_ends_the_pause_by_itself(hass, freezer):
    coord = await _setup(hass, freezer)
    await _step(hass, freezer, 60)
    await B.async_pre_backup(hass)
    assert coord._paused
    freezer.tick(dt.timedelta(seconds=CO.PAUSE_MAX_S + 1))
    _set_states(hass)
    async_fire_time_changed(hass)  # the time-out fires, the queued tick then runs
    await hass.async_block_till_done(wait_background_tasks=True)
    assert not coord._paused and coord._gate.is_open
    await _step(hass, freezer, 60)
    assert coord.site.next_minute == _now() // 60 * 60  # a restore after the long gap
    await B.async_post_backup(hass)  # a late end changes nothing
    assert not coord._paused and coord._gate.is_open


async def test_an_unload_during_a_backup_ends_the_pause(hass, freezer):
    coord = await _setup(hass, freezer)
    await B.async_pre_backup(hass)
    entry = hass.config_entries.async_entries("betterybalance")[0]
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert not coord._paused and coord._gate.is_open and coord._store is None
    await B.async_post_backup(hass)  # no loaded entry: nothing to do


async def test_the_backup_manager_finds_the_platform(hass, freezer):
    from homeassistant.components.backup import DATA_MANAGER
    from homeassistant.setup import async_setup_component

    coord = await _setup(hass, freezer)
    assert await async_setup_component(hass, "backup", {})
    await hass.async_block_till_done()
    manager = hass.data[DATA_MANAGER]
    assert "betterybalance" in manager.platforms
    await manager.async_pre_backup_actions()
    assert coord._paused
    await manager.async_post_backup_actions()
    assert not coord._paused
