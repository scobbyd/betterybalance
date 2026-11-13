"""Home Assistant backups: the store is paused and checkpointed while the backup copies it.

Home Assistant calls ``async_pre_backup`` before it copies the config folder (also on a
Supervisor backup, through the ``backup/start`` command) and ``async_post_backup`` after it,
also when the backup failed. The store is one SQLite file in WAL mode written every minute; a
copy taken while it is written can hold a database file and a WAL that do not belong
together. So the pre-backup closes every store call and moves the WAL into the database file
(``BetteryCoordinator.async_pause_for_backup``), and the post-backup opens them again. A
pause ends by itself after ``coordinator.PAUSE_MAX_S``.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from .const import DOMAIN


async def async_pre_backup(hass: HomeAssistant) -> None:
    """Pause the store writes of the loaded entry and checkpoint its WAL."""
    for entry in hass.config_entries.async_loaded_entries(DOMAIN):
        await entry.runtime_data.async_pause_for_backup()


async def async_post_backup(hass: HomeAssistant) -> None:
    """Resume the store writes."""
    for entry in hass.config_entries.async_loaded_entries(DOMAIN):
        entry.runtime_data.async_resume_after_backup()
