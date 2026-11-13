"""The period of the results views: day, week, month or year, restored after a restart."""

from __future__ import annotations

from homeassistant.components.select import SelectEntity
from homeassistant.helpers.restore_state import RestoreEntity

from .entity import device_info, fixed_id

PERIODS = ("day", "week", "month", "year")
KEY = "period"


async def async_setup_entry(hass, entry, async_add_entities) -> None:
    async_add_entities([PeriodSelect(entry)])


class PeriodSelect(SelectEntity, RestoreEntity):
    _attr_has_entity_name = True
    _attr_options = list(PERIODS)

    def __init__(self, entry) -> None:
        self.entity_id = fixed_id("select", KEY)
        self._attr_unique_id = f"{entry.entry_id}_{KEY}"
        self._attr_translation_key = KEY
        self._attr_device_info = device_info(entry)
        self._attr_current_option = PERIODS[0]

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        last = await self.async_get_last_state()
        if last is not None and last.state in PERIODS:
            self._attr_current_option = last.state

    async def async_select_option(self, option: str) -> None:
        self._attr_current_option = option
        self.async_write_ha_state()
