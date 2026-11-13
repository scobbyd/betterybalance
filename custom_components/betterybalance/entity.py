"""The base of every BetteryBalance entity: one service device, fixed entity ids."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import callback
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN

DEVICE_NAME = "BetteryBalance"


def device_info(entry: ConfigEntry) -> DeviceInfo:
    return DeviceInfo(
        identifiers={(DOMAIN, entry.entry_id)},
        name=DEVICE_NAME,
        entry_type=DeviceEntryType.SERVICE,
    )


def fixed_id(platform: str, key: str) -> str:
    """The entity id of ``key``: fixed in code, the same in every language."""
    return f"{platform}.{DOMAIN}_{key}"


class BetteryEntity(CoordinatorEntity):
    """A coordinator entity whose state ``_refresh`` reads from ``coordinator.data``."""

    _attr_has_entity_name = True
    _platform = ""

    def __init__(self, coordinator, key: str) -> None:
        super().__init__(coordinator)
        entry = coordinator.config_entry
        self.key = key
        self.entity_id = fixed_id(self._platform, key)
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_translation_key = key
        self._attr_device_info = device_info(entry)
        if coordinator.data is not None:
            self._refresh(coordinator.data)

    def _refresh(self, data: dict) -> None:
        raise NotImplementedError

    @callback
    def _handle_coordinator_update(self) -> None:
        if self.coordinator.data is not None:
            self._refresh(self.coordinator.data)
        super()._handle_coordinator_update()
