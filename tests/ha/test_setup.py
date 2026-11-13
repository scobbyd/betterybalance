from homeassistant.config_entries import ConfigEntryState

from tests.ha.test_coordinator import _setup


async def test_setup_and_unload(hass, freezer):
    coord = await _setup(hass, freezer)
    entry = hass.config_entries.async_entries("betterybalance")[0]
    assert entry.state is ConfigEntryState.LOADED and entry.runtime_data is coord
    assert await hass.config_entries.async_unload(entry.entry_id)
    assert entry.state is ConfigEntryState.NOT_LOADED
