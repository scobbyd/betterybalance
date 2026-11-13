"""The plan card: served from the integration's static path, registered once as a Lovelace
module resource (updated in place, never duplicated), dropped with the last entry."""

from unittest.mock import patch

from homeassistant.components.lovelace.const import LOVELACE_DATA
from homeassistant.config_entries import ConfigEntryState
from homeassistant.setup import async_setup_component

from custom_components.betterybalance import __version__
from custom_components.betterybalance import frontend as FE
from custom_components.betterybalance.const import CARD_URL
from tests.ha.test_coordinator import (
    _setup,
    offline,  # noqa: F401  (every feed offline, the store under tmp_path)
)


def _ours(hass) -> list[dict]:
    items = hass.data[LOVELACE_DATA].resources.async_items()
    return [i for i in items if i["url"].startswith(CARD_URL)]


async def test_reload_leaves_one_resource_and_the_card_is_served(hass, hass_client, freezer):
    coord = await _setup(hass, freezer)
    assert await hass.config_entries.async_reload(coord.entry.entry_id)
    await hass.async_block_till_done()
    ours = _ours(hass)
    assert len(ours) == 1
    assert ours[0]["url"] == f"{CARD_URL}?v={__version__}" and ours[0]["type"] == "module"
    resp = await (await hass_client()).get(CARD_URL)
    assert resp.status == 200
    body = await resp.text()
    assert "betterybalance-plan-card" in body


async def test_an_older_resource_is_updated_in_place(hass, freezer):
    assert await async_setup_component(hass, "lovelace", {})
    resources = hass.data[LOVELACE_DATA].resources
    await resources.async_get_info()
    old = await resources.async_create_item({"res_type": "module", "url": f"{CARD_URL}?v=0.0.1"})
    dup = await resources.async_create_item({"res_type": "module", "url": CARD_URL})
    other = await resources.async_create_item({"res_type": "module", "url": "/local/other.js"})
    await _setup(hass, freezer)
    ours = _ours(hass)
    assert [i["id"] for i in ours] == [old["id"]]
    assert ours[0]["url"] == f"{CARD_URL}?v={__version__}"
    assert dup["id"] not in [i["id"] for i in resources.async_items()]
    assert other["id"] in [i["id"] for i in resources.async_items()]


async def test_yaml_mode_adds_an_extra_module_url(hass, freezer):
    from homeassistant.components.frontend import DATA_EXTRA_MODULE_URL

    assert await async_setup_component(hass, "lovelace", {"lovelace": {"resource_mode": "yaml"}})
    assert hass.data[LOVELACE_DATA].resource_mode == "yaml"
    await _setup(hass, freezer)
    assert f"{CARD_URL}?v={__version__}" in hass.data[DATA_EXTRA_MODULE_URL].urls


async def test_removing_the_last_entry_drops_the_resource(hass, freezer):
    coord = await _setup(hass, freezer)
    assert len(_ours(hass)) == 1
    assert await hass.config_entries.async_remove(coord.entry.entry_id)
    await hass.async_block_till_done()
    assert _ours(hass) == []


async def test_register_twice_registers_the_static_path_once(hass, freezer):
    await _setup(hass, freezer)
    await FE.async_register_card(hass)
    await FE.async_register_card(hass)
    assert len(_ours(hass)) == 1


async def test_a_failed_registration_does_not_stop_the_entry(hass, freezer, caplog):
    with patch.object(FE, "async_register_card", side_effect=RuntimeError("boom")):
        entry = await _setup(hass, freezer, loaded=False)
    assert entry.state is ConfigEntryState.LOADED
    assert "plan card could not be registered" in caplog.text
