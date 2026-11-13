import json
from http import HTTPStatus

from homeassistant.setup import async_setup_component

from custom_components.betterybalance.diagnostics import EVENTS_N
from tests.ha.test_coordinator import (
    _setup,
    offline,  # noqa: F401  (every feed offline, the store under tmp_path)
)
from tests.ha.test_services_http import _run_minutes, _token

KEY = "synthetic-tennet-key-0123"


def _walk(x, keys: set, strings: list) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            keys.add(k)
            _walk(k, keys, strings)
            _walk(v, keys, strings)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _walk(v, keys, strings)
    elif isinstance(x, str):
        strings.append(x)


async def _diagnostics(hass, hass_client_no_auth, user, entry) -> dict:
    """The diagnostics download of ``entry`` through HA's own endpoint."""
    assert await async_setup_component(hass, "diagnostics", {})
    await hass.async_block_till_done()
    client = await hass_client_no_auth()
    headers = {"Authorization": f"Bearer {await _token(hass, user)}"}
    resp = await client.get(f"/api/diagnostics/config_entry/{entry.entry_id}", headers=headers)
    assert resp.status == HTTPStatus.OK
    return (await resp.json())["data"]


async def test_diagnostics_are_redacted(hass, freezer, hass_client_no_auth, hass_admin_user):
    coord = await _setup(hass, freezer)
    entry = coord.entry
    hass.config_entries.async_update_entry(entry, options={**entry.options, "tennet_key": KEY})
    await hass.async_block_till_done(wait_background_tasks=True)
    coord = entry.runtime_data
    await _run_minutes(hass, freezer, 3)
    diag = await _diagnostics(hass, hass_client_no_auth, hass_admin_user, entry)

    keys: set = set()
    strings: list = []
    _walk(diag, keys, strings)
    dump = json.dumps(diag)
    # the location the feeds see, nothing finer
    assert diag["site"]["latitude"] == 52.1 and diag["site"]["longitude"] == 5.2
    assert "52.149" not in dump and "5.231" not in dump
    # the mapping as role -> present, no entity id of the house
    assert diag["mapping"]["grid_power"] is True and diag["mapping"]["soc"] is True
    assert not [s for s in strings if s.startswith("sensor.")
                and not s.startswith("sensor.betterybalance_")]
    ids = [i for v in entry.data["entities"].values() for i in ([v] if isinstance(v, str) else v)]
    assert ids and not [i for i in ids if i in dump]
    # never the store contents, never a token or an identifier
    assert "bookings" not in keys and "decisions" not in keys and "minutes" not in keys
    assert KEY not in dump and diag["options"]["tennet_key"] == "**REDACTED**"
    assert entry.data["install_id"] not in dump
    # the household's monthly revenue line stays out of a public bug report
    assert diag["site"]["terms"]["balans_revenue_eur_month"] == "**REDACTED**"
    assert diag["site"]["terms"]["btw"] == entry.data["site"]["terms"]["btw"]
    # what the diagnostics are for
    head = coord.site.store.head()
    assert head is not None and diag["chain_head"] == {"seq": head[0], "hash": head[1]}
    assert diag["versions"]["integration"] == "0.3.0" and diag["pack_version"] is None
    assert set(diag["feeds"]) == {"day_ahead", "ida", "settlement", "live", "irradiance", "pack"}
    assert "offline" in (diag["feeds"]["day_ahead"]["last_error"] or "")
    assert diag["timing"]["minute_p50_ms"] is not None
    assert 1 <= len(diag["events"]) <= EVENTS_N == 10
    assert diag["events"][-1]["kind"] == "restart"


async def test_diagnostics_of_an_entry_that_is_not_loaded(hass, freezer):
    from custom_components.betterybalance.diagnostics import async_get_config_entry_diagnostics

    coord = await _setup(hass, freezer)
    entry = coord.entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    diag = await async_get_config_entry_diagnostics(hass, entry)
    assert diag["site"]["latitude"] == 52.1 and diag["loaded"] is False
    assert diag["chain_head"] is None and diag["events"] == []


async def test_the_pack_source_is_scrubbed_from_the_feed_errors(
    hass, freezer, hass_client_no_auth, hass_admin_user, tmp_path
):
    coord = await _setup(hass, freezer)
    entry = coord.entry
    source = str(tmp_path / "private-releases" / "folder")
    hass.config_entries.async_update_entry(entry, options={**entry.options, "pack_source": source})
    await hass.async_block_till_done(wait_background_tasks=True)
    coord = entry.runtime_data
    coord.feeds["pack"].last_error = f"pack source {source!r} does not read"
    diag = await _diagnostics(hass, hass_client_no_auth, hass_admin_user, entry)
    dump = json.dumps(diag)
    assert source not in dump and "private-releases" not in dump
    assert diag["options"]["pack_source"] == "**REDACTED**"
    assert "does not read" in diag["feeds"]["pack"]["last_error"]
