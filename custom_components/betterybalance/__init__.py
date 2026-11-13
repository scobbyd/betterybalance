"""BetteryBalance: imbalance-price shadow trader for Dutch home batteries.

Home Assistant is imported inside functions only, so ``import betterybalance``
works without Home Assistant installed; the one module-level import (for ``CONFIG_SCHEMA``)
is skipped when Home Assistant is absent.
"""

import logging

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)
__version__ = "0.3.0"

try:  # Home Assistant present: the integration defines async_setup, so it declares its schema
    from homeassistant.helpers import config_validation as _cv
except ImportError:  # imported without Home Assistant (the core, the tools, the core tests)
    CONFIG_SCHEMA = None
else:
    CONFIG_SCHEMA = _cv.config_entry_only_config_schema(DOMAIN)


async def async_setup(hass, config) -> bool:
    """Register the services, the ledger download view and the ledger tail websocket command
    once for the integration (they find the loaded entry when called), and sweep the
    temporary ledger zips."""
    from .http import async_register_http, sweep_temp
    from .services import async_register_services

    async_register_services(hass)
    async_register_http(hass)
    await hass.async_add_executor_job(sweep_temp)  # zips a crash left behind
    return True


async def async_setup_entry(hass, entry) -> bool:
    """Set up one BetteryBalance config entry: the plan card (served and registered once),
    the coordinator, its minute clock, the entity platforms and the options listener."""
    from .const import PLATFORMS
    from .coordinator import BetteryCoordinator
    from .frontend import async_register_card

    try:
        await async_register_card(hass)
    except Exception:  # the card is a convenience: the shadow trader runs without it
        _LOGGER.warning("The plan card could not be registered", exc_info=True)
    coordinator = BetteryCoordinator(hass, entry)
    await coordinator.async_start()
    entry.runtime_data = coordinator
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception:
        await coordinator.async_stop()
        raise
    entry.async_on_unload(entry.add_update_listener(_async_entry_updated))
    return True


async def _async_entry_updated(hass, entry) -> None:
    """Reload the entry when the options the site runs on changed (``run_options``): the
    knobs, the feeds and the pack source take effect at the reload's restart event, which
    names the knobs of the new run. An unchanged save and a change of the export choices
    (read at each service call) reload nothing."""
    from .config_flow import run_options

    coordinator = getattr(entry, "runtime_data", None)
    if coordinator is None or run_options(entry.options) == run_options(coordinator.options):
        return
    hass.config_entries.async_schedule_reload(entry.entry_id)


async def async_remove_entry(hass, entry) -> None:
    """Remove the entry's repair issues (the fill-in scale check, a failing minute loop) with
    the entry, and the
    card's Lovelace resource with the last entry."""
    from homeassistant.helpers import issue_registry as ir

    from .coordinator import loop_issue_id, scale_issue_id
    from .frontend import async_unregister_card

    ir.async_delete_issue(hass, DOMAIN, scale_issue_id(entry.entry_id))
    ir.async_delete_issue(hass, DOMAIN, loop_issue_id(entry.entry_id))
    others = [e for e in hass.config_entries.async_entries(DOMAIN) if e.entry_id != entry.entry_id]
    if not others:
        await async_unregister_card(hass)


async def async_unload_entry(hass, entry) -> bool:
    """Unload one BetteryBalance config entry: the platforms, then stop the clock and close
    the store. The "stopped trading" repair issue goes with the coordinator that raised it:
    the next coordinator counts its own failures and would never remove it."""
    from homeassistant.helpers import issue_registry as ir

    from .const import PLATFORMS
    from .coordinator import loop_issue_id

    if not await hass.config_entries.async_unload_platforms(entry, PLATFORMS):
        return False
    await entry.runtime_data.async_stop()
    ir.async_delete_issue(hass, DOMAIN, loop_issue_id(entry.entry_id))
    return True
