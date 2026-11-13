"""The plan card: served from the integration's own static path and registered as a Lovelace
resource (design 14.2).

- The static path ``CARD_URL`` serves ``www/betterybalance-plan-card.js`` without cache
  headers; it is registered once per Home Assistant run (a second registration of the same
  route is refused by the router).
- Lovelace storage mode: exactly one ``module`` resource at ``CARD_URL?v=<version>``. A
  resource on the same path (any or no ``v``) is updated in place; further copies on that path
  are deleted, so a dashboard never loads the card twice.
- Lovelace YAML mode: the frontend's extra module URL list (``add_extra_js_url``), which loads
  the card on every dashboard without a resource entry.
- ``async_unregister_card`` deletes the storage resource when the last entry is removed.
"""

from __future__ import annotations

import logging
from pathlib import Path

from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.components.lovelace.const import LOVELACE_DATA
from homeassistant.core import HomeAssistant

from . import __version__
from .const import CARD_URL, DOMAIN

_LOGGER = logging.getLogger(__name__)
CARD_FILE = Path(__file__).parent / "www" / "betterybalance-plan-card.js"
_STATIC_DONE = f"{DOMAIN}_card_static"
MODE_YAML = "yaml"


def card_url() -> str:
    """The resource URL with the version as cache key."""
    return f"{CARD_URL}?v={__version__}"


def _ours(items) -> list[dict]:
    return [i for i in items if str(i.get("url", "")).split("?", 1)[0] == CARD_URL]


async def async_register_card(hass: HomeAssistant) -> None:
    """Serve the card file and register it once (module docstring)."""
    if not hass.data.get(_STATIC_DONE):
        await hass.http.async_register_static_paths(
            [StaticPathConfig(CARD_URL, str(CARD_FILE), cache_headers=False)]
        )
        hass.data[_STATIC_DONE] = True
    url = card_url()
    data = hass.data.get(LOVELACE_DATA)
    if data is None or data.resource_mode == MODE_YAML:
        add_extra_js_url(hass, url)
        return
    resources = data.resources
    await resources.async_get_info()  # loads the storage collection
    ours = _ours(resources.async_items())
    if not ours:
        await resources.async_create_item({"res_type": "module", "url": url})
        return
    first, *extra = ours
    if first.get("url") != url or first.get("type") != "module":
        await resources.async_update_item(first["id"], {"res_type": "module", "url": url})
    for item in extra:
        _LOGGER.info("Removing a duplicate card resource %s", item.get("url"))
        await resources.async_delete_item(item["id"])


async def async_unregister_card(hass: HomeAssistant) -> None:
    """Delete the card's storage resources (the last entry was removed)."""
    data = hass.data.get(LOVELACE_DATA)
    if data is None or data.resource_mode == MODE_YAML:
        return
    resources = data.resources
    await resources.async_get_info()
    for item in _ours(resources.async_items()):
        await resources.async_delete_item(item["id"])
