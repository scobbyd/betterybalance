"""HA's diagnostics download of a BetteryBalance entry (design 15).

What a bug report needs and nothing that describes the house beyond it: the versions, the site
with the location rounded to 0,1 degree and the monthly Balans revenue redacted, the role
mapping as role -> present (never an entity id), the grid source and signs, the options with
the TenneT key and the pack source redacted, the feed states (both also scrubbed from each
``last_error``), the chain head, the pack version, the loop timing,
the last backtest's status and the last ``EVENTS_N`` events. Never the store's contents
(minutes, flows, decisions, bookings, settlements), never the ledger, never a token or the
install id.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any

import numpy as np
from homeassistant.components.diagnostics import REDACTED, async_redact_data
from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.core import HomeAssistant

from . import __version__
from .core.site import contract as C
from .core.store import SCHEMA_VERSION

EVENTS_N = 10
TO_REDACT = {"tennet_key", "pack_source"}
# the household's own money line: diagnostics get pasted into public issues
SITE_REDACT = {"balans_revenue_eur_month"}
RUN_FIELDS = ("status", "started", "finished", "span_from", "span_to")


def _site(data: dict) -> dict:
    site = copy.deepcopy(dict(data.get("site") or {}))
    if "latitude" in site and "longitude" in site:
        site["latitude"], site["longitude"] = C.round_location(
            float(site["latitude"]), float(site["longitude"])
        )
    return site


def _scrub(text: str | None, secrets: list[str]) -> str | None:
    """``text`` with every secret replaced by the redaction marker."""
    if text is None:
        return None
    for secret in secrets:
        text = text.replace(secret, REDACTED)
    return text


def _store_part(coord) -> dict[str, Any]:
    """The store facts (blocking: run in the executor)."""
    store = coord.site.store
    head = store.head()
    return {
        "chain_head": None if head is None else {"seq": head[0], "hash": head[1]},
        "events": store.recent_events(EVENTS_N),
    }


async def async_get_config_entry_diagnostics(hass: HomeAssistant, entry: ConfigEntry) -> dict:
    """The redacted diagnostics of ``entry`` (module docstring)."""
    data = dict(entry.data)
    options = dict(entry.options)
    secrets = [s for s in (options.get("tennet_key"), options.get("pack_source")) if s]
    entities = dict(data.get("entities") or {})
    coord = getattr(entry, "runtime_data", None)
    loaded = entry.state is ConfigEntryState.LOADED and coord is not None and coord.site is not None
    out: dict[str, Any] = {
        "loaded": loaded,
        "versions": {
            "integration": __version__,
            "home_assistant": HA_VERSION,
            "numpy": np.__version__,
            "store_schema": SCHEMA_VERSION,
            "model": None,
        },
        "site": async_redact_data(_site(data), SITE_REDACT),
        "grid_source": data.get("grid_source"),
        "pv_mode": data.get("pv_mode"),
        "signs": dict(data.get("signs") or {}),
        "mapping": {role: bool(value) for role, value in sorted(entities.items())},
        "options": async_redact_data(options, TO_REDACT),
        "feeds": {},
        "chain_head": None,
        "pack_version": None,
        "timing": None,
        "backtest": None,
        "events": [],
    }
    if not loaded:
        return out
    lc = coord.site.lc
    out["versions"]["model"] = lc.model_version
    out["pack_version"] = lc.pack_version
    out["feeds"] = {
        name: {**asdict(st), "last_error": _scrub(st.last_error, secrets)}
        for name, st in coord.feeds.items()
    }
    snap = coord.data or {}
    out["timing"] = snap.get("timing")
    last = (snap.get("backtest") or {}).get("last")
    out["backtest"] = None if last is None else {k: last.get(k) for k in RUN_FIELDS}
    out.update(await hass.async_add_executor_job(_store_part, coord))
    return out
