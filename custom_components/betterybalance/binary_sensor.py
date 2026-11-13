"""One problem sensor per feed (``live.feeds.problem``), from the coordinator's feed states
and the store facts of its snapshot."""

from __future__ import annotations

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.util import dt as dt_util

from .const import FEEDS
from .entity import BetteryEntity
from .live import feeds as FD


async def async_setup_entry(hass, entry, async_add_entities) -> None:
    coord = entry.runtime_data
    async_add_entities([FeedProblemSensor(coord, feed) for feed in FEEDS])


class FeedProblemSensor(BetteryEntity, BinarySensorEntity):
    """On when the feed has a problem; attributes its last success (``last_ok``, epoch s),
    the age of it (``age_s``), its source, its last error and ``off_by_choice`` (the IDA feed
    with the ``fetch_ida`` option off: never fetched, never a problem). ``age_s`` changes every
    minute, so the recorder skips it; ``last_ok`` changes only with a success."""

    _platform = "binary_sensor"
    _attr_device_class = BinarySensorDeviceClass.PROBLEM
    _unrecorded_attributes = frozenset({"age_s"})

    def __init__(self, coordinator, feed: str) -> None:
        self.feed = feed
        super().__init__(coordinator, f"feed_{feed}")

    def _refresh(self, data: dict) -> None:
        now = int(dt_util.utcnow().timestamp())
        st = data["feeds"][self.feed]
        facts = data.get("facts") or {}
        self._attr_is_on = FD.problem(
            self.feed,
            now,
            st,
            bool(facts.get("have_tomorrow_da")),
            bool(facts.get("yesterday_settled")),
            bool(facts.get("pack_installed")),
        )
        self._attr_extra_state_attributes = {
            "last_ok": None if st.last_ok is None else int(st.last_ok),
            "age_s": None if st.last_ok is None else max(0, now - int(st.last_ok)),
            "source": st.source or None,
            "last_error": st.last_error,
            "off_by_choice": not st.enabled,
        }
