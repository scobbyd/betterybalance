"""Minute recorder: power and SoC samples to the minute flows rows ``LiveSite.on_minute`` reads.

Samples are ``state_changed`` and ``state_reported`` events, so an unchanged value still counts
as a sample and the sample rate is the integration's polling rate. That rate differs per box: a
Solarman profile can report every 5 s or every 32 s, and the SoC every 64 s. Each series is
integrated sample-and-hold over ``[minute_ts, minute_ts + 60)``: the value before the minute
holds until the first sample inside it, each sample holds until the next. A series is valid in
a minute when every span of positive length inside the minute holds a number and, at every
instant of the minute, the holding sample is no older than the series' hold age:
``MAX_HOLD_S`` for the power series, ``MAX_HOLD_SOC_S`` for the SoC. A series without a sample
at or before the minute start holds no value there and is invalid. An unavailable or unknown
state (``None``) is never integrated as zero: its span makes the series invalid, and an invalid
required series makes the minute missing (``close`` returns ``None``). An invalid or unmapped
optional series gives NaN. ``n_samples`` is the smallest number of samples inside the minute
over the required series (0 when one of them only holds a value from before the minute).

Load is ``Load Power`` alone: on the ``deye_p3`` profile it already holds the UPS port (it
equals Total Load Consumption), so ``load_ups_power`` is recorded but not added, as on the
hourly path (``solarman.statistic_roles``).

With ``grid_counters`` (grid source ``p1`` or ``custom``) the grid comes from energy counters
in kWh instead of ``grid_power``: per direction the delta of the summed counters between their
held values at the two minute boundaries (the last value at or before each boundary). The grid
is valid when every counter has a number at the start boundary and no unavailable state up to
the end boundary; a counter without a sample inside the minute did not change (no hold age
applies to a counter). A negative delta of any counter (a reset) makes the minute missing.
``grid_power`` is then not required.
"""
from __future__ import annotations

import bisect
import math

MINUTE_S = 60
MAX_HOLD_S = 150  # the oldest a held power sample may be at any instant of a valid minute
MAX_HOLD_SOC_S = 600  # the same for the SoC, which moves slowly and may report every 64 s
REQUIRED_SERIES = ("grid_power", "battery_power", "soc")
OPTIONAL_SERIES = (
    "load_power",
    "load_ups_power",
    "pv_power",
    *(f"pv{i}_power" for i in range(1, 9)),
    "gen_power",
)
_ENERGY_COLUMNS = (
    ("load_kwh", "load_power"),
    ("pv_kwh", "pv_power"),
    *((f"pv{i}_kwh", f"pv{i}_power") for i in range(1, 9)),
    ("gen_kwh", "gen_power"),
)
_DIRECTIONS = ("import", "export")
_UNSEEN = object()  # an entity with no sample yet: no held value
_Sample = tuple[float, float | None]  # (ts, value); value None is unavailable
_W_S_PER_KWH = 3_600_000.0


def _clean(value) -> float | None:
    """A finite float, else None (unavailable, unknown, a non-number, NaN or infinity)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _key(sample: tuple[float, float | None]) -> float:
    return sample[0]


class MinuteRecorder:
    """Sample buffer per entity, closed one minute at a time (rules in the module docstring)."""

    def __init__(
        self,
        mapping: dict[str, str],
        signs: dict[str, int],
        grid_counters: dict[str, list[str]] | None = None,
    ) -> None:
        self._counters = (
            None if grid_counters is None else {d: list(grid_counters[d]) for d in _DIRECTIONS}
        )
        if self._counters is not None and not all(self._counters.values()):
            raise ValueError("grid_counters needs at least one import and one export counter")
        self._required = REQUIRED_SERIES if self._counters is None else REQUIRED_SERIES[1:]
        missing = [r for r in self._required if not isinstance(mapping.get(r), str)]
        if missing:
            raise ValueError(f"required series not mapped: {', '.join(missing)}")
        self._series = {
            role: mapping[role]
            for role in (*self._required, *OPTIONAL_SERIES)
            if isinstance(mapping.get(role), str) and mapping[role]
        }
        self._grid_sign = int(signs.get("grid_power", 1))
        self._battery_sign = int(signs.get("battery_power", 1))
        ids = set(self._series.values())
        if self._counters is not None:
            ids.update(i for d in _DIRECTIONS for i in self._counters[d])
        self._held: dict[str, object] = {eid: _UNSEEN for eid in ids}  # _UNSEEN or a _Sample
        self._buf: dict[str, list[_Sample]] = {eid: [] for eid in ids}
        self._closed_end: int | None = None

    def sample(self, entity_id: str, ts: float, value: float | None) -> None:
        """One state sample (``state_changed`` or ``state_reported``); ``None`` is unavailable.

        Samples of entities the recorder does not read, and samples older than the end of the
        last closed minute, are dropped.
        """
        buf = self._buf.get(entity_id)
        if buf is None:
            return
        ts = float(ts)
        if self._closed_end is not None and ts < self._closed_end:
            return
        bisect.insort_right(buf, (ts, _clean(value)), key=_key)

    def close(self, minute_ts: int) -> dict | None:
        """The flows row of ``[minute_ts, minute_ts + 60)``, or None for a missing minute.

        Samples before the end of the minute are folded into the held values and dropped.
        Raises ``ValueError`` for a minute that starts before the end of the last closed one.
        """
        m = int(minute_ts)
        end = m + MINUTE_S
        if self._closed_end is not None and m < self._closed_end:
            raise ValueError(f"minute {m} starts before the closed end {self._closed_end}")
        try:
            return self._row(m, end)
        finally:
            self._fold(end)

    # --- internals --------------------------------------------------------------------------

    def _sample_at(self, eid: str, t: float, inclusive: bool) -> object:
        """The holding sample at ``t``: the last at (``inclusive``) or before ``t``, or _UNSEEN."""
        buf = self._buf[eid]
        i = (bisect.bisect_right if inclusive else bisect.bisect_left)(buf, t, key=_key)
        return buf[i - 1] if i else self._held[eid]

    def _value_at(self, eid: str, t: float, inclusive: bool) -> object:
        """The held value at ``t`` (None for unavailable), or _UNSEEN."""
        got = self._sample_at(eid, t, inclusive)
        return got if got is _UNSEEN else got[1]

    def _segments(
        self, eid: str, m: int, end: int, max_hold: float
    ) -> tuple[list[tuple[float, float]], int] | None:
        """Sample-and-hold spans of positive length in ``[m, end)`` and the sample count.

        None when a span of positive length has no number (unavailable, or no value yet) or
        its holding sample is older than ``max_hold`` at the end of the span.
        """
        buf = self._buf[eid]
        lo = bisect.bisect_left(buf, m, key=_key)
        hi = bisect.bisect_left(buf, end, key=_key)
        inside = buf[lo:hi]
        spans: list[tuple[float, float]] = []
        held = self._sample_at(eid, m, inclusive=False)
        if held is _UNSEEN:
            held = (-math.inf, None)
        t = float(m)
        for nxt in (*inside, (float(end), None)):
            ts = nxt[0]
            if ts > t:
                if held[1] is None or ts - held[0] > max_hold:
                    return None
                spans.append((ts - t, held[1]))
            t, held = ts, nxt
        return spans, len(inside)

    def _valid(self, role: str, m: int, end: int) -> tuple[list[tuple[float, float]], int] | None:
        """The spans and sample count of a mapped series, None when it is invalid."""
        hold = MAX_HOLD_SOC_S if role == "soc" else MAX_HOLD_S
        return self._segments(self._series[role], m, end, hold)

    def _counter_delta(self, eid: str, m: int, end: int) -> float | None:
        start = self._value_at(eid, m, inclusive=True)
        if start is None or start is _UNSEEN:
            return None
        buf = self._buf[eid]
        lo = bisect.bisect_right(buf, m, key=_key)
        hi = bisect.bisect_right(buf, end, key=_key)
        if any(v is None for _, v in buf[lo:hi]):
            return None
        last = buf[hi - 1][1] if hi > lo else start
        delta = last - start
        return delta if delta >= 0 else None

    def _row(self, m: int, end: int) -> dict | None:
        valid = {role: self._valid(role, m, end) for role in self._series}
        if any(valid[role] is None for role in self._required):
            return None
        n = min(valid[role][1] for role in self._required)
        spans = {role: got[0] for role, got in valid.items() if got is not None}
        row: dict = {"ts": m}
        if self._counters is None:
            row["grid_import_kwh"], row["grid_export_kwh"] = _split(
                spans["grid_power"], self._grid_sign
            )
        else:
            for d in _DIRECTIONS:
                deltas = [self._counter_delta(eid, m, end) for eid in self._counters[d]]
                if any(x is None for x in deltas):
                    return None
                row[f"grid_{d}_kwh"] = float(sum(deltas))
        row["batt_charge_kwh"], row["batt_discharge_kwh"] = _split(
            spans["battery_power"], self._battery_sign
        )
        soc = spans["soc"]
        row["soc_pct"] = sum(dt * v for dt, v in soc) / MINUTE_S
        row["soc_min_pct"] = min(v for _, v in soc)
        row["soc_max_pct"] = max(v for _, v in soc)
        for column, role in _ENERGY_COLUMNS:
            s = spans.get(role)
            row[column] = math.nan if s is None else sum(dt * v for dt, v in s) / _W_S_PER_KWH
        row["n_samples"] = n
        return row

    def _fold(self, end: int) -> None:
        """Fold the samples before ``end`` into the held values and drop them."""
        for eid, buf in self._buf.items():
            i = bisect.bisect_left(buf, end, key=_key)
            if i:
                self._held[eid] = buf[i - 1]
                del buf[:i]
        self._closed_end = end


def _split(spans: list[tuple[float, float]], sign: int) -> tuple[float, float]:
    """Positive and negative parts (both as positive kWh) of ``sign * W`` over the spans."""
    pos = sum(dt * sign * v for dt, v in spans if sign * v > 0) / _W_S_PER_KWH
    neg = sum(-dt * sign * v for dt, v in spans if sign * v < 0) / _W_S_PER_KWH
    return pos, neg
