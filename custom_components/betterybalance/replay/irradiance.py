"""The replay's PV regressors per slot from the forecast a live box would have had.

``IrrByLead`` serves the window's irradiance at a decision time from the open-meteo Previous
Runs API (``core.feeds.openmeteo.previous_runs``), one table per lead, and falls back to the
archive (the historical-forecast API's day-0 output, ERA5 where missing) where no honest
lead has a value. Site-model fits keep reading the archive.

Weather data by Open-Meteo.com (CC BY 4.0).
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from ..core.fmt import eu
from ..core.timegrid import ZONE, Table, align
from .phases import IRR_COLS

# The Previous Runs archive starts in January 2024. At the generic point 52,1 / 5,2 the first
# values are lead 1 from 2024-01-19 12 UTC, lead 2 from 01-20 12 UTC, lead 3 from 01-21 12 UTC,
# complete from February 2024; earlier rows are null and fall back as ``missing_value``. Slots
# before this date never read a lead table (``pre_2024``).
PREVIOUS_RUNS_START = dt.date(2024, 1, 1)
MAX_LEAD = 3  # the leads fetched (previous_day1 to previous_day3)
DAY0_LEAD = 1  # day-0 slots read previous_day1 (the IrrByLead docstring has the finding)
# A model run publishes 3 to 5 h after its run time; 6 h is the conservative bound. Lead d at
# valid hour h is the run of h minus 24 d hours, so a box has it from h - 24 d h + LAG_S on.
LAG_S = 6 * 3600
FALLBACK_REASONS = ("pre_2024", "beyond_lead_3", "missing_value")
_DAY_S = 86400
_HOUR_S = 3600
_START_TS = int(
    dt.datetime(
        PREVIOUS_RUNS_START.year, PREVIOUS_RUNS_START.month, PREVIOUS_RUNS_START.day, tzinfo=ZONE
    ).timestamp()
)


class IrrByLead:
    """Irradiance per window slot from the Previous Runs lead a live box had at ``now``.

    Day-0 finding (live check at 52,1 / 5,2 for 2026-10-20, Task 18 report): the API accepts
    ``*_previous_day0`` but answers it under the plain variable name, and the plain variable
    equals the historical-forecast series hour for hour. For a past hour that is the newest
    run before the hour, issued within hours of it, which a live box did not have that
    morning. Day-0 slots therefore read ``previous_day1`` (``day0_lead=1``, ``DAY0_LEAD``) and
    ``previous_runs`` never requests a ``previous_day0`` variable. ``previous_day<d>`` at valid
    hour ``h`` is the newest run initialised at or before ``h - 24 d`` hours (the open-meteo
    page, and the ``icon_eu`` single runs of 2026-10-19 00, 06, 12 and 18 UTC).

    Lead rule: a quarter slot belongs to the hourly value that ends at ``hour_end`` (the next
    full hour); its lead is the smallest ``d >= day0_lead`` with
    ``hour_end - d * 86400 + LAG_S <= now``. A slot that needs a lead beyond ``MAX_LEAD``,
    a slot whose lead row has no value (NaN in any of the four columns) and a slot before
    ``PREVIOUS_RUNS_START`` (Amsterdam) take the archive row and count in ``fallback_rows``.
    ``fallback_rows`` sums over every ``at`` call, so a slot read in several windows counts
    once per window. ``reads`` counts every slot read; ``fallbacks`` splits ``fallback_rows``
    by reason: ``pre_2024`` (slot before ``PREVIOUS_RUNS_START``), ``beyond_lead_3`` (the
    lead rule needs a lead above ``MAX_LEAD``) and ``missing_value`` (no lead table, or NaN in
    any of the four columns), checked in that order.
    """

    def __init__(self, by_lead: dict[int, Table], archive: Table, day0_lead: int = DAY0_LEAD):
        self.by_lead = {int(d): t for d, t in by_lead.items()}
        self.archive = archive
        self.day0_lead = int(day0_lead)
        self.reads = 0
        self.fallbacks = dict.fromkeys(FALLBACK_REASONS, 0)

    @property
    def fallback_rows(self) -> int:
        """Slot reads that took the archive, all reasons."""
        return sum(self.fallbacks.values())

    def leads(self, now: int, slots_ts: np.ndarray) -> np.ndarray:
        """The lead per slot by the lead rule (values above ``MAX_LEAD`` included)."""
        slots = np.asarray(slots_ts, dtype=np.int64)
        hour_end = (slots // _HOUR_S) * _HOUR_S + _HOUR_S
        need = -((int(now) - hour_end - LAG_S) // _DAY_S)  # ceil((end + LAG - now) / day)
        return np.maximum(need, self.day0_lead)

    def at(self, now: int, slots_ts: np.ndarray) -> Table:
        """``ts`` plus the four ``IRR_COLS`` on ``slots_ts`` as a box had them at ``now``."""
        slots = np.asarray(slots_ts, dtype=np.int64)
        lead = self.leads(now, slots)
        out = align(self.archive, slots, IRR_COLS)
        use = np.zeros(len(slots), dtype=bool)
        pre = slots < _START_TS
        beyond = ~pre & (lead > MAX_LEAD)
        for d in np.unique(lead[~pre & ~beyond]).tolist():
            table = self.by_lead.get(int(d))
            if table is None:
                continue
            rows = ~pre & (lead == d)
            got = align(table, slots[rows], IRR_COLS)
            ok = np.ones(int(rows.sum()), dtype=bool)
            for col in IRR_COLS:
                ok &= np.isfinite(got[col])
            idx = np.flatnonzero(rows)[ok]
            for col in IRR_COLS:
                out[col][idx] = got[col][ok]
            use[idx] = True
        self.reads += len(slots)
        self.fallbacks["pre_2024"] += int(pre.sum())
        self.fallbacks["beyond_lead_3"] += int(beyond.sum())
        self.fallbacks["missing_value"] += int((~use & ~pre & ~beyond).sum())
        return out

    def note(self) -> str:
        """The ``site_report["irradiance"]`` sentence: the day-0 lead and the archive share of
        the slot reads, split by reason (percent of all reads, 0,0 % without reads)."""

        def pct(n: int) -> str:
            return eu(100.0 * n / self.reads if self.reads else 0.0, 1) + " %"

        f = self.fallbacks
        return (
            f"previous runs (day 0 from lead {self.day0_lead}); archive for "
            f"{pct(self.fallback_rows)} of reads: {pct(f['pre_2024'])} before 2024, "
            f"{pct(f['beyond_lead_3'])} beyond lead {MAX_LEAD}, {pct(f['missing_value'])} missing"
        )
