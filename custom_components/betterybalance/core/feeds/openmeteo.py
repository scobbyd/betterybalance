"""Irradiance and temperature from open-meteo (CC BY 4.0, no key)."""

from __future__ import annotations

import csv
import datetime as dt
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from ..timegrid import ISP_S, Table
from .base import DayCache, FeedError, FeedHealth, http_get, json_object, local_days

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
# lead 0 is not requested: the API answers ``*_previous_day0`` with the plain variable, the
# newest run before each hour, which a live box did not have that morning (replay.irradiance)
PREVIOUS_RUNS_LEADS = (1, 2, 3)
PREVIOUS_RUNS_SRC = "previous_runs"  # the src of each cached row, never an archive src

IRRADIANCE_COLS = ("ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c")
STALE_S = 21600
_VARIABLES = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "temperature_2m",
)
_HOUR_S = 3600
_DAY_ROWS = 86400 // ISP_S
_CACHE_KEYS = ("ghi", "dni", "dhi", "temp")


class IrradianceSource(Protocol):
    def forecast(self, lat: float, lon: float, now: int) -> Table: ...

    def archive(
        self, lat: float, lon: float, start: dt.date, end: dt.date, cache: DayCache
    ) -> Table: ...

    def health(self) -> FeedHealth: ...


def relabel_start(ts: np.ndarray) -> np.ndarray:
    """Open-meteo labels an interval at its end; the core labels it at its start."""
    return np.asarray(ts, dtype=np.int64) - ISP_S


def _block(doc: dict, key: str) -> dict:
    block = doc.get(key)
    if not isinstance(block, dict) or "time" not in block:
        raise FeedError(f"open-meteo: response has no {key} block")
    return block


def _column(block: dict, name: str) -> np.ndarray:
    values = block.get(name)
    if values is None or len(values) != len(block["time"]):
        raise FeedError(f"open-meteo: column {name} is missing or has the wrong length")
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _table(ts: np.ndarray, block: dict, repeat: int = 1) -> Table:
    table: Table = {"ts": ts}
    for col, name in zip(IRRADIANCE_COLS, _VARIABLES, strict=True):
        table[col] = np.repeat(_column(block, name), repeat)
    return table


def parse_minutely15(doc: dict) -> Table:
    block = _block(doc, "minutely_15")
    return _table(relabel_start(np.array(block["time"], dtype=np.int64)), block)


def parse_hourly(doc: dict) -> Table:
    block = _block(doc, "hourly")
    starts = np.array(block["time"], dtype=np.int64) - _HOUR_S
    quarters = starts[:, None] + ISP_S * np.arange(_HOUR_S // ISP_S, dtype=np.int64)
    return _table(quarters.reshape(-1), block, repeat=_HOUR_S // ISP_S)


def _is_interpolated(doc: dict) -> bool:
    """True when the first 96 raw rows hold flat groups of four (hourly data upsampled)."""
    values = _column(_block(doc, "minutely_15"), "shortwave_radiation")[:_DAY_ROWS]
    groups = len(values) // 4
    if groups == 0:
        return False
    flat = values[: groups * 4].reshape(groups, 4)
    return bool(np.all(flat == flat[:, :1]))


def _request(url: str, params: dict) -> dict:
    return json_object(http_get(url, params=params), "open-meteo")


def _day_start(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.UTC).timestamp())


def _archive_params(lat: float, lon: float, first: dt.date, last: dt.date) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "timeformat": "unixtime",
        "timezone": "UTC",
    }


def _historical_table(lat: float, lon: float, first: dt.date, last: dt.date) -> Table | None:
    """Return the historical-forecast table, or None when that API has no 15 minute data."""
    params = {**_archive_params(lat, lon, first, last), "minutely_15": ",".join(_VARIABLES)}
    try:
        doc = _request(HISTORICAL_URL, params)
        if "minutely_15" in doc and not doc.get("error"):
            return parse_minutely15(doc)
    except FeedError:
        pass
    return None


def _hourly_table(lat: float, lon: float, first: dt.date, last: dt.date) -> Table:
    params = {**_archive_params(lat, lon, first, last), "hourly": ",".join(_VARIABLES)}
    doc = _request(ARCHIVE_URL, params)
    if doc.get("error"):
        raise FeedError(f"open-meteo archive: {doc.get('reason', 'error')}")
    return parse_hourly(doc)


def _day_mask(table: Table, day: dt.date) -> np.ndarray:
    lo = _day_start(day)
    return (table["ts"] >= lo) & (table["ts"] < lo + 86400)


def _day_is_usable(table: Table | None, day: dt.date) -> bool:
    """A day is usable with 96 rows and at least one radiation value."""
    if table is None:
        return False
    mask = _day_mask(table, day)
    if int(mask.sum()) != _DAY_ROWS:
        return False
    return any(not np.all(np.isnan(table[c][mask])) for c in IRRADIANCE_COLS[:3])


def _cache_day(table: Table, src: str, day: dt.date, cache: DayCache) -> None:
    mask = _day_mask(table, day)
    columns = [table[c][mask] for c in IRRADIANCE_COLS]
    rows = [
        {
            "ts": int(t),
            **{
                key: None if np.isnan(v) else float(v)
                for key, v in zip(_CACHE_KEYS, values, strict=True)
            },
            "src": src,
        }
        for t, *values in zip(table["ts"][mask], *columns, strict=True)
    ]
    cache.put(day, rows)


def _fill_month(lat: float, lon: float, days: list[dt.date], cache: DayCache) -> None:
    """Cache each day from the historical API, retrying unusable days through the ERA5 archive.

    open-meteo labels the END of the preceding interval; the API's day D response carries the
    interval ending at 00:00 of D as its first row and ends at the one labelled 23:45, so the
    last interval of D arrives with D + 1. Both requests therefore reach one day past the last
    day cached, and each cached day keeps its 96 start-labelled rows.
    """
    after = dt.timedelta(days=1)
    table = _historical_table(lat, lon, days[0], days[-1] + after)
    bad = []
    for day in days:
        if _day_is_usable(table, day):
            _cache_day(table, "m15", day, cache)
        else:
            bad.append(day)
    if not bad:
        return
    hourly = _hourly_table(lat, lon, bad[0], bad[-1] + after)
    for day in bad:
        if not _day_is_usable(hourly, day):
            raise FeedError(f"open-meteo: no complete irradiance data for {day.isoformat()}")
        _cache_day(hourly, "hourly", day, cache)


def _lead_name(name: str, lead: int) -> str:
    """The Previous Runs variable of ``name`` at ``lead``; lead 0 is the plain variable."""
    return name if lead == 0 else f"{name}_previous_day{lead}"


def parse_previous_runs(doc: dict, leads: tuple[int, ...]) -> dict[int, Table]:
    """One quarter table per lead from a Previous Runs ``hourly`` block.

    Lead ``d`` reads ``<variable>_previous_day<d>`` into ``IRRADIANCE_COLS``. Lead 0 reads
    ``<variable>_previous_day0`` when present, else the plain variable (the key under which
    the API answers a ``previous_day0`` request). Hourly values repeat to four quarters with
    the start relabel of ``parse_hourly``.
    """
    block = _block(doc, "hourly")
    starts = np.array(block["time"], dtype=np.int64) - _HOUR_S
    quarters = (starts[:, None] + ISP_S * np.arange(_HOUR_S // ISP_S, dtype=np.int64)).reshape(-1)
    repeat = _HOUR_S // ISP_S
    out: dict[int, Table] = {}
    for lead in leads:
        table: Table = {"ts": quarters.copy()}
        for col, name in zip(IRRADIANCE_COLS, _VARIABLES, strict=True):
            key = _lead_name(name, lead)
            if lead == 0 and f"{name}_previous_day0" in block:
                key = f"{name}_previous_day0"
            table[col] = np.repeat(_column(block, key), repeat)
        out[lead] = table
    return out


def _lead_keys(lead: int) -> list[str]:
    return [f"{key}{lead}" for key in _CACHE_KEYS]


def _previous_runs_month(
    lat: float, lon: float, days: list[dt.date], leads: tuple[int, ...], cache: DayCache | None
) -> dict[dt.date, list[dict]]:
    """Fetch ``days`` (one month) in one request; return and cache 96 rows per day.

    The request reaches one day past the last day, because the last hour of a day arrives
    labelled 00:00 of the next (``_fill_month``). A day is cached only when it lies before
    yesterday in UTC, so a day still inside the forecast horizon is fetched again later.
    """
    hourly = [_lead_name(name, lead) for lead in leads for name in _VARIABLES]
    params = {
        **_archive_params(lat, lon, days[0], days[-1] + dt.timedelta(days=1)),
        "hourly": ",".join(hourly),
    }
    doc = _request(PREVIOUS_RUNS_URL, params)
    if doc.get("error"):
        raise FeedError(f"open-meteo previous runs: {doc.get('reason', 'error')}")
    by = parse_previous_runs(doc, leads)
    settled = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)
    got: dict[dt.date, list[dict]] = {}
    for day in days:
        mask = _day_mask(by[leads[0]], day)
        if int(mask.sum()) != _DAY_ROWS:
            raise FeedError(f"open-meteo previous runs: no complete day {day.isoformat()}")
        rows = [{"ts": int(t), "src": PREVIOUS_RUNS_SRC} for t in by[leads[0]]["ts"][mask]]
        for lead in leads:
            for col, key in zip(IRRADIANCE_COLS, _lead_keys(lead), strict=True):
                for row, v in zip(rows, by[lead][col][mask], strict=True):
                    row[key] = None if np.isnan(v) else float(v)
        got[day] = rows
        if cache is not None and day < settled:
            cache.put(day, rows)
    return got


def previous_runs(
    lat: float,
    lon: float,
    first: dt.date,
    last: dt.date,
    leads: tuple[int, ...] = PREVIOUS_RUNS_LEADS,
    cache: DayCache | None = None,
) -> dict[int, Table]:
    """Previous Runs irradiance per lead for the UTC days ``first`` to ``last`` (inclusive).

    One request per month of days the cache lacks; each day is cached with its 96
    start-labelled rows and one value per lead and column. Null values (days before the
    Previous Runs archive starts in January 2024) stay NaN. Lead 0 (the plain variable, the
    run a live box did not have) and an empty ``leads`` raise ``ValueError``. Cached rows
    carry ``src = PREVIOUS_RUNS_SRC``; a cached day without it (an archive row) is fetched again.
    Weather data by Open-Meteo.com (CC BY 4.0).
    """
    leads = tuple(int(d) for d in leads)
    if not leads or any(d < 1 for d in leads):
        raise ValueError(f"previous runs: leads must be 1 or more and not empty, got {leads}")
    days = local_days(first, last)
    rows_of: dict[dt.date, list[dict]] = {}
    missing = []
    for day in days:
        rows = cache.get(day) if cache is not None else None
        need = [k for lead in leads for k in _lead_keys(lead)]
        if (
            rows is not None
            and len(rows) == _DAY_ROWS
            and all(r.get("src") == PREVIOUS_RUNS_SRC for r in rows)
            and all(k in rows[0] for k in need)
        ):
            rows_of[day] = rows
        else:
            missing.append(day)
    for year, month in sorted({(d.year, d.month) for d in missing}):
        month_days = [d for d in missing if (d.year, d.month) == (year, month)]
        rows_of.update(_previous_runs_month(lat, lon, month_days, leads, cache))
    ts = np.array([int(r["ts"]) for day in days for r in rows_of[day]], dtype=np.int64)
    out: dict[int, Table] = {}
    for lead in leads:
        table: Table = {"ts": ts.copy()}
        for col, key in zip(IRRADIANCE_COLS, _lead_keys(lead), strict=True):
            vals = [r[key] for day in days for r in rows_of[day]]
            table[col] = np.array([np.nan if v is None else v for v in vals], dtype=float)
        out[lead] = table
    return out


def _archive_rows(cache: DayCache, day: dt.date) -> list[dict] | None:
    """The cached archive rows of ``day``; None without them or for Previous Runs rows."""
    rows = cache.get(day)
    if rows is None or any(r.get("src") == PREVIOUS_RUNS_SRC for r in rows):
        return None
    return rows


class OpenMeteo(IrradianceSource):
    def __init__(self) -> None:
        self._ok = False
        self._fetched_at: int | None = None
        self._note = ""

    def forecast(self, lat: float, lon: float, now: int) -> Table:
        params = {
            "latitude": lat,
            "longitude": lon,
            "minutely_15": ",".join(_VARIABLES),
            "forecast_days": 3,
            "past_days": 1,
            "timeformat": "unixtime",
            "timezone": "UTC",
        }
        try:
            doc = _request(FORECAST_URL, params)
            table = parse_minutely15(doc)
            note = "interpolated" if _is_interpolated(doc) else ""
        except FeedError:
            self._ok = False
            raise
        self._ok, self._fetched_at, self._note = True, now, note
        return table

    def archive(
        self, lat: float, lon: float, start: dt.date, end: dt.date, cache: DayCache
    ) -> Table:
        days = local_days(start, end)
        missing = [d for d in days if _archive_rows(cache, d) is None]
        for year, month in sorted({(d.year, d.month) for d in missing}):
            month_days = [d for d in missing if (d.year, d.month) == (year, month)]
            _fill_month(lat, lon, month_days, cache)
        ts_list: list[int] = []
        columns: dict[str, list[float]] = {c: [] for c in IRRADIANCE_COLS}
        for day in days:
            for row in _archive_rows(cache, day) or []:
                ts_list.append(int(row["ts"]))
                for col, key in zip(IRRADIANCE_COLS, _CACHE_KEYS, strict=True):
                    columns[col].append(np.nan if row[key] is None else row[key])
        out: Table = {"ts": np.array(ts_list, dtype=np.int64)}
        for col in IRRADIANCE_COLS:
            out[col] = np.array(columns[col], dtype=float)
        return out

    def health(self) -> FeedHealth:
        age = None if self._fetched_at is None else max(0.0, time.time() - self._fetched_at)
        return FeedHealth(self._ok, age, self._note)


def stale_fallback(clear_sky_ghi: np.ndarray, trailing_clearness: float) -> np.ndarray:
    """Clear-sky shape times the trailing 3-day clearness, for a forecast older than STALE_S."""
    return np.asarray(clear_sky_ghi, dtype=float) * trailing_clearness


def write_irradiance_csv(table: Table, path: Path) -> None:
    """Write ts plus IRRADIANCE_COLS as plain decimals. NaN becomes an empty cell."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", *IRRADIANCE_COLS])
        for i, t in enumerate(table["ts"]):
            cells = [table[c][i] for c in IRRADIANCE_COLS]
            writer.writerow([int(t), *("" if np.isnan(v) else repr(float(v)) for v in cells)])


def read_irradiance_csv(path: Path) -> Table:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    table: Table = {"ts": np.array([int(r["ts"]) for r in rows], dtype=np.int64)}
    for col in IRRADIANCE_COLS:
        cells = [float(r[col]) if r[col] != "" else np.nan for r in rows]
        table[col] = np.array(cells, dtype=float)
    return table
