"""The ledger export, the live results export and the ledger tail (design 10.5, 13.3).

- ``export_ledger``: the files of a range ``[start, end)`` (epoch s on quarter boundaries):
  ``ledger.csv`` (the seq-contiguous span of chained rows, ``LEDGER_COLUMNS``),
  ``settlements.csv`` (every published version of the range's quarters), ``manifest.json``
  and, on request, ``flows_minute.csv``. The span runs from the lowest to the highest seq of
  the decisions with ``ts`` and the bookings with ``quarter_ts`` in range, and every row
  between them is included whatever its time: decisions and bookings share one chain and a
  day's settled bookings land among the next day's decisions (R19).
- ``write_export`` writes the files row by row (the chain in pages of ``CHAIN_PAGE`` rows,
  the minute flows from the store's columns) through a file opener: ``write_export_dir``
  writes a folder, ``write_export_zip`` streams one zip file, ``export_ledger`` keeps the
  files in memory (small ranges, tests). ``export_range`` checks a request: minute flows
  cover at most ``MAX_FLOW_DAYS`` days.
- ``zip_bytes``: the files as one zip, byte-identical for identical files and to
  ``write_export_zip`` of the same range.
- ``export_results_live``: the anonymised plan-2 results of the live ledger (``source`` live).
- ``ledger_tail``: the newest trading decisions joined with their quarter's settlement.

Machine files: CSV cells by ``FIELD_TYPES``: bools ``true``/``false``, floats as ``repr``
(they parse back to the same float, so the hash chain verifies over the rows read back), NaN
and None as an empty cell. JSON with sorted keys, non-finite numbers as ``null``. No file
carries the location: the manifest's site dict drops latitude and longitude.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import zipfile
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import TextIO

import numpy as np

from .. import __version__
from ..core.features import ELAPSED_MIN, TIERS
from ..core.ledger import (
    FIELD_TYPES,
    LEDGER_COLUMNS,
    LedgerRow,
    bootstrap_interval,
    daily_pairs,
    genesis_hash,
    quarter_result,
)
from ..core.results import export_results
from ..core.site.contract import SiteConfig, tax_mode_at
from ..core.store import Store
from ..core.timegrid import ISP_S, isp_start
from ..replay.phases import drop_empty_days, local_date, local_midnight, modal_tiers
from .bands import StoredBandSource
from .site import LiveConfig

__all__ = [
    "LEDGER_COLUMNS",
    "day_range",
    "export_ledger",
    "export_range",
    "export_results_live",
    "ledger_tail",
    "site_dict",
    "write_export",
    "write_export_dir",
    "write_export_zip",
    "zip_bytes",
]

MANIFEST_FORMAT = 1
DAY_S = 86400
_END = 2**62  # past every row
ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # the zip epoch: fixed member times keep the bytes stable
SETTLEMENT_COLUMNS = ("quarter_ts", "take", "feed", "regulation_state", "published_at", "source")
_SETTLEMENT_TYPES = {
    "quarter_ts": int, "take": float, "feed": float, "regulation_state": int,
    "published_at": int, "source": str,
}
_FLOW_INTS = ("ts", "n_samples", "step_s")
MAX_FLOW_DAYS = 31  # minute flows per export request: bounds the memory and the store lock
CHAIN_PAGE = 2000  # chained rows read per store call while writing ledger.csv


def day_range(first: dt.date, last: dt.date) -> tuple[int, int]:
    """``[start, end)`` of the Amsterdam days ``first`` to ``last`` (both included)."""
    return local_midnight(first), local_midnight(last + dt.timedelta(days=1))


# the translation keys of the range refusals (the integration's ``exceptions`` strings)
REFUSAL_KEYS = ("export_order", "export_dates", "export_flow_days")


class RangeRefusal(ValueError):
    """A refused export range: the English sentence, the translation ``key`` (one of
    ``REFUSAL_KEYS``) and its ``placeholders``."""

    def __init__(self, message: str, key: str, **placeholders: object) -> None:
        if key not in REFUSAL_KEYS:
            raise ValueError(f"unknown refusal key: {key}")
        super().__init__(message)
        self.key = key
        self.placeholders = {k: str(v) for k, v in placeholders.items()}


def export_range(first: dt.date, last: dt.date, include_flows: bool) -> tuple[int, int]:
    """``day_range`` of a requested export, checked: ``RangeRefusal`` (a ``ValueError`` with
    a sentence the caller shows) when ``first`` is after ``last`` (``export_order``), when a
    date lies outside the clock's range (``export_dates``), or when minute flows are asked for
    more than ``MAX_FLOW_DAYS`` days (``export_flow_days``)."""
    if first > last:
        raise RangeRefusal(
            f"from ({first}) must be on or before to ({last})",
            "export_order", **{"from": first, "to": last},
        )
    try:
        span = day_range(first, last)
    except (OverflowError, ValueError) as err:
        raise RangeRefusal(
            f"the dates {first} to {last} are out of range",
            "export_dates", **{"from": first, "to": last},
        ) from err
    days = (last - first).days + 1
    if include_flows and days > MAX_FLOW_DAYS:
        raise RangeRefusal(
            f"minute flows cover at most {MAX_FLOW_DAYS} days per export, {days} were asked",
            "export_flow_days", max_days=MAX_FLOW_DAYS, days=days,
        )
    return span


def _cell(typ: type, v) -> str:
    """One CSV cell (module docstring): an empty cell for None and non-finite floats."""
    if v is None:
        return ""
    if isinstance(v, np.generic):
        v = v.item()
    if typ is bool:
        return "true" if v else "false"
    if typ is float:
        f = float(v)
        return repr(f) if math.isfinite(f) else ""
    if typ is int:
        return str(int(v))
    return str(v)


def _write_rows(f: TextIO, columns, rows, types: dict[str, type]) -> None:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([_cell(types[c], r.get(c)) for c in columns])


def _jsonable(x):
    """Tuples as lists, numpy scalars unwrapped, non-finite floats as None, recursively."""
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    return x


def _json_text(d: dict) -> str:
    return json.dumps(_jsonable(d), sort_keys=True, indent=2, allow_nan=False) + "\n"


def site_dict(cfg: SiteConfig) -> dict:
    """The ``contract.from_dict`` dict of ``cfg`` without latitude and longitude.

    ``from_dict({**site_dict(cfg), "latitude": lat, "longitude": lon})`` rebuilds ``cfg`` at
    that location (the location enters neither the money nor the battery).
    """
    d = asdict(cfg)
    d.pop("latitude")
    d.pop("longitude")
    plan = d["ev"]["plan"]
    if plan is not None:
        d["ev"]["plan"] = {
            "energy_kwh": plan["energy_kwh"], "by": plan["by_local"], "days": list(plan["days"]),
        }
    return _jsonable(d)


def _install_hash(store: Store, lc: LiveConfig) -> str:
    """The chain's genesis: ``genesis_hash`` of the store's install id (the anchor it chains on)."""
    return genesis_hash(store.meta("install_id") or lc.install_id)


def _settled_result(newest: dict) -> float:
    """The settled result of the newest bookings: ``quarter_result`` summed over the quarters
    whose house and shadow bookings are both settled and priced."""
    total = 0.0
    for q in sorted({q for q, _ in newest}):
        house, shadow = newest.get((q, "house")), newest.get((q, "shadow"))
        if house is None or shadow is None:
            continue
        if house["price_source"] != "settled" or shadow["price_source"] != "settled":
            continue
        r = quarter_result(house, shadow)
        if not math.isnan(r):
            total += r
    return total


def _restarts(store: Store, start: int, end: int, span: tuple[int, int] | None) -> list[dict]:
    """The ``restart`` events of the export: ``ts``, ``epoch`` (the event's seq, the run it
    starts), the anchor ``soc_kwh``, ``head_seq``, ``last_minute`` and the run's ``knobs``
    (None for an event written before it named them).

    An event belongs to the export when its rows lie in the seq span (``head_seq`` from the
    span's first seq minus 1 to its last seq minus 1: the gap bookings and the run after it
    start inside the span), or when its ``ts`` lies in ``[start, end)`` and it precedes the
    span's last row. A restart on the next day that books the range's last quarters as a gap
    is in the first group. Without rows: the restarts with ``ts`` in range.
    """
    out = []
    for e in store.events():
        if e["kind"] != "restart":
            continue
        head = e["detail"].get("head_seq")
        in_time = start <= int(e["ts"]) < end
        if span is None:
            keep = in_time
        elif head is None:
            keep = in_time
        else:
            keep = span[0] - 1 <= int(head) <= span[1] - 1 or (in_time and int(head) < span[1])
        if keep:
            out.append(
                {
                    "ts": int(e["ts"]), "epoch": int(e["seq"]),
                    "soc_kwh": e["detail"].get("soc_kwh"), "head_seq": head,
                    "last_minute": e["detail"].get("last_minute"),
                    "knobs": e["detail"].get("knobs"),
                }
            )
    return out


def _knobs_at_start(store: Store, span: tuple[int, int] | None) -> dict | None:
    """The ``knobs`` of the run the span's first row belongs to: the last ``restart`` event
    whose ``head_seq`` lies before that row (None without rows, without such an event or for
    an event written before it named its knobs)."""
    if span is None:
        return None
    out = None
    for e in store.events():
        head = e["detail"].get("head_seq") if e["kind"] == "restart" else None
        if head is not None and int(head) <= span[0] - 1:
            out = e["detail"].get("knobs")
    return out


def _settles(store: Store, start: int, end: int) -> list[dict]:
    """The ``settle`` events that name a quarter in ``[start, end)``, in seq order: ``ts``,
    ``seq``, ``quarters`` and ``published_at`` (the settlement version each quarter was
    rebooked at, an empty list for an event written before it named them). The verifier
    pairs the k-th settled booking of a quarter and path with the k-th event naming it.

    Every settled row of the range's quarters comes from an event that names its quarter, so
    this covers the settled rows of the span. ``write_export`` reads the events before the
    ledger: an event and its rows are stored in one transaction, so every event listed has its
    rows in ``ledger.csv``."""
    out = []
    for e in store.events():
        if e["kind"] != "settle":
            continue
        quarters = [int(q) for q in e["detail"].get("quarters", [])]
        if any(start <= q < end for q in quarters):
            out.append(
                {
                    "ts": int(e["ts"]), "seq": int(e["seq"]), "quarters": quarters,
                    "published_at": [int(p) for p in e["detail"].get("published_at", [])],
                }
            )
    return out


def _soc0_kwh(store: Store, cfg: SiteConfig, start: int, restarts: list[dict]) -> float | None:
    """The shadow's SoC at ``start``: the SoC of the newest shadow booking of the quarter before
    ``start``, else the anchor of the first restart in range (the chain starts there), else None.
    """
    prev = [r for r in store.bookings(start - ISP_S, start) if r["path"] == "shadow"]
    if prev and math.isfinite(prev[-1]["soc_pct"]):
        return float(prev[-1]["soc_pct"]) / 100.0 * cfg.pack.capacity_kwh
    if restarts and restarts[0]["soc_kwh"] is not None:
        return float(restarts[0]["soc_kwh"])
    return None


def _write_flows(f: TextIO, m: dict) -> None:
    """``flows_minute.csv`` straight from the store's columns (no per-row dicts)."""
    columns = tuple(m)
    types = [int if c in _FLOW_INTS else float for c in columns]
    arrays = [m[c] for c in columns]
    w = csv.writer(f, lineterminator="\n")
    w.writerow(columns)
    for i in range(len(m["ts"])):
        w.writerow([_cell(t, a[i]) for t, a in zip(types, arrays, strict=True)])


def _write_ledger(f: TextIO, store: Store, start: int, end: int) -> dict:
    """``ledger.csv`` of the seq span, read ``CHAIN_PAGE`` rows at a time; the facts the
    manifest needs. Rows up to the span's last seq never change, so paging is consistent."""
    span = store.chain_span(start, end)
    facts = {
        "span": span, "first_prev_hash": None, "last_hash": None, "n_rows": 0,
        "model_versions": set(), "newest": {},
    }
    w = csv.writer(f, lineterminator="\n")
    w.writerow(LEDGER_COLUMNS)
    if span is None:
        return facts
    newest = facts["newest"]
    for lo in range(span[0], span[1] + 1, CHAIN_PAGE):
        for kind, row in store.chain(lo, min(lo + CHAIN_PAGE - 1, span[1])):
            row = {**row, "kind": kind}
            w.writerow([_cell(FIELD_TYPES[c], row.get(c)) for c in LEDGER_COLUMNS])
            if facts["first_prev_hash"] is None:
                facts["first_prev_hash"] = row["prev_hash"]
            facts["last_hash"] = row["hash"]
            facts["n_rows"] += 1
            if kind == "decision":
                facts["model_versions"].add(row["model_version"])
            elif start <= int(row["quarter_ts"]) < end:
                # seq order: the last booking of a quarter and path is its newest
                newest[(int(row["quarter_ts"]), row["path"])] = {
                    k: row[k] for k in ("price_source", "money_eur", "missing")
                }
    return facts


def _manifest(
    store: Store,
    lc: LiveConfig,
    start: int,
    end: int,
    include_flows: bool,
    facts: dict,
    settles: list[dict],
) -> dict:
    span, newest = facts["span"], facts["newest"]
    head = store.head()
    restarts = _restarts(store, start, end, span)
    return {
        "format": MANIFEST_FORMAT,
        "install_id_hash": _install_hash(store, lc),
        "first_prev_hash": facts["first_prev_hash"],
        "last_hash": facts["last_hash"],
        "n_rows": facts["n_rows"],
        "chain_head": None if head is None else head[1],
        "chain_head_seq": None if head is None else int(head[0]),
        "range": [start, end],
        "seq_span": None if span is None else [int(span[0]), int(span[1])],
        "model_versions": sorted(facts["model_versions"]),
        "pack_version": lc.pack_version,
        "integration_version": __version__,
        "site": site_dict(lc.cfg),
        "knobs_at_start": _knobs_at_start(store, span),
        "events": restarts,
        "settles": settles,
        "soc0_kwh": _soc0_kwh(store, lc.cfg, start, restarts),
        "result_settled_eur": _settled_result(newest),
        "n_quarters": len({q for q, _ in newest}),
        "flows_included": bool(include_flows),
    }


def write_export(
    store: Store,
    lc: LiveConfig,
    start: int,
    end: int,
    include_flows: bool,
    open_text: Callable[[str], AbstractContextManager[TextIO]],
) -> list[str]:
    """Write the export files of ``[start, end)`` (epoch s on quarter boundaries) row by row,
    each through ``open_text(name)``, in name order (``EXPORT_NAMES``); the names written.

    ``ledger.csv``: the seq span (module docstring). ``settlements.csv``: every version.
    ``flows_minute.csv``: only with ``include_flows``. ``manifest.json``: ``format`` 1;
    ``install_id_hash`` (the genesis); ``first_prev_hash`` (the anchor of the span's first
    row, the genesis when the span starts at seq 1); ``last_hash`` and ``n_rows`` (the
    span's last hash and row count: they anchor its end); ``chain_head`` and
    ``chain_head_seq`` (the store's head when exported, the value the status sensor
    publishes as ``ledger_head``); ``range`` ``[start, end]`` (half-open); ``seq_span``
    ``[first, last]`` (None without rows); ``model_versions`` of the span's decisions;
    ``pack_version``; ``integration_version``; ``site`` (``site_dict``, the terms included,
    the knobs as now); ``knobs_at_start`` (the knobs of the run the span starts in,
    ``_knobs_at_start``); ``events`` (the restarts of the span with each run's ``knobs``,
    ``_restarts``); ``settles`` (the settle events of the range's quarters, ``_settles``);
    ``soc0_kwh`` (``_soc0_kwh``); ``result_settled_eur`` (the settled result of the range's
    quarters) and ``n_quarters`` (the quarters with a booking in range), both from the
    exported rows, so they agree with ``ledger.csv`` while the minute loop appends;
    ``flows_included``.
    """
    start, end = int(start), int(end)
    names = []
    if include_flows:
        with open_text("flows_minute.csv") as f:
            _write_flows(f, store.minutes(start, end))
        names.append("flows_minute.csv")
    settles = _settles(store, start, end)  # before the ledger (``_settles``)
    with open_text("ledger.csv") as f:
        facts = _write_ledger(f, store, start, end)
    with open_text("manifest.json") as f:
        f.write(_json_text(_manifest(store, lc, start, end, include_flows, facts, settles)))
    with open_text("settlements.csv") as f:
        rows = store.settlements(start, end, newest=False)
        _write_rows(f, SETTLEMENT_COLUMNS, rows, _SETTLEMENT_TYPES)
    return [*names, "ledger.csv", "manifest.json", "settlements.csv"]


def export_ledger(
    store: Store, lc: LiveConfig, start: int, end: int, include_flows: bool
) -> dict[str, bytes]:
    """The files of ``write_export`` in memory as ``{name: bytes}`` (small ranges, tests)."""
    files: dict[str, bytes] = {}

    @contextmanager
    def open_text(name: str):
        buf = io.StringIO(newline="")
        yield buf
        files[name] = buf.getvalue().encode("utf-8")

    write_export(store, lc, start, end, include_flows, open_text)
    return files


def write_export_dir(
    store: Store, lc: LiveConfig, start: int, end: int, include_flows: bool, folder: Path
) -> list[str]:
    """``write_export`` into ``folder`` (created); an export file of an earlier export that this
    one lacks (``flows_minute.csv``) is removed, so the folder matches its manifest."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not include_flows:
        (folder / "flows_minute.csv").unlink(missing_ok=True)

    def open_text(name: str):
        return open(folder / name, "w", encoding="utf-8", newline="")

    return write_export(store, lc, start, end, include_flows, open_text)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def write_export_zip(
    store: Store, lc: LiveConfig, start: int, end: int, include_flows: bool, path: Path
) -> list[str]:
    """``write_export`` streamed into a zip file at ``path``: the same bytes as ``zip_bytes``
    of ``export_ledger`` (same members, order, times and modes), row by row."""
    with zipfile.ZipFile(path, "w") as z:

        @contextmanager
        def open_text(name: str):
            with z.open(_zip_info(name), "w") as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                yield text
                text.flush()
                text.detach()

        return write_export(store, lc, start, end, include_flows, open_text)


def zip_bytes(files: dict[str, bytes]) -> bytes:
    """``files`` as a deflated zip: members in name order, fixed times, modes and system."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name in sorted(files):
            z.writestr(_zip_info(name), files[name])
    return buf.getvalue()


# -- the live results ---------------------------------------------------------------------


def _ledger_row(b: dict) -> LedgerRow:
    return LedgerRow(
        ts=int(b["quarter_ts"]), path=b["path"], buy_eur_mwh=float(b["buy"]),
        sell_eur_mwh=float(b["sell"]), state=int(b["state"]),
        settled=b["price_source"] == "settled", import_kwh=float(b["import_kwh"]),
        export_kwh=float(b["export_kwh"]), money_eur=float(b["money_eur"]),
        tax_eur=float(b["tax_eur"]), learning=False, ev_passthrough_kwh=0.0,
        missing=bool(b["missing"]),
    )


def _slot0_tier(store: Store, lc: LiveConfig, qts: int) -> str | None:
    """The tier the quarter's ranking read for slot 0 (``build_window``: the highest healthy
    tier whose h0 cell at the setup's elapsed minute was served); None without a ranking."""
    w = store.window(qts)
    if w is None or w["healthy"] is None:
        return None
    src = StoredBandSource(store, w["model_version"], w["healthy"], lc.model.tables)
    src.set_now(w["now"], w["epoch"])
    minutes = (int(w["now"]) - int(isp_start(int(w["now"])))) // 60
    elapsed = max([e for e in ELAPSED_MIN if e <= minutes], default=ELAPSED_MIN[0])
    for tier in reversed(TIERS):
        if tier in w["healthy"] and src.bands(tier, 0, elapsed, qts) is not None:
            return tier
    return None


def _daily_heads(store: Store, start: int, end: int, genesis: str) -> list[dict]:
    """Per Amsterdam day inside ``[start, end)``: ``day``, ``seq`` and ``hash`` of the chain
    head just before the first row of the next day (a decision with ``ts`` or a booking with
    ``quarter_ts`` at or after the day's end), the head when the box began the next day. A day
    whose next day has no row yet is left out. ``seq`` 0 is the genesis.

    The head covers the rows written before the next day's first row, not every row of the
    day: a row of the day written after that row (the day's settled rebookings, days later)
    lies after the head. In the normal minute order the 23:45 quarter's provisional booking
    (minute 23:59) precedes the 00:00 decision, so the day's provisional bookings are
    covered."""
    keyed = [(int(r["ts"]), int(r["seq"])) for r in store.decisions(start, _END)]
    keyed += [(int(r["quarter_ts"]), int(r["seq"])) for r in store.bookings(start, _END)]
    if not keyed:
        return []
    keyed.sort()
    keys = np.array([k for k, _ in keyed], dtype=np.int64)
    first = np.minimum.accumulate(np.array([s for _, s in keyed], dtype=np.int64)[::-1])[::-1]
    out = []
    day = local_date(start)
    if local_midnight(day) < start:
        day += dt.timedelta(days=1)
    while local_midnight(day + dt.timedelta(days=1)) <= end:
        m = local_midnight(day + dt.timedelta(days=1))
        k = int(np.searchsorted(keys, m))
        if k == len(keys):
            break
        seq = int(first[k]) - 1
        h = genesis if seq == 0 else store.chain(seq, seq)[0][1]["hash"]
        out.append({"day": day.isoformat(), "seq": seq, "hash": h})
        day += dt.timedelta(days=1)
    return out


def export_results_live(
    store: Store, lc: LiveConfig, start: int, end: int, version: str, heads: bool
) -> dict:
    """The plan-2 results JSON (``core.results.export_results``) of the live ledger in
    ``[start, end)``, ``source`` ``live``.

    ``daily``: per Amsterdam day ``diff_eur = actual - shadow`` (``ledger.daily_pairs``, the
    Balans revenue credited to the actual path) over the quarters whose newest shadow and
    actual bookings are both settled; days whose every such quarter is missing are dropped.
    ``interval``: the seeded bootstrap of the daily differences. ``tiers_by_month``: the modal
    slot-0 tier of the quarters booked in range. ``learning_days``: the days with a learning
    decision. ``terminal_eur`` 0 (terminal valuation applies to a backtest window only,
    design 10.4). ``contract_mode`` at ``start``. With ``heads`` also ``daily_heads``
    (``_daily_heads``), the anchor a pooled export posts.
    """
    start, end = int(start), int(end)
    cfg = lc.cfg
    newest = store.newest_bookings(start, end)
    quarters = sorted({q for q, _ in newest})
    virtual, actual = [], []
    for q in quarters:
        s, a = newest.get((q, "shadow")), newest.get((q, "actual"))
        if s is None or a is None:
            continue
        if s["price_source"] != "settled" or a["price_source"] != "settled":
            continue
        virtual.append(_ledger_row(s))
        actual.append(_ledger_row(a))
    pairs = drop_empty_days(daily_pairs(virtual, actual, cfg.terms), virtual)
    interval = bootstrap_interval(np.asarray(pairs["diff_eur"], dtype=float), seed=0)
    tiers = [(q, t) for q in quarters if (t := _slot0_tier(store, lc, q)) is not None]
    learning = {local_date(int(r["ts"])) for r in store.decisions(start, end) if r["learning"]}
    result = SimpleNamespace(
        pairs=pairs, virtual=virtual, tiers_by_month=modal_tiers(tiers),
        learning_days=len(learning), terminal_eur=0.0, interval=tuple(interval),
    )
    out = export_results(result, cfg, version, source="live")
    out["contract_mode"] = str(tax_mode_at(cfg.terms, start))
    if heads:
        out["daily_heads"] = _daily_heads(store, start, end, _install_hash(store, lc))
    return out


# -- the ledger tail ----------------------------------------------------------------------


def _tail_row(store: Store, r: dict) -> dict:
    q = int(r["quarter_ts"])
    s = store.settlements(q, q + ISP_S, newest=True)
    s = s[-1] if s else {}
    out = _jsonable({k: v for k, v in r.items()})
    out["take"] = _jsonable(s.get("take"))
    out["feed"] = _jsonable(s.get("feed"))
    rs = s.get("regulation_state")
    out["regulation_state"] = None if rs is None else int(rs)
    return out


def ledger_tail(store: Store, limit: int, now: int) -> list[dict]:
    """The newest ``limit`` decision rows with ``committed`` true or an action other than
    ``hold``, newest first, each with ``take``, ``feed`` (EUR/MWh) and ``regulation_state`` of
    the newest settlement of its quarter (None while unsettled). NaN values are None.

    Reads backwards from ``now`` in growing windows (one day, then four times longer each
    step) so a long quiet spell costs a few queries, not the whole chain.
    """
    out: list[dict] = []
    lo, hi, span = int(now) - DAY_S, _END, DAY_S
    while True:
        rows = [r for r in store.decisions(lo, hi) if r["committed"] or r["action"] != "hold"]
        out.extend(reversed(rows))
        if len(out) >= limit or lo <= 0:
            break
        hi, span = lo, span * 4
        lo = max(0, lo - span)
    return [_tail_row(store, r) for r in out[: int(limit)]]
