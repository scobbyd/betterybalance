"""The ledger verifier (design 10.1, 10.5): ``python -m betterybalance.verify <export dir>``.

``verify_export`` reads an export folder (``ledger.csv``, ``settlements.csv``,
``manifest.json`` and, when present, ``flows_minute.csv``, as ``live.export`` writes them) and
runs these checks in order. It stops at the first failure and names it:

1. The chain: every row of ``ledger.csv`` from ``manifest["first_prev_hash"]`` (equal to
   ``install_id_hash`` when the span starts at seq 1) through ``ledger.verify_chain``, then
   the end against the manifest (``n_rows``, ``seq_span``, ``last_hash``), because a chain
   without a key cannot see a dropped last row: ``"chain broken at seq N: <reason>"``.
2. The structure of the range (the live site books every quarter it ran on the three paths,
   a gap after a restart as missing): every booked quarter has the three paths, the quarters
   from the first to the last booked one are all booked, a quarter with decisions is booked
   (only the running quarter after the last booked one may wait, when the chain ends with
   the export), and the count agrees with ``manifest["n_quarters"]``:
   ``"bookings missing at quarter <iso>"``.
3. The settled prices of the range's bookings. Each quarter that the k-th ``settle`` event
   in ``manifest["settles"]`` names has a k-th settled booking on every path:
   ``"settled booking missing at quarter <iso> path <p>"``. The newest bookings of a quarter
   carry one price source and one price: ``"mixed price sources at quarter <iso>"``. The k-th
   settled booking of a quarter and path matches the settlement version that the k-th event
   naming the quarter booked, else any version of the quarter with the same take, feed and
   regulation state; a version stored but not booked yet is no mismatch:
   ``"price mismatch at quarter <iso> path <p>"``.
4. The money of every booking of the range, ``ledger.money_eur`` with the manifest's terms
   (NaN when missing or unpriced): ``"money mismatch at quarter <iso> path <p>"``
   (tolerance 1e-9 EUR). Rows of the seq span outside the range are checked by the chain only.
5. The settled result over the range's quarters (``quarter_result`` of the newest house and
   shadow bookings, both settled) against ``manifest["result_settled_eur"]``:
   ``"total mismatch"`` (tolerance 0,005 EUR).
6. The energies: every booking of a quarter and path keeps the energies of the first one (a
   settled rebooking changes the price only). With ``flows_minute.csv`` also the re-run (below):
   ``"energy mismatch at quarter <iso>: <path> <field>"`` (tolerance 1e-9 kWh).

The re-run is the live site's own minute code: ``replay.steps.Stepper`` (the executor and
the virtual battery) and ``replay.steps.minute_flows``, fed with the recorded decisions
instead of the trader. A minute without a decision row repeats the previous hold (0 kW), a
minute without flows holds without a battery step. Restarts come from ``manifest["events"]``
and re-anchor at their SoC, each run on its own ``knobs`` (the executor's latency, dwell and
trade floor; the run the range starts in on ``manifest["knobs_at_start"]``; an export or a
run without them on the knobs of ``manifest["site"]``); the bookings between a restart's
``head_seq`` and its run's first decision are the gap, booked missing without a minute. The
shadow, house (``grid - batt_actual``) and actual bookings of each quarter go through
``ledger.booking_row`` and are compared with the booked energies, SoC and missing flag.

A range that does not start at a restart starts inside a run, so the executor's state at the
start (its setpoint, its queue and the sign and start of its last action, which the dwell
reads) comes from decisions outside the export. No fixed warm-up makes it known: the start
of the last action can carry a difference across later reversals. So the re-run tracks every
state the run can be in: ``core.battery.Executor.unknown`` with an unknown (NaN) setpoint,
for each sign and each action start the dwell can still reach (from the start of the range,
or with a deadband from the first 0 kW request, which sets every target to 0). A minute's
setpoint is known when all of them deliver the same finite value, and they become one
executor when their states are equal. The shadow of a quarter is judged only when every
minute's setpoint was known; otherwise the battery re-anchors at the quarter's booked shadow
SoC (a chained value) and the report says how many quarters were not judged. The first
quarter of such a range is never judged. The house and actual paths come from the flows
alone and are judged in every quarter.

What the chain proves: the rows are in the order they were written, and no row was inserted,
changed or dropped without a rewrite of every row after it, except rows cut from the end: a
shorter chain is valid too, and only the manifest's anchors (``n_rows``, ``seq_span``,
``last_hash``) or a recorded ``ledger_head`` show such a cut. Decisions and quarter energies
are chained before the quarter's settled price is known, so a good result cannot be picked
afterwards without a rewrite. What it cannot prove: a person with root on the box can rewrite
the files and rechain them. The external anchors are the ``ledger_head`` history that Home
Assistant's recorder keeps and a manifest or daily head that someone else holds: compare the
last hash that this verifier prints with them. Three inputs are not chained and the report
names each: ``settlements.csv`` is the box's copy of the settlement prices (compare it with
TenneT's publication), the terms come from ``manifest.json`` (compare them with the contract),
and ``flows_minute.csv`` holds the house's own minute measurements (the energies agree with
these flows, not with a meter someone else reads). The flows are read whatever run stored them;
the live site reads a minute only from its own run or an earlier one, which differs only when a
later run stored a minute that an earlier run booked without it.

The verifier imports numpy, ``core`` and ``replay.steps`` only, never ``live`` and never
Home Assistant, and its output is the same on the same files.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.battery import ExecutionProfile, Executor, execution_profile, load_profile
from ..core.fmt import eu
from ..core.ledger import (
    BOOKING_FIELDS,
    DECISION_FIELDS,
    FIELD_TYPES,
    LEDGER_COLUMNS,
    PATHS,
    PRICE_SOURCES,
    booking_row,
    money_eur,
    quarter_result,
    verify_chain,
)
from ..core.rule.trader import Decision
from ..core.site.contract import SiteConfig, from_dict, tax_mode_at, with_knobs
from ..core.timegrid import ISP_S, ZONE, Table, isp_start
from ..replay.steps import FLOW_REQUIRED, MinuteOut, Stepper, minute_flows

__all__ = ["LEDGER_COLUMNS", "VerifyReport", "main", "quarter_iso", "read_export", "verify_export"]

MINUTE_S = 60
MONEY_TOL_EUR = 1e-9
TOTAL_TOL_EUR = 0.005
ENERGY_TOL_KWH = 1e-9
ENERGY_FIELDS = (
    "import_kwh", "export_kwh", "batt_charge_kwh", "batt_discharge_kwh", "soc_pct", "missing",
)
_KIND_FIELDS = {"decision": DECISION_FIELDS, "booking": BOOKING_FIELDS}
_CHAIN_FIELDS = ("seq", "prev_hash", "hash")
_SETTLEMENT_TYPES = {
    "quarter_ts": int, "take": float, "feed": float, "regulation_state": int,
    "published_at": int, "source": str,
}
_FLOW_INTS = ("ts", "n_samples", "step_s")
_NO_PRICE = {"buy": math.nan, "sell": math.nan, "state": 0}
NO_FLOWS = "energies not checked (no flows_minute.csv)"
PROOF_NOTE = (
    "The chain shows that the file is consistent and that no row was changed without a "
    "rewrite of every row after it. Decisions and quarter energies are chained before the "
    "quarter's settled price is known. Rows cut from the end leave a valid chain: only a kept "
    "manifest or a recorded ledger_head shows such a cut. A person with root on the box can "
    "rewrite the file and rechain it: compare the last hash with a ledger_head that Home "
    "Assistant recorded or that someone else holds."
)
SETTLEMENTS_NOTE = (
    "The settled prices are checked against settlements.csv, the copy on the box, which is "
    "not chained: compare it with TenneT's publication."
)
FLOWS_NOTE = (
    "The energies agree with flows_minute.csv, the house's own minute measurements, which "
    "are not chained."
)


@dataclass
class VerifyReport:
    ok: bool
    quarters: int
    lines: list[str]


def quarter_iso(qts: int) -> str:
    """The start of quarter ``qts`` as ISO 8601 in Amsterdam time."""
    return dt.datetime.fromtimestamp(int(qts), ZONE).isoformat()


# -- reading ------------------------------------------------------------------------------

_TRUE = ("true", "1")
_FALSE = ("false", "0")


def _typed(typ: type, cell: str):
    """One CSV cell as ``typ``: an empty cell is None (int, bool) or NaN (float), a non-finite
    float is NaN (the chain hashes both as null). Raises ValueError when it does not parse."""
    if typ is str:
        return cell
    t = cell.strip()
    if typ is float:
        if t == "":
            return math.nan
        f = float(t)
        return f + 0.0 if math.isfinite(f) else math.nan
    if t == "":
        return None
    if typ is bool:
        if t.lower() in _TRUE:
            return True
        if t.lower() in _FALSE:
            return False
        raise ValueError(f"not a bool: {cell!r}")
    try:
        return int(t)
    except ValueError:
        f = float(t)
        if not (math.isfinite(f) and f.is_integer()):
            raise
        return int(f)


def _cell(typ: type, cell: str):
    """``_typed``, or the raw cell when it does not parse (the chain check then fails on it)."""
    try:
        return _typed(typ, cell)
    except ValueError:
        return cell


def _ledger_item(raw: dict) -> tuple[str, dict]:
    kind = raw.get("kind") or ""
    fields = _KIND_FIELDS.get(kind, ())
    row = {f: _cell(FIELD_TYPES[f], raw.get(f) or "") for f in fields}
    for f in _CHAIN_FIELDS:
        row[f] = _cell(FIELD_TYPES[f], raw.get(f) or "")
    return kind, row


def _csv_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _flows_table(path: Path) -> Table:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        cells = list(reader)
    out: Table = {}
    for k, name in enumerate(header):
        col = [r[k] if k < len(r) else "" for r in cells]
        if name in _FLOW_INTS:
            out[name] = np.array([_typed(int, c) or 0 for c in col], dtype=np.int64)
        else:
            out[name] = np.array([_typed(float, c) for c in col], dtype=float)
    if "ts" not in out:
        raise ValueError("flows_minute.csv has no ts column")
    return out


def read_export(folder: Path) -> tuple[list[tuple[str, dict]], list[dict], dict, Table | None]:
    """The export in ``folder``: the ledger items ``(kind, row)`` in file order, each row with
    its kind's fields typed by ``ledger.FIELD_TYPES`` plus ``seq``, ``prev_hash`` and ``hash``
    (a cell that does not parse stays a string); the settlement rows typed; the manifest; the
    minute flows as a table, None without ``flows_minute.csv``. Raises OSError for a missing
    ledger, settlements or manifest file and ValueError for a ledger whose header is not
    ``LEDGER_COLUMNS`` or a manifest or flows file that does not parse."""
    folder = Path(folder)
    with open(folder / "ledger.csv", newline="", encoding="utf-8") as f:
        header = tuple(next(csv.reader(f), []))
    if header != LEDGER_COLUMNS:
        raise ValueError("the columns of ledger.csv are not LEDGER_COLUMNS")
    items = [_ledger_item(r) for r in _csv_rows(folder / "ledger.csv")]
    settlements = [
        {k: _typed(t, r.get(k) or "") for k, t in _SETTLEMENT_TYPES.items()}
        for r in _csv_rows(folder / "settlements.csv")
    ]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    path = folder / "flows_minute.csv"
    flows = _flows_table(path) if path.exists() else None
    return items, settlements, manifest, flows


# -- the checks ---------------------------------------------------------------------------


def _close(a: float, b: float, tol: float) -> bool:
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= tol


def _check_chain(items: list[tuple[str, dict]], m: dict) -> str | None:
    """Check 1: the rows chain from the manifest's anchor and end where the manifest says."""
    span, n_rows = m.get("seq_span"), m.get("n_rows")
    anchor = m.get("first_prev_hash")
    if span is not None and int(span[0]) == 1 and anchor != m.get("install_id_hash"):
        return "chain broken at seq 1: prev_hash mismatch"
    if not items:
        if span is not None or n_rows:
            first = 1 if span is None else int(span[0])
            return f"chain broken at seq {first}: chain truncated"
        return None
    check = verify_chain(items, anchor)
    if not check.ok:
        seq = check.first_bad_seq
        if seq is None:
            seq = 1 if span is None else int(span[0])
        return f"chain broken at seq {seq}: {check.reason}"
    last_seq, last_hash = int(items[-1][1]["seq"]), items[-1][1]["hash"]
    if span is None or n_rows is None:
        return f"chain broken at seq {last_seq}: the manifest names no end"
    if last_seq < int(span[1]) or len(items) < int(n_rows):
        return f"chain broken at seq {last_seq + 1}: chain truncated"
    if last_seq > int(span[1]) or len(items) > int(n_rows):
        return f"chain broken at seq {int(span[1]) + 1}: rows after the manifest's last row"
    if last_hash != m.get("last_hash"):
        return f"chain broken at seq {last_seq}: last hash differs from the manifest"
    return None


def _terms_line(cfg: SiteConfig, quarters: list[int]) -> str:
    """The terms the money used, from the manifest (not chained)."""
    t = cfg.terms
    modes = sorted({tax_mode_at(t, q) for q in quarters}) or [tax_mode_at(t, 0)]
    return (
        f"Terms used (manifest.json, not chained): import fee {eu(t.fee_import_eur_kwh, 4)} "
        f"EUR/kWh, export fee {eu(t.fee_export_eur_kwh, 4)} EUR/kWh, BTW {eu(t.btw * 100, 0)} "
        f"%, energy tax {eu(t.tax_eur_kwh, 5)} EUR/kWh excl. BTW, tax mode "
        f"{' and '.join(modes)}, dual pricing {'on' if t.dual_pricing else 'off'}. Compare "
        "them with the contract."
    )


def _check_structure(
    items: list[tuple[str, dict]], bookings: list[dict], m: dict, start: int, end: int
) -> str | None:
    """Check 2: every quarter the site ran is booked on the three paths (module docstring)."""
    booked: dict[int, set[str]] = {}
    for b in bookings:
        booked.setdefault(int(b["quarter_ts"]), set()).add(b["path"])
    for q in sorted(booked):
        if booked[q] != set(PATHS):
            return f"bookings missing at quarter {quarter_iso(q)}"
    qs = sorted(booked)
    if qs:
        for q in range(qs[0], qs[-1], ISP_S):
            if q not in booked:
                return f"bookings missing at quarter {quarter_iso(q)}"
    decided = sorted(
        {
            int(isp_start(int(r["ts"])))
            for kind, r in items
            if kind == "decision" and start <= int(r["ts"]) < end
        }
        - set(booked)
    )
    span, head_seq = m.get("seq_span"), m.get("chain_head_seq")
    chain_ends_here = span is None or head_seq is None or int(head_seq) == int(span[1])
    for q in decided:
        running = (
            q == decided[-1]
            and chain_ends_here
            and (not qs or q == qs[-1] + ISP_S)
        )
        if not running:
            return f"bookings missing at quarter {quarter_iso(q)}"
    want = m.get("n_quarters")
    if want is not None and int(want) != len(qs):
        return (
            f"bookings missing: the manifest names {eu(int(want), 0)} quarters, the ledger "
            f"holds {eu(len(qs), 0)}"
        )
    return None


def _named_versions(m: dict) -> dict[int, list[int | None]]:
    """Per quarter, the ``published_at`` each ``settle`` event naming it booked, in seq order
    (None where an older event names no version)."""
    named: dict[int, list[int | None]] = {}
    for ev in sorted(m.get("settles") or [], key=lambda e: int(e["seq"])):
        published = ev.get("published_at") or []
        for k, q in enumerate(ev.get("quarters") or []):
            named.setdefault(int(q), []).append(int(published[k]) if k < len(published) else None)
    return named


def _same_price(b: dict, s: dict) -> bool:
    state = 0 if s["regulation_state"] is None else int(s["regulation_state"])
    return (
        _close(b["buy"], s["take"], 0.0)
        and _close(b["sell"], s["feed"], 0.0)
        and b["state"] == state
        and not math.isnan(float(s["take"]))
    )


def _check_prices(
    bookings: list[dict], settlements: list[dict], m: dict, start: int, end: int
) -> str | None:
    """Check 3: settle events against settled rows, one price per quarter, settled prices."""
    named = _named_versions(m)
    settled_n: dict[tuple[int, str], int] = {}
    for b in bookings:
        q, path = int(b["quarter_ts"]), b["path"]
        if path not in PATHS or b["price_source"] not in PRICE_SOURCES:
            return f"price mismatch at quarter {quarter_iso(q)} path {path}"
        if b["price_source"] == "settled":
            settled_n[(q, path)] = settled_n.get((q, path), 0) + 1
    for q in sorted(named):
        if not start <= q < end:
            continue
        for path in PATHS:
            if settled_n.get((q, path), 0) < len(named[q]):
                return f"settled booking missing at quarter {quarter_iso(q)} path {path}"
    newest = {(int(b["quarter_ts"]), b["path"]): b for b in bookings}  # seq order: last wins
    for q in sorted({q for q, _ in newest}):
        rows = [newest[(q, p)] for p in PATHS if (q, p) in newest]
        first = rows[0]
        for r in rows[1:]:
            if (
                r["price_source"] != first["price_source"]
                or not _close(r["buy"], first["buy"], 0.0)
                or not _close(r["sell"], first["sell"], 0.0)
                or r["state"] != first["state"]
            ):
                return f"mixed price sources at quarter {quarter_iso(q)}"
    versions: dict[int, list[dict]] = {}
    for s in settlements:
        if s["quarter_ts"] is not None:
            versions.setdefault(int(s["quarter_ts"]), []).append(s)
    seen: dict[tuple[int, str], int] = {}
    for b in bookings:
        if b["price_source"] != "settled":
            continue
        q, path = int(b["quarter_ts"]), b["path"]
        k = seen.get((q, path), 0)
        seen[(q, path)] = k + 1
        names = named.get(q, [])
        pub = names[k] if k < len(names) else None
        cands = versions.get(q, [])
        if pub is not None:
            cands = [s for s in cands if s["published_at"] == pub]
        if not any(_same_price(b, s) for s in cands):
            return f"price mismatch at quarter {quarter_iso(q)} path {path}"
    return None


def _check_money(bookings: list[dict], cfg: SiteConfig) -> str | None:
    """Check 4: every ``money_eur`` from the booking's energies and price and the terms."""
    terms = cfg.terms
    for b in bookings:
        q = int(b["quarter_ts"])
        if b["missing"] or b["price_source"] == "none":
            want = math.nan
        else:
            want, _ = money_eur(
                b["import_kwh"], b["export_kwh"], b["buy"], b["sell"], b["state"], terms,
                tax_mode_at(terms, q),
            )
        if not _close(want, b["money_eur"], MONEY_TOL_EUR):
            return f"money mismatch at quarter {quarter_iso(q)} path {b['path']}"
    return None


def _settled_total(bookings: list[dict]) -> float:
    newest = {(int(b["quarter_ts"]), b["path"]): b for b in bookings}  # seq order: last wins
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


def _same_energies(path: str, q: int, got: dict, want: dict) -> str | None:
    for f in ENERGY_FIELDS:
        if f == "missing":
            same = bool(got[f]) == bool(want[f])
        else:
            same = _close(got[f], want[f], ENERGY_TOL_KWH)
        if not same:
            return f"energy mismatch at quarter {quarter_iso(q)}: {path} {f}"
    return None


def _check_rebookings(bookings: list[dict]) -> str | None:
    """Check 6, first part: every booking of a quarter and path has the first one's energies."""
    first: dict[tuple[int, str], dict] = {}
    for b in bookings:
        key = (int(b["quarter_ts"]), b["path"])
        if key not in first:
            first[key] = b
            continue
        bad = _same_energies(b["path"], key[0], b, first[key])
        if bad is not None:
            return bad
    return None


class _Recorded:
    """The trader of the re-run: ``Stepper.minute`` gets the minute's recorded decision."""

    def __init__(self) -> None:
        self.decision: Decision | None = None

    def minute(self, ts, nowcast, residual_kw, soc_kwh, plan_slot, learning) -> Decision:
        return self.decision


class _RerunStepper(Stepper):
    """``Stepper`` (executor and virtual battery as live) with ``_Recorded`` as its trader."""

    def reanchor(self, soc_kwh: float) -> None:
        super().reanchor(soc_kwh)
        self.trader = _Recorded()


class _Unsure:
    """The executor of a run that started before the range: every state it can be in
    (module docstring). ``request`` returns the known setpoint, else 0 kW with ``known``
    False; ``settled`` is the one executor once all of them are in the same finite state."""

    def __init__(self, profile: ExecutionProfile, t0: int):
        self.p = profile
        self.cands: list[Executor] | None = None
        self.known = False
        if profile.deadband_w <= 0:
            self._start(t0)

    def _start(self, t0: int) -> None:
        """The states at the request at ``t0``: an unknown setpoint and target, nothing
        queued, and each sign with each action start the dwell can reach after ``t0`` (an
        earlier request ``r`` took effect at ``r + latency``), or long ago."""
        p = self.p
        sinces = [
            t0 - MINUTE_S * k + p.latency_s
            for k in range(1, math.ceil((p.latency_s + p.dwell_s) / MINUTE_S) + 1)
            if t0 - MINUTE_S * k + p.latency_s > t0 - p.dwell_s
        ]
        self.cands = [Executor.unknown(p, 0, None)] + [
            Executor.unknown(p, sign, since) for sign in (-1, 1) for since in (None, *sinces)
        ]

    def request(self, ts: int, kw: float, grid_trade: bool = True) -> float:
        if self.cands is None:
            if kw != 0:  # with a deadband, only a 0 kW request makes every target known
                self.known = False
                return 0.0
            self._start(ts)
        outs = [e.request(ts, kw, grid_trade) for e in self.cands]
        self.known = math.isfinite(outs[0]) and all(o == outs[0] for o in outs)
        return outs[0] if self.known else 0.0

    def settled(self, now: int) -> Executor | None:
        if not self.cands:
            return None
        states = [e.state(now) for e in self.cands]
        cur, queue, _, _ = states[0]
        if not (math.isfinite(cur) and all(math.isfinite(v) for _, v in queue)):
            return None
        return self.cands[0] if all(s == states[0] for s in states[1:]) else None


def _decision(row: dict) -> Decision:
    return Decision(
        int(row["ts"]), row["action"], float(row["kw"]), bool(row["committed"]), row["reason"],
        row["margin_w"], row["value_stored"], row["charge_threshold"],
        row["discharge_threshold"], bool(row["learning"]),
    )


class _Rerun:
    """Check 6, second part: the minute walk over the chain in seq order (module docstring).

    The chain holds each run's minutes in order: a minute's decision row (none for a repeat
    hold), and after minute 14 of a quarter its three bookings. A ``restart`` re-anchors the
    stepper; the bookings between its ``head_seq`` and the run's first decision (the first
    minute of a run is always written) are the gap, booked missing without a minute.
    """

    def __init__(self, cfg: SiteConfig, m: dict, flows: Table, start: int, end: int):
        self.cfg, self.start, self.end = cfg, start, end
        soc0 = m.get("soc0_kwh")
        if soc0 is None:
            soc0 = cfg.pack.soc_floor * cfg.pack.capacity_kwh  # as a new live site
        run = self._run_cfg(m.get("knobs_at_start"))
        self.stepper = _RerunStepper(run, load_profile(cfg.loss_profile, cfg.pack), float(soc0))
        # inside a run until a restart says otherwise (``_restart`` replaces it)
        self.stepper.executor = _Unsure(execution_profile(run.execution_profile, run.knobs), start)
        self.flows = flows
        self.index = {int(t): i for i, t in enumerate(flows["ts"].tolist())}
        self.restarts = sorted(
            m.get("events") or [], key=lambda e: (int(e.get("head_seq") or 0), int(e["epoch"]))
        )
        self.cursor: int | None = start  # the next minute; None after a restart
        self.prev: Decision | None = None
        self._reset(None)
        self.expected: dict[int, dict[str, dict | None]] = {}
        self.unjudged: list[int] = []  # quarters whose shadow was not judged
        self.judged: list[int] = []

    def _reset(self, qts: int | None) -> None:
        self.qts = qts
        self.outs: list[MinuteOut] = []
        self.grid: list[float] = []
        self.batt: list[float] = []
        self.soc_rec = math.nan
        self.missing = False
        self.unsure = False  # a setpoint of this quarter was not known

    def _run_cfg(self, knobs: dict | None) -> SiteConfig:
        """The site on a run's knobs; the manifest's ``site`` knobs for a run without them (an
        export or an event written before runs named their knobs)."""
        return self.cfg if knobs is None else with_knobs(self.cfg, knobs)

    def _restart(self, ev: dict) -> None:
        soc = ev.get("soc_kwh")
        self.stepper.cfg = self._run_cfg(ev.get("knobs"))  # the run's executor and battery
        self.stepper.reanchor(self.stepper.battery.soc_kwh if soc is None else float(soc))
        self.cursor, self.prev = None, None
        self._reset(None)
        self.expected = {}

    def _flows_at(self, ts: int) -> dict | None:
        i = self.index.get(ts)
        if i is None:
            return None
        return {c: (self.flows[c][i] if c in self.flows else math.nan) for c in FLOW_REQUIRED}

    def _step(self, ts: int, row: dict | None) -> None:
        qts = int(isp_start(ts))
        if qts != self.qts:
            self._reset(qts)
        grid, batt, soc_rec, missing = minute_flows(self._flows_at(ts), self.cfg.pack)
        if row is not None:
            d = _decision(row)
        else:  # a repeat hold of the previous minute
            reason = "" if self.prev is None else self.prev.reason
            nan = math.nan
            d = Decision(ts, "hold", 0.0, False, reason, nan, nan, nan, nan, False)
        self.stepper.trader.decision = d
        out = self.stepper.minute(ts, None, grid, batt, missing, d.learning)
        ex = self.stepper.executor
        if isinstance(ex, _Unsure):
            self.unsure = self.unsure or not ex.known
            one = ex.settled(ts + MINUTE_S)
            if one is not None:
                self.stepper.executor = one
        self.prev = out.decision
        self.outs.append(out)
        self.grid.append(grid)
        self.batt.append(batt)
        self.soc_rec = soc_rec
        self.missing = self.missing or missing
        self.cursor = ts + MINUTE_S

    def _fill(self, until: int) -> None:
        while self.cursor is not None and self.cursor < until:
            self._step(self.cursor, None)

    def _book(self, q: int) -> dict[str, dict | None]:
        """The three bookings of quarter ``q`` as ``live.site`` books them (no price); the
        shadow None when a setpoint of the quarter was not known."""
        cfg = self.cfg
        if self.cursor is not None:
            self._fill(q + ISP_S)
        if self.outs and self.qts == q:
            net = np.array([o.net_kw for o in self.outs], dtype=float)
            house = np.array([o.house_kw for o in self.outs], dtype=float)
            ac = np.array([o.batt_ac_kw for o in self.outs], dtype=float)
            grid, batt = np.array(self.grid), np.array(self.batt)
            missing, soc_rec, unsure = self.missing, self.soc_rec, self.unsure
        else:  # a gap quarter: booked missing without a minute
            net = house = ac = grid = batt = np.full(ISP_S // MINUTE_S, np.nan)
            missing, soc_rec, unsure = True, math.nan, False
        soc = self.stepper.battery.soc_kwh / cfg.pack.capacity_kwh * 100.0
        terms = cfg.terms
        out = {
            "shadow": None if unsure else booking_row(
                q, "shadow", net, ac, soc, _NO_PRICE, terms, missing
            ),
            "house": booking_row(
                q, "house", house, np.zeros(len(house)), math.nan, _NO_PRICE, terms, missing
            ),
            "actual": booking_row(q, "actual", grid, batt, soc_rec, _NO_PRICE, terms, missing),
        }
        (self.unjudged if unsure else self.judged).append(q)
        self._reset(None)
        return out

    def run(self, items: list[tuple[str, dict]]) -> str | None:
        k = 0
        for kind, row in items:
            seq = int(row["seq"])
            while k < len(self.restarts) and int(self.restarts[k].get("head_seq") or 0) < seq:
                self._restart(self.restarts[k])
                k += 1
            if kind == "decision":
                ts = int(row["ts"])
                if not self.start <= ts < self.end:
                    continue
                if self.cursor is None:
                    self.cursor = ts  # the first minute of a run
                if ts < self.cursor:
                    iso = quarter_iso(isp_start(ts))
                    return f"energy mismatch at quarter {iso}: shadow minute order"
                self._fill(ts)
                self._step(ts, row)
                continue
            q = int(row["quarter_ts"])
            if row["price_source"] == "settled" or not self.start <= q < self.end:
                continue
            if q not in self.expected:
                self.expected[q] = self._book(q)
            path = row["path"]
            want = self.expected[q].get(path, False)
            if want is None:  # a shadow not judged: the battery goes on from its booked SoC
                if math.isfinite(row["soc_pct"]):
                    self.stepper.set_soc(row["soc_pct"] / 100.0 * self.cfg.pack.capacity_kwh)
                continue
            bad = (
                f"energy mismatch at quarter {quarter_iso(q)}: {path} import_kwh"
                if want is False
                else _same_energies(path, q, row, want)
            )
            if bad is not None:
                return bad
        return None

    def verdict(self) -> str:
        """``", energies match"``, or ``", energies match on J of M quarters"`` when the
        shadow of some quarters was not judged."""
        unjudged = set(self.unjudged)
        total = len(unjudged | set(self.judged))
        if not unjudged:
            return ", energies match"
        return f", energies match on {eu(total - len(unjudged), 0)} of {eu(total, 0)} quarters"

    def note(self) -> str | None:
        """The line on the quarters whose shadow was not judged (None when all were)."""
        unjudged = sorted(set(self.unjudged))
        n = len(unjudged)
        if not n:
            return None
        why = "(the range does not start at a restart)"
        first = sorted(set(self.unjudged) | set(self.judged))[:n]
        if unjudged == first:
            what = "the first quarter" if n == 1 else f"the first {eu(n, 0)} quarters"
        else:
            what = f"{eu(n, 0)} quarters"
        return f"energies not checked for {what} {why}"


def verify_export(folder: Path) -> VerifyReport:
    """Run the checks of the module docstring on the export in ``folder``."""
    try:
        items, settlements, m, flows = read_export(Path(folder))
        cfg = from_dict({**m["site"], "latitude": 0.0, "longitude": 0.0})
        start, end = (int(x) for x in m["range"])
    except (OSError, ValueError, KeyError, TypeError) as err:
        return VerifyReport(False, 0, [f"export unreadable: {err}"])
    bad = _check_chain(items, m)
    if bad is not None:
        return VerifyReport(False, 0, [bad])
    try:
        return _after_chain(items, settlements, m, flows, cfg, start, end)
    except (ValueError, KeyError, TypeError) as err:  # a chained cell of the wrong kind
        return VerifyReport(False, 0, [f"export unreadable: {err}"])


def _after_chain(
    items: list[tuple[str, dict]],
    settlements: list[dict],
    m: dict,
    flows: Table | None,
    cfg: SiteConfig,
    start: int,
    end: int,
) -> VerifyReport:
    """Checks 2 to 6 on rows that chain (module docstring)."""
    bookings = [
        r for kind, r in items if kind == "booking" and start <= int(r["quarter_ts"]) < end
    ]
    quarters = sorted({int(b["quarter_ts"]) for b in bookings})
    n = len(quarters)
    bad = (
        _check_structure(items, bookings, m, start, end)
        or _check_prices(bookings, settlements, m, start, end)
        or _check_money(bookings, cfg)
    )
    if bad is not None:
        return VerifyReport(False, n, [bad])
    total, want = _settled_total(bookings), float(m["result_settled_eur"])
    if not abs(total - want) <= TOTAL_TOL_EUR:
        detail = f"settled result of the ledger {eu(total)} EUR, of the manifest {eu(want)} EUR"
        return VerifyReport(False, n, ["total mismatch", detail])
    bad = _check_rebookings(bookings)
    rerun = None
    if bad is None and flows is not None:
        rerun = _Rerun(cfg, m, flows, start, end)
        bad = rerun.run(items)
    if bad is not None:
        return VerifyReport(False, n, [bad])
    head = f"chain intact, {eu(n, 0)} quarter{'' if n == 1 else 's'} recomputed, totals match"
    if rerun is not None:
        lines = [head + rerun.verdict()]
        note = rerun.note()
        if note is not None:
            lines.append(note)
    else:
        lines = [head, NO_FLOWS]
    if items:
        last = items[-1][1]
        lines.append(f"last hash {last['hash']} at seq {int(last['seq'])}")
    lines += [PROOF_NOTE, SETTLEMENTS_NOTE, _terms_line(cfg, quarters)]
    if rerun is not None:
        lines.append(FLOWS_NOTE)
    return VerifyReport(True, n, lines)


def main(argv: list[str] | None = None) -> int:
    """Print the report of ``verify_export(argv[0])``; 0 when every check passes, 1 when one
    fails, 2 for a wrong call."""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        print("usage: python -m betterybalance.verify <export dir>", file=sys.stderr)
        return 2
    report = verify_export(Path(args[0]))
    for line in report.lines:
        print(line)
    return 0 if report.ok else 1
