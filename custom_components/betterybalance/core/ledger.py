"""Ledgers of the two paths, the Balans money formula and the proof arithmetic."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from calendar import monthrange
from dataclasses import dataclass

import numpy as np

from .rule.thresholds import tax_legs_for_mode
from .rule.trader import Decision
from .rule.window import Nowcast
from .site.contract import Terms, tax_mode_at
from .timegrid import ZONE, Table, isp_start


@dataclass
class LedgerRow:
    """One quarter of one path. ``money_eur`` positive means the house pays."""

    ts: int
    path: str
    buy_eur_mwh: float
    sell_eur_mwh: float
    state: int
    settled: bool
    import_kwh: float
    export_kwh: float
    money_eur: float
    tax_eur: float
    learning: bool
    ev_passthrough_kwh: float
    missing: bool


def money_eur(
    import_kwh: float,
    export_kwh: float,
    buy_eur_mwh: float,
    sell_eur_mwh: float,
    state: int,
    terms: Terms,
    tax_mode: str,
) -> tuple[float, float]:
    """Cost of one quarter in EUR incl. BTW, as ``(total, tax_part)``.

    ``total = (supplier + tax) * (1 + btw)`` with
    ``supplier = import * (buy / 1000 + fee_import) - export * (sell / 1000 - fee_export)``
    and ``tax = import * tax_in - export * tax_out`` (legs by ``tax_mode``).
    ``tax_part = tax * (1 + btw)``. With ``dual_pricing`` off both legs take the mean of
    buy and sell. It never assumes ``buy >= sell`` (state 2 can publish buy below sell);
    ``state`` is kept for the caller's row and does not change the price.
    """
    if terms.dual_pricing:
        buy, sell = buy_eur_mwh, sell_eur_mwh
    else:
        buy = sell = (buy_eur_mwh + sell_eur_mwh) / 2.0
    tax_in, tax_out = tax_legs_for_mode(terms, tax_mode)
    supplier = import_kwh * (buy / 1000.0 + terms.fee_import_eur_kwh) - export_kwh * (
        sell / 1000.0 - terms.fee_export_eur_kwh
    )
    tax = import_kwh * tax_in - export_kwh * tax_out
    factor = 1.0 + terms.btw
    return (supplier + tax) * factor, tax * factor


def gross_split(net_kw_minutes: np.ndarray) -> tuple[float, float]:
    """``(import_kwh, export_kwh)`` from minute net powers (positive import), ``kWh = kW / 60``."""
    net = np.asarray(net_kw_minutes, dtype=float)
    return float(np.maximum(net, 0.0).sum() / 60.0), float(np.maximum(-net, 0.0).sum() / 60.0)


def virtual_net_kw(residual_kw: np.ndarray, batt_ac_kw: np.ndarray) -> np.ndarray:
    """Net grid power of the virtual path per minute: ``residual + batt``."""
    return np.asarray(residual_kw, dtype=float) + np.asarray(batt_ac_kw, dtype=float)


def rows_for_quarter(
    ts: int,
    path: str,
    net_kw_minutes: np.ndarray,
    prices: dict,
    terms: Terms,
    learning: bool = False,
    ev_passthrough_kwh: float = 0.0,
    missing: bool = False,
) -> LedgerRow:
    """Book one quarter. ``prices`` has ``buy``, ``sell``, ``state`` and ``settled``.

    A ``missing`` quarter keeps its energies and has NaN money and tax.
    """
    imp, exp = gross_split(net_kw_minutes)
    if missing:
        total = tax = float("nan")
    else:
        total, tax = money_eur(
            imp, exp, prices["buy"], prices["sell"], prices["state"], terms, tax_mode_at(terms, ts)
        )
    return LedgerRow(
        ts=int(ts),
        path=path,
        buy_eur_mwh=float(prices["buy"]),
        sell_eur_mwh=float(prices["sell"]),
        state=int(prices["state"]),
        settled=bool(prices["settled"]),
        import_kwh=imp,
        export_kwh=exp,
        money_eur=total,
        tax_eur=tax,
        learning=learning,
        ev_passthrough_kwh=ev_passthrough_kwh,
        missing=missing,
    )


def _local_day(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(ts, ZONE).date()


def daily_pairs(virtual: list[LedgerRow], actual: list[LedgerRow], terms: Terms) -> Table:
    """Per local day: ``virtual_eur``, ``actual_eur``, ``diff_eur``, ``n_missing``.

    ``diff_eur = actual - virtual``: positive means the virtual path cost less.

    Rows pair by ``ts``. A quarter missing on either path is excluded from both and counted
    in ``n_missing``. The Balans revenue ``balans_revenue_eur_month / days_in_month`` is
    credited to the actual path each day, so it lowers ``actual_eur``.
    """
    act = {r.ts: r for r in actual}
    sums: dict[dt.date, list[float]] = {}
    for v in virtual:
        a = act.get(v.ts)
        if a is None:
            continue
        s = sums.setdefault(_local_day(v.ts), [0.0, 0.0, 0.0])
        if v.missing or a.missing:
            s[2] += 1
        else:
            s[0] += v.money_eur
            s[1] += a.money_eur
    days = sorted(sums)
    v_eur = np.array([sums[d][0] for d in days], dtype=float)
    a_eur = np.array([sums[d][1] for d in days], dtype=float)
    credit = np.array(
        [terms.balans_revenue_eur_month / monthrange(d.year, d.month)[1] for d in days], dtype=float
    )
    a_eur = a_eur - credit
    return {
        "day": np.array([d.isoformat() for d in days], dtype=object),
        "virtual_eur": v_eur,
        "actual_eur": a_eur,
        "diff_eur": a_eur - v_eur,
        "n_missing": np.array([int(sums[d][2]) for d in days], dtype=np.int64),
    }


def terminal_valuation(
    soc_virtual_kwh: float, soc_actual_kwh: float, trailing_buy_eur_mwh: float, ed: float
) -> float:
    """Value of the SoC difference at the end: ``(virtual - actual) * trailing_buy / 1000 * ed``."""
    return (soc_virtual_kwh - soc_actual_kwh) * trailing_buy_eur_mwh / 1000.0 * ed


def trailing_mean_buy(settled_buy: np.ndarray, ts: np.ndarray, end: int, days: int = 7) -> float:
    """Mean non-NaN settled buy price in ``[end - days * 86400, end)``; NaN with none."""
    ts = np.asarray(ts, dtype=np.int64)
    buy = np.asarray(settled_buy, dtype=float)
    keep = (ts >= end - days * 86400) & (ts < end) & ~np.isnan(buy)
    return float(buy[keep].mean()) if keep.any() else float("nan")


def bootstrap_interval(
    diff: np.ndarray, level: float = 0.9, n: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """``(mean, lo, hi)``: percentile bootstrap of the mean daily difference, seeded.

    ``lo`` and ``hi`` are the ``(1 - level) / 2`` and ``(1 + level) / 2`` percentiles of
    ``n`` resampled means. Fewer than 2 days give ``(mean, nan, nan)``.
    """
    d = np.asarray(diff, dtype=float)
    mean = float(d.mean()) if len(d) else float("nan")
    if len(d) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return mean, float(lo), float(hi)


def drift_eur_mwh(rows: list[LedgerRow], provisional: list[LedgerRow]) -> float:
    """Mean ``settled buy - provisional buy`` over quarters in both (by ``ts``); NaN with none."""
    prov = {r.ts: r.buy_eur_mwh for r in provisional}
    d = [r.buy_eur_mwh - prov[r.ts] for r in rows if r.ts in prov]
    return float(np.mean(d)) if d else float("nan")


# --- The hash chain (plan 3) -------------------------------------------------------------

DECISION_FIELDS = (
    "ts", "quarter_ts", "action", "kw", "committed", "reason", "margin_w", "value_stored",
    "charge_threshold", "discharge_threshold", "band_take_q10", "band_take_q50",
    "band_take_q90", "band_feed_q10", "band_feed_q50", "band_feed_q90", "nowcast_take",
    "nowcast_feed", "soc_pct", "learning", "model_version",
)
BOOKING_FIELDS = (
    "quarter_ts", "path", "import_kwh", "export_kwh", "batt_charge_kwh", "batt_discharge_kwh",
    "soc_pct", "missing", "price_source", "buy", "sell", "state", "money_eur", "tax_eur",
)
# the columns of the exported ledger.csv: one row per chained row of either kind, the fields
# the row's kind lacks left empty (``live.export`` writes it, ``verify`` reads it)
LEDGER_COLUMNS = (
    ("seq", "kind")
    + DECISION_FIELDS
    + tuple(f for f in BOOKING_FIELDS if f not in DECISION_FIELDS)
    + ("prev_hash", "hash")
)
PATHS = ("shadow", "house", "actual")
PRICE_SOURCES = ("provisional", "settled", "none")
_KIND_FIELDS = {"decision": DECISION_FIELDS, "booking": BOOKING_FIELDS}

FIELD_TYPES: dict[str, type] = {
    "seq": int, "prev_hash": str, "hash": str, "kind": str,
    "ts": int, "quarter_ts": int, "action": str, "kw": float, "committed": bool, "reason": str,
    "margin_w": float, "value_stored": float, "charge_threshold": float,
    "discharge_threshold": float, "band_take_q10": float, "band_take_q50": float,
    "band_take_q90": float, "band_feed_q10": float, "band_feed_q50": float,
    "band_feed_q90": float, "nowcast_take": float, "nowcast_feed": float, "soc_pct": float,
    "learning": bool, "model_version": str,
    "path": str, "import_kwh": float, "export_kwh": float, "batt_charge_kwh": float,
    "batt_discharge_kwh": float, "missing": bool, "price_source": str, "buy": float,
    "sell": float, "state": int, "money_eur": float, "tax_eur": float,
}


def genesis_hash(install_id: str) -> str:
    """The anchor of an install's chain: ``sha256(install_id)`` as hex."""
    return hashlib.sha256(install_id.encode()).hexdigest()


_TRUE = ("true", "1")
_FALSE = ("false", "0")


def _as_int(v) -> int:
    """An integer from an int, an integral float or an integral numeric string; else ValueError."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, str):
        t = v.strip()
        try:
            return int(t)
        except ValueError:
            v = float(t)
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    f = float(v)
    if not (math.isfinite(f) and f.is_integer()):
        raise ValueError(f"not an integral value: {v!r}")
    return int(f)


def _canonical_value(field: str, v):
    """A value as its ``FIELD_TYPES`` type in plain Python; None and non-finite floats as None.

    The coercion makes the hash survive a round trip through SQLite (an integer 0 read back
    from a REAL column is 0.0, a bool comes back as 0 or 1) and through the ledger CSV (every
    value a string). Strings parse as: bools from ``"true"``/``"1"``/``"false"``/``"0"`` in
    any case, ints from integral numeric strings (``"1"``, ``"1.0"``), floats from numeric
    strings with ``""`` and ``"nan"`` as null. ``-0.0`` hashes as ``0.0``. A float32 value
    hashes at float32 precision (``np.float32(0.1)`` is ``0.10000000149011612``). A value that
    does not parse, a bool other than 0 or 1 and a non-integral int raise ValueError.
    """
    if v is None:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    typ = FIELD_TYPES[field]
    if typ is float:
        if isinstance(v, str):
            t = v.strip()
            if t == "" or t.lower() == "nan":
                return None
            v = t
        f = float(v)
        return f + 0.0 if math.isfinite(f) else None
    if typ is bool:
        if isinstance(v, str):
            t = v.strip().lower()
            if t in _TRUE:
                return True
            if t in _FALSE:
                return False
            raise ValueError(f"{field}: not a bool: {v!r}")
        if v in (0, 1):
            return bool(v)
        raise ValueError(f"{field}: not a bool: {v!r}")
    if typ is int:
        return _as_int(v)
    return str(v)


def canonical_json(kind: str, seq: int, row: dict) -> bytes:
    """The hashed bytes of a row: its own fields plus ``prev_hash``, ``kind`` and ``seq``.

    Keys sorted, no whitespace, ASCII, non-finite floats as ``null``. Values follow
    ``_canonical_value``; a missing field raises KeyError, a value that does not parse as its
    ``FIELD_TYPES`` type raises ValueError.
    """
    fields = _KIND_FIELDS[kind]
    obj = {f: _canonical_value(f, row[f]) for f in fields}
    obj["prev_hash"] = str(row["prev_hash"])
    obj["kind"] = kind
    obj["seq"] = _as_int(seq)
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def chain_row(kind: str, seq: int, row: dict, prev_hash: str) -> dict:
    """The row with ``seq``, ``prev_hash`` and ``hash = sha256(prev_hash + canonical_json)``."""
    body = canonical_json(kind, seq, {**row, "prev_hash": prev_hash})
    digest = hashlib.sha256(prev_hash.encode("ascii") + body).hexdigest()
    return {**row, "seq": seq, "prev_hash": prev_hash, "hash": digest}


@dataclass
class ChainCheck:
    """Result of ``verify_chain``: ``n`` is the number of items, also on a failure."""

    ok: bool
    n: int
    first_bad_seq: int | None
    reason: str


def verify_chain(items: list[tuple[str, dict]], anchor: str) -> ChainCheck:
    """Check ``(kind, row)`` items in seq order against ``anchor``; stop at the first failure.

    Per row: ``seq`` is the previous seq plus 1 (``"seq gap"``), ``prev_hash`` is the previous
    hash or the anchor for the first row (``"prev_hash mismatch"``), the recomputed hash
    equals ``hash`` (``"hash mismatch"``). A row whose ``seq`` is absent or not an integer is
    a ``"seq gap"`` at the expected seq (None for the first row). A row with a missing field
    or a value that does not parse as its type is a ``"hash mismatch"`` at its seq.
    """
    n = len(items)
    prev_hash, prev_seq = anchor, None
    for kind, row in items:
        try:
            seq = _as_int(row["seq"])
        except (KeyError, TypeError, ValueError):
            return ChainCheck(False, n, None if prev_seq is None else prev_seq + 1, "seq gap")
        if prev_seq is not None and seq != prev_seq + 1:
            return ChainCheck(False, n, seq, "seq gap")
        if row.get("prev_hash") != prev_hash:
            return ChainCheck(False, n, seq, "prev_hash mismatch")
        try:
            ok = chain_row(kind, seq, row, prev_hash)["hash"] == row["hash"]
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            return ChainCheck(False, n, seq, "hash mismatch")
        prev_hash, prev_seq = row["hash"], seq
    return ChainCheck(True, n, None, "")


def decision_row(
    d: Decision, quarter_band: dict, nowcast: Nowcast | None, soc_pct: float, model_version: str
) -> dict:
    """One decision as a ``DECISION_FIELDS`` dict, written before the quarter's price is known.

    ``quarter_band`` is ``window.band_for_slot(window, 0)``; the band columns take its q10,
    q50 and q90. Without a nowcast ``nowcast_take`` and ``nowcast_feed`` are NaN.
    """
    take, feed = quarter_band["take"], quarter_band["feed"]
    nan = float("nan")
    return {
        "ts": int(d.ts),
        "quarter_ts": int(isp_start(np.int64(d.ts))),
        "action": d.action,
        "kw": float(d.kw),
        "committed": bool(d.committed),
        "reason": d.reason,
        "margin_w": float(d.margin_w),
        "value_stored": float(d.value_stored),
        "charge_threshold": float(d.charge_threshold),
        "discharge_threshold": float(d.discharge_threshold),
        "band_take_q10": float(take[0.1]),
        "band_take_q50": float(take[0.5]),
        "band_take_q90": float(take[0.9]),
        "band_feed_q10": float(feed[0.1]),
        "band_feed_q50": float(feed[0.5]),
        "band_feed_q90": float(feed[0.9]),
        "nowcast_take": float(nowcast.take_point) if nowcast is not None else nan,
        "nowcast_feed": float(nowcast.feed_point) if nowcast is not None else nan,
        "soc_pct": float(soc_pct),
        "learning": bool(d.learning),
        "model_version": model_version,
    }


def is_repeat_hold(prev: Decision | None, d: Decision) -> bool:
    """True when ``prev`` and ``d`` are both a zero-kW hold with the same reason."""
    return (
        prev is not None
        and prev.action == "hold"
        and d.action == "hold"
        and prev.kw == 0
        and d.kw == 0
        and prev.reason == d.reason
    )


def booking_row(
    qts: int,
    path: str,
    net_kw: np.ndarray,
    batt_ac_kw: np.ndarray,
    soc_pct: float,
    prices: dict,
    terms: Terms,
    missing: bool,
) -> dict:
    """One quarter of one path as a ``BOOKING_FIELDS`` dict, booked at the provisional price.

    Energies, money and tax come from ``rows_for_quarter`` (the one money path). The battery
    energies are AC side: ``sum(max(batt, 0)) / 60`` charged, ``sum(max(-batt, 0)) / 60``
    discharged. ``price_source`` is ``"provisional"`` with finite buy and sell, else
    ``"none"``; money and tax are NaN when ``missing`` or ``"none"``. ``prices["state"]`` must
    be an integer also with NaN prices (pass 0), because ``rows_for_quarter`` books it as int.
    """
    if path not in PATHS:
        raise ValueError(f"path must be one of {PATHS}, got {path!r}")
    finite = math.isfinite(float(prices["buy"])) and math.isfinite(float(prices["sell"]))
    source = "provisional" if finite else "none"
    lr = rows_for_quarter(qts, path, net_kw, {**prices, "settled": False}, terms, missing=missing)
    batt = np.asarray(batt_ac_kw, dtype=float)
    money, tax = (lr.money_eur, lr.tax_eur) if finite else (float("nan"), float("nan"))
    return {
        "quarter_ts": lr.ts,
        "path": path,
        "import_kwh": lr.import_kwh,
        "export_kwh": lr.export_kwh,
        "batt_charge_kwh": float(np.maximum(batt, 0.0).sum() / 60.0),
        "batt_discharge_kwh": float(np.maximum(-batt, 0.0).sum() / 60.0),
        "soc_pct": float(soc_pct),
        "missing": bool(missing),
        "price_source": source,
        "buy": lr.buy_eur_mwh,
        "sell": lr.sell_eur_mwh,
        "state": lr.state,
        "money_eur": money,
        "tax_eur": tax,
    }


def resettle(booking: dict, buy: float, sell: float, state: int, terms: Terms) -> dict:
    """A settled rebooking: same energies, ``price_source = "settled"``, money at the new price.

    Returns the booking's ``BOOKING_FIELDS`` only (no chain columns); money and tax are NaN
    when the booking is ``missing``.
    """
    out = {f: booking[f] for f in BOOKING_FIELDS}
    if bool(booking["missing"]):
        total = tax = float("nan")
    else:
        qts = int(booking["quarter_ts"])
        total, tax = money_eur(
            booking["import_kwh"], booking["export_kwh"], buy, sell, state, terms,
            tax_mode_at(terms, qts),
        )
    out.update(
        price_source="settled", buy=float(buy), sell=float(sell), state=int(state),
        money_eur=total, tax_eur=tax,
    )
    return out


def quarter_result(house: dict, shadow: dict) -> float:
    """The shadow's result of one quarter: ``money(house) - money(shadow)``, positive earned.

    NaN when either booking is missing or has NaN money.
    """
    if bool(house["missing"]) or bool(shadow["missing"]):
        return float("nan")
    h, s = float(house["money_eur"]), float(shadow["money_eur"])
    if math.isnan(h) or math.isnan(s):
        return float("nan")
    return h - s
