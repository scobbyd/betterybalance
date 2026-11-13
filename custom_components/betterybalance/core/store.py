"""The store: one SQLite file with the inputs the box saw, the chained ledger and the runs.

Every write runs in one transaction (``with self._db:``). NaN is stored as NULL and read
back as NaN. Arrays are stored as ``blob``: zlib over little-endian float64 bytes. Ranges
``(lo, hi)`` are half-open ``[lo, hi)`` except ``chain(lo_seq, hi_seq)``, which is inclusive.
"""

from __future__ import annotations

import json
import math
import sqlite3
import struct
import zlib
from pathlib import Path

import numpy as np

from .ledger import BOOKING_FIELDS, DECISION_FIELDS, FIELD_TYPES, chain_row, genesis_hash
from .pricemodel import QUANTILES
from .site.flows import COLUMNS, STRING_COLUMNS
from .timegrid import Table

SCHEMA_VERSION = 1
RETENTION_DAYS = 400
# 400-day retention for the replay inputs; decisions and bookings share the hash chain, so both
# are kept forever (a pruned decision would turn every older verification into a seq gap)
PRUNED = (
    "flows_minute", "live_minutes", "features", "bands", "windows", "nowcasts",
    "forecast_fetches",
)
FEATURE_COLUMNS = (
    "da_eur_mwh", "ida1_eur_mwh", "ida2_eur_mwh", "ida3_eur_mwh", "buy_live_end",
    "sell_live_end", "buy_live_e1", "sell_live_e1", "buy_live_e5", "sell_live_e5",
    "buy_live_e10", "sell_live_e10",
)
NOWCAST_FIELDS = ("take_point", "feed_point", "w_take", "w_feed", "p_dual", "take_q50", "feed_q50")
# the plan-2 greed columns (replay.phases.GREED_COLUMNS; core does not import replay)
GREED_COLUMNS = (
    "day", "soc_kwh", "charge_threshold", "sale_threshold", "night_median_take",
    "night_median_feed", "surplus_tomorrow_kwh", "low_soc_applies", "low_soc_ok",
    "high_soc_applies", "high_soc_ok",
)

_FLOW_FLOATS = tuple(c for c in COLUMNS if c not in ("ts", "step_s")) + STRING_COLUMNS
_BAND_QS = tuple(f"{side}_q{round(q * 100)}" for side in ("take", "feed") for q in QUANTILES)
_BAND_KEY = ("target_ts", "tier", "h", "e", "model_version")
_BAND_REALS = (*_BAND_QS, "p_dual", "cov")
_WINDOW_ARRAYS = ("load_kw", "pv_kw", "ev_kw")
_WINDOW_MASKS = ("da_known", "plan_mask")
_WINDOW_COLUMNS = (
    "ts", "epoch", "now", "model_version", "pack_version", "healthy", "da_known", "soc_kwh",
    "widening", "irr_issue_ts", "load_kw", "pv_kw", "ev_kw", "plan_mask", "flags",
)
_NOWCAST_COLUMNS = ("ts", "epoch", "quarter_ts", *NOWCAST_FIELDS, "live_age_s")
_LIVE_COLUMNS = ("ts", "buy_live", "sell_live", "state", "source")
_SETTLEMENT_COLUMNS = ("quarter_ts", "take", "feed", "regulation_state", "published_at", "source")
_RUN_COLUMNS = (
    "run_id", "started", "finished", "status", "span_from", "span_to", "headline", "report_path",
)
_RUN_QUARTER_COLUMNS = ("quarter_ts", "path", "import_kwh", "export_kwh", "money_eur", "missing")
_KIND_TABLE = {"decision": ("decisions", DECISION_FIELDS), "booking": ("bookings", BOOKING_FIELDS)}
_SQL_TYPE = {float: "REAL", int: "INTEGER", bool: "INTEGER", str: "TEXT"}


def _cols(fields, types=FIELD_TYPES) -> str:
    return ", ".join(f"{f} {_SQL_TYPE[types[f]]}" for f in fields)


def _real(names) -> str:
    return ", ".join(f"{c} REAL" for c in names)


_GREED_SQL = ", ".join(
    f"{c} INTEGER" if c.endswith(("_applies", "_ok")) else f"{c} REAL" for c in GREED_COLUMNS[1:]
)
_SCHEMA = (
    f"""CREATE TABLE IF NOT EXISTS flows_minute (ts INTEGER PRIMARY KEY, {_real(_FLOW_FLOATS)},
        n_samples INTEGER, epoch INTEGER NOT NULL DEFAULT 0)""",
    f"CREATE TABLE IF NOT EXISTS flows_hour (ts INTEGER PRIMARY KEY, {_real(_FLOW_FLOATS)})",
    f"CREATE TABLE IF NOT EXISTS features (ts INTEGER PRIMARY KEY, {_real(FEATURE_COLUMNS)})",
    """CREATE TABLE IF NOT EXISTS live_minutes (ts INTEGER PRIMARY KEY, buy_live REAL,
        sell_live REAL, state INTEGER, source TEXT)""",
    f"""CREATE TABLE IF NOT EXISTS bands (target_ts INTEGER, tier TEXT, h INTEGER, e INTEGER,
        model_version TEXT, recorded_at INTEGER, epoch INTEGER NOT NULL DEFAULT 0,
        {_real(_BAND_QS)}, p_dual REAL, cov REAL,
        PRIMARY KEY (target_ts, tier, h, e, model_version, recorded_at, epoch))""",
    """CREATE TABLE IF NOT EXISTS windows (ts INTEGER, epoch INTEGER NOT NULL DEFAULT 0,
        now INTEGER, model_version TEXT, pack_version TEXT, healthy TEXT, da_known BLOB,
        soc_kwh REAL, widening REAL, irr_issue_ts INTEGER, load_kw BLOB, pv_kw BLOB,
        ev_kw BLOB, plan_mask BLOB, flags TEXT, PRIMARY KEY (ts, epoch))""",
    f"""CREATE TABLE IF NOT EXISTS nowcasts (ts INTEGER, epoch INTEGER NOT NULL DEFAULT 0,
        quarter_ts INTEGER, {_real(NOWCAST_FIELDS)}, live_age_s INTEGER,
        PRIMARY KEY (ts, epoch))""",
    """CREATE TABLE IF NOT EXISTS forecast_fetches (issue_ts INTEGER, source TEXT,
        fetched_at INTEGER, payload BLOB, PRIMARY KEY (issue_ts, source))""",
    f"""CREATE TABLE IF NOT EXISTS decisions (seq INTEGER PRIMARY KEY, {_cols(DECISION_FIELDS)},
        prev_hash TEXT, hash TEXT)""",
    "CREATE INDEX IF NOT EXISTS decisions_ts ON decisions (ts)",
    f"""CREATE TABLE IF NOT EXISTS bookings (seq INTEGER PRIMARY KEY, {_cols(BOOKING_FIELDS)},
        prev_hash TEXT, hash TEXT)""",
    "CREATE INDEX IF NOT EXISTS bookings_qpath ON bookings (quarter_ts, path, seq)",
    """CREATE TABLE IF NOT EXISTS settlements (quarter_ts INTEGER, take REAL, feed REAL,
        regulation_state INTEGER, published_at INTEGER, source TEXT,
        PRIMARY KEY (quarter_ts, published_at))""",
    "CREATE TABLE IF NOT EXISTS registry (key TEXT PRIMARY KEY, value TEXT, updated INTEGER)",
    """CREATE TABLE IF NOT EXISTS pack (version TEXT PRIMARY KEY, installed_at INTEGER,
        manifest TEXT, path TEXT, active INTEGER)""",
    """CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER,
        kind TEXT, detail TEXT)""",
    f"CREATE TABLE IF NOT EXISTS greed (day TEXT PRIMARY KEY, {_GREED_SQL})",
    """CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        started INTEGER, finished INTEGER, status TEXT, span_from INTEGER, span_to INTEGER,
        headline TEXT, report_path TEXT)""",
    """CREATE TABLE IF NOT EXISTS run_quarters (run_id INTEGER, quarter_ts INTEGER, path TEXT,
        import_kwh REAL, export_kwh REAL, money_eur REAL, missing INTEGER,
        PRIMARY KEY (run_id, quarter_ts, path))""",
    f"""CREATE TABLE IF NOT EXISTS run_decisions (run_id INTEGER, {_cols(DECISION_FIELDS)},
        PRIMARY KEY (run_id, ts))""",
)
_PRUNE_COLUMN = {"bands": "target_ts", "forecast_fetches": "issue_ts"}


def blob(a: np.ndarray) -> bytes:
    """``zlib.compress`` over the little-endian float64 bytes of ``a`` (NaN kept)."""
    return zlib.compress(np.ascontiguousarray(a, dtype="<f8").tobytes())


def unblob(b: bytes) -> np.ndarray:
    """The float64 array of ``blob``, bit-exact."""
    return np.frombuffer(zlib.decompress(b), dtype="<f8").astype(np.float64)


def _table_blob(table: Table) -> bytes:
    """A Table as zlib over a JSON header (name, dtype, length per column) and the raw bytes."""
    head, parts = [], []
    for name in table:
        a = np.asarray(table[name])
        if a.dtype.kind not in "biuf" or a.ndim != 1:
            raise ValueError(f"column {name!r}: only 1-d numeric or bool arrays are stored")
        a = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<"))
        head.append([name, a.dtype.str, len(a)])
        parts.append(a.tobytes())
    h = json.dumps(head, separators=(",", ":")).encode("ascii")
    return zlib.compress(struct.pack("<I", len(h)) + h + b"".join(parts))


def _table_unblob(b: bytes) -> Table:
    raw = zlib.decompress(b)
    (n,) = struct.unpack("<I", raw[:4])
    pos, out = 4 + n, {}
    for name, dtype, length in json.loads(raw[4 : 4 + n]):
        dt = np.dtype(dtype)
        a = np.frombuffer(raw, dtype=dt, count=length, offset=pos)
        out[name] = a.astype(dt.newbyteorder("="))
        pos += dt.itemsize * length
    return out


def _sql(v):
    """A value as SQLite takes it: numpy scalars unwrapped, non-finite floats as NULL."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, bool):
        return int(v)
    return v


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def _same(stored: float, new) -> bool:
    """A stored band value equals a new one (NaN and a missing value equal NaN)."""
    a, b = _nan(stored), _nan(new)
    return a == b or (a != a and b != b)


def _typed(field: str, v):
    """A ledger value read back as its ``FIELD_TYPES`` type; NULL floats as NaN."""
    typ = FIELD_TYPES[field]
    if typ is float:
        return _nan(v)
    if v is None:
        return None
    return typ(v)


def _check_kinds(items) -> None:
    for kind, _ in items:
        if kind not in _KIND_TABLE:
            raise ValueError(f"kind must be one of {tuple(_KIND_TABLE)}, got {kind!r}")


def _check(keys, allowed, what: str) -> None:
    bad = sorted(set(keys) - set(allowed))
    if bad:
        raise ValueError(f"unknown {what} column(s): {', '.join(bad)}")


class Store:
    """The integration's SQLite store. One executor job at a time calls it."""

    def __init__(self, path: str | Path, timeout: float = 5.0):
        # ``timeout``: seconds a write waits for another connection's write lock (the
        # backtest opens its own connection with a longer one, coordinator.BT_BUSY_S)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=timeout)
        if str(path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")

    def close(self) -> None:
        self._db.close()

    def migrate(self) -> None:
        """Create the schema; refuse a file written by a newer schema."""
        with self._db:
            self._db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            found = self._meta("schema_version")
            if found is not None and int(found) > SCHEMA_VERSION:
                raise ValueError(
                    f"store schema {int(found)} is newer than this integration ({SCHEMA_VERSION})"
                )
            for stmt in _SCHEMA:
                self._db.execute(stmt)
            if found is None:
                self._set_meta("schema_version", str(SCHEMA_VERSION))

    # --- helpers ---------------------------------------------------------------------------

    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def _set_meta(self, key: str, value) -> None:
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def _insert(self, table: str, row: dict, columns, verb: str = "INSERT OR REPLACE") -> int:
        _check(row, columns, table)
        keys = [c for c in columns if c in row]
        cur = self._db.execute(
            f"{verb} INTO {table} ({', '.join(keys)}) VALUES ({', '.join('?' * len(keys))})",
            [_sql(row[c]) for c in keys],
        )
        return cur.rowcount

    def _rows(self, sql: str, params=()) -> list[dict]:
        cur = self._db.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]

    def _table(self, sql: str, params, floats, ints=("ts",), objects=()) -> Table:
        rows = self._db.execute(sql, params).fetchall()
        names = (*ints, *floats, *objects)
        cols = list(zip(*rows)) if rows else [()] * len(names)
        out: Table = {}
        for name, vals in zip(names, cols):
            if name in ints:
                out[name] = np.array([0 if v is None else v for v in vals], dtype=np.int64)
            elif name in objects:
                out[name] = np.array(vals, dtype=object)
            else:
                out[name] = np.array(vals, dtype=float).reshape(-1)
        return out

    # --- flows -----------------------------------------------------------------------------

    def put_minute(self, row: dict) -> None:
        """One minute of flows; a minute already stored is kept (insert-or-ignore).

        ``epoch`` is the run that wrote it (default 0: written outside a live run). A stored
        minute row never changes (no update path exists; ``prune`` only deletes rows older
        than the retention): ``LiveSite._minute_quarters`` caches aggregated minutes and
        detects a later write by the row count alone, so keep it that way.
        """
        row = {k: v for k, v in row.items() if k != "step_s"}
        with self._db:
            self._insert(
                "flows_minute", row, ("ts", *_FLOW_FLOATS, "n_samples", "epoch"),
                "INSERT OR IGNORE",
            )

    def minutes(self, lo: int, hi: int, max_epoch: int | None = None) -> Table:
        """Minute flows in ``[lo, hi)``: every flows column, the string columns, ``step_s`` 60.

        ``n_samples`` comes along as int64 (0 where none was stored). With ``max_epoch`` only
        the rows written by runs with ``epoch <= max_epoch``.
        """
        sql, args = "WHERE ts >= ? AND ts < ?", (int(lo), int(hi))
        if max_epoch is not None:
            sql, args = sql + " AND epoch <= ?", (*args, int(max_epoch))
        t = self._table(
            f"SELECT ts, n_samples, {', '.join(_FLOW_FLOATS)} FROM flows_minute "
            f"{sql} ORDER BY ts",
            args, _FLOW_FLOATS, ints=("ts", "n_samples"),
        )
        t["step_s"] = np.full(len(t["ts"]), 60, dtype=np.int64)
        return t

    def minute_count(self, lo: int, hi: int) -> int:
        """The number of stored minute rows in ``[lo, hi)`` (every epoch)."""
        row = self._db.execute(
            "SELECT COUNT(*) FROM flows_minute WHERE ts >= ? AND ts < ?", (int(lo), int(hi))
        ).fetchone()
        return int(row[0])

    def put_hours(self, table: Table) -> None:
        """Upsert hourly flows; a column the table lacks keeps its stored value."""
        _check(table, ("ts", "step_s", *_FLOW_FLOATS), "flows_hour")
        keys = [c for c in _FLOW_FLOATS if c in table]
        cols = ["ts", *keys]
        update = ", ".join(f"{c} = excluded.{c}" for c in keys) or "ts = excluded.ts"
        sql = (
            f"INSERT INTO flows_hour ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
            f"ON CONFLICT(ts) DO UPDATE SET {update}"
        )
        arrays = [np.asarray(table["ts"], dtype=np.int64)] + [
            np.asarray(table[c], dtype=float) for c in keys
        ]
        with self._db:
            self._db.executemany(sql, ([_sql(a[i]) for a in arrays] for i in range(len(arrays[0]))))

    def hours(self, lo: int, hi: int) -> Table:
        """Hourly flows in ``[lo, hi)`` with ``step_s`` 3600."""
        t = self._table(
            f"SELECT ts, {', '.join(_FLOW_FLOATS)} FROM flows_hour "
            "WHERE ts >= ? AND ts < ? ORDER BY ts",
            (int(lo), int(hi)), _FLOW_FLOATS,
        )
        t["step_s"] = np.full(len(t["ts"]), 3600, dtype=np.int64)
        return t

    # --- features and live minutes ---------------------------------------------------------

    def upsert_features(self, ts: int, **cols) -> None:
        """Merge the non-NaN values of ``cols`` into the feature row of ``ts``."""
        _check(cols, FEATURE_COLUMNS, "features")
        keys = [c for c in FEATURE_COLUMNS if c in cols and _sql(float(cols[c])) is not None]
        with self._db:
            if not keys:
                self._db.execute("INSERT OR IGNORE INTO features (ts) VALUES (?)", (int(ts),))
                return
            self._db.execute(
                f"INSERT INTO features (ts, {', '.join(keys)}) "
                f"VALUES ({', '.join('?' * (len(keys) + 1))}) ON CONFLICT(ts) DO UPDATE SET "
                + ", ".join(f"{c} = excluded.{c}" for c in keys),
                [int(ts), *(float(cols[c]) for c in keys)],
            )

    def features(self, lo: int, hi: int) -> Table:
        return self._table(
            f"SELECT ts, {', '.join(FEATURE_COLUMNS)} FROM features "
            "WHERE ts >= ? AND ts < ? ORDER BY ts",
            (int(lo), int(hi)), FEATURE_COLUMNS,
        )

    def put_live_minutes(self, rows: list[dict]) -> None:
        """Live price minutes; a minute stored again is replaced."""
        with self._db:
            for r in rows:
                self._insert("live_minutes", r, _LIVE_COLUMNS)

    def live_minutes(self, lo: int, hi: int) -> Table:
        """Live minutes in ``[lo, hi)``; ``state`` is float with NaN for none, ``source`` object."""
        return self._table(
            "SELECT ts, buy_live, sell_live, state, source FROM live_minutes "
            "WHERE ts >= ? AND ts < ? ORDER BY ts",
            (int(lo), int(hi)), ("buy_live", "sell_live", "state"), objects=("source",),
        )

    # --- bands, windows, nowcasts, fetches -------------------------------------------------

    def put_band(self, row: dict, recorded_at: int, epoch: int = 0) -> bool:
        """One band cell as served at ``recorded_at`` (epoch s) in run ``epoch`` (the seq of
        the ``restart`` event that started the run); ``e`` None is -1.

        Bands are as-of: the row is inserted only when its values differ from the band of
        the same cell as run ``epoch`` sees it at ``recorded_at`` (``band``; an identical
        re-serve adds nothing). A row with every value NaN records that the cell has no band
        from ``recorded_at`` on; it is inserted only after a band of the cell (a cell never
        served needs no row). True when a row was inserted. Raises ``ValueError`` when a
        different band arrives for a cell already recorded at the same ``recorded_at`` in
        the same run: one serve time of one run has one band, so re-execution sees exactly
        what the live box saw. A re-run after a restart records its own band at a serve time
        the earlier run also served.
        """
        row = dict(
            row,
            e=-1 if row.get("e") is None else row["e"],
            recorded_at=int(recorded_at),
            epoch=int(epoch),
        )
        _check(row, (*_BAND_KEY, "recorded_at", "epoch", *_BAND_REALS), "bands")
        with self._db:
            prev = self.band(
                *(row.get(k) for k in _BAND_KEY), as_of=int(recorded_at), epoch=int(epoch)
            )
            if prev is not None and all(_same(prev[k], row.get(k)) for k in _BAND_REALS):
                return False
            if prev is None and all(math.isnan(_nan(row.get(k))) for k in _BAND_REALS):
                return False
            same = prev is not None and int(prev["recorded_at"]) == int(recorded_at)
            if same and int(prev["epoch"]) == int(epoch):
                key = ", ".join(f"{k}={row.get(k)!r}" for k in _BAND_KEY)
                raise ValueError(
                    f"band ({key}) already recorded_at {int(recorded_at)} in epoch "
                    f"{int(epoch)} with other values"
                )
            cols = (*_BAND_KEY, "recorded_at", "epoch", *_BAND_REALS)
            return self._insert("bands", row, cols, "INSERT") > 0

    def band(
        self,
        target_ts: int,
        tier: str,
        h: int,
        e: int | None,
        model_version: str,
        as_of: int | None = None,
        epoch: int | None = None,
    ) -> dict | None:
        """The band of the cell as run ``epoch`` saw it at ``as_of``.

        The newest row by ``(recorded_at, epoch)`` among the rows with ``recorded_at <= as_of``
        and ``epoch <= epoch``: the rows that existed when that run served at ``as_of`` (a
        later run's rows never, whatever their serve time). None drops that bound.
        """
        key = (int(target_ts), tier, int(h), -1 if e is None else int(e), model_version)
        sql = (
            "SELECT * FROM bands WHERE target_ts = ? AND tier = ? AND h = ? AND e = ? "
            "AND model_version = ?"
        )
        if as_of is not None:
            sql, key = sql + " AND recorded_at <= ?", (*key, int(as_of))
        if epoch is not None:
            sql, key = sql + " AND epoch <= ?", (*key, int(epoch))
        rows = self._rows(sql + " ORDER BY recorded_at DESC, epoch DESC LIMIT 1", key)
        if not rows:
            return None
        return {k: (_nan(v) if k in _BAND_REALS else v) for k, v in rows[0].items()}

    def put_window(self, row: dict) -> None:
        """The window row of a quarter (``ts``) in run ``epoch`` (default 0); a quarter set up
        again in the same run replaces it.

        ``da_known`` and ``plan_mask`` are bool masks, ``load_kw``, ``pv_kw`` and ``ev_kw``
        float arrays, ``healthy`` and ``flags`` sequences of strings (stored as JSON).
        """
        _check(row, _WINDOW_COLUMNS, "windows")
        out = dict(row)
        for k in (*_WINDOW_ARRAYS, *_WINDOW_MASKS):
            if out.get(k) is not None:
                out[k] = blob(np.asarray(out[k], dtype=float))
        for k in ("healthy", "flags"):
            if out.get(k) is not None:
                out[k] = json.dumps(list(out[k]))
        with self._db:
            self._insert("windows", out, _WINDOW_COLUMNS)

    def window(self, qts: int, epoch: int | None = None) -> dict | None:
        """The window row of quarter ``qts`` of the newest run with ``epoch <= epoch`` (of all
        runs when None), blobs decoded (masks as bool)."""
        sql, args = "SELECT * FROM windows WHERE ts = ?", (int(qts),)
        if epoch is not None:
            sql, args = sql + " AND epoch <= ?", (*args, int(epoch))
        rows = self._rows(sql + " ORDER BY epoch DESC LIMIT 1", args)
        if not rows:
            return None
        w = rows[0]
        for k in _WINDOW_ARRAYS:
            w[k] = None if w[k] is None else unblob(w[k])
        for k in _WINDOW_MASKS:
            w[k] = None if w[k] is None else unblob(w[k]) != 0.0
        if w["healthy"] is not None:
            w["healthy"] = tuple(json.loads(w["healthy"]))
        if w["flags"] is not None:
            w["flags"] = list(json.loads(w["flags"]))
        for k in ("soc_kwh", "widening"):
            w[k] = _nan(w[k])
        return w

    def put_nowcast(self, row: dict) -> None:
        """The nowcast of one minute (``ts``) in run ``epoch`` (default 0); a minute stored
        again in the same run is replaced."""
        with self._db:
            self._insert("nowcasts", row, _NOWCAST_COLUMNS)

    def nowcasts(self, lo: int, hi: int, epoch: int | None = None) -> dict:
        """Nowcasts in ``[lo, hi)``: ``quarter_ts``, ``epoch``, the fields, ``live_age_s``.

        Keyed by minute ``ts`` for one run ``epoch``, by ``(ts, epoch)`` for all runs (None).
        """
        sql, args = "SELECT * FROM nowcasts WHERE ts >= ? AND ts < ?", (int(lo), int(hi))
        if epoch is not None:
            sql, args = sql + " AND epoch = ?", (*args, int(epoch))
        out = {}
        for r in self._rows(sql + " ORDER BY ts, epoch", args):
            ts = r.pop("ts")
            row = {k: (_nan(v) if k in NOWCAST_FIELDS else v) for k, v in r.items()}
            out[ts if epoch is not None else (ts, r["epoch"])] = row
        return out

    def last_nowcast_ts(self, epoch: int) -> int | None:
        """The newest minute with a nowcast row in run ``epoch`` (its last decided minute);
        None without one."""
        row = self._db.execute(
            "SELECT MAX(ts) FROM nowcasts WHERE epoch = ?", (int(epoch),)
        ).fetchone()
        return None if row[0] is None else int(row[0])

    def put_fetch(self, issue_ts: int, source: str, fetched_at: int, table: Table) -> None:
        """A fetched forecast table (numeric columns, dtypes kept); a re-fetch replaces it."""
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO forecast_fetches (issue_ts, source, fetched_at, payload) "
                "VALUES (?, ?, ?, ?)",
                (int(issue_ts), source, int(fetched_at), _table_blob(table)),
            )

    def fetch_at(self, now: int, source: str) -> tuple[int, Table] | None:
        """The newest fetch of ``source`` with ``fetched_at <= now`` as ``(issue_ts, table)``."""
        row = self._db.execute(
            "SELECT issue_ts, payload FROM forecast_fetches WHERE source = ? AND fetched_at <= ? "
            "ORDER BY fetched_at DESC, issue_ts DESC LIMIT 1",
            (source, int(now)),
        ).fetchone()
        return None if row is None else (int(row[0]), _table_unblob(row[1]))

    # --- the chain -------------------------------------------------------------------------

    def append_chained(self, kind: str, row: dict) -> dict:
        """Chain ``row`` after the head with the next seq, store it and return the chained row.

        Decisions and bookings share one seq counter (``meta.chain_seq``, head hash in
        ``meta.chain_head``). The first row chains on ``genesis_hash(meta.install_id)``.
        """
        return self.append_chained_many([(kind, row)])[0]

    def append_chained_many(self, items: list[tuple[str, dict]]) -> list[dict]:
        """Chain ``(kind, row)`` items in order in one transaction and return the chained rows.

        All rows are stored or none: a failure on any row rolls back the rows before it and
        leaves the head unchanged. Kinds and the genesis rule as ``append_chained``.
        """
        _check_kinds(items)
        if not items:
            return []
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            return self._chain(items)

    def append_chained_with_event(
        self, items: list[tuple[str, dict]], ts: int, kind: str, detail: dict
    ) -> tuple[list[dict], int]:
        """Chain ``items`` (as ``append_chained_many``, none allowed) and append the event
        ``(ts, kind, detail)`` after them in one transaction: the chained rows and the seq of
        the event. A failure on any row or on the event stores none of them."""
        _check_kinds(items)
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            out = self._chain(items) if items else []
            return out, self._insert_event(ts, kind, detail)

    def _chain(self, items: list[tuple[str, dict]]) -> list[dict]:
        """Chain ``items`` after the head inside the caller's transaction."""
        head = self._head()
        if head is None:
            install_id = self._meta("install_id")
            if install_id is None:
                raise ValueError("meta install_id is not set: the chain has no genesis anchor")
            seq, prev = 0, genesis_hash(install_id)
        else:
            seq, prev = head
        out = []
        for kind, row in items:
            table, fields = _KIND_TABLE[kind]
            seq += 1
            chained = chain_row(kind, seq, {f: row[f] for f in fields}, prev)
            self._insert(table, chained, ("seq", *fields, "prev_hash", "hash"), "INSERT")
            out.append(chained)
            prev = chained["hash"]
        self._set_meta("chain_seq", seq)
        self._set_meta("chain_head", prev)
        return out

    def _head(self) -> tuple[int, str] | None:
        seq, head = self._meta("chain_seq"), self._meta("chain_head")
        return None if seq is None or head is None else (int(seq), head)

    def head(self) -> tuple[int, str] | None:
        """``(seq, hash)`` of the newest chained row; None before the first."""
        return self._head()

    def _ledger_rows(self, kind: str, where: str, params) -> list[dict]:
        table, fields = _KIND_TABLE[kind]
        cols = ("seq", *fields, "prev_hash", "hash")
        sql = f"SELECT {', '.join(cols)} FROM {table} WHERE {where} ORDER BY seq"
        rows = self._rows(sql, params)
        return [{k: _typed(k, v) for k, v in r.items()} for r in rows]

    def chain(self, lo_seq: int, hi_seq: int) -> list[tuple[str, dict]]:
        """``(kind, row)`` for seq in ``[lo_seq, hi_seq]`` (inclusive), in seq order."""
        params = (int(lo_seq), int(hi_seq))
        items = [
            (kind, r)
            for kind in _KIND_TABLE
            for r in self._ledger_rows(kind, "seq >= ? AND seq <= ?", params)
        ]
        return sorted(items, key=lambda kr: kr[1]["seq"])

    def chain_span(self, lo_ts: int, hi_ts: int) -> tuple[int, int] | None:
        """``(lowest, highest)`` seq of the decisions with ``ts`` and the bookings with
        ``quarter_ts`` in ``[lo_ts, hi_ts)``; None without one."""
        params = (int(lo_ts), int(hi_ts))
        lo = hi = None
        for table, col in (("decisions", "ts"), ("bookings", "quarter_ts")):
            row = self._db.execute(
                f"SELECT MIN(seq), MAX(seq) FROM {table} WHERE {col} >= ? AND {col} < ?", params
            ).fetchone()
            if row[0] is not None:
                lo = row[0] if lo is None else min(lo, row[0])
                hi = row[1] if hi is None else max(hi, row[1])
        return None if lo is None else (int(lo), int(hi))

    def decisions(self, lo_ts: int, hi_ts: int) -> list[dict]:
        """Decision rows with ``ts`` in ``[lo_ts, hi_ts)``, in seq order."""
        return self._ledger_rows("decision", "ts >= ? AND ts < ?", (int(lo_ts), int(hi_ts)))

    def bookings(self, lo_ts: int, hi_ts: int) -> list[dict]:
        """Booking rows with ``quarter_ts`` in ``[lo_ts, hi_ts)``, in seq order."""
        where = "quarter_ts >= ? AND quarter_ts < ?"
        return self._ledger_rows("booking", where, (int(lo_ts), int(hi_ts)))

    def newest_bookings(self, lo_ts: int, hi_ts: int) -> dict[tuple[int, str], dict]:
        """The booking with the highest seq per ``(quarter_ts, path)`` in ``[lo_ts, hi_ts)``."""
        rows = self._ledger_rows(
            "booking",
            "seq IN (SELECT MAX(seq) FROM bookings WHERE quarter_ts >= ? AND quarter_ts < ? "
            "GROUP BY quarter_ts, path)",
            (int(lo_ts), int(hi_ts)),
        )
        return {(r["quarter_ts"], r["path"]): r for r in rows}

    # --- settlements -----------------------------------------------------------------------

    def put_settlement(self, row: dict) -> bool:
        """Append a published settlement; False when ``(quarter_ts, published_at)`` exists."""
        with self._db:
            return self._insert("settlements", row, _SETTLEMENT_COLUMNS, "INSERT OR IGNORE") == 1

    def settlements(self, lo: int, hi: int, newest: bool = True) -> list[dict]:
        """Settlements of quarters in ``[lo, hi)``; with ``newest`` the latest publication only."""
        where = "quarter_ts >= ? AND quarter_ts < ?"
        if newest:
            where += (
                " AND published_at = (SELECT MAX(published_at) FROM settlements s2 "
                "WHERE s2.quarter_ts = settlements.quarter_ts)"
            )
        rows = self._rows(
            f"SELECT * FROM settlements WHERE {where} ORDER BY quarter_ts, published_at",
            (int(lo), int(hi)),
        )
        return [{k: (_nan(v) if k in ("take", "feed") else v) for k, v in r.items()} for r in rows]

    # --- registry, meta, events, greed -----------------------------------------------------

    def set_registry(self, key: str, value: str, now: int) -> None:
        with self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO registry (key, value, updated) VALUES (?, ?, ?)",
                (key, value, int(now)),
            )

    def registry(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM registry WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def put_pack(self, version: str, installed_at: int, manifest: dict, path: str) -> None:
        """Record an installed data pack as the active one (every other row inactive)."""
        with self._db:
            self._db.execute("UPDATE pack SET active = 0")
            self._db.execute(
                "INSERT OR REPLACE INTO pack (version, installed_at, manifest, path, active) "
                "VALUES (?, ?, ?, ?, 1)",
                (version, int(installed_at), json.dumps(manifest, sort_keys=True), str(path)),
            )

    def active_pack(self) -> dict | None:
        """The active installed pack, ``manifest`` decoded; None without one."""
        rows = self._rows("SELECT * FROM pack WHERE active = 1 ORDER BY version DESC LIMIT 1")
        return dict(rows[0], manifest=json.loads(rows[0]["manifest"])) if rows else None

    def set_meta(self, key: str, value) -> None:
        with self._db:
            self._set_meta(key, value)

    def meta(self, key: str) -> str | None:
        return self._meta(key)

    def put_event(self, ts: int, kind: str, detail: dict) -> int:
        """Append an event; returns its seq."""
        with self._db:
            return self._insert_event(ts, kind, detail)

    def _insert_event(self, ts: int, kind: str, detail: dict) -> int:
        cur = self._db.execute(
            "INSERT INTO events (ts, kind, detail) VALUES (?, ?, ?)",
            (int(ts), kind, json.dumps(detail, sort_keys=True)),
        )
        return int(cur.lastrowid)

    def last_event(self, kind: str) -> dict | None:
        """The newest event of ``kind``, ``detail`` decoded; None without one."""
        rows = self._rows("SELECT * FROM events WHERE kind = ? ORDER BY seq DESC LIMIT 1", (kind,))
        return dict(rows[0], detail=json.loads(rows[0]["detail"])) if rows else None

    def events(self, lo_seq: int = 0) -> list[dict]:
        """Events with ``seq >= lo_seq`` in seq order, ``detail`` decoded."""
        rows = self._rows("SELECT * FROM events WHERE seq >= ? ORDER BY seq", (int(lo_seq),))
        return [dict(r, detail=json.loads(r["detail"])) for r in rows]

    def recent_events(self, n: int) -> list[dict]:
        """The newest ``n`` events in seq order, ``detail`` decoded."""
        sql = "SELECT * FROM (SELECT * FROM events ORDER BY seq DESC LIMIT ?) ORDER BY seq"
        rows = self._rows(sql, (max(0, int(n)),))
        return [dict(r, detail=json.loads(r["detail"])) for r in rows]

    def put_greed(self, row: dict) -> None:
        """The greed row of a local day; a day stored again is replaced."""
        with self._db:
            self._insert("greed", row, GREED_COLUMNS)

    # --- backtest runs ---------------------------------------------------------------------

    def new_run(self, started: int, span_from: int, span_to: int) -> int:
        """A new run in status ``running``; returns its ``run_id``."""
        with self._db:
            cur = self._db.execute(
                "INSERT INTO runs (started, status, span_from, span_to) "
                "VALUES (?, 'running', ?, ?)",
                (int(started), int(span_from), int(span_to)),
            )
        return int(cur.lastrowid)

    def finish_run(
        self, run_id: int, finished: int, status: str, headline: dict, report_path
    ) -> None:
        with self._db:
            self._db.execute(
                "UPDATE runs SET finished = ?, status = ?, headline = ?, report_path = ? "
                "WHERE run_id = ?",
                (
                    int(finished), status, json.dumps(headline, sort_keys=True),
                    None if report_path is None else str(report_path), int(run_id),
                ),
            )

    def put_run_quarters(self, run_id: int, rows: list[dict]) -> None:
        with self._db:
            for r in rows:
                self._insert(
                    "run_quarters", {"run_id": run_id, **{c: r[c] for c in _RUN_QUARTER_COLUMNS}},
                    ("run_id", *_RUN_QUARTER_COLUMNS),
                )

    def put_run_decisions(self, run_id: int, rows: list[dict]) -> None:
        """Decision rows of a run: ``DECISION_FIELDS`` only, other keys (chain columns) ignored."""
        with self._db:
            for r in rows:
                self._insert(
                    "run_decisions", {"run_id": run_id, **{f: r[f] for f in DECISION_FIELDS}},
                    ("run_id", *DECISION_FIELDS),
                )

    def delete_run_rows(self, run_id: int) -> None:
        """Delete the ``run_quarters`` and ``run_decisions`` rows of one run (a failed write)."""
        with self._db:
            self._db.execute("DELETE FROM run_quarters WHERE run_id = ?", (int(run_id),))
            self._db.execute("DELETE FROM run_decisions WHERE run_id = ?", (int(run_id),))

    def runs(self) -> list[dict]:
        """Every run in ``run_id`` order, ``headline`` decoded (None while running)."""
        rows = self._rows(f"SELECT {', '.join(_RUN_COLUMNS)} FROM runs ORDER BY run_id")
        return [
            dict(r, headline=None if r["headline"] is None else json.loads(r["headline"]))
            for r in rows
        ]

    # --- retention -------------------------------------------------------------------------

    def prune(self, now: int) -> dict[str, int]:
        """Delete rows older than ``now - RETENTION_DAYS`` days in ``PRUNED``; counts per table."""
        cutoff = int(now) - RETENTION_DAYS * 86400
        out = {}
        with self._db:
            for table in PRUNED:
                col = _PRUNE_COLUMN.get(table, "ts")
                cur = self._db.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
                out[table] = cur.rowcount
        return out

    def vacuum(self) -> None:
        """``VACUUM``, then a truncating checkpoint, so the WAL does not stay the size of the
        database."""
        self._db.execute("VACUUM")
        self.checkpoint()

    def checkpoint(self) -> tuple[int, int, int]:
        """``PRAGMA wal_checkpoint(TRUNCATE)``: every WAL page into the database file and the
        WAL cut to zero bytes; SQLite's ``(busy, log pages, checkpointed pages)``."""
        busy, log, done = self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return int(busy), int(log), int(done)
