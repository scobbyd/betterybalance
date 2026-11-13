"""Settled ISP prices from TenneT's public settlement-prices report (Oracle APEX)."""

from __future__ import annotations

import datetime as dt
import html as html_lib
import http.cookiejar
import json
import re
import time
import urllib.parse

import numpy as np

from ..timegrid import ISP_S, ZONE, Table, align, isp_grid
from .base import DayCache, FeedError, http_get, http_post, local_days

BASE = "https://publications.tennet.eu/ords"
APP = "100"
PAGE = "90035"
CONTEXT = "aetherweb-2-0100/atr_06_table"
DATE_ITEM = "P90035_PTU_DATE_LOC"
PACE_S = 1.0
RETRIES = 3

_NAN = float("nan")
_DAY_MARK = re.compile(r"<!--\s*day:\s*(\d{4}-\d{2}-\d{2})\s*-->")
_last_request = [0.0]


def _pace() -> None:
    wait = PACE_S - (time.monotonic() - _last_request[0])
    if wait > 0 and _last_request[0] > 0:
        time.sleep(wait)
    _last_request[0] = time.monotonic()


def _hidden(page: str, item_id: str) -> str | None:
    for match in re.finditer(r"<input[^>]*>", page):
        tag = match.group(0)
        if f'id="{item_id}"' in tag:
            value = re.search(r'value="([^"]*)"', tag)
            return html_lib.unescape(value.group(1)) if value else ""
    return None


def _num(value: str | None) -> float:
    text = (value or "").replace("\xa0", "").strip()
    if not text:
        return _NAN
    return float(text.replace(",", "."))


def _parse_cells(fragment: str) -> list[dict[str, str]]:
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", fragment, re.S):
        if 'headers="PTU"' not in tr:
            continue
        cells = dict(re.findall(r'headers="([A-Z_]+)"[^>]*>([^<]*)<', tr))
        rows.append({key: html_lib.unescape(val).strip() for key, val in cells.items()})
    return rows


def _midnight(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=ZONE).timestamp())


def parse_day_fragment(html: str) -> list[dict]:
    """Parse one day fragment into rows. The fragment starts with a `<!-- day: YYYY-MM-DD -->` mark.

    The report gives the ISP number of the local day. The start is local midnight plus
    (number - 1) quarters, which stays right on days with 92 or 100 ISPs.
    """
    mark = _DAY_MARK.search(html)
    if mark is None:
        raise FeedError("settlement fragment has no day mark")
    day = mark.group(1)
    base = _midnight(dt.date.fromisoformat(day))
    out = []
    for cells in _parse_cells(html):
        try:
            state = _num(cells.get("REGULATION_STATE"))
            out.append(
                {
                    "ts": base + (int(cells["PTU"]) - 1) * ISP_S,
                    "buy_eur_mwh": _num(cells.get("PRICE_SHORTAGE")),
                    "sell_eur_mwh": _num(cells.get("PRICE_SURPLUS")),
                    "dispatch_up_eur_mwh": _num(cells.get("PRICE_DISPATCH_UP")),
                    "dispatch_down_eur_mwh": _num(cells.get("PRICE_DISPATCH_DOWN")),
                    "regulation_state": _NAN if np.isnan(state) else int(state),
                    "incident_reserve_up": bool(cells.get("IND_INCIDENT_RESERVE_UP", "")),
                    "incident_reserve_down": bool(cells.get("IND_INCIDENT_RESERVE_DOWN", "")),
                }
            )
        except (ValueError, KeyError, OverflowError) as exc:
            raise FeedError(f"settlement {day}: malformed row {cells!r}: {exc!r}") from exc
    return out


class _Session:
    """One APEX session against the settlement-prices report region."""

    def __init__(self) -> None:
        self.jar = http.cookiejar.CookieJar()
        self.open()

    def open(self) -> None:
        _pace()
        page = http_get(
            f"{BASE}/f?p={APP}:{PAGE}::::::", jar=self.jar
        ).decode("utf-8", "replace")
        self.instance = _hidden(page, "pInstance")
        self.salt = _hidden(page, "pSalt")
        self.protected = _hidden(page, "pPageItemsProtected")
        found = re.search(
            r"apex\.widget\.report\.paginate\('(\d+)', '([^']+)'", html_lib.unescape(page)
        )
        if not (found and self.instance and self.salt is not None):
            raise FeedError("TenneT APEX page layout changed; cannot bootstrap")
        self.region, self.ajax_id = found.groups()

    def day(self, day: dt.date) -> str:
        payload = {
            "pageItems": {
                "itemsToSubmit": [{"n": DATE_ITEM, "v": day.strftime("%d-%m-%Y")}],
                "protected": self.protected,
                "rowVersion": "",
                "formRegionChecksums": [],
            },
            "salt": self.salt,
        }
        form = {
            "p_flow_id": APP,
            "p_flow_step_id": PAGE,
            "p_instance": self.instance,
            "p_debug": "",
            "p_request": "PLUGIN=" + self.ajax_id,
            "p_widget_action": "paginate",
            "x01": self.region,
            "p_pg_min_row": "1",
            "p_pg_max_rows": "200",
            "p_pg_rows_fetched": "200",
            "p_json": json.dumps(payload),
        }
        _pace()
        raw = http_post(
            f"{BASE}/wwv_flow.ajax?p_context={CONTEXT}/{self.instance}",
            urllib.parse.urlencode(form).encode(),
            headers={"X-Requested-With": "XMLHttpRequest"},
            jar=self.jar,
        )
        return f"<!-- day: {day.isoformat()} -->\n" + raw.decode("utf-8", "replace")


def _fetch_day(session: _Session, day: dt.date) -> list[dict]:
    for attempt in range(RETRIES):
        try:
            rows = parse_day_fragment(session.day(day))
        except FeedError:
            rows = []
        if rows or attempt == RETRIES - 1:
            return rows
        try:
            session.open()  # the session may have expired
        except FeedError:
            pass
    return []


_COLUMNS = (
    "buy_eur_mwh",
    "sell_eur_mwh",
    "dispatch_up_eur_mwh",
    "dispatch_down_eur_mwh",
    "regulation_state",
    "incident_reserve_up",
    "incident_reserve_down",
)


def _to_cache(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        item = {"ts": int(row["ts"])}
        for col in _COLUMNS:
            val = row[col]
            item[col] = None if isinstance(val, float) and np.isnan(val) else val
        out.append(item)
    return out


def fetch_settlement(start: dt.date, end: dt.date, cache: DayCache, offline: bool = False) -> Table:
    """Settled prices on the 15 minute grid for the local days start..end, NaN where missing.

    `buy` is Price Shortage (take), `sell` is Price Surplus (feed). Boolean columns are 0/1.
    """
    days = local_days(start, end)
    cached: dict[dt.date, list[dict]] = {}
    missing: list[dt.date] = []
    for day in days:
        rows = cache.get(day)
        if rows is None:
            missing.append(day)
        else:
            cached[day] = rows
    if missing and not offline:
        session = _Session()
        for day in missing:
            rows = _fetch_day(session, day)
            if rows:
                cache.put(day, _to_cache(rows))
                cached[day] = cache.get(day) or []
    records = sorted((r for day in days for r in cached.get(day, [])), key=lambda r: r["ts"])
    source: Table = {"ts": np.array([r["ts"] for r in records], dtype=np.int64)}
    for col in _COLUMNS:
        source[col] = np.array(
            [_NAN if r[col] is None else float(r[col]) for r in records], dtype=float
        )
    grid = isp_grid(_midnight(days[0]), _midnight(days[-1] + dt.timedelta(days=1)))
    return align(source, grid, list(_COLUMNS))
