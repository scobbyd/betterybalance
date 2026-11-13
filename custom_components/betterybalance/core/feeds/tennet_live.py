"""Live minute prices from the TenneT publications API, for users who register a free key.

Endpoint: GET https://api.tennet.eu/publications/v1/balance-delta-high-res/latest
Documentation: developer.tennet.eu/specs/v1/balance-delta-high-res ("Aether - Balance
Delta High resolution", OpenAPI 1.5.0, read 2026-10-31; operation `request_get_latest`).
Authentication: the header `apikey` (security scheme `apikey`, in header). A key comes from a
developer account at developer.tennet.eu. Rate limits of `latest`: 1 request per second and
10 requests per minute; the box polls once per minute.

The documented response (schema `response_json`) is
`{"Response": {"informationType", "period.timeInterval", "TimeSeries": [{"mRID",
"quantity_Measurement_Unit_name", "currency_Unit_name", "Period": [{"timeInterval",
"points": [...]}]}]}}`. Each point covers 12 s and holds strings: `timeInterval_start`,
`timeInterval_end` (UTC, `Z`), `sequence`, `power_afrr_in`, `power_afrr_out`, `power_igcc_in`,
`power_igcc_out`, `power_mfrrda_in`, `power_mfrrda_out` (nullable), `power_picasso_in`,
`power_picasso_out`, `power_mari_in`, `power_mari_out`, `max_upw_regulation_price` (nullable),
`min_downw_regulation_price` (nullable) and `mid_price`. A response holds the most recent
available 30 minutes, after a delay TenneT sets (2 minutes in the press, not in the spec). With
no data the body is `{"error": "No data found"}` (schema `no-data`).

The response has no take price, feed price or regulation state. This module derives them per
ISP to date, after TenneT's Imbalance Pricing System (version 6.3, 1 September 2026, sections
4.2 and 4.3, Table 2), so a minute row carries the running ISP value, as the TenEnergy page
does. Over the points that start in the ISP up to and including the newest point of the minute:

- a point is upward active when `power_afrr_in + power_mfrrda_in > 0` and its
  `max_upw_regulation_price` is a number, downward active when `power_afrr_out +
  power_mfrrda_out > 0` and its `min_downw_regulation_price` is a number (the dummy-energy
  filter of the `aritmeester/tennet_balance` integration; `in` is upward, `out` downward;
  IGCC, PICASSO and MARI volumes do not set the state);
- Pup is the highest upward price of the upward active points, Pdown the lowest downward price
  of the downward active points, Pmid the newest `mid_price` that is a number;
- state 0 without activation, 1 with upward only, -1 with downward only. With both, the series
  of balance deltas `(afrr_in + mfrrda_in) - (afrr_out + mfrrda_out)` of all points decides:
  rising only gives 1, falling only gives -1, rising and falling gives 2, and a constant
  series gives 1 (my call: TenneT names constant under both 1 and -1);
- prices: state 0 both Pmid; state 1 both Pup; state -1 both Pdown; state 2 buy
  `max(Pup, Pmid)` and sell `min(Pdown, Pmid)` (reverse pricing; without Pmid, Pup and Pdown).

A missing volume counts as no activation. A missing or unparseable price is NaN, never 0. The
ISP of the window's first point is dropped when that point starts after the ISP start: the
window cut it, so its aggregate would be partial. Residual differences from settlement, which a
nowcast accepts: TenneT prices 4 s activations, the incident-reserve rules, the previous-ISP
price when no price exists, and the ISP is not complete yet.

Labels: `parse_balance_delta` returns point times. The publication comes after the point, so a
recorder labels a sample by the minute of the poll that saw it (``live.fetch.poll_live``), as
for the TenEnergy page.
"""

from __future__ import annotations

import datetime as dt
import math

from ..timegrid import ISP_S
from . import base

URL = "https://api.tennet.eu/publications/v1/balance-delta-high-res/latest"
SOURCE = "tennet"
MINUTE_S = 60
_UP_VOLUMES = ("power_afrr_in", "power_mfrrda_in")
_DOWN_VOLUMES = ("power_afrr_out", "power_mfrrda_out")


def _num(value) -> float:
    """A documented string field as a float; None, empty or unparseable as NaN."""
    if value is None or isinstance(value, bool):
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _volume(point: dict, names: tuple[str, ...]) -> float:
    return sum(v for v in (_num(point.get(n)) for n in names) if not math.isnan(v))


def _epoch(stamp) -> int:
    try:
        parsed = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError as exc:
        raise base.FeedError(f"tennet: bad timeInterval_start {stamp!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return int(parsed.timestamp())


def _activity(point: dict) -> tuple[float, float, float, float]:
    """(up price, down price, mid price, balance delta) of one point; the up or down price is
    NaN when that direction is not active."""
    up_price = _num(point.get("max_upw_regulation_price"))
    down_price = _num(point.get("min_downw_regulation_price"))
    up_vol, down_vol = _volume(point, _UP_VOLUMES), _volume(point, _DOWN_VOLUMES)
    up = up_price if up_vol > 0 else math.nan
    down = down_price if down_vol > 0 else math.nan
    return up, down, _num(point.get("mid_price")), up_vol - down_vol


def _isp_value(points: list[dict]) -> dict:
    """``buy_live``, ``sell_live`` and ``state`` of an ISP to date (rules in the docstring);
    ``points`` are the ISP's points in start order."""
    acts = [_activity(p) for p in points]
    ups = [a[0] for a in acts if not math.isnan(a[0])]
    downs = [a[1] for a in acts if not math.isnan(a[1])]
    mids = [a[2] for a in acts if not math.isnan(a[2])]
    p_up = max(ups) if ups else math.nan
    p_down = min(downs) if downs else math.nan
    p_mid = mids[-1] if mids else math.nan
    if ups and downs:
        deltas = [a[3] for a in acts]
        rises = any(b > a for a, b in zip(deltas, deltas[1:], strict=False))
        falls = any(b < a for a, b in zip(deltas, deltas[1:], strict=False))
        if rises and falls:
            buy = p_up if math.isnan(p_mid) else max(p_up, p_mid)
            sell = p_down if math.isnan(p_mid) else min(p_down, p_mid)
            return {"buy_live": buy, "sell_live": sell, "state": 2}
        if falls:
            return {"buy_live": p_down, "sell_live": p_down, "state": -1}
        return {"buy_live": p_up, "sell_live": p_up, "state": 1}
    if ups:
        return {"buy_live": p_up, "sell_live": p_up, "state": 1}
    if downs:
        return {"buy_live": p_down, "sell_live": p_down, "state": -1}
    return {"buy_live": p_mid, "sell_live": p_mid, "state": 0}


def point_row(point: dict) -> dict:
    """The value of one 12 s point read as an ISP of its own (diagnostics and tests)."""
    return _isp_value([point])


def _points(doc: dict) -> list[dict]:
    response = doc.get("Response")
    if not isinstance(response, dict) or not isinstance(response.get("TimeSeries"), list):
        raise base.FeedError(f"tennet: unexpected response keys {sorted(doc)[:5]!r}")
    out = []
    for series in response["TimeSeries"]:
        for period in (series or {}).get("Period") or []:
            out.extend(p for p in (period or {}).get("points") or [] if isinstance(p, dict))
    return out


def parse_balance_delta(doc: dict) -> list[dict]:
    """One row per minute of point time: ``ts`` (minute start), ``buy_live``, ``sell_live``,
    ``state``, the running ISP value as of the newest point that starts in that minute.
    Sorted by ``ts``; a window-cut first ISP is dropped (module docstring).

    A `no-data` body gives []. Any other body without ``Response.TimeSeries`` raises FeedError.
    """
    if "error" in doc and "Response" not in doc:
        return []
    timed = sorted(
        ((_epoch(p.get("timeInterval_start")), i, p) for i, p in enumerate(_points(doc))),
        key=lambda x: (x[0], x[1]),
    )
    if not timed:
        return []
    first = timed[0][0]
    cut_isp = first - first % ISP_S if first % ISP_S else None
    rows: dict[int, dict] = {}
    isp, so_far = None, []
    for start, _, point in timed:
        q = start - start % ISP_S
        if q == cut_isp:
            continue
        if q != isp:
            isp, so_far = q, []
        so_far.append(point)
        rows[start - start % MINUTE_S] = _isp_value(so_far)  # the newest point of the minute wins
    return [{"ts": m, **rows[m]} for m in sorted(rows)]


def poll_tennet(key: str) -> list[dict]:
    """GET the `latest` endpoint with the user's key and parse it. Raises FeedError on a
    failed request, a body that is not a JSON object or a `no-data` body."""
    payload = base.http_get(URL, headers={"apikey": key, "Accept": "application/json"})
    doc = base.json_object(payload, "tennet")
    if "error" in doc and "Response" not in doc:
        raise base.FeedError(f"tennet: {doc['error']}")
    return parse_balance_delta(doc)
