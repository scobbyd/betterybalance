import copy
import datetime as dt
import json
import math
import pathlib

import pytest
from betterybalance.core.feeds import base
from betterybalance.core.feeds import tennet_live as TL

FIX = pathlib.Path(__file__).parent / "fixtures" / "synthetic_tennet_balance_delta.json"
T0 = int(dt.datetime(2026, 11, 2, 11, 0, tzinfo=dt.UTC).timestamp())
MINUTES = (0, 1, 2, 15, 16, 30, 31, 45, 46, 60, 61)


def _doc():
    return json.loads(FIX.read_text())


def _points(doc):
    return doc["Response"]["TimeSeries"][0]["Period"][0]["points"]


def _rows(doc=None):
    return {(r["ts"] - T0) // 60: r for r in TL.parse_balance_delta(doc or _doc())}


def _vals(r):
    return (r["buy_live"], r["sell_live"], r["state"])


def test_url_is_the_documented_latest_endpoint():
    assert TL.URL == "https://api.tennet.eu/publications/v1/balance-delta-high-res/latest"


def test_one_row_per_minute_first_point_values():
    rows = TL.parse_balance_delta(_doc())
    assert [r["ts"] for r in rows] == [T0 + 60 * m for m in MINUTES]
    first = rows[0]  # the fixture's first point: upward only, upward price 231,5
    assert first == {"ts": T0, "buy_live": 231.5, "sell_live": 231.5, "state": 1}


def test_a_minute_carries_the_running_isp_not_the_last_point():
    rows = _rows()
    # the newest point of 11:01 and the only point of 11:02 have no activation (a point
    # alone reads state 0 at the mid price 88,1); the ISP so far is upward with Pup 250
    assert _vals(rows[1]) == (250.0, 250.0, 1)
    assert _vals(rows[2]) == (250.0, 250.0, 1)
    assert TL.point_row(_points(_doc())[3]) == {"buy_live": 88.1, "sell_live": 88.1, "state": 0}


def test_downward_isp_to_date_is_minus_one_at_the_lowest_downward_price():
    assert _vals(_rows()[15]) == (-10.0, -10.0, -1)


def test_both_directions_with_a_rising_delta_is_plus_one():
    # deltas -30, -10, 20: both directions active, the delta only rises
    assert _vals(_rows()[16]) == (120.0, 120.0, 1)


def test_delta_rising_then_falling_is_state_two():
    rows = _rows()
    assert _vals(rows[30]) == (180.0, 180.0, 1)
    assert _vals(rows[31]) == (180.0, -25.0, 2)


def test_state_two_reverse_pricing_falls_back_to_the_mid_price():
    # Pup 60 < Pmid 100 and Pdown 110 > Pmid 100
    assert _vals(_rows()[46]) == (100.0, 100.0, 2)


def test_no_activation_is_state_zero_at_the_mid_price_and_nan_without_one():
    rows = _rows()
    idle, dummy = rows[60], rows[61]
    assert idle["state"] == 0 and math.isnan(idle["buy_live"]) and math.isnan(idle["sell_live"])
    assert _vals(dummy) == (93.0, 93.0, 0)  # upward volume without an upward price


def test_constant_delta_with_both_directions_is_plus_one():
    doc = _doc()
    pts = _points(doc)
    for p in pts[4:7]:  # ISP 11:15: up 10 and down 10 at every point, delta 0 throughout
        p.update(power_afrr_in="10.0", power_afrr_out="10.0", max_upw_regulation_price="130.0",
                 min_downw_regulation_price="-5.0")
    assert _vals(_rows(doc)[16]) == (130.0, 130.0, 1)


def test_a_truncated_first_isp_is_dropped():
    doc = _doc()
    del _points(doc)[0]  # the window now starts at 11:01, inside the ISP of 11:00
    rows = _rows(doc)
    assert sorted(rows) == list(MINUTES[3:])


def test_missing_price_field_is_nan_never_zero():
    doc = _doc()
    point = _points(doc)[0]
    del point["max_upw_regulation_price"]
    first = TL.parse_balance_delta(doc)[0]
    # the upward price is missing, so the upward direction is not counted and the mid price
    # stands; with the mid price missing as well, both prices are NaN
    assert first["state"] == 0 and first["buy_live"] == 88.1
    del point["mid_price"]
    first = TL.parse_balance_delta(doc)[0]
    assert math.isnan(first["buy_live"]) and math.isnan(first["sell_live"])
    assert first["buy_live"] != 0


def test_unparseable_price_is_nan():
    doc = _doc()
    point = _points(doc)[0]
    point["max_upw_regulation_price"] = "n/a"
    point["mid_price"] = ""
    first = TL.parse_balance_delta(doc)[0]
    assert math.isnan(first["buy_live"]) and first["state"] == 0


def test_no_data_and_bad_shape():
    assert TL.parse_balance_delta({"error": "No data found"}) == []
    with pytest.raises(base.FeedError):
        TL.parse_balance_delta({"something": 1})
    doc = copy.deepcopy(_doc())
    _points(doc)[0]["timeInterval_start"] = "x"
    with pytest.raises(base.FeedError):
        TL.parse_balance_delta(doc)


def test_poll_sends_the_apikey_header(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=20.0, headers=None, jar=None):
        seen.update(url=url, headers=headers)
        return FIX.read_bytes()

    monkeypatch.setattr(base, "http_get", fake_get)
    rows = TL.poll_tennet("synthetic-key")
    assert seen["url"] == TL.URL and seen["headers"]["apikey"] == "synthetic-key"
    assert seen["headers"]["Accept"] == "application/json" and len(rows) == len(MINUTES)


def test_poll_no_data_is_a_feed_error(monkeypatch):
    monkeypatch.setattr(base, "http_get", lambda url, **k: b'{"error": "No data found"}')
    with pytest.raises(base.FeedError, match="No data found"):
        TL.poll_tennet("synthetic-key")
