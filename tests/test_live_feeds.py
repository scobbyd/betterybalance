import datetime as dt
import json
import math
import pathlib

import numpy as np
import pytest
from betterybalance.core.feeds import base
from betterybalance.core.feeds import openmeteo as OM
from betterybalance.core.feeds import tennet_settlement as TS
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import feeds as FD
from betterybalance.live import fetch as FX

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _t(s): return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZONE).timestamp())


# --- cadences (the brief's tests) ---------------------------------------------------------------

def test_day_ahead_cadence():
    st = FD.FeedState("day_ahead")
    assert not FD.due("day_ahead", _t("2026-11-02T13:04:30"), st)
    assert FD.due("day_ahead", _t("2026-11-02T13:05:10"), st)
    st.last_try = _t("2026-11-02T13:05:10")
    assert FD.due("day_ahead", _t("2026-11-02T13:35:00"), st) and not FD.due(
        "day_ahead", _t("2026-11-02T13:35:00"), st, have_tomorrow_da=True
    )

def test_ida_twenty_minutes_after_each_gate():
    st = FD.FeedState("ida")
    assert FD.due("ida", _t("2026-11-02T15:20:00"), st)
    assert FD.due("ida", _t("2026-11-02T22:20:00"), st)
    assert FD.due("ida", _t("2026-11-02T10:20:00"), st)
    assert not FD.due("ida", _t("2026-11-02T15:19:00"), st)

def test_live_backoff_and_problem():
    st = FD.FeedState("live", failures=3)
    assert FD.next_backoff(st) == 480 and FD.next_backoff(FD.FeedState("live", failures=5)) == 600
    now = _t("2026-11-02T12:00:00")
    assert FD.problem("live", now, FD.FeedState("live", last_ok=now - 181), False, True, True)
    assert not FD.problem("live", now, FD.FeedState("live", last_ok=now - 179), False, True, True)

def test_gates_follow_amsterdam_across_the_dst_end():
    st = FD.FeedState("settlement")
    assert FD.due("settlement", _t("2026-10-25T09:00:20"), st)
    assert not FD.due("settlement", _t("2026-10-25T08:00:20"), st)

def test_irradiance_and_pack_slots():
    assert FD.due("irradiance", _t("2026-11-02T05:07:00"), FD.FeedState("irradiance"))
    assert FD.due("irradiance", _t("2026-11-02T15:07:00"), FD.FeedState("irradiance"))
    assert FD.due("pack", _t("2026-11-02T04:10:00"), FD.FeedState("pack"))


# --- cadences, further cases --------------------------------------------------------------------

def test_a_slot_tried_in_its_minute_is_not_due_again():
    now = _t("2026-11-02T09:00:40")
    tried = FD.FeedState("settlement", last_try=_t("2026-11-02T09:00:05"))
    assert not FD.due("settlement", now, tried)
    assert FD.due("settlement", now, FD.FeedState("settlement", last_try=now - 41))
    assert FD.due("settlement", _t("2026-11-02T15:00:00"), FD.FeedState("settlement"))


def test_a_missed_slot_is_caught_up_for_45_minutes():
    st = FD.FeedState("day_ahead", last_try=_t("2026-11-01T14:05:00"))
    assert FD.due("day_ahead", _t("2026-11-02T13:15:00"), st)  # a tick 10 min after 13:05
    assert FD.due("settlement", _t("2026-11-02T09:01:00"), FD.FeedState("settlement"))
    assert FD.due("settlement", _t("2026-11-02T09:44:59"), FD.FeedState("settlement"))
    assert not FD.due("settlement", _t("2026-11-02T09:45:00"), FD.FeedState("settlement"))
    assert FD.due("pack", _t("2026-11-02T04:11:00"), FD.FeedState("pack"))
    assert not FD.due("pack", _t("2026-11-02T04:55:00"), FD.FeedState("pack"))
    tried = FD.FeedState("settlement", last_try=_t("2026-11-02T09:00:02"))
    assert not FD.due("settlement", _t("2026-11-02T09:20:00"), tried)
    # overlapping day-ahead windows: tried at 13:05, due again at 13:35, not before
    da = FD.FeedState("day_ahead", last_try=_t("2026-11-02T13:05:02"))
    assert not FD.due("day_ahead", _t("2026-11-02T13:34:00"), da)
    assert FD.due("day_ahead", _t("2026-11-02T13:36:00"), da)
    assert not FD.due("day_ahead", _t("2026-11-02T13:36:00"), da, have_tomorrow_da=True)


def test_ida_failure_is_retried_every_five_minutes_until_gate_plus_60():
    gate = _t("2026-11-02T15:00:00")  # IDA1 for 11-03
    st = FD.FeedState("ida", last_try=gate + 1202, failures=1, last_error="no result yet")
    assert not FD.due("ida", gate + 1440, st)
    assert FD.due("ida", gate + 1502, st)  # gate + 25
    st.last_try, st.failures = gate + 3302, 2  # the try at gate + 55 failed as well
    assert not FD.due("ida", gate + 3602, st)  # gate + 60: give up
    assert FD.due_auctions(gate + 1502) == [("IDA1", dt.date(2026, 11, 3))]
    ok = FD.FeedState("ida", last_ok=gate + 1202, last_try=gate + 1202)
    assert not FD.due("ida", gate + 1502, ok)
    missed = FD.FeedState("ida", last_try=gate - 3600)
    assert FD.due("ida", gate + 1800, missed)  # a missed gate + 20 slot is caught up


def test_irradiance_pack_and_day_ahead_off_slot():
    assert not FD.due("irradiance", _t("2026-11-02T04:07:00"), FD.FeedState("irradiance"))
    assert FD.due("irradiance", _t("2026-11-02T15:08:00"), FD.FeedState("irradiance"))
    assert not FD.due("irradiance", _t("2026-11-02T15:52:00"), FD.FeedState("irradiance"))
    assert FD.due("irradiance", _t("2026-11-02T00:07:59"), FD.FeedState("irradiance"))
    assert not FD.due("pack", _t("2026-11-02T03:10:00"), FD.FeedState("pack"))
    assert FD.due("day_ahead", _t("2026-11-02T14:05:00"), FD.FeedState("day_ahead"))
    assert not FD.due("day_ahead", _t("2026-11-02T14:05:00"), FD.FeedState("day_ahead"), True)
    assert FD.due("day_ahead", _t("2026-11-02T13:05:00"), FD.FeedState("day_ahead"), True)


def test_live_due_on_the_backoff():
    now = _t("2026-11-02T12:00:00")
    assert FD.due("live", now, FD.FeedState("live"))
    assert FD.next_backoff(FD.FeedState("live")) == 60
    assert FD.next_backoff(FD.FeedState("live", failures=1)) == 120
    assert FD.due("live", now, FD.FeedState("live", last_try=now - 60))
    assert not FD.due("live", now + 59, FD.FeedState("live", last_try=now))
    assert not FD.due("live", now, FD.FeedState("live", last_try=now - 180, failures=2))
    assert FD.due("live", now, FD.FeedState("live", last_try=now - 240, failures=2))


def test_live_backoff_counts_minutes_not_seconds():
    tick = _t("2026-11-02T12:00:02")
    st = FD.FeedState("live", last_try=tick - 60 + 1)  # stored after the fetch at 11:59:03
    assert FD.due("live", tick, st)  # 59 s, but the next minute
    st = FD.FeedState("live", last_try=tick - 120 + 5, failures=1)  # 11:58:07, backoff 120 s
    assert FD.due("live", tick, st) and not FD.due("live", tick - 60, st)


def test_gates_ignore_the_process_time_zone(monkeypatch):
    import time

    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        assert FD.due("pack", _t("2026-11-02T04:10:00"), FD.FeedState("pack"))
        assert FD.due_auctions(_t("2026-11-02T15:20:00")) == [("IDA1", dt.date(2026, 11, 3))]
    finally:
        monkeypatch.delenv("TZ")
        time.tzset()


def test_due_auctions_names_the_delivery_day():
    assert FD.due_auctions(_t("2026-11-02T22:20:30")) == [("IDA2", dt.date(2026, 11, 3))]
    assert FD.due_auctions(_t("2026-11-02T10:20:00")) == [("IDA3", dt.date(2026, 11, 2))]
    assert FD.due_auctions(_t("2026-11-02T10:21:00")) == [("IDA3", dt.date(2026, 11, 2))]
    assert FD.due_auctions(_t("2026-11-02T11:05:00")) == []


def test_ida_gates_across_the_dst_end():
    # the IDA1 gate of delivery day 2026-10-26 is 15:00 CET on 10-25, after the clock change
    assert FD.due("ida", _t("2026-10-25T15:20:00"), FD.FeedState("ida"))
    assert not FD.due("ida", _t("2026-10-25T14:20:00"), FD.FeedState("ida"))


def test_unknown_feed_is_refused():
    with pytest.raises(ValueError):
        FD.due("weather", 0, FD.FeedState("weather"))
    with pytest.raises(ValueError):
        FD.problem("weather", 0, FD.FeedState("weather"), False, True, True)


# --- problem rules ------------------------------------------------------------------------------

def _p(feed, now, st=None, tomorrow=False, settled=True, pack=True):
    return FD.problem(feed, now, st or FD.FeedState(feed), tomorrow, settled, pack)


def test_day_ahead_problem_from_1430_without_tomorrow():
    assert not _p("day_ahead", _t("2026-11-02T14:29:59"))
    assert _p("day_ahead", _t("2026-11-02T14:30:00"))
    assert not _p("day_ahead", _t("2026-11-02T14:30:00"), tomorrow=True)
    assert not _p("day_ahead", _t("2026-11-02T00:30:00"))


def test_settlement_problem_from_1600_when_yesterday_is_unsettled():
    assert not _p("settlement", _t("2026-11-02T15:59:00"), settled=False)
    assert _p("settlement", _t("2026-11-02T16:00:00"), settled=False)
    assert not _p("settlement", _t("2026-11-02T16:00:00"), settled=True)


def test_ida_and_irradiance_age_rules():
    now = _t("2026-11-02T12:00:00")
    assert _p("ida", now, FD.FeedState("ida", last_ok=now - 26 * 3600 - 1))
    assert not _p("ida", now, FD.FeedState("ida", last_ok=now - 26 * 3600))
    assert _p("irradiance", now, FD.FeedState("irradiance", last_ok=now - 86401))
    assert not _p("irradiance", now, FD.FeedState("irradiance", last_ok=now - 86400))
    assert _p("live", now) and _p("ida", now) and _p("irradiance", now)  # never succeeded


def test_pack_problem():
    now = _t("2026-11-02T12:00:00")
    assert _p("pack", now, pack=False)
    assert _p("pack", now, FD.FeedState("pack", failures=3))
    assert not _p("pack", now, FD.FeedState("pack", failures=2))


# --- fetchers (no network: core.feeds.base.http_get and the bound copies are replaced) ----------

class Net:
    """One fake for every HTTP call; ``answers`` maps a URL prefix to bytes or a callable."""

    def __init__(self, monkeypatch, answers):
        self.calls = []
        self.answers = answers
        for mod in (base, TS, OM):
            monkeypatch.setattr(mod, "http_get", self.get)
        for mod in (base, TS):
            monkeypatch.setattr(mod, "http_post", self.post)
        monkeypatch.setattr(TS, "PACE_S", 0.0)

    def _answer(self, url, **kw):
        self.calls.append((url, kw))
        for prefix, body in self.answers.items():
            if url.startswith(prefix):
                return body(url, **kw) if callable(body) else body
        raise base.FeedError(f"unexpected url {url}")

    def get(self, url, params=None, timeout=20.0, headers=None, jar=None):
        return self._answer(url, params=params, headers=headers)

    def post(self, url, data, timeout=20.0, headers=None, jar=None):
        return self._answer(url, data=data)


def _store():
    s = Store(":memory:")
    s.migrate()
    return s


def _da_payload(day, skip=(), null=()):
    start = _t(f"{day.isoformat()}T00:00:00")
    stamps = [start + 900 * i for i in range(96) if i not in skip]
    prices = [None if (t - start) // 900 in null else 50.0 + (t - start) // 900 for t in stamps]
    return json.dumps({"unix_seconds": stamps, "price": prices}).encode()


def test_fetch_day_ahead_writes_the_day(monkeypatch):
    day = dt.date(2026, 11, 3)
    net = Net(monkeypatch, {"https://api.energy-charts.info/price": _da_payload(day)})
    s = _store()
    assert FX.fetch_day_ahead(s, day) == 96
    lo = _t("2026-11-03T00:00:00")
    f = s.features(lo, lo + 86400)
    assert len(f["ts"]) == 96 and f["da_eur_mwh"][5] == 55.0
    (url, kw), = net.calls
    assert kw["params"] == {"bzn": "NL", "start": "2026-11-03", "end": "2026-11-03"}


def test_fetch_day_ahead_incomplete_day_is_a_feed_error_and_nan_stays_nan(monkeypatch):
    day = dt.date(2026, 11, 3)
    Net(monkeypatch, {"https://api.energy-charts.info": _da_payload(day, null=(7,))})
    s = _store()
    with pytest.raises(base.FeedError, match="95 of 96"):
        FX.fetch_day_ahead(s, day)
    lo = _t("2026-11-03T00:00:00")
    f = s.features(lo, lo + 86400)
    got = dict(zip(f["ts"].tolist(), f["da_eur_mwh"].tolist(), strict=True))
    assert lo + 7 * 900 not in got and got[lo + 8 * 900] == 58.0 and len(got) == 95
    Net(monkeypatch, {"https://api.energy-charts.info": b'{"unix_seconds": [], "price": []}'})
    with pytest.raises(base.FeedError, match="0 of 96"):
        FX.fetch_day_ahead(_store(), day)


def test_fetch_day_ahead_counts_the_quarters_of_a_dst_day(monkeypatch):
    day = dt.date(2026, 10, 25)  # 25 hours, 100 quarters
    start = _t("2026-10-25T00:00:00")
    stamps = [start + 900 * i for i in range(100)]
    body = json.dumps({"unix_seconds": stamps, "price": [1.0] * 100}).encode()
    Net(monkeypatch, {"https://api.energy-charts.info": body})
    assert FX.fetch_day_ahead(_store(), day) == 100


def test_fetch_ida_writes_its_column(monkeypatch):
    body = (FIXTURES / "synthetic_ida_IDA1_2026-10-15.json").read_bytes()
    net = Net(monkeypatch, {"https://dataportal-api.nordpoolgroup.com": body})
    s = _store()
    n = FX.fetch_ida(s, "IDA1", dt.date(2026, 10, 15))
    first = int(dt.datetime(2026, 10, 14, 22, tzinfo=dt.UTC).timestamp())
    f = s.features(first, first + 86400)
    assert n > 0 and f["ida1_eur_mwh"][0] == 100.0 and math.isnan(f["ida2_eur_mwh"][0])
    assert net.calls[0][1]["params"]["market"] == "SIDC_IntradayAuction1"
    Net(monkeypatch, {"https://dataportal-api.nordpoolgroup.com": b""})
    with pytest.raises(base.FeedError):
        FX.fetch_ida(_store(), "IDA1", dt.date(2026, 10, 15))


_APEX_PAGE = (
    '<input type="hidden" id="pInstance" value="1">'
    '<input type="hidden" id="pSalt" value="9">'
    '<input type="hidden" id="pPageItemsProtected" value="a">'
    "apex.widget.report.paginate('42', 'AJAXID'"
).encode()


def _settlement_net(monkeypatch, fragment):
    return Net(
        monkeypatch,
        {
            "https://publications.tennet.eu/ords/f": _APEX_PAGE,
            "https://publications.tennet.eu/ords/wwv_flow.ajax": fragment.encode(),
        },
    )


def test_fetch_settlement_appends_new_publications_only(monkeypatch):
    frag = (FIXTURES / "tennet_settlement_fragment.html").read_text()
    day = dt.date(2026, 10, 28)
    s = _store()
    _settlement_net(monkeypatch, frag)
    rows = FX.fetch_settlement(s, day, now=1000)
    assert len(rows) == 8 and rows[0]["published_at"] == 1000
    assert set(rows[0]) == {
        "quarter_ts", "take", "feed", "regulation_state", "published_at", "source",
    }
    assert rows[0]["quarter_ts"] == _t("2026-10-28T00:00:00") and rows[0]["take"] == 180.0
    assert FX.fetch_settlement(s, day, now=2000) == []  # same prices: nothing new
    corrected = frag.replace('"PRICE_SHORTAGE">180<', '"PRICE_SHORTAGE">181<', 1)
    assert corrected != frag
    _settlement_net(monkeypatch, corrected)
    rows = FX.fetch_settlement(s, day, now=3000)
    assert [(r["take"], r["published_at"]) for r in rows] == [(181.0, 3000)]
    lo = _t("2026-10-28T00:00:00")
    assert len(s.settlements(lo, lo + 900, newest=False)) == 2
    assert s.settlements(lo, lo + 900)[0]["take"] == 181.0


def test_fetch_settlement_unpublished_day_is_a_feed_error(monkeypatch):
    _settlement_net(monkeypatch, "<table></table>")
    with pytest.raises(base.FeedError):
        FX.fetch_settlement(_store(), dt.date(2026, 10, 28), now=1000)


def test_poll_live_tenergy_labels_the_poll_minute(monkeypatch):
    page = (FIXTURES / "tenergy_page.html").read_bytes()
    net = Net(monkeypatch, {"https://services.tenergy.nl": page})
    s = _store()
    now = 1793308000 + 25
    assert FX.poll_live(s, now, None) == 1
    t = s.live_minutes(now - 3600, now + 60)
    assert t["ts"].tolist() == [now - now % 60]
    assert t["buy_live"][0] == 292.43 and t["sell_live"][0] == 235.0
    assert np.isnan(t["state"][0]) and t["source"].tolist() == ["tenergy"]
    assert all("apikey" not in (kw.get("headers") or {}) for _, kw in net.calls)


def test_poll_live_tennet_key_path(monkeypatch):
    body = (FIXTURES / "synthetic_tennet_balance_delta.json").read_bytes()
    net = Net(monkeypatch, {"https://api.tennet.eu/publications/v1/": body})
    s = _store()
    t0 = int(dt.datetime(2026, 11, 2, 11, 0, tzinfo=dt.UTC).timestamp())
    now = t0 + 61 * 60 + 150  # the newest point (minute 12:01) ended 90 s ago
    assert FX.poll_live(s, now, "synthetic-key") == 1
    t = s.live_minutes(now - 3600, now + 60)
    assert t["ts"].tolist() == [now - now % 60] and t["buy_live"][0] == 93.0
    assert t["state"][0] == 0 and t["source"].tolist() == ["tennet"]
    assert net.calls[0][1]["headers"]["apikey"] == "synthetic-key"


def test_poll_live_tennet_stale_window_is_a_feed_error(monkeypatch):
    body = (FIXTURES / "synthetic_tennet_balance_delta.json").read_bytes()
    Net(monkeypatch, {"https://api.tennet.eu/publications/v1/": body})
    s = _store()
    t0 = int(dt.datetime(2026, 11, 2, 11, 0, tzinfo=dt.UTC).timestamp())
    with pytest.raises(base.FeedError, match="old"):
        FX.poll_live(s, t0 + 3 * 3600, "synthetic-key")
    assert len(s.live_minutes(0, 2**40)["ts"]) == 0


def test_poll_live_empty_page_is_a_feed_error(monkeypatch):
    Net(monkeypatch, {"https://services.tenergy.nl": b"<html>maintenance</html>"})
    with pytest.raises(base.FeedError):
        FX.poll_live(_store(), 1793308000, None)


def test_fetch_irradiance_rounds_the_location_and_stores_the_forecast(monkeypatch):
    body = (FIXTURES / "synthetic_openmeteo_minutely15.json").read_bytes()
    net = Net(monkeypatch, {"https://api.open-meteo.com/v1/forecast": body})
    s = _store()
    now = 1788307200
    FX.fetch_irradiance(s, 52.14, 5.24, now)
    params = net.calls[0][1]["params"]
    assert (params["latitude"], params["longitude"]) == (52.1, 5.2)
    issue, table = s.fetch_at(now, "forecast")
    assert issue == now and len(table["ts"]) == 96 * 4 and "ghi_w_m2" in table
    Net(monkeypatch, {"https://api.open-meteo.com": b"<html>"})
    with pytest.raises(base.FeedError):
        FX.fetch_irradiance(_store(), 52.1, 5.2, now)


def test_poll_live_unpriced_newest_point_writes_nothing(monkeypatch):
    doc = json.loads((FIXTURES / "synthetic_tennet_balance_delta.json").read_text())
    points = doc["Response"]["TimeSeries"][0]["Period"][0]["points"]
    del points[-1]  # the newest point is now minute 12:00: no activation, no mid price
    Net(monkeypatch, {"https://api.tennet.eu/publications/v1/": json.dumps(doc).encode()})
    s = _store()
    t0 = int(dt.datetime(2026, 11, 2, 11, 0, tzinfo=dt.UTC).timestamp())
    with pytest.raises(base.FeedError, match="no price"):
        FX.poll_live(s, t0 + 61 * 60 + 30, "synthetic-key")
    assert len(s.live_minutes(0, 2**40)["ts"]) == 0


def test_poll_live_without_now_labels_by_the_wall_time_after_the_response(monkeypatch):
    page = (FIXTURES / "tenergy_page.html").read_bytes()
    Net(monkeypatch, {"https://services.tenergy.nl": page})
    seen = 1793308000 + 125  # the response arrives in the minute after the stale tick time
    monkeypatch.setattr(FX.time, "time", lambda: float(seen))
    s = _store()
    assert FX.poll_live(s, None, None) == 1
    assert s.live_minutes(0, 2**40)["ts"].tolist() == [seen - seen % 60]
