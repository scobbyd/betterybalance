# tests/test_live_results.py
import datetime as dt
import math

import pytest
from betterybalance.core import ledger as LG
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import results as R
from betterybalance.live.site import LiveSite
from betterybalance.replay import reexec as RX

from tests.test_live_site import (
    FIRST_DAY,
    _flows,
    _local_midnight,
    _negative_take,
    _run,
    _site,
)

_START = _local_midnight(FIRST_DAY + dt.timedelta(days=21))  # _site's default start


def _settle_rows(store, start, quarters, take=100.0, feed=50.0, published=None):
    return [
        {
            "quarter_ts": start + 900 * k, "take": take, "feed": feed, "regulation_state": 1,
            "published_at": published or start + 86400, "source": "tennet",
        }
        for k in quarters
    ]


def test_settle_twice_books_once_and_a_correction_books_again():
    site, store, start = _site()
    _run(site, start, 4 * 15)
    assert site.settle(_settle_rows(store, start, range(4)), start + 90000) == [
        start + 900 * k for k in range(4)
    ]
    assert site.settle(_settle_rows(store, start, range(4)), start + 90000) == []
    assert site.settle(
        _settle_rows(store, start, [2], take=300.0, published=start + 99999), start + 100000
    ) == [start + 1800]
    newest = store.newest_bookings(start, start + 3600)
    assert newest[(start + 1800, "house")]["buy"] == 300.0
    assert newest[(start, "house")]["buy"] == 100.0


def test_settle_books_the_three_paths_chained_and_puts_one_event():
    site, store, start = _site()
    _run(site, start, 2 * 15)
    head = store.head()[0]
    site.settle(_settle_rows(store, start, range(2)), start + 90000)
    items = store.chain(head + 1, store.head()[0])
    assert [(r["quarter_ts"], r["path"]) for _, r in items] == [
        (q, p) for q in (start, start + 900) for p in LG.PATHS
    ]
    assert all(r["price_source"] == "settled" and r["state"] == 1 for _, r in items)
    assert LG.verify_chain(store.chain(1, store.head()[0]), LG.genesis_hash("install-1")).ok
    ev = store.last_event("settle")
    assert ev["ts"] == start + 90000 and ev["detail"] == {
        "quarters": [start, start + 900], "published_at": [start + 86400, start + 86400]
    }
    # a quarter without a booking (the running quarter) and a call without news: no event
    n = len(store.events())
    assert site.settle(_settle_rows(store, start, [5]), start + 90000) == []
    assert len(store.events()) == n


def test_partial_day_stays_provisional():
    site, store, start = _site()
    _run(site, start, 4 * 15)
    rows = _settle_rows(store, start, range(4))
    rows[3]["take"] = float("nan")
    site.settle(rows, start + 90000)
    newest = store.newest_bookings(start, start + 3600)
    assert newest[(start + 2700, "shadow")]["price_source"] == "provisional"
    assert newest[(start + 1800, "shadow")]["price_source"] == "settled"
    assert store.settlements(start + 2700, start + 3600) == []  # the NaN row is not stored


def test_a_stored_settlement_rebooks_a_provisional_quarter_later():
    # the pull stored the rows, the apply was lost (a restart): the next settle books them
    site, store, start = _site()
    _run(site, start, 2 * 15)
    for row in _settle_rows(store, start, range(2)):
        store.put_settlement(row)
    assert site.settle(store.settlements(start, start + 1800), start + 90000) == [
        start, start + 900
    ]


def test_settle_refuses_a_row_published_after_now():
    site, store, start = _site()
    _run(site, start, 15)
    with pytest.raises(ValueError, match="published"):
        site.settle(_settle_rows(store, start, [0], published=start + 90001), start + 90000)
    assert store.settlements(start, start + 900) == []


def test_totals_never_double_count_a_correction(monkeypatch):
    _negative_take(monkeypatch)  # the shadow charges: results are not 0
    site, store, start = _site()
    _run(site, start, 4 * 15)
    site.settle(_settle_rows(store, start, range(4)), start + 90000)
    first = R.result_totals(store, start + 90000).settled_eur
    assert first != 0.0
    site.settle(_settle_rows(store, start, range(4), published=start + 99999), start + 100000)
    assert R.result_totals(store, start + 100000).settled_eur == pytest.approx(first)


def _results(store, lo, hi):
    new = store.newest_bookings(lo, hi)
    return {
        q: LG.quarter_result(new[(q, "house")], new[(q, "shadow")]) for q, p in new if p == "house"
    }


def test_totals_split_settled_provisional_and_today(monkeypatch):
    _negative_take(monkeypatch)  # the shadow charges: results are not 0
    site, store, start = _site()
    _run(site, start, 6 * 15)
    site.settle(_settle_rows(store, start, range(2), take=250.0, feed=-20.0), start + 90000)
    res = _results(store, start, start + 6 * 900)
    t = R.result_totals(store, start + 6 * 900)
    assert res[start] != 0.0
    assert t.settled_eur == pytest.approx(res[start] + res[start + 900])
    assert t.provisional_eur == pytest.approx(sum(res.values()))
    assert t.today_eur == pytest.approx(t.provisional_eur)
    assert t.today_settled_share == pytest.approx(2 / 6) and t.today_missing == 0
    # the next Amsterdam day has no quarter yet
    t2 = R.result_totals(store, start + 86400)
    assert (t2.today_eur, t2.today_settled_share, t2.today_missing) == (0.0, 0.0, 0)
    assert t2.provisional_eur == pytest.approx(t.provisional_eur)


def test_totals_count_a_missing_quarter_and_leave_it_out():
    site, store, start = _site()
    _run(site, start, 2 * 15, flows=lambda ts: None if ts == start + 300 else _flows(ts))
    t = R.result_totals(store, start + 1800)
    res = _results(store, start, start + 1800)
    assert math.isnan(res[start]) and t.today_missing == 1
    assert t.provisional_eur == pytest.approx(res[start + 900])


def test_the_totals_cache_follows_the_chain(monkeypatch):
    _negative_take(monkeypatch)  # the shadow charges: results are not 0
    site, store, start = _site()
    cache = R.TotalsCache()
    _run(site, start, 3 * 15)
    assert cache.totals(store, start + 2700) == R.result_totals(store, start + 2700)
    rows = _settle_rows(store, start, range(2), take=400.0, published=start + 2600)
    site.settle(rows, start + 2700)
    _run(site, start + 2700, 2 * 15)
    assert cache.totals(store, start + 4500) == R.result_totals(store, start + 4500)
    assert cache.totals(store, start + 4500).settled_eur != 0.0


def test_hourly_sums_cover_the_25_hour_day(monkeypatch):
    # the live site over local 2026-10-25 (25 hours), every quarter settled at prices that
    # make each quarter's result non-negative: the cumulative sums never decrease, and each
    # hour adds exactly the settled results of its four quarters
    _negative_take(monkeypatch)  # the shadow charges: results are not 0
    site, store, start = _site(days=14, start_day=11)
    end = _local_midnight(dt.date(2026, 10, 26))
    assert dt.datetime.fromtimestamp(start, ZONE).date() == dt.date(2026, 10, 25)
    _run(site, start, (end - start) // 60)
    new = store.newest_bookings(start, end)
    rows = []
    for q in range(start, end, 900):
        h, s = new[(q, "house")], new[(q, "shadow")]
        more_in = s["import_kwh"] > h["import_kwh"]
        more_out = s["export_kwh"] > h["export_kwh"]
        rows.append(
            {
                "quarter_ts": q, "take": -2000.0 if more_in else 2000.0,
                "feed": 2000.0 if more_out else -2000.0, "regulation_state": 1,
                "published_at": end + 3600, "source": "tennet",
            }
        )
    assert len(site.settle(rows, end + 7200)) == 100
    res = _results(store, start, end)
    assert all(r >= 0.0 for r in res.values()) and len(res) == 100
    assert any(r > 0.0 for r in res.values())
    sums = R.hourly_sums(store, start, end)
    assert len(sums) == 25 and [h for h, _ in sums] == list(range(start, end, 3600))
    values = [v for _, v in sums]
    assert all(b >= a for a, b in zip(values, values[1:]))
    prev = 0.0
    for h, v in sums:
        assert v - prev == pytest.approx(sum(res[q] for q in range(h, h + 3600, 900)))
        prev = v
    assert values[-1] == pytest.approx(R.result_totals(store, end).settled_eur)


def test_hourly_sums_start_from_the_first_booking(monkeypatch):
    _negative_take(monkeypatch)  # the shadow charges: results are not 0
    site, store, start = _site()
    _run(site, start, 8 * 15)
    site.settle(_settle_rows(store, start, range(8), take=300.0), start + 90000)
    res = _results(store, start, start + 7200)
    late = R.hourly_sums(store, start + 3600, start + 7200)
    assert late == [(start + 3600, pytest.approx(sum(res.values())))]


def test_the_provisional_price_ignores_a_live_row_stored_after_the_booking():
    # the booking at qts + 840 reads live rows labelled at or before qts + 780 only (known by
    # qts + 840): a row labelled qts + 840 that lands after the booking changes nothing, live
    # and re-executed
    site, store, start = _site(live_until=_START + 840)
    _run(site, start, 15)
    booked = store.newest_bookings(start, start + 900)[(start, "house")]
    store.put_live_minutes(
        [{"ts": start + 840, "buy_live": 9999.0, "sell_live": 9999.0, "state": 2, "source": "x"}]
    )
    last = store.live_minutes(start + 780, start + 840)
    assert booked["buy"] == float(last["buy_live"][0]) and booked["state"] == 0
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, start + 900, out)
    assert r.differences == [] and out.head() == store.head()


def test_settle_lands_byte_identical_in_reexecution_mid_quarter(monkeypatch):
    _negative_take(monkeypatch)  # the shadow charges: results are not 0
    site, store, start = _site()
    _run(site, start, 15 * 4 + 7)  # a settle event at elapsed 7 of the fifth quarter
    at = site.next_minute
    assert site.settle(_settle_rows(store, start, range(4), published=at - 600), at) == [
        start + 900 * k for k in range(4)
    ]
    for ts in range(at, at + 60 * 38, 60):
        if ts % 900 == 60:
            site.setup_quarter(ts)
        site.on_minute(ts, dict(_flows(ts), ts=ts))
    end = at + 60 * 38
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, end, out)
    assert r.differences == [] and out.head() == store.head()
    assert r.n_bookings == len(store.bookings(start, end)) == (7 + 4) * 3


def _fail_event(monkeypatch, store):
    def boom(*args, **kwargs):
        raise RuntimeError("power cut before the event")

    monkeypatch.setattr(store, "_insert_event", boom)


def test_a_failure_at_the_settle_event_leaves_neither_bookings_nor_event(monkeypatch):
    site, store, start = _site()
    _run(site, start, 2 * 15)
    head, n_events = store.head(), len(store.events())
    rows = _settle_rows(store, start, range(2))
    with monkeypatch.context() as m:
        _fail_event(m, store)
        with pytest.raises(RuntimeError, match="power cut"):
            site.settle(rows, start + 90000)
    assert store.head() == head and len(store.events()) == n_events
    assert all(b["price_source"] == "provisional" for b in store.bookings(start, start + 1800))
    # the rows were stored: the next settle books them, with the event
    assert site.settle(rows, start + 90060) == [start, start + 900]
    assert store.last_event("settle")["ts"] == start + 90060


def test_a_failure_at_the_restart_event_leaves_neither_gap_bookings_nor_event(monkeypatch):
    site, store, start = _site()
    _run(site, start, 15)
    head, n_events = store.head(), len(store.events())
    site2 = LiveSite(store, site.lc)
    with monkeypatch.context() as m:
        _fail_event(m, store)
        with pytest.raises(RuntimeError, match="power cut"):
            site2.restore(start + 5 * 900)
    assert store.head() == head and len(store.events()) == n_events
    site2.restore(start + 5 * 900)
    assert len(store.bookings(start + 900, start + 5 * 900)) == 4 * 3
    assert store.events()[-1]["kind"] == "restart" and site2.epoch == store.events()[-1]["seq"]


def test_reexecution_books_the_settlement_row_the_event_names(monkeypatch):
    # a correction stored after the event, published before it: live never saw it, so
    # re-execution must not either
    _negative_take(monkeypatch)
    site, store, start = _site()
    _run(site, start, 2 * 15)
    at = site.next_minute
    site.settle(_settle_rows(store, start, range(2), published=at - 600), at)
    _run(site, at, 15)
    store.put_settlement(
        {
            "quarter_ts": start, "take": 999.0, "feed": 999.0, "regulation_state": 2,
            "published_at": at - 300, "source": "tennet",
        }
    )
    out = Store(":memory:")
    out.migrate()
    r = RX.reexecute(store, site.lc, start, at + 900, out)
    assert r.differences == [] and out.head() == store.head()
