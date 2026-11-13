# tests/test_live_site_cache.py
"""The view's minute-quarter cache (``LiveSite._minute_quarters``) gives the full read's table."""

import datetime as dt

import numpy as np
from betterybalance.core.site.flows import to_quarters
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import site as LS
from betterybalance.replay.reexec import _SplitStore

from tests.test_live_site import _flows, _run, _site


def _full(store, lo, hi):
    m = store.minutes(lo, hi)
    return to_quarters(m) if len(m["ts"]) else {"ts": np.zeros(0, dtype=np.int64)}


def _same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.dtype != y.dtype or x.shape != y.shape:
            return False
        if x.dtype.kind == "f":
            if not np.array_equal(x, y, equal_nan=True):
                return False
        elif not np.array_equal(x, y):
            return False
    return True


class _Checked(LS.LiveSite):
    """Every cached read is compared with a full read of the store; the spans aggregated."""

    def __init__(self, store, lc):
        super().__init__(store, lc)
        self.reads, self.bad, self.spans = 0, [], []

    def _minute_quarters(self, lo, hi):
        got = super()._minute_quarters(lo, hi)
        self.reads += 1
        if not _same(got, _full(self.store, lo, hi)):
            self.bad.append(hi)
        return got

    def _aggregate(self, lo, hi):
        self.spans.append(hi - lo)
        return super()._aggregate(lo, hi)


class _Uncached(LS.LiveSite):
    def _minute_quarters(self, lo, hi):
        return _full(self.store, lo, hi)


def _imp(ts: int) -> float:
    """Import that changes per quarter and per day (0,30 to 1,10 kW), per minute in kWh."""
    return (0.3 + 0.1 * ((ts // 900) % 7) + 0.05 * ((ts // 86400) % 5)) / 60


DST = {"days": 14, "start_day": 11}  # start: local midnight of 2026-10-25, the 25-hour day


def _seeded(**kw):
    """``_site`` plus 8 days of minute rows before ``start``, so the load median reads them."""
    site, store, start = _site(**kw)
    for t in range(start - 8 * 86400, start, 60):
        store.put_minute(dict(_flows(t, imp=_imp(t)), ts=t))
    return site, store, start


def _checked(**kw):
    site, store, start = _seeded(**kw)
    chk = _Checked(store, site.lc)
    chk.restore(start)
    return chk, store, start


def _gappy(start):
    """``_imp``, a missing minute every 97 minutes and a missing quarter on day 2."""
    def flows(ts):
        k = (ts - start) // 60
        if k % 97 == 5 or 1440 + 300 <= k < 1440 + 315:
            return None
        return _flows(ts, imp=_imp(ts))
    return flows


def test_cached_view_equals_the_full_read_over_two_days_with_missing_minutes():
    site, store, start = _checked(**DST)
    assert dt.datetime.fromtimestamp(start, ZONE).date() == dt.date(2026, 10, 25)
    _run(site, start, 2 * 1440, flows=_gappy(start))
    assert site.reads == 2 * 96 and site.bad == []
    # after the first read every setup aggregates only the minutes of the last quarter
    assert site.spans[0] == LS.EV_DAYS * LS.DAY_S and set(site.spans[1:]) == {900}


def test_cached_and_uncached_runs_chain_the_same_rows():
    heads, windows = [], []
    for cls in (LS.LiveSite, _Uncached):
        s, store, start = _seeded(**DST)
        site = cls(store, s.lc)
        site.restore(start)
        _run(site, start, 25 * 60 + 60, flows=_gappy(start))  # the 25-hour day and one hour
        heads.append(store.head())
        windows.append([store.window(q) for q in range(start, start + 26 * 3600, 900)])
    assert heads[0] == heads[1]
    for a, b in zip(*windows, strict=True):
        assert a.keys() == b.keys()
        for k in a:
            if isinstance(a[k], np.ndarray):
                assert np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k
            else:
                assert a[k] == b[k], k


def test_restore_drops_the_cache_and_the_view_stays_the_full_read():
    site, store, start = _checked()
    _run(site, start, 8 * 15)
    assert site._qcache is not None
    later = start + 11 * 900 + 60  # three quarters booked missing, minutes from quarter 11
    site.restore(later)
    assert site._qcache is None
    n = len(site.spans)
    q = start + 11 * 900
    for ts in range(q, q + 4 * 900, 60):
        if (ts - q) % 900 == 60:
            site.setup_quarter(ts)
        site.on_minute(ts, dict(_flows(ts), ts=ts))
    assert site.spans[n] == LS.EV_DAYS * LS.DAY_S and site.bad == []


def test_adopting_a_refit_drops_the_cache():
    site, store, start = _checked()
    _run(site, start, 4 * 15)
    n = 96 * 3
    ts = start - n * 900 + 900 * np.arange(n, dtype=np.int64)
    irr = {
        "ts": ts, "ghi_w_m2": np.zeros(n), "dni_w_m2": np.zeros(n), "dhi_w_m2": np.zeros(n),
        "temp_c": np.full(n, 10.0),
    }
    site.adopt_refit(site.fit(start + 3600, irr), start + 3600, irr)
    assert site._qcache is None
    k = len(site.spans)
    for t in range(start + 3600, start + 3600 + 2 * 900, 60):
        if (t - start) % 900 == 60:
            site.setup_quarter(t)
        site.on_minute(t, dict(_flows(t), ts=t))
    assert site.spans[k] == LS.EV_DAYS * LS.DAY_S and site.bad == []


def test_a_minute_written_behind_the_cache_rebuilds_it():
    site, store, start = _checked()
    _run(site, start, 3 * 15, flows=lambda ts: None if ts == start + 420 else _flows(ts))
    view = site._minute_quarters(start - LS.EV_DAYS * LS.DAY_S, start + 2700)
    assert bool(view["missing"][view["ts"] == start][0])
    store.put_minute(dict(_flows(start + 420), ts=start + 420))  # a late row for that minute
    n = len(site.spans)
    view = site._minute_quarters(start - LS.EV_DAYS * LS.DAY_S, start + 2700)
    assert site.spans[n:] == [LS.EV_DAYS * LS.DAY_S + 2700]  # rebuilt from a full read
    assert not bool(view["missing"][view["ts"] == start][0]) and site.bad == []


def test_an_earlier_span_rebuilds_and_returned_tables_are_copies():
    site, store, start = _checked()
    _run(site, start, 3 * 15)
    lo = start - LS.EV_DAYS * LS.DAY_S
    got = site._minute_quarters(lo + 900, start + 1800)  # ends before the cached end
    got["grid_import_kwh"][:] = -1.0  # a caller writing into the table
    got.pop("missing")
    again = site._minute_quarters(lo + 900, start + 2700)
    assert site.bad == [] and _same(again, _full(store, lo + 900, start + 2700))


def test_a_span_off_the_quarter_grid_is_a_full_read_that_keeps_the_cache():
    site, store, start = _checked()
    _run(site, start, 2 * 15)
    cached = site._qcache
    lo = start - LS.EV_DAYS * LS.DAY_S
    got = site._minute_quarters(lo, start + 900 + 300)  # ends inside a quarter
    assert site._qcache is cached and site.bad == []
    assert int(got["ts"][-1]) == start + 900 and bool(got["missing"][-1])  # 5 of 15 minutes


def test_reexecution_store_passes_the_minute_count_through():
    site, store, start = _checked()
    _run(site, start, 20)
    split = _SplitStore(store, Store(":memory:"), 0, [])
    lo = start - LS.EV_DAYS * LS.DAY_S
    assert split.minute_count(lo, start + 1200) == store.minute_count(lo, start + 1200) > 0
