# tests/test_backtest.py
import datetime as dt
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core import ledger as LG
from betterybalance.core import pack as PK
from betterybalance.core import pricemodel as PM
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import backtest as BT
from betterybalance.replay import bands as RB
from betterybalance.replay.irradiance import IrrByLead
from betterybalance.replay.phases import IRR_COLS

from tests.test_features import _synthetic_master
from tests.test_live_site import FIRST_DAY, _local_midnight, _run, _site
from tests.test_pack import _quarters

DAY_S = 86400


def _hours(first: int, n_days: int, imp: float = 0.5) -> dict:
    ts = np.arange(first, first + n_days * DAY_S, 3600, dtype=np.int64)
    n = len(ts)
    return {
        "ts": ts, "step_s": np.full(n, 3600), "grid_import_kwh": np.full(n, imp),
        "grid_export_kwh": np.zeros(n), "batt_charge_kwh": np.zeros(n),
        "batt_discharge_kwh": np.zeros(n), "soc_pct": np.full(n, 50.0),
        "soc_min_pct": np.full(n, 50.0), "soc_max_pct": np.full(n, 50.0),
    }


def _stub_pack(first: int, last: int, train_end: int):
    """What plan_backtest reads of a pack: the training ends and the quarter grid."""
    return SimpleNamespace(
        train_ends=np.array([train_end], dtype=np.int64),
        quarters={"ts": np.arange(first, last, 900, dtype=np.int64)},
    )


def _day(ts: int) -> dt.date:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date()


def test_plan_caps_and_reasons():
    first = _local_midnight(dt.date(2025, 1, 1))
    hours = _hours(first, 500)
    now = first + 500 * DAY_S + 5 * 3600  # 05:00 local the day after the last hour
    pack = _stub_pack(first - DAY_S, now, first - 30 * DAY_S)
    plan = BT.plan_backtest(hours, pack, now)
    assert plan.reason is None and plan.days == BT.MAX_DAYS == 365
    assert plan.end == _local_midnight(_day(now))  # the whole days before today
    assert (_day(plan.end) - _day(plan.start)).days == 365
    # ten days of hours are too short
    short = BT.plan_backtest(_hours(first, 10), pack, first + 10 * DAY_S + 3600)
    assert short.reason == "short_history" and short.days < BT.MIN_DAYS
    # no pack: refused before anything else
    assert BT.plan_backtest(hours, None, now).reason == "no_pack"
    # the start waits for the first band row the pack can price
    late = _stub_pack(first - DAY_S, now, first + 400 * DAY_S)
    lp = BT.plan_backtest(hours, late, now)
    priced = first + 400 * DAY_S + RB.embargo_lag(F.HORIZONS)
    assert lp.start >= priced and lp.start - priced < DAY_S
    assert lp.reason is None and lp.days == (_day(lp.end) - _day(lp.start)).days
    # NaN grid import hours do not count: the span is that of the finite ones
    gappy = _hours(first, 500)
    gappy["grid_import_kwh"][: 300 * 24] = np.nan
    gp = BT.plan_backtest(gappy, pack, now)
    assert gp.start == first + 300 * DAY_S  # 2025-10-28 00:00, the same offset as January
    assert gp.days == 200 and gp.end == plan.end
    # no recorded SoC in the hour before the end: the terminal valuation has no anchor
    # (the stored hours run one hour past the end: index -2 is the span's final hour)
    no_soc = _hours(first, 500)
    assert int(no_soc["ts"][-2]) == plan.end - 3600
    no_soc["soc_pct"][-2] = np.nan
    ns = BT.plan_backtest(no_soc, pack, now)
    assert ns.reason == "no_end_soc" and ns.days == 365 and "no_end_soc" in BT.REASONS
    no_soc["soc_pct"][-2], no_soc["soc_pct"][-3] = 50.0, np.nan  # only the final hour counts
    assert BT.plan_backtest(no_soc, pack, now).reason is None


# --- the run on a synthetic site ---------------------------------------------------------


@pytest.fixture(scope="module")
def pack_dir(tmp_path_factory):
    """A synthetic 41-day pack: models trained to days 3 and 10 on tiers A and C."""
    m = _synthetic_master(days=41)
    series = []
    for d in (3, 10):
        train = np.zeros(len(m["ts"]), bool)
        train[: 96 * d] = True
        series.append(
            (int(m["ts"][96 * d]), PM.fit_model(m, ("A", "C"), F.HORIZONS, train_mask=train))
        )
    out = tmp_path_factory.mktemp("pack")
    PK.write_pack(out, "2026-11", _quarters(m), series, built="2026-11-01")
    return out


def _irr(first: int, last: int) -> dict:
    ts = np.arange(first, last, 900, dtype=np.int64)
    return {"ts": ts, **{c: np.zeros(len(ts)) for c in IRR_COLS}}


def test_backtest_never_touches_the_live_ledger(tmp_path, pack_dir):
    site, store, start = _site()
    pack = PK.load_pack(pack_dir)
    # 20 days of hours in the store: the site's hours to day 21 and more to day 31
    first = _local_midnight(FIRST_DAY)
    store.put_hours(_hours(first, 31))
    now = _local_midnight(FIRST_DAY + dt.timedelta(days=28)) + 3600
    hours = store.hours(0, 2**40)
    full = BT.plan_backtest(hours, pack, now)
    assert full.reason is None and full.days == 20
    # the replay runs on three of those days (a unit-test budget), across the 25-hour day of
    # 2026-10-25 so the per-day progress and the quarter count meet the DST change
    lo = _local_midnight(dt.date(2026, 10, 24))
    plan = replace(full, start=lo, end=_local_midnight(dt.date(2026, 10, 27)), days=3)
    n_quarters = (plan.end - plan.start) // 900
    assert full.start < plan.start and plan.end < full.end and n_quarters == 3 * 96 + 4
    irr = _irr(first - DAY_S, first + 41 * DAY_S)
    by_lead = IrrByLead({d: irr for d in (1, 2, 3)}, irr)
    calls = []
    head = store.head()
    bookings = len(store.bookings(0, 2**40))
    headline, report = BT.run_backtest(
        store, site.lc, pack, plan, irr, by_lead, lambda d, n: calls.append((d, n)),
        store.new_run(start, plan.start, plan.end),
    )
    assert store.head() == head and len(store.bookings(0, 2**40)) == bookings == 0
    assert {"result_eur", "lo_eur_day", "hi_eur_day", "indicative"} <= set(headline)
    assert BT.INDICATIVE_HOURLY in report
    assert "Weather data by Open-Meteo.com (CC BY 4.0)" in report
    # progress once per Amsterdam day, in order, ending at the plan's day count
    assert [c[0] for c in calls] == sorted(c[0] for c in calls)
    assert calls[-1] == (plan.days, plan.days) and len(calls) == plan.days
    # the runs tables hold the three paths per quarter and the non-repeat-hold decisions
    run_id = store.runs()[-1]["run_id"]
    q = store._rows("SELECT path, COUNT(*) AS n FROM run_quarters WHERE run_id = ? GROUP BY path",
                    (run_id,))
    assert {r["path"]: r["n"] for r in q} == dict.fromkeys(("shadow", "house", "actual"),
                                                          n_quarters)
    d = store._rows("SELECT * FROM run_decisions WHERE run_id = ? ORDER BY ts", (run_id,))
    assert 0 < len(d) <= n_quarters * 15
    assert all(r["model_version"].startswith("pack-2026-11-") for r in d)
    # the headline: the sum of the daily result, the interval around the mean, the terminal
    # valuation beside it (not added), JSON-safe values
    assert headline["days"] == plan.days and headline["missing_quarters"] == 0
    assert headline["lo_eur_day"] <= headline["mean_eur_day"] <= headline["hi_eur_day"]
    assert headline["result_eur"] == pytest.approx(headline["mean_eur_day"] * plan.days)
    assert np.isfinite(headline["terminal_eur"]) and headline["learning_days"] == 0
    assert headline["irradiance"].startswith("previous runs")
    assert "EUR" in report and "actual" not in report.lower()


def test_a_stop_ends_the_run_before_it_writes(pack_dir):
    site, store, start = _site()
    pack = PK.load_pack(pack_dir)
    first = _local_midnight(FIRST_DAY)
    store.put_hours(_hours(first, 31))
    now = _local_midnight(FIRST_DAY + dt.timedelta(days=28)) + 3600
    plan = BT.plan_backtest(store.hours(0, 2**40), pack, now)
    irr = _irr(first - DAY_S, first + 41 * DAY_S)
    asked = []

    def stop() -> bool:  # true from the second question on: the first check passes
        asked.append(1)
        return len(asked) > 1

    run_id = store.new_run(start, plan.start, plan.end)
    with pytest.raises(BT.Stopped):
        BT.run_backtest(store, site.lc, pack, plan, irr, IrrByLead({}, irr), lambda d, n: None,
                        run_id, should_stop=stop)
    assert len(asked) == 2  # stopped at the first band cell, long before the replay
    rows = store._rows("SELECT COUNT(*) AS n FROM run_quarters WHERE run_id = ?", (run_id,))
    assert rows[0]["n"] == 0


def _run_rows(run_id: int, n: int, bad_at: int | None = None) -> list[dict]:
    rows = [
        {"quarter_ts": 1_700_000_000 + 900 * i, "path": "shadow", "import_kwh": 0.1,
         "export_kwh": 0.0, "money_eur": 0.01, "missing": False}
        for i in range(n)
    ]
    if bad_at is not None:
        del rows[bad_at]["money_eur"]
    return rows


def test_run_rows_on_their_own_connection_never_mix_with_the_live_chain(tmp_path):
    path = tmp_path / "store.sqlite3"
    live = Store(path)
    site, store, start = _site(store=live)
    bt = Store(path, timeout=60.0)
    try:
        run_id = bt.new_run(start, start, start + DAY_S)
        rows = _run_rows(run_id, 60_000)
        writer = threading.Thread(target=BT._write_runs, args=(bt, run_id, rows, []))
        writer.start()
        _run(site, start, 45)  # three quarters of live minutes while the run rows go in
        writer.join(120)
        assert not writer.is_alive()
        items = live.chain(1, live.head()[0])
        assert LG.verify_chain(items, LG.genesis_hash("install-1")).ok and len(items) > 3
        n = live._rows("SELECT COUNT(*) AS n FROM run_quarters WHERE run_id = ?", (run_id,))
        assert n[0]["n"] == 60_000
        # a failed write removes that run's rows only: the first run and the chain stay
        head = live.head()
        bad = bt.new_run(start, start, start + DAY_S)
        with pytest.raises(KeyError):
            BT._write_runs(bt, bad, _run_rows(bad, 9_000, bad_at=8_500), [])
        counts = live._rows("SELECT run_id, COUNT(*) AS n FROM run_quarters GROUP BY run_id")
        assert {r["run_id"]: r["n"] for r in counts} == {run_id: 60_000}
        assert live.head() == head
    finally:
        bt.close()
        live.close()
