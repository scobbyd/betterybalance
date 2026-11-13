# tests/test_live_site.py
import datetime as dt
import math
from dataclasses import replace

import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core import ledger as LG
from betterybalance.core import pricemodel as PM
from betterybalance.core.site import contract as C
from betterybalance.core.site.flows import ac_legs
from betterybalance.core.store import Store
from betterybalance.core.timegrid import ZONE
from betterybalance.live import site as LS

from tests.test_features import _synthetic_master

CFG = C.from_dict(
    {
        "latitude": 52.1,
        "longitude": 5.2,
        "pack": {
            "capacity_kwh": 48,
            "charge_kw": 12,
            "discharge_kw": 12,
            "soc_floor": 0.1,
            "soc_ceiling": 1.0,
        },
        "pv": {"none": True},
    }
)
FIRST_DAY = dt.date(2026, 10, 14)  # the synthetic master's first local day
_LIVE = (
    "buy_live_end", "sell_live_end", "buy_live_e1", "sell_live_e1",
    "buy_live_e5", "sell_live_e5", "buy_live_e10", "sell_live_e10",
)
_MODELS: dict = {}


def _local_midnight(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=ZONE).timestamp())


def _model(days: int, start_day: int):
    key = (days, start_day)
    if key not in _MODELS:
        m = _synthetic_master(days=days)
        train = np.zeros(len(m["ts"]), bool)
        train[: 96 * (start_day - 1)] = True
        _MODELS[key] = PM.fit_model(m, ("A", "C"), F.HORIZONS, train_mask=train)
    return _MODELS[key]


def _flows(ts, imp=0.5 / 60, charge=0.0):
    return {
        "grid_import_kwh": imp, "grid_export_kwh": 0.0, "batt_charge_kwh": charge,
        "batt_discharge_kwh": 0.0, "soc_pct": 50.0,
    }


def _site(store=None, days=30, start_day=21, live_until=None):
    """A store with `days` days of synthetic features, settlements only for quarters before `start`,
    hours as flows_hour, a model fitted on days [0, start_day - 1).

    Feature rows of `start` and later carry the day-ahead and IDA prices only (their live values
    are written by the site as it runs); live minutes (da + 1, state 0) run from one hour before
    `start` to `live_until` (default: the master's end). Flat 0,5 kW import, SoC 50 %.
    """
    m = _synthetic_master(days=days)
    ts = m["ts"]
    start = _local_midnight(FIRST_DAY + dt.timedelta(days=start_day))
    store = store or Store(":memory:")
    store.migrate()
    store.set_meta("install_id", "install-1")
    for i, t in enumerate(ts.tolist()):
        cols = {"da_eur_mwh": m["da_eur_mwh"][i]}
        cols.update({f"ida{k}_eur_mwh": m[f"ida{k}_eur_mwh"][i] for k in (1, 2, 3)})
        if t < start:
            cols.update({c: m[c][i] for c in _LIVE})
            store.put_settlement(
                {
                    "quarter_ts": t, "take": m["buy_eur_mwh"][i], "feed": m["sell_eur_mwh"][i],
                    "regulation_state": int(m["regulation_state"][i]), "published_at": t + 86400,
                    "source": "synthetic",
                }
            )
        store.upsert_features(t, **cols)
    hours = np.arange(int(ts[0]), start, 3600, dtype=np.int64)
    n = len(hours)
    store.put_hours(
        {
            "ts": hours, "step_s": np.full(n, 3600), "grid_import_kwh": np.full(n, 0.5),
            "grid_export_kwh": np.zeros(n), "batt_charge_kwh": np.zeros(n),
            "batt_discharge_kwh": np.zeros(n), "soc_pct": np.full(n, 50.0),
            "soc_min_pct": np.full(n, 50.0), "soc_max_pct": np.full(n, 50.0),
        }
    )
    end = int(ts[-1]) + 900 if live_until is None else live_until
    rows = []
    for t in range(start - 3600, end, 60):
        k = int(np.searchsorted(ts, t - t % 900))
        p = float(m["da_eur_mwh"][min(k, len(ts) - 1)]) + 1.0
        rows.append({"ts": t, "buy_live": p, "sell_live": p, "state": 0, "source": "synthetic"})
    store.put_live_minutes(rows)
    lc = LS.LiveConfig(CFG, "install-1", _model(days, start_day), "v-test", pack_version="p1")
    site = LS.LiveSite(store, lc)
    site.restore(start)
    return site, store, start


def _run(site, start, minutes, flows=lambda ts: _flows(ts)):
    out = []
    for ts in range(start, start + 60 * minutes, 60):
        if (ts - start) % 900 == 60:
            site.setup_quarter(ts)
        f = flows(ts)
        out.append(site.on_minute(ts, None if f is None else dict(f, ts=ts)))
    return out


def test_two_days_book_three_paths_and_chain_verifies():
    site, store, start = _site()
    _run(site, start, 2 * 1440)
    b = store.bookings(start, start + 2 * 86400)
    assert len(b) == 2 * 96 * 3 and {r["path"] for r in b} == {"shadow", "house", "actual"}
    assert [r["path"] for r in b[:3]] == ["shadow", "house", "actual"]
    assert all(r["price_source"] == "provisional" and not r["missing"] for r in b)
    items = store.chain(1, store.head()[0])
    assert LG.verify_chain(items, LG.genesis_hash("install-1")).ok
    assert 0 < len(store.decisions(start, start + 2 * 86400)) < 2 * 1440
    # every input of the day is in the store: a window per quarter, a nowcast per minute
    assert store.window(start + 60 * 15) is not None and store.window(start + 900)["now"] == (
        start + 960
    )
    assert len(store.nowcasts(start, start + 86400)) == 1440
    feats = store.features(start, start + 900)
    assert np.isfinite(feats["buy_live_end"][0]) and np.isfinite(feats["sell_live_e10"][0])


def test_missing_minute_holds_and_books_missing():
    site, store, start = _site()

    def flows(ts):
        return None if ts == start + 7 * 60 else _flows(ts)

    out = _run(site, start, 15, flows=flows)
    assert out[7].decision.reason == "missing" and out[7].decision.action == "hold"
    assert store.minutes(start + 420, start + 480)["ts"].size == 0  # absent row = missing
    assert len(out[14].booked) == 3
    assert all(r["missing"] for r in store.bookings(start, start + 900))


def test_on_minute_refuses_a_minute_that_is_not_the_next():
    site, store, start = _site()
    _run(site, start, 3)
    with pytest.raises(ValueError):
        site.on_minute(start + 4 * 60, dict(_flows(0), ts=start + 4 * 60))


def test_restart_mid_quarter_books_whole_gap_missing_once():
    site, store, start = _site()
    _run(site, start, 15 + 7)
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 900 * 3 + 60)
    quarters = [
        (r["quarter_ts"], r["path"])
        for r in store.bookings(start, start + 4 * 900)
        if r["price_source"] != "settled"
    ]
    assert len(quarters) == len(set(quarters)) == 3 * 3
    assert store.events()[-1]["kind"] == "restart"
    gap = store.bookings(start + 900, start + 2700)
    assert all(r["missing"] for r in gap)
    # the next minute is the start of the running quarter, fed as missing
    with pytest.raises(ValueError):
        site2.on_minute(start + 900 * 3 + 60, None)
    for ts in range(start + 2700, start + 3600, 60):
        site2.on_minute(ts, None)
    assert len(store.bookings(start + 2700, start + 3600)) == 3
    assert LG.verify_chain(
        store.chain(1, store.head()[0]), LG.genesis_hash("install-1")
    ).ok


def test_a_failed_booking_writes_no_row_of_the_triple(monkeypatch):
    site, store, start = _site()
    _run(site, start, 14)
    head = store.head()
    real, calls = store._insert, []

    def failing(table, row, columns, verb="INSERT OR REPLACE"):
        if table == "bookings":
            calls.append(row["path"])
            if len(calls) == 2:
                raise RuntimeError("crash after the shadow row")
        return real(table, row, columns, verb)

    monkeypatch.setattr(store, "_insert", failing)
    with pytest.raises(RuntimeError):
        site.on_minute(start + 14 * 60, dict(_flows(0), ts=start + 14 * 60))
    monkeypatch.setattr(store, "_insert", real)
    assert store.bookings(start, start + 900) == []
    after = store.chain(head[0] + 1, store.head()[0])  # at most the minute's decision row
    assert len(after) <= 1 and all(kind == "decision" for kind, _ in after)
    # a restart books the quarter on all three paths, once
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 900 + 60)
    b = store.bookings(start, start + 900)
    assert [r["path"] for r in b] == ["shadow", "house", "actual"] and all(r["missing"] for r in b)


def test_a_restart_before_the_first_booking_books_the_partial_first_quarter():
    site, store, start = _site()
    _run(site, start, 7)
    assert store.head() is not None and store.bookings(start, start + 86400) == []
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 3 * 900 + 60)
    b = store.bookings(start, start + 86400)
    keys = [(r["quarter_ts"], r["path"]) for r in b]
    assert len(keys) == len(set(keys)) == 3 * 3
    assert {q for q, _ in keys} == {start, start + 900, start + 1800}
    assert all(r["missing"] for r in b)
    site3 = LS.LiveSite(store, site.lc)  # restored again: nothing is booked twice
    site3.restore(start + 3 * 900 + 120)
    assert len(store.bookings(start, start + 86400)) == 9


def _fresh(soc_sensor_pct):
    """A site on a store without any run, restored with the SoC sensor's current value."""
    store = Store(":memory:")
    store.migrate()
    store.set_meta("install_id", "install-1")
    lc = LS.LiveConfig(CFG, "install-1", _model(30, 21), "v-test", pack_version="p1")
    site = LS.LiveSite(store, lc)
    start = _local_midnight(FIRST_DAY + dt.timedelta(days=21))
    site.restore(start, soc_sensor_pct)
    return site, store, start


def test_the_first_restore_anchors_at_the_soc_sensor():
    site, store, _ = _fresh(100.0)
    assert site.stepper.battery.soc_kwh == pytest.approx(48.0)
    ev = store.last_event("restart")
    assert ev["detail"]["soc_kwh"] == pytest.approx(48.0)
    assert ev["detail"]["soc_sensor_pct"] == 100.0


@pytest.mark.parametrize("sensor", [None, math.nan, -3.0, 140.0])
def test_without_a_valid_soc_sensor_the_first_restore_anchors_at_the_floor(sensor):
    site, store, _ = _fresh(sensor)
    assert site.stepper.battery.soc_kwh == pytest.approx(4.8)
    assert store.last_event("restart")["detail"]["soc_sensor_pct"] is None


def test_a_shadow_that_never_stepped_re_anchors_at_the_soc_sensor():
    # every minute missing: the shadow never stepped, its booked SoC is the first anchor
    site, store, start = _fresh(None)
    _run(site, start, 2 * 15, flows=lambda ts: None)
    assert store.bookings(start, start + 1800)[0]["soc_pct"] == pytest.approx(10.0)
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 3600, 80.0)
    assert site2.stepper.battery.soc_kwh == pytest.approx(0.8 * 48)
    assert store.last_event("restart")["detail"]["soc_kwh"] == pytest.approx(0.8 * 48)
    # the gap quarters are booked at the new anchor; the chain still verifies
    gap = [r for r in store.bookings(start + 1800, start + 3600) if r["path"] == "shadow"]
    assert len(gap) == 2 and all(r["soc_pct"] == pytest.approx(80.0) for r in gap)
    assert LG.verify_chain(store.chain(1, store.head()[0]), LG.genesis_hash("install-1")).ok
    # without a sensor value it keeps its booked SoC, not the floor of a fresh start
    site3 = LS.LiveSite(store, site.lc)
    site3.restore(start + 4500, None)
    assert site3.stepper.battery.soc_kwh == pytest.approx(0.8 * 48)


def test_a_shadow_that_stepped_keeps_its_booked_soc_over_the_sensor():
    site, store, start = _site()
    _run(site, start, 15)
    booked = [r for r in store.bookings(start, start + 900) if r["path"] == "shadow"][0]
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 1800, 95.0)
    assert site2.stepper.battery.soc_kwh == pytest.approx(booked["soc_pct"] / 100 * 48)


def test_restart_after_setup_ranks_the_running_quarter_from_its_stored_window():
    site, store, start = _site()
    _run(site, start, 15 + 7)  # quarter 2 was set up at its minute 1
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start + 900 + 7 * 60)
    ranked = site2.status()["ranked"]
    assert ranked is not None and int(ranked.window.ts[0]) == start + 900
    assert site.ranked.window.take == pytest.approx(ranked.window.take, nan_ok=True)
    for ts in range(start + 900, start + 1800, 60):  # fed as missing, the rereads still run
        site2.on_minute(ts, None)
    assert len(store.bookings(start + 900, start + 1800)) == 3


def test_dst_end_day_has_100_quarters():
    # the synthetic master starts 2026-10-14; start is local midnight 2026-10-25
    site, store, start = _site(days=14, start_day=11)
    assert dt.datetime.fromtimestamp(start, ZONE).date() == dt.date(2026, 10, 25)
    end = _local_midnight(dt.date(2026, 10, 26))
    assert end - start == 25 * 3600
    _run(site, start, (end - start) // 60)
    assert len(store.bookings(start, end)) == 300
    greed = store._rows("SELECT day FROM greed")
    assert [g["day"] for g in greed] == ["2026-10-25"]


def _negative_take(monkeypatch):
    """Every ranking's nowcast takes at -200 EUR/MWh, so a fresh feed commits at minute 1."""
    real = LS.rank_quarter

    def rank(*args, **kwargs):
        r = real(*args, **kwargs)
        r.nowcast = replace(r.nowcast, take_point=-200.0, take_q50=-200.0)
        return r

    monkeypatch.setattr(LS, "rank_quarter", rank)


@pytest.mark.parametrize("stale", [False, True])
def test_stale_live_feed_blocks_new_commits(monkeypatch, stale):
    _negative_take(monkeypatch)
    q = 4  # the quarter under test
    target = _local_midnight(FIRST_DAY + dt.timedelta(days=21)) + q * 900
    site, store, start = _site(live_until=target - 600 if stale else None)
    assert target == start + q * 900
    _run(site, start, (q + 1) * 15)
    window = store.decisions(target, target + 900)
    nc = store.nowcasts(target, target + 900)
    if not stale:  # the control: the same quarter commits with a fresh feed
        assert any(d["committed"] and d["reason"] == "negative_take" for d in window)
        assert all(r["live_age_s"] == 0 for r in nc.values())
        return
    # live_minutes end 10 minutes before the quarter: no nowcast, no commit
    assert window
    assert all(d["reason"] in ("wait", "default", "late") for d in window)
    assert not any(d["committed"] for d in window)
    assert all(math.isnan(r["take_point"]) for r in nc.values())
    assert all(r["live_age_s"] is None or r["live_age_s"] > 180 for r in nc.values())
    assert "live_stale" in site.status()["flags"]


def test_house_books_grid_minus_battery_and_a_missing_quarter_leaves_the_daily_result():
    site, store, start = _site()

    def flows(ts):
        # 0,5 kW import while the real battery charges 2 kW; quarter 1 has a missing minute
        if ts == start + 900 + 5 * 60:
            return None
        return _flows(ts, imp=2.5 / 60, charge=2.0 / 60)

    _run(site, start, 3 * 15, flows=flows)
    new = store.newest_bookings(start, start + 3 * 900)
    ec, _ = ac_legs(CFG.pack)  # a dc-side pack: the AC charge is the DC charge over the leg
    for qts in (start, start + 1800):
        house, actual = new[(qts, "house")], new[(qts, "actual")]
        assert house["import_kwh"] == pytest.approx((2.5 - 2.0 / ec) * 0.25)
        assert house["batt_charge_kwh"] == 0.0 and house["batt_discharge_kwh"] == 0.0
        assert actual["import_kwh"] == pytest.approx(2.5 * 0.25)
        assert actual["batt_charge_kwh"] == pytest.approx(2.0 / ec * 0.25)
    assert new[(start + 900, "house")]["missing"] and new[(start + 900, "shadow")]["missing"]
    results = [
        LG.quarter_result(new[(q, "house")], new[(q, "shadow")])
        for q in (start, start + 900, start + 1800)
    ]
    assert math.isnan(results[1]) and all(math.isfinite(r) for r in (results[0], results[2]))
    day = LS.daily_result(store, dt.datetime.fromtimestamp(start, ZONE).date())
    assert day["n_missing"] == 1
    assert day["result_eur"] == pytest.approx(results[0] + results[2])


def test_refit_stores_the_site_state_and_restore_reads_it():
    site, store, start = _site()
    n = 96 * 10
    ts = start - n * 900 + 900 * np.arange(n, dtype=np.int64)
    irr = {
        "ts": ts, "ghi_w_m2": np.zeros(n), "dni_w_m2": np.zeros(n), "dhi_w_m2": np.zeros(n),
        "temp_c": np.full(n, 10.0),
    }
    state = site.refit(start, irr)
    assert state.learning is False and store.registry("site_state") is not None
    assert store.events()[-1]["kind"] == "refit"
    site2 = LS.LiveSite(store, site.lc)
    site2.restore(start)
    st = site2.status()
    assert st["learning"] is False and st["drift"] is False
    assert st["head"] == store.head()
    assert set(st["flags"]) <= set(LS.FLAGS)


def test_fit_writes_nothing_and_adopting_it_later_stamps_the_refit_there():
    site, store, start = _site()
    n = 96 * 10
    ts = start - n * 900 + 900 * np.arange(n, dtype=np.int64)
    irr = {
        "ts": ts, "ghi_w_m2": np.zeros(n), "dni_w_m2": np.zeros(n), "dhi_w_m2": np.zeros(n),
        "temp_c": np.full(n, 10.0),
    }
    def written():
        return store.registry("site_state"), len(store.events()), store.fetch_at(2**40, "archive")

    before, old = written(), site.state
    state = site.fit(start, irr)
    assert written() == before
    assert site.state is old and state.learning is False and state.last_fit_ts is not None
    at = start + 600  # the minute the live loop processes next when the fit is adopted
    site.adopt_refit(state, at, irr)
    ev = store.events()[-1]
    assert ev["kind"] == "refit" and ev["ts"] == at and site.state == state
    assert store.fetch_at(at, "archive")[0] == at and store.fetch_at(at - 1, "archive") is None


def _string_hours(site, store, start):
    """pv1 (35/135, 4,0 kWp) and the GEN port (15/180, 1,0 kWp) in the hourly backfill, from a
    seeded clear-sky irradiance (clearness 0,3 to 1,0 per day) of the backfill window."""
    from betterybalance.core.site import solar as S
    from betterybalance.core.site.plant import plane_shape

    ts = np.arange(_local_midnight(FIRST_DAY), start, 900, dtype=np.int64)
    z, _ = S.solar_position(ts, CFG.latitude, CFG.longitude)
    kt = np.random.default_rng(5).uniform(0.3, 1.0, len(ts) // 96 + 2)[(ts - ts[0]) // 86400]
    ghi = S.clear_sky_ghi(z) * kt
    dhi = ghi * np.where(kt > 0.6, 0.25, 0.7)
    irr = {
        "ts": ts, "ghi_w_m2": ghi, "dhi_w_m2": dhi, "temp_c": np.full(len(ts), 10.0),
        "dni_w_m2": np.where(z < 89, (ghi - dhi) / np.cos(np.radians(np.minimum(z, 89.0))), 0.0),
    }
    lat, lon = CFG.latitude, CFG.longitude
    pv1 = 4.0 * plane_shape(irr, C.Plane(35, 135), lat, lon) / 4.0
    gen = 1.0 * plane_shape(irr, C.Plane(15, 180), lat, lon) / 4.0
    n = len(pv1) // 4
    store.put_hours(
        {
            "ts": irr["ts"][::4], "step_s": np.full(n, 3600),
            "pv1_kwh": pv1.reshape(n, 4).sum(axis=1), "gen_kwh": gen.reshape(n, 4).sum(axis=1),
        }
    )
    return irr


def test_refit_fits_the_strings_and_the_gen_port_of_the_hourly_backfill():
    site, store, start = _site()
    site.lc = replace(site.lc, cfg=replace(CFG, pv=C.PvHints()))
    irr = _string_hours(site, store, start)
    state = site.fit(start, irr)
    m = state.challenger
    assert [p.string for p in m.planes] == ["pv1", "gen"]
    assert m.scales_kwp.tolist() == pytest.approx([4.0, 1.0], rel=0.1)
    assert site.scale_off(state) is None  # nothing entered: no verdict
    # the live view still reads the GEN port as 0: the site models learn all PV
    assert (site._quarters(start - 86400, start)["gen_kwh"] == 0.0).all()


def test_scale_off_names_the_entered_strings_that_disagree():
    site, store, start = _site()
    planes = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(15, 180, 2.0, "gen"))
    site.lc = replace(site.lc, cfg=replace(CFG, pv=C.PvHints(planes=planes)))
    irr = _string_hours(site, store, start)
    state = site.fit(start, irr)
    assert state.challenger.planes == planes
    assert site.scale_off(state) == "GEN"
    site.lc = replace(site.lc, cfg=replace(CFG, pv=C.PvHints(planes=planes[:1])))
    assert site.scale_off(state) == ""


def test_a_blind_refit_gives_no_scale_verdict():
    site, store, start = _site()
    planes = (C.Plane(35, 135, 4.0, "pv1"), C.Plane(15, 180, 2.0, "gen"))
    site.lc = replace(site.lc, cfg=replace(CFG, pv=C.PvHints(planes=planes)))
    irr = _string_hours(site, store, start)
    state = site.fit(start, irr)
    assert state.challenger.supervised and site.scale_off(state) == "GEN"
    # the same planes and scales from the blind path: no verdict, neither raise nor clear
    blind = replace(state, challenger=replace(state.challenger, supervised=False))
    assert site.scale_off(blind) is None
    assert site.scale_off(replace(state, challenger=None)) is None
