# tests/test_replay_engine.py
import numpy as np
import pytest
from betterybalance.core import features as F
from betterybalance.core.site import contract as C
from betterybalance.replay import bands as BC
from betterybalance.replay import engine as E
from betterybalance.replay import steps as ST

from tests.test_features import _synthetic_master


def _inputs(days_replay=3, missing_minute=None, pv_none=True):
    m = _synthetic_master(days=41)
    ts = m["ts"]
    # the synthetic master steps 86.400 s across the DST end, so day 35 starts at 23:00 local;
    # plus 3.600 s is local midnight 2026-11-18
    start = int(ts[96 * 35]) + 3600
    end = start + 86400 * days_replay
    cache = BC.build_band_cache(
        m,
        int(ts[96 * 30]),
        int(ts[-1]) + 900,
        tiers=("A",),
        horizons=F.HORIZONS,
        refit_every_days=7,
        train_days=20,
    )
    n = len(ts)
    q = {
        "ts": ts,
        "step_s": np.full(n, 900),
        "missing": np.zeros(n, bool),
        "grid_import_kwh": np.full(n, 0.125),
        "grid_export_kwh": np.zeros(n),
        "batt_charge_kwh": np.zeros(n),
        "batt_discharge_kwh": np.zeros(n),
        "soc_pct": np.full(n, 50.0),
        "soc_min_pct": np.full(n, 50.0),
        "soc_max_pct": np.full(n, 50.0),
        "load_kwh": np.full(n, np.nan),
        "gen_kwh": np.full(n, np.nan),
        "pv_kwh": np.full(n, np.nan),
    }
    mts = np.arange(ts[0], ts[-1] + 900, 60, dtype=np.int64)
    mn = len(mts)
    minute = {
        "ts": mts,
        "step_s": np.full(mn, 60),
        "grid_import_kwh": np.full(mn, 0.5 / 60),
        "grid_export_kwh": np.zeros(mn),
        "batt_charge_kwh": np.zeros(mn),
        "batt_discharge_kwh": np.zeros(mn),
        "soc_pct": np.full(mn, 50.0),
        "soc_min_pct": np.full(mn, 50.0),
        "soc_max_pct": np.full(mn, 50.0),
        "load_kwh": np.full(mn, np.nan),
        "gen_kwh": np.full(mn, np.nan),
        "pv_kwh": np.full(mn, np.nan),
    }
    if missing_minute is not None:
        for k in minute:
            if k != "ts" and k != "step_s":
                minute[k][missing_minute] = np.nan
        q["missing"][missing_minute // 15] = True
    irr = {
        "ts": ts,
        "ghi_w_m2": np.zeros(n),
        "dni_w_m2": np.zeros(n),
        "dhi_w_m2": np.zeros(n),
        "temp_c": np.full(n, 10.0),
    }
    cfg = C.from_dict(
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
            "pv": {"none": pv_none},
        }
    )
    return E.ReplayInputs(cfg, q, m, cache, irr, start, end, flows_minute=minute)


@pytest.mark.parametrize("impl", ["ranking", "dp"])
def test_replay_is_deterministic_and_books_both_paths(impl):
    # both marginal value implementations run end to end: dp is the shipped default, ranking
    # is a ladder rung and a user knob
    knobs = C.Knobs(impl=impl, kappa=0.5 if impl == "ranking" else 1.0)
    r1 = E.replay(_inputs(), knobs=knobs)
    r2 = E.replay(_inputs(), knobs=knobs)
    assert r1.knobs.impl == impl
    assert len(r1.virtual) == len(r1.actual) == 96 * 3 and len(r1.decisions) == 96 * 3 * 15
    assert np.array_equal([x.money_eur for x in r1.virtual], [x.money_eur for x in r2.virtual])
    assert all(a.import_kwh == pytest.approx(0.125) for a in r1.actual)
    assert (r1.soc_virtual_kwh >= 4.8 - 1e-9).all() and (r1.soc_virtual_kwh <= 48 + 1e-9).all()
    assert (
        set(r1.tiers_by_month.values()) == {"A"}
        and r1.learning_days == 0
        and r1.pairs["day"].shape == (3,)
    )
    assert len(r1.interval) == 3 and r1.site_report["pv"] == "none"


def test_missing_minute_flags_the_quarter_on_both_paths():
    idx = 96 * 35 * 15 + 60 + 7  # seven minutes into the first replay quarter
    r = E.replay(_inputs(missing_minute=idx))
    assert r.virtual[0].missing and r.actual[0].missing and not r.virtual[1].missing
    assert np.isnan(r.virtual[0].money_eur) and r.pairs["n_missing"][0] == 1


def test_da_known_mask_follows_the_thirteen_hundred_rule():
    import datetime as dt

    from betterybalance.core.timegrid import ZONE

    def _ts(s):
        return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZONE).timestamp())

    slots = np.array(
        [_ts("2026-10-20T15:00:00"), _ts("2026-10-21T03:00:00"), _ts("2026-10-22T03:00:00")]
    )
    assert E.da_known_mask(_ts("2026-10-20T12:00:00"), slots).tolist() == [True, False, False]
    assert E.da_known_mask(_ts("2026-10-20T13:00:00"), slots).tolist() == [True, True, False]


def test_greed_rows_exist_per_day_and_decisions_table_has_the_columns():
    r = E.replay(_inputs())
    assert len(r.greed["day"]) == 3 and {
        "charge_threshold",
        "sale_threshold",
        "night_median_take",
        "low_soc_ok",
    } <= set(r.greed)
    t = E.decisions_table(r.decisions)
    assert {"ts", "action", "kw", "committed", "reason", "margin_w", "value_stored"} <= set(t)


def test_replay_refuses_a_start_row_without_a_band():
    ts = np.arange(0, 96 * 900, 900, dtype=np.int64)
    cache = BC.BandCache(
        ts=ts,
        take={},
        feed={},
        p_dual={},
        cov={},
        fold_of=np.full(len(ts), -1),
        fold_train_end=np.zeros(0, dtype=np.int64),
        tables=[],
        meta={"horizons": [0]},
    )
    cfg = C.from_dict(
        {
            "latitude": 52.1,
            "longitude": 5.2,
            "pack": {
                "capacity_kwh": 10,
                "charge_kw": 5,
                "discharge_kw": 5,
                "soc_floor": 0.1,
                "soc_ceiling": 1.0,
            },
        }
    )
    inputs = E.ReplayInputs(cfg, {"ts": ts}, {"ts": ts}, cache, {"ts": ts}, int(ts[4]), int(ts[8]))
    with pytest.raises(ValueError, match=f"replay start {int(ts[4])}"):
        E.replay(inputs)


def test_wrong_side_counts_each_committed_quarter_once():
    from betterybalance.core.rule.trader import Decision
    from betterybalance.replay import phases as P

    ts = np.arange(4, dtype=np.int64) * 900
    master = {
        "ts": ts,
        "buy_eur_mwh": np.array([50.0, 50.0, 10.0, np.nan]),
        "sell_eur_mwh": np.array([80.0, 80.0, 80.0, 80.0]),
        "regulation_state": np.zeros(4),
        "buy_live_end": np.full(4, 50.0),
        "sell_live_end": np.full(4, 80.0),
    }
    cfg = C.from_dict(
        {
            "latitude": 52.1,
            "longitude": 5.2,
            "pack": {
                "capacity_kwh": 10,
                "charge_kw": 5,
                "discharge_kw": 5,
                "soc_floor": 0.1,
                "soc_ceiling": 1.0,
            },
        }
    )

    def d(t, action, reason, charge_th=40.0, discharge_th=60.0):
        committed = reason in ("commit", "precommit", "negative_take")
        return Decision(
            int(t), action, 0.0, committed, reason, 1.0, 0.1, charge_th, discharge_th, False
        )

    decisions = [
        # quarter 0: buy 50 above the charge threshold 40 on 14 committed minutes -> counts once
        *[d(60 * e, "charge", "commit") for e in range(1, 15)],
        # quarter 1: a discharge at sell 80 above 60 is right; a negative take at buy 50 is wrong
        d(900 + 60, "discharge", "precommit"),
        d(900 + 120, "charge", "negative_take"),
        # quarter 2: buy 10 at or below 40 is right; an uncommitted wait never counts
        d(1800 + 60, "charge", "commit"),
        d(1800 + 120, "hold", "wait", charge_th=0.0),
        # quarter 3: unsettled buy does not count
        d(2700 + 60, "charge", "commit"),
    ]
    assert P.wrong_side_count(decisions, P.Prices(master), cfg) == 2


@pytest.fixture(scope="module")
def one_day():
    """One replay day on one band cache, shared (tests copy before they change a column)."""
    return _inputs(days_replay=1)


def _copied(table):
    return {k: np.array(v, copy=True) for k, v in table.items()}


def test_a_virtual_battery_that_copies_the_actual_one_books_identical_rows(monkeypatch, one_day):
    import dataclasses

    from betterybalance.core.site.flows import ac_legs
    from betterybalance.core.timegrid import isp_start

    base = one_day
    ec, ed = ac_legs(base.cfg.pack)
    i = np.arange(len(base.flows_minute["ts"]))
    m = dict(base.flows_minute)  # two-way grid flow and a battery that changes direction
    m["grid_import_kwh"] = np.where(i % 3 == 0, 0.0, 0.5 / 60)
    m["grid_export_kwh"] = np.where(i % 3 == 0, 0.2 / 60, 0.0)
    m["batt_charge_kwh"] = np.where(i % 2 == 0, 0.1 / 60, 0.0)
    m["batt_discharge_kwh"] = np.where(i % 2 == 0, 0.0, 0.05 / 60)
    q = dict(base.flows_quarter, batt_charge_kwh=np.full(len(base.flows_quarter["ts"]), 0.02))
    for minute in (m, None):
        table, per_hour = (m, 60.0) if minute is not None else (q, 4.0)
        batt = (table["batt_charge_kwh"] / ec - table["batt_discharge_kwh"] * ed) * per_hour

        class Copy:  # replays the recorded battery AC flow, minute by minute from the start
            def __init__(self, *args, **kwargs):
                self.soc_kwh, self.throughput_kwh, self.n = 24.0, 0.0, 0

            def step(self, ac_kw, dt_s, table=table, batt=batt, minute=minute):
                t = base.start + 60 * self.n
                self.n += 1
                at = t if minute is not None else int(isp_start(t))
                return float(batt[np.searchsorted(table["ts"], at)])

        monkeypatch.setattr(ST, "VirtualBattery", Copy)
        r = E.replay(dataclasses.replace(base, flows_quarter=q, flows_minute=minute))
        assert len(r.virtual) == 96
        for v, a in zip(r.virtual, r.actual, strict=True):
            assert (v.import_kwh, v.export_kwh, v.money_eur) == (
                a.import_kwh,
                a.export_kwh,
                a.money_eur,
            )
        bias = r.site_report["gross_split_bias_eur"]
        # the recorded quarter gross (0,125 kWh one way) differs from the minute split
        assert np.isfinite(bias) and (bias != 0.0 if minute is not None else True)


def test_soc_anchors_are_the_instants_at_start_and_end(monkeypatch, one_day):
    import dataclasses

    from betterybalance.core.ledger import terminal_valuation, trailing_mean_buy
    from betterybalance.core.site.flows import ac_legs

    base = one_day
    seen = []
    original = ST.VirtualBattery

    class Recording(original):
        def __init__(self, pack, loss, soc_kwh, standby=False):
            seen.append(soc_kwh)
            super().__init__(pack, loss, soc_kwh, standby)

    monkeypatch.setattr(ST, "VirtualBattery", Recording)
    m, q = _copied(base.flows_minute), _copied(base.flows_quarter)
    m["soc_pct"][m["ts"] == base.start] = 30.0  # the start minute
    m["soc_pct"][m["ts"] == base.end - 60] = 70.0  # the last minute
    q["soc_pct"][q["ts"] == base.start - 900] = 20.0  # the quarter that ends at start
    q["soc_pct"][q["ts"] == base.end - 900] = 80.0  # the last quarter
    _, ed = ac_legs(base.cfg.pack)
    trailing = trailing_mean_buy(base.master["buy_eur_mwh"], base.master["ts"], base.end)
    for minute, soc0, soc_end in ((m, 30.0, 70.0), (None, 20.0, 80.0)):
        seen.clear()
        r = E.replay(dataclasses.replace(base, flows_quarter=q, flows_minute=minute))
        assert seen == [pytest.approx(soc0 / 100 * 48)]
        expected = terminal_valuation(r.soc_virtual_kwh[-1], soc_end / 100 * 48, trailing, ed)
        assert r.terminal_eur == pytest.approx(expected)


def test_a_missing_quarter_holds_the_pack_and_a_nan_gross_is_booked_missing(one_day):
    import dataclasses

    base = one_day
    knobs = dataclasses.replace(base.cfg.knobs, standby=True)
    cfg = dataclasses.replace(base.cfg, loss_profile="deye", knobs=knobs)  # standby drains
    m, q = _copied(base.flows_minute), _copied(base.flows_quarter)
    j = int(np.searchsorted(m["ts"], base.start)) + 7
    for col in m:
        if col not in ("ts", "step_s"):
            m[col][j] = np.nan
    k = int(np.searchsorted(q["ts"], base.start))
    q["missing"][k] = True
    q["grid_import_kwh"][k + 1] = np.nan  # a NaN recorded gross without the flag
    r = E.replay(dataclasses.replace(base, cfg=cfg, flows_quarter=q, flows_minute=m))
    assert r.virtual[0].missing and r.soc_virtual_kwh[0] == 0.5 * 48  # held, no drain
    assert r.virtual[1].missing and r.actual[1].missing and np.isnan(r.actual[1].money_eur)
    assert not r.virtual[2].missing and r.soc_virtual_kwh[2] < r.soc_virtual_kwh[1]


def test_gen_port_production_never_reaches_the_engine(one_day):
    import dataclasses

    base = one_day
    q = _copied(base.flows_quarter)
    q["gen_kwh"] = np.full(len(q["ts"]), 0.3)  # a GEN-port string, validation data only
    with_gen = E.replay(dataclasses.replace(base, flows_quarter=q))
    zeroed = E.replay(
        dataclasses.replace(base, flows_quarter=dict(q, gen_kwh=np.zeros(len(q["ts"]))))
    )
    a, b = E.decisions_table(with_gen.decisions), E.decisions_table(zeroed.decisions)
    for col in a:
        same = np.array_equal(a[col], b[col], equal_nan=a[col].dtype == np.float64)
        assert same, col
    assert [r.money_eur for r in with_gen.virtual] == [r.money_eur for r in zeroed.virtual]
    assert with_gen.site_report["gen_handling"] == "learned"


def test_a_nan_soc_at_the_end_falls_back_to_the_final_hour_or_raises(one_day):
    import dataclasses

    from betterybalance.core.ledger import terminal_valuation, trailing_mean_buy
    from betterybalance.core.site.flows import ac_legs

    base = one_day
    _, ed = ac_legs(base.cfg.pack)
    trailing = trailing_mean_buy(base.master["buy_eur_mwh"], base.master["ts"], base.end)
    m, q = _copied(base.flows_minute), _copied(base.flows_quarter)
    m["soc_pct"][m["ts"] == base.end - 240] = 64.0  # the last finite minute: 3 min before end
    m["soc_pct"][(m["ts"] > base.end - 240) & (m["ts"] < base.end)] = np.nan
    q["soc_pct"][q["ts"] == base.end - 1800] = 72.0  # the last finite quarter
    q["soc_pct"][q["ts"] == base.end - 900] = np.nan
    for minute, soc_end, lag in ((m, 64.0, 3), (None, 72.0, 15)):
        r = E.replay(dataclasses.replace(base, flows_quarter=q, flows_minute=minute))
        assert r.site_report["soc_end_anchor"]["minutes_before_end"] == lag
        expected = terminal_valuation(r.soc_virtual_kwh[-1], soc_end / 100 * 48, trailing, ed)
        assert np.isfinite(r.terminal_eur) and r.terminal_eur == pytest.approx(expected)
    exact = E.replay(dataclasses.replace(base, flows_minute=None))
    assert exact.site_report["soc_end_anchor"] == {"ts": base.end - 900, "minutes_before_end": 0}
    # no finite SoC in the final 60 minutes: a ValueError that names the last finite minute
    m2 = _copied(base.flows_minute)
    m2["soc_pct"][(m2["ts"] >= base.end - 3660) & (m2["ts"] < base.end)] = np.nan
    with pytest.raises(ValueError, match=r"end anchor .*last finite SoC is at 2026-11-18 22:58"):
        E.replay(dataclasses.replace(base, flows_minute=m2))
