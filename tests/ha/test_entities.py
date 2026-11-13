import datetime as dt
import json
import math

from homeassistant.core import State
from homeassistant.helpers import entity_platform
from homeassistant.helpers.json import JSONEncoder
from pytest_homeassistant_custom_component.common import mock_restore_cache

from custom_components.betterybalance.const import DOMAIN, STORE_DIRNAME, STORE_FILE
from custom_components.betterybalance.core.store import Store
from tests.ha.test_coordinator import (
    START,
    _set_states,
    _setup,
    _step,
    offline,  # noqa: F401  (every feed offline, the store under tmp_path)
)

RECORDER_LIMIT = 16384
PLAN_ARRAYS = {
    "ts", "take_q10", "take_q50", "take_q90", "feed_q10", "feed_q50", "feed_q90", "soc_pct",
    "action",
}
SENSORS = (
    "state", "plan", "result_today", "result", "result_provisional", "shadow_soc",
    "take_nowcast", "feed_nowcast", "pv_expected", "backtest",
)
FEEDS = ("day_ahead", "ida", "settlement", "live", "irradiance", "pack")


def _seed_prices(tmp_path) -> None:
    """Synthetic day-ahead and IDA prices from two days back to three days ahead of START and
    live minutes for the hour before it, so the coordinator ranks its first quarter."""
    start = int(dt.datetime.fromisoformat(START).timestamp())
    q0 = start - start % 900
    (tmp_path / STORE_DIRNAME).mkdir(exist_ok=True)
    store = Store(tmp_path / STORE_DIRNAME / STORE_FILE)
    store.migrate()
    for t in range(q0 - 2 * 86400, q0 + 3 * 86400, 900):
        p = 90.0 + 40.0 * math.sin(2 * math.pi * (t % 86400) / 86400)
        store.upsert_features(t, da_eur_mwh=p, ida1_eur_mwh=p, ida2_eur_mwh=p, ida3_eur_mwh=p)
    rows = [
        {"ts": t, "buy_live": 95.0, "sell_live": 95.0, "state": 0, "source": "synthetic"}
        for t in range(q0 - 3600, start + 600, 60)
    ]
    store.put_live_minutes(rows)
    store.close()


def _entities(hass) -> list:
    platforms = entity_platform.async_get_platforms(hass, DOMAIN)
    return [e for p in platforms for e in p.entities.values()]


async def test_entity_ids_are_fixed_in_dutch(hass, freezer):
    hass.config.language = "nl"
    await _setup(hass, freezer)  # from test_coordinator
    for key in SENSORS:
        assert hass.states.get(f"sensor.betterybalance_{key}") is not None
    for feed in FEEDS:
        assert hass.states.get(f"binary_sensor.betterybalance_feed_{feed}") is not None
    assert hass.states.get("select.betterybalance_period").state == "day"
    names = {
        "sensor.betterybalance_state": "BetteryBalance Actie",
        "binary_sensor.betterybalance_feed_live": "BetteryBalance Bron live onbalansprijs",
        "select.betterybalance_period": "BetteryBalance Resultaatperiode",
    }
    for entity_id, name in names.items():
        assert hass.states.get(entity_id).attributes["friendly_name"] == name


async def test_entity_names_come_from_the_translations(hass, freezer):
    await _setup(hass, freezer)
    names = {
        "sensor.betterybalance_state": "BetteryBalance Action",
        "sensor.betterybalance_shadow_soc": "BetteryBalance Shadow state of charge",
        "binary_sensor.betterybalance_feed_day_ahead": "BetteryBalance Day-ahead price feed",
        "select.betterybalance_period": "BetteryBalance Results period",
    }
    for entity_id, name in names.items():
        assert hass.states.get(entity_id).attributes["friendly_name"] == name


async def test_recorded_attributes_stay_under_the_recorder_limit(hass, freezer, tmp_path):
    _seed_prices(tmp_path)
    await _setup(hass, freezer)
    _set_states(hass)
    for _ in range(3 * 12):  # three minutes: a quarter's minute 1 sets up the ranking
        await _step(hass, freezer)
    plan = hass.states.get("sensor.betterybalance_plan")
    assert plan.state not in ("unknown", "unavailable")
    assert PLAN_ARRAYS <= set(plan.attributes)
    ents = _entities(hass)
    assert len(ents) == len(SENSORS) + len(FEEDS) + 1
    for ent in ents:
        st = hass.states.get(ent.entity_id)
        unrecorded = ent._unrecorded_attributes | ent._entity_component_unrecorded_attributes
        recorded = {k: v for k, v in st.attributes.items() if k not in unrecorded}
        assert len(json.dumps(recorded, cls=JSONEncoder).encode()) < RECORDER_LIMIT, ent.entity_id
        if ent.entity_id == "sensor.betterybalance_plan":
            assert PLAN_ARRAYS <= unrecorded
            full = json.dumps(dict(st.attributes), cls=JSONEncoder).encode()
            assert len(full) > len(json.dumps(recorded, cls=JSONEncoder).encode())


async def test_entities_read_the_snapshot(hass, freezer):
    coord = await _setup(hass, freezer)
    _set_states(hass)
    for _ in range(3 * 12):
        await _step(hass, freezer)
    totals = coord.data["totals"]
    result = hass.states.get("sensor.betterybalance_result")
    assert float(result.state) == round(totals.settled_eur, 2)
    assert result.attributes["state_class"] == "total"
    assert result.attributes["device_class"] == "monetary"
    assert result.attributes["unit_of_measurement"] == "EUR"
    assert "balans_revenue_eur_month" in result.attributes
    prov = hass.states.get("sensor.betterybalance_result_provisional")
    assert prov.attributes["state_class"] == "total"
    today = hass.states.get("sensor.betterybalance_result_today")
    assert {"settled_share", "missing_quarters"} <= set(today.attributes)
    soc = hass.states.get("sensor.betterybalance_shadow_soc")
    assert soc.attributes["unit_of_measurement"] == "%" and "kwh" in soc.attributes
    for side in ("take", "feed"):
        nc = hass.states.get(f"sensor.betterybalance_{side}_nowcast")
        assert nc.attributes["unit_of_measurement"] == "ct/kWh"
        assert {"q10", "q25", "q50", "q75", "q90", "horizon", "tier"} <= set(nc.attributes)
    pv = hass.states.get("sensor.betterybalance_pv_expected")
    assert pv.attributes["unit_of_measurement"] == "kW"
    assert {"per_string", "model_version", "learning", "drift"} <= set(pv.attributes)
    bt = hass.states.get("sensor.betterybalance_backtest")
    assert bt.state == "idle" and {"progress", "eta_s", "headline"} <= set(bt.attributes)
    state = hass.states.get("sensor.betterybalance_state")
    assert state.state in ("hold", "charge", "discharge")
    assert {"kw", "committed", "reason", "verdict", "next_window", "last_trade", "ledger_head",
            "status", "status_text"} <= set(state.attributes)
    for feed in FEEDS:
        b = hass.states.get(f"binary_sensor.betterybalance_feed_{feed}")
        assert b.attributes["device_class"] == "problem"
        assert {"last_ok", "age_s", "source", "last_error"} <= set(b.attributes)
    # the feeds are offline in these tests and the box has no pack
    assert hass.states.get("binary_sensor.betterybalance_feed_pack").state == "on"
    # the IDA fetch is off by choice (the default): never a problem
    ida = hass.states.get("binary_sensor.betterybalance_feed_ida")
    assert ida.state == "off" and ida.attributes["off_by_choice"] is True
    assert hass.states.get("binary_sensor.betterybalance_feed_live").attributes[
        "off_by_choice"
    ] is False
    for st in hass.states.async_all():
        if st.entity_id.split(".")[1].startswith("betterybalance_"):
            assert "actual" not in repr(dict(st.attributes))


async def test_period_select_is_restored(hass, freezer):
    mock_restore_cache(hass, [State("select.betterybalance_period", "month")])
    await _setup(hass, freezer)
    assert hass.states.get("select.betterybalance_period").state == "month"
    await hass.services.async_call(
        "select", "select_option",
        {"entity_id": "select.betterybalance_period", "option": "year"}, blocking=True,
    )
    assert hass.states.get("select.betterybalance_period").state == "year"


def _recorded(hass) -> dict:
    out = {}
    for ent in _entities(hass):
        st = hass.states.get(ent.entity_id)
        unrecorded = ent._unrecorded_attributes | ent._entity_component_unrecorded_attributes
        out[ent.entity_id] = {k: v for k, v in st.attributes.items() if k not in unrecorded}
    return out


async def test_feed_sensors_record_no_new_attributes_between_two_minutes(hass, freezer, tmp_path):
    _seed_prices(tmp_path)  # the seeded store gives day_ahead, ida and live a last_ok
    await _setup(hass, freezer)
    _set_states(hass)
    for _ in range(2 * 12):
        await _step(hass, freezer)
    before = _recorded(hass)
    ages = {f: hass.states.get(f"binary_sensor.betterybalance_feed_{f}").attributes["age_s"]
            for f in FEEDS}
    for _ in range(12):  # one more minute; every feed is offline, so no fetch succeeds
        await _step(hass, freezer)
    after = _recorded(hass)
    for feed in FEEDS:
        eid = f"binary_sensor.betterybalance_feed_{feed}"
        assert after[eid] == before[eid], eid
        assert "age_s" not in after[eid] and "last_ok" in after[eid]
        age = hass.states.get(eid).attributes["age_s"]
        assert (age is None) == (ages[feed] is None)
        if age is not None:
            assert age > ages[feed]  # the shown age moves, the recorded attributes do not


async def test_the_verdict_window_follows_the_plan(hass, freezer, tmp_path):
    _seed_prices(tmp_path)
    coord = await _setup(hass, freezer)
    _set_states(hass)
    for _ in range(3 * 12):
        await _step(hass, freezer)
    plan = coord.data["plan"]
    assert plan is not None
    state = hass.states.get("sensor.betterybalance_state").attributes
    action = plan["planned"].action
    window = state["next_window"]
    assert window is not None  # the seeded prices give the plan a charge run
    slots = list(coord.data["status"]["ranked"].window.ts)
    j, e = slots.index(window["start"]), slots.index(window["end"] - 900)
    # the window is the plan's first run of trade actions, as the card draws them
    assert all(a == "hold" for a in action[:j])
    assert all(a == window["side"] for a in action[j : e + 1])
    assert e + 1 == len(action) or action[e + 1] != window["side"]
