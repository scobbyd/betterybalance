import asyncio
import io
import json
import pathlib
import tempfile
import threading
import time
import zipfile
from unittest.mock import patch

import numpy as np
import pytest
from homeassistant.core import Context
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError, Unauthorized
from homeassistant.helpers import issue_registry as ir

from custom_components.betterybalance import coordinator as CO
from custom_components.betterybalance import http as HT
from custom_components.betterybalance.const import DOMAIN, SERVICES, STORE_DIRNAME
from custom_components.betterybalance.live import backtest as BT
from custom_components.betterybalance.replay.phases import IRR_COLS
from tests.ha.test_coordinator import (
    _setup,
    _step,
    offline,  # noqa: F401  (every feed offline, the store under tmp_path)
)
from tests.ha.test_entities import _seed_prices

CLIENT_ID = "https://example.com/app"

DAY = "2026-11-02"


async def _token(hass, user) -> str:
    """An access token minted at the frozen time (the fixture's token predates the freeze)."""
    refresh = await hass.auth.async_create_refresh_token(user, CLIENT_ID)
    return hass.auth.async_create_access_token(refresh)


async def _run_minutes(hass, freezer, minutes: int) -> None:
    for _ in range(minutes * 12):
        await _step(hass, freezer)


async def test_services_are_registered(hass, freezer):
    await _setup(hass, freezer)
    for name in SERVICES:
        assert hass.services.has_service(DOMAIN, name)


async def test_download_needs_auth(
    hass, hass_client, hass_client_no_auth, hass_admin_user, freezer
):
    coord = await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 3)
    url = f"/api/betterybalance/{coord.entry.entry_id}/ledger.zip?from={DAY}&to={DAY}"
    assert (await (await hass_client_no_auth()).get(url)).status == 401
    client = await hass_client(await _token(hass, hass_admin_user))
    temps = set(HT.temp_dir().glob("betterybalance-ledger-*.zip"))
    resp = await client.get(url)
    assert resp.status == 200 and resp.headers["Content-Type"] == "application/zip"
    assert f'filename="betterybalance-ledger-{DAY}-{DAY}.zip"' in resp.headers[
        "Content-Disposition"
    ]
    names = zipfile.ZipFile(io.BytesIO(await resp.read())).namelist()
    assert sorted(names) == ["ledger.csv", "manifest.json", "settlements.csv"]
    # the streamed temporary zip is gone once it was sent (the server removes it after the
    # body, so the client can finish reading first)
    for _ in range(100):
        left = set(HT.temp_dir().glob("betterybalance-ledger-*.zip"))
        if left == temps:
            break
        await hass.async_add_executor_job(time.sleep, 0.01)  # real time (the clock is frozen)
    assert left == temps
    resp = await client.get(url + "&flows=1")
    assert "flows_minute.csv" in zipfile.ZipFile(io.BytesIO(await resp.read())).namelist()
    base = f"/api/betterybalance/{coord.entry.entry_id}/ledger.zip"
    for query in (
        f"from=2026-11-03&to={DAY}",  # reversed
        "from=2026-11-01&to=9999-12-31",  # past the clock's range
        "from=2026-10-01&to=2026-11-01&flows=1",  # 32 days of minute flows
        "from=2026-11-02",  # no to
    ):
        resp = await client.get(f"{base}?{query}")
        assert resp.status == 400, query
    big = await client.get(f"{base}?from=2026-10-01&to=2026-11-01&flows=0")  # no cap
    assert big.status == 200
    assert (await client.get("/api/betterybalance/nope/ledger.zip?from=a&to=b")).status == 404


async def test_export_ledger_service_writes_the_files(hass, freezer, tmp_path):
    coord = await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 3)
    resp = await hass.services.async_call(
        DOMAIN, "export_ledger", {"from": DAY, "to": DAY}, blocking=True, return_response=True
    )
    folder = tmp_path / STORE_DIRNAME / "exports" / f"{DAY}_{DAY}"
    assert resp["path"] == str(folder)
    assert sorted(p.name for p in folder.iterdir()) == [
        "ledger.csv", "manifest.json", "settlements.csv",
    ]
    m = json.loads((folder / "manifest.json").read_text())
    assert m["chain_head"] == coord.site.store.head()[1] and m["flows_included"] is False
    # include_flows defaults to the export_flows option
    hass.config_entries.async_update_entry(
        coord.entry, options={**coord.entry.options, "export_flows": True}
    )
    await hass.services.async_call(DOMAIN, "export_ledger", {"from": DAY, "to": DAY}, blocking=True)
    assert (folder / "flows_minute.csv").exists()
    for data, match, key in (
        ({"from": "2026-11-03", "to": DAY}, "on or before", "export_order"),
        ({"from": DAY, "to": "9999-12-31"}, "out of range", "export_dates"),
        (
            {"from": "2026-10-01", "to": "2026-11-01", "include_flows": True},
            "at most 31",
            "export_flow_days",
        ),
    ):
        with pytest.raises(ServiceValidationError, match=match) as err:
            await hass.services.async_call(DOMAIN, "export_ledger", data, blocking=True)
        assert err.value.translation_domain == DOMAIN and err.value.translation_key == key


async def test_export_results_refuses_without_pooling(hass, freezer, tmp_path):
    coord = await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 3)
    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(DOMAIN, "export_results", {}, blocking=True)
    assert err.value.translation_key == "pooling_off"
    hass.config_entries.async_update_entry(
        coord.entry, options={**coord.entry.options, "pool_results": True}
    )
    resp = await hass.services.async_call(
        DOMAIN, "export_results", {}, blocking=True, return_response=True
    )
    path = tmp_path / STORE_DIRNAME / "exports" / f"results-{DAY}.json"
    assert resp["path"] == str(path)
    d = json.loads(path.read_text())
    assert d["source"] == "live" and "daily_heads" in d and d["version"] == "0.3.0"


async def test_backtest_without_pack_is_refused_with_the_reason(hass, freezer):
    coord = await _setup(hass, freezer)
    assert hass.states.get("sensor.betterybalance_backtest").state == "idle"
    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
    assert err.value.translation_domain == DOMAIN and err.value.translation_key == "no_pack"
    await hass.async_block_till_done(wait_background_tasks=True)
    # nothing started: no run row, no running flag, the sensor still idle
    assert coord.site.store.runs() == [] and coord.data["backtest"]["running"] is False
    assert hass.states.get("sensor.betterybalance_backtest").state == "idle"


async def test_backtest_runs_in_the_background_and_refuses_a_second_request(
    hass, freezer, offline  # noqa: F811
):
    coord = await _setup(hass, freezer)
    plan = BT.BacktestPlan(start=1_790_000_000, end=1_790_000_000 + 20 * 86400, days=20,
                           reason=None)
    release, started, seen = threading.Event(), threading.Event(), {}

    def fake_run(store, lc, pack, plan_, archive, by_lead, progress, run_id, should_stop=None):
        # the backtest's own connection on the same file, never the live one
        assert store._db is not coord._store._db and should_stop() is False
        seen.update(pack=pack, run_id=run_id, by_lead=by_lead)
        progress(1, plan_.days)
        started.set()
        assert release.wait(60)
        if seen.get("fail"):
            raise ValueError("synthetic failure")
        return {"result_eur": 12.5, "days": 20}, "# Backtest report\n"

    pack = object()
    with patch.object(CO.BetteryCoordinator, "_plan_job", lambda self, now: (plan, pack)), \
            patch.object(CO, "_backtest_irradiance", lambda *a: ({"ts": []}, "by-lead")), \
            patch.object(CO.BT, "run_backtest", fake_run):
        await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
        # the job reaches the replay and reports day 1; block_till_done would wait for it
        assert await asyncio.get_running_loop().run_in_executor(None, started.wait, 60)
        for _ in range(5):
            await asyncio.sleep(0)
        st = hass.states.get("sensor.betterybalance_backtest")
        assert st.state == "running" and st.attributes["progress"] == pytest.approx(0.05)
        assert st.attributes["eta_s"] is not None
        with pytest.raises(ServiceValidationError) as err:
            await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
        assert err.value.translation_key == "backtest_running"
        release.set()
        await hass.async_block_till_done(wait_background_tasks=True)
        st = hass.states.get("sensor.betterybalance_backtest")
        assert st.state == "done" and st.attributes["headline"]["result_eur"] == 12.5
        report = pathlib.Path(st.attributes["report"])
        assert report.read_text() == "# Backtest report\n"
        assert report.parent.parent == pathlib.Path(hass.config.path(STORE_DIRNAME, "runs"))
        assert report.parent.name.endswith(f"-{seen['run_id']}")  # one folder per run
        assert seen["pack"] is pack and seen["by_lead"] == "by-lead"
        runs = coord.site.store.runs()
        assert [r["status"] for r in runs] == ["done"] and runs[0]["run_id"] == seen["run_id"]
        # a failing run ends "failed" with the error; the live ledger never saw either run
        release.clear()
        seen["fail"] = True
        await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
        release.set()
        await hass.async_block_till_done(wait_background_tasks=True)
        st = hass.states.get("sensor.betterybalance_backtest")
        assert st.state == "failed" and st.attributes["error"] == "synthetic failure"
        assert st.attributes["headline"] is None
        assert coord.site.store.bookings(0, 2**40) == []


async def test_an_unload_stops_a_running_backtest_and_a_stopping_entry_refuses(
    hass, freezer, offline  # noqa: F811
):
    coord = await _setup(hass, freezer)
    plan = BT.BacktestPlan(start=1_790_000_000, end=1_790_000_000 + 20 * 86400, days=20,
                           reason=None)
    started = threading.Event()

    def fake_run(store, lc, pack, plan_, archive, by_lead, progress, run_id, should_stop=None):
        started.set()
        for _ in range(12000):  # a year-long replay that asks at every quarter (120 s)
            if should_stop():
                raise BT.Stopped("the backtest was stopped")
            time.sleep(0.01)
        raise AssertionError("the unload did not stop the run")

    with patch.object(CO.BetteryCoordinator, "_plan_job", lambda self, now: (plan, object())), \
            patch.object(CO, "_backtest_irradiance", lambda *a: ({"ts": []}, None)), \
            patch.object(CO.BT, "run_backtest", fake_run):
        await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
        assert await asyncio.get_running_loop().run_in_executor(None, started.wait, 60)
        store_path = hass.config.path(STORE_DIRNAME, "store.sqlite3")
        t0 = time.monotonic()
        assert await hass.config_entries.async_unload(coord.entry.entry_id)
        assert time.monotonic() - t0 < 60  # well before the 120 s the run would take
    from custom_components.betterybalance.core.store import Store

    s = Store(store_path)
    try:
        runs = s.runs()
    finally:
        s.close()
    assert [r["status"] for r in runs] == ["failed"]
    assert runs[0]["headline"] == {"error": CO.STOPPED}


async def test_a_stopping_entry_refuses_a_backtest_with_its_own_reason(hass, freezer):
    coord = await _setup(hass, freezer)
    coord._stopped = True  # async_stop began: the unload is under way
    try:
        with pytest.raises(ServiceValidationError) as err:
            await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
        assert err.value.translation_key == "backtest_stopping"
    finally:
        coord._stopped = False


async def test_a_span_without_an_end_soc_is_refused_with_the_reason(hass, freezer):
    await _setup(hass, freezer)
    plan = BT.BacktestPlan(start=0, end=20 * 86400, days=20, reason="no_end_soc")
    with patch.object(CO.BetteryCoordinator, "_plan_job", lambda self, now: (plan, object())):
        with pytest.raises(ServiceValidationError) as err:
            await hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
    assert err.value.translation_key == "no_end_soc"
    assert hass.states.get("sensor.betterybalance_backtest").state == "idle"


async def test_verify_ledger_posts_the_report_and_leaves_no_folder(hass, freezer):
    await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 20)  # one whole quarter booked
    temps = set(pathlib.Path(tempfile.gettempdir()).glob("betterybalance-verify-*"))
    posted = []
    with patch(
        "custom_components.betterybalance.services.persistent_notification.async_create",
        lambda hass, message, title=None, notification_id=None: posted.append(
            (message, title, notification_id)
        ),
    ):
        await hass.services.async_call(
            DOMAIN, "verify_ledger", {"from": DAY, "to": DAY}, blocking=True
        )
    assert len(posted) == 1
    message, title, notification_id = posted[0]
    assert title == "BetteryBalance ledger check" and notification_id.startswith(DOMAIN)
    lines = message.split("\n\n")
    assert lines[0] == f"{DAY} to {DAY}"
    assert lines[1].startswith("chain intact, ") and lines[1].endswith("energies match"), lines
    assert set(pathlib.Path(tempfile.gettempdir()).glob("betterybalance-verify-*")) == temps
    with pytest.raises(ServiceValidationError) as err:  # the range refusals of the export
        await hass.services.async_call(
            DOMAIN, "verify_ledger", {"from": "2026-11-03", "to": DAY}, blocking=True
        )
    assert err.value.translation_key == "export_order"


def _capture_notifications():
    posted = []
    target = patch(
        "custom_components.betterybalance.services.persistent_notification.async_create",
        lambda hass, message, title=None, notification_id=None: posted.append(message),
    )
    return posted, target


async def test_verify_ledger_says_why_long_ranges_have_no_energy_check(hass, freezer):
    await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 20)
    posted, target = _capture_notifications()
    with target:
        await hass.services.async_call(
            DOMAIN, "verify_ledger", {"from": "2026-10-01", "to": DAY}, blocking=True
        )
    lines = posted[0].split("\n\n")
    assert lines[1].startswith("chain intact, ")
    assert lines[2] == (
        "energies not checked (minute flows are checked for ranges up to 31 days, "
        "this range has 33)"
    )
    assert "flows_minute.csv" not in posted[0]


async def test_a_verify_ledger_that_cannot_run_posts_a_translated_sentence(hass, freezer):
    await _setup(hass, freezer)

    def boom(*args):
        raise OSError("disk full")

    posted, target = _capture_notifications()
    with target, patch("custom_components.betterybalance.services._verify", boom):
        await hass.services.async_call(
            DOMAIN, "verify_ledger", {"from": DAY, "to": DAY}, blocking=True
        )
    assert posted == [f"{DAY} to {DAY}\n\nThe ledger check could not run: disk full"]


async def test_service_refusals_carry_translation_keys(hass, freezer):
    await _setup(hass, freezer)
    with pytest.raises(ServiceValidationError) as err:
        await hass.services.async_call(DOMAIN, "export_results", {"config_entry_id": "nope"},
                                       blocking=True)
    assert err.value.translation_domain == DOMAIN and err.value.translation_key == "not_loaded"


async def test_refit_service_adopts_at_the_next_minute(hass, freezer, offline):  # noqa: F811
    coord = await _setup(hass, freezer)
    offline["archive"].calls.clear()
    with pytest.raises(HomeAssistantError, match="offline") as err:
        await hass.services.async_call(DOMAIN, "refit_site_models", {}, blocking=True)
    assert err.value.translation_key == "refit_failed"
    assert "offline" in err.value.translation_placeholders["error"]
    assert len(offline["archive"].calls) == 1 and coord.site.store.last_event("refit") is None
    archive = {"ts": np.zeros(0, dtype=np.int64), **{c: np.zeros(0) for c in IRR_COLS}}
    fitted = []

    def fit(now, irr):
        fitted.append((now, irr))
        return coord.site.state

    with patch.object(CO, "_irradiance_archive", lambda *a: archive), patch.object(
        coord.site, "fit", fit
    ):
        await hass.services.async_call(DOMAIN, "refit_site_models", {}, blocking=True)
    ev = coord.site.store.last_event("refit")
    assert fitted and fitted[0][1] is archive
    assert ev is not None and ev["ts"] == coord.site.next_minute


async def test_refit_raises_the_scale_issue_on_a_fill_in_house_only(hass, freezer, offline):  # noqa: F811
    coord = await _setup(hass, freezer)
    archive = {"ts": np.zeros(0, dtype=np.int64), **{c: np.zeros(0) for c in IRR_COLS}}
    issue_id = CO.scale_issue_id(coord.entry.entry_id)
    assert issue_id == f"pv_scale_off_{coord.entry.entry_id}"
    registry = ir.async_get(hass)

    async def refit(strings: str | None) -> None:
        with patch.object(CO, "_irradiance_archive", lambda *a: archive), patch.object(
            coord.site, "fit", lambda now, irr: coord.site.state
        ), patch.object(coord.site, "scale_off", lambda state: strings):
            await hass.services.async_call(DOMAIN, "refit_site_models", {}, blocking=True)

    await refit("1, 2")  # a learning house: the entered figures do not exist
    assert registry.async_get_issue(DOMAIN, issue_id) is None
    ir.async_create_issue(  # an issue left over from a fill-in entry goes with the next refit
        hass, DOMAIN, issue_id, is_fixable=False, is_persistent=True,
        severity=ir.IssueSeverity.WARNING, translation_key="pv_scale_off",
        translation_placeholders={"strings": "1"},
    )
    await refit("1, 2")
    assert registry.async_get_issue(DOMAIN, issue_id) is None
    hass.config_entries.async_update_entry(
        coord.entry, data={**coord.entry.data, "pv_mode": "fill"}
    )
    await refit("1, 2")
    issue = registry.async_get_issue(DOMAIN, issue_id)
    assert issue is not None and issue.translation_key == "pv_scale_off"
    assert issue.translation_placeholders == {"strings": "1, 2"} and not issue.is_fixable
    assert issue.severity == ir.IssueSeverity.WARNING
    assert issue.is_persistent  # only a refit raises it: a restart keeps it
    await refit(None)  # a blind refit has no verdict: the issue stays as it was
    assert registry.async_get_issue(DOMAIN, issue_id).translation_placeholders == {
        "strings": "1, 2"
    }
    await refit("")  # the next refit agrees with the entered figures: the issue goes
    assert registry.async_get_issue(DOMAIN, issue_id) is None
    await refit("GEN")
    assert registry.async_get_issue(DOMAIN, issue_id).translation_placeholders == {
        "strings": "GEN"
    }
    assert await hass.config_entries.async_remove(coord.entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)
    assert registry.async_get_issue(DOMAIN, issue_id) is None  # the entry took it along


async def test_ledger_tail_websocket(hass, hass_ws_client, hass_admin_user, freezer, tmp_path):
    _seed_prices(tmp_path)
    coord = await _setup(hass, freezer)
    # the seeded IDA prices serve tier B only with the fetch on (the default is off)
    hass.config_entries.async_update_entry(
        coord.entry, options={**coord.entry.options, "fetch_ida": True}
    )
    await hass.async_block_till_done(wait_background_tasks=True)
    coord = coord.entry.runtime_data
    assert coord.site.lc.skip_tiers == ()
    await _run_minutes(hass, freezer, 20)
    client = await hass_ws_client(hass, await _token(hass, hass_admin_user))
    await client.send_json({"id": 1, "type": "betterybalance/ledger_tail"})
    msg = await client.receive_json()
    assert msg["success"], msg
    rows = msg["result"]
    store = coord.site.store
    dec = [r for r in store.decisions(0, 2**40) if r["committed"] or r["action"] != "hold"]
    assert len(rows) == min(20, len(dec)) and rows
    for r in rows:
        assert {"take", "feed", "regulation_state"} <= set(r)
        assert r["take"] is None and r["feed"] is None and r["regulation_state"] is None
        assert r["committed"] or r["action"] != "hold"
    await client.send_json({"id": 2, "type": "betterybalance/ledger_tail", "limit": 1})
    msg = await client.receive_json()
    assert msg["success"] and len(msg["result"]) == 1


async def test_non_admin_users_get_neither_the_download_nor_any_service(
    hass, hass_client, hass_ws_client, hass_read_only_user, freezer
):
    coord = await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 3)
    token = await _token(hass, hass_read_only_user)
    url = f"/api/betterybalance/{coord.entry.entry_id}/ledger.zip?from={DAY}&to={DAY}"
    assert (await (await hass_client(token)).get(url)).status in (401, 403)
    context = Context(user_id=hass_read_only_user.id)
    for name, data in (
        ("export_ledger", {"from": DAY, "to": DAY}),
        ("export_results", {}),
        ("refit_site_models", {}),
        ("verify_ledger", {"from": DAY, "to": DAY}),
        ("run_backtest", {}),
    ):
        with pytest.raises(Unauthorized):
            await hass.services.async_call(DOMAIN, name, data, blocking=True, context=context)
    # the tail stays open to any user
    client = await hass_ws_client(hass, token)
    await client.send_json({"id": 1, "type": "betterybalance/ledger_tail"})
    assert (await client.receive_json())["success"]


async def test_a_cancelled_download_removes_its_zip_when_the_build_ends(hass, freezer):
    import asyncio
    import threading
    from unittest.mock import patch

    from custom_components.betterybalance.live import export as X

    coord = await _setup(hass, freezer)
    await _run_minutes(hass, freezer, 3)
    before = set(HT.temp_dir().glob("betterybalance-ledger-*.zip"))
    started, release = threading.Event(), threading.Event()
    real = X.write_export_zip

    def slow(*args):
        started.set()
        assert release.wait(30)
        return real(*args)

    import datetime as dt

    start, end = X.export_range(dt.date(2026, 11, 2), dt.date(2026, 11, 2), False)
    with patch.object(X, "write_export_zip", slow):
        task = hass.async_create_task(HT.build_zip(hass, coord, start, end, False))
        await hass.async_add_executor_job(started.wait, 30)
        mid = set(HT.temp_dir().glob("betterybalance-ledger-*.zip")) - before
        assert len(mid) == 1  # the zip being built
        task.cancel()  # the client went away
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        for _ in range(200):
            await hass.async_block_till_done()
            if not (set(HT.temp_dir().glob("betterybalance-ledger-*.zip")) - before):
                break
            await hass.async_add_executor_job(time.sleep, 0.01)
    assert not (set(HT.temp_dir().glob("betterybalance-ledger-*.zip")) - before)


async def test_setup_sweeps_the_zips_a_crash_left(hass, freezer):
    folder = HT.temp_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stale = folder / "betterybalance-ledger-left.zip"
    stale.write_bytes(b"x")
    other = folder / "not-ours.txt"
    other.write_text("x")
    try:
        await _setup(hass, freezer)
        assert not stale.exists() and other.exists()
    finally:
        other.unlink(missing_ok=True)
