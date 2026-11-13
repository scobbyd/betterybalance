"""The integration's services (design 13.3).

- ``export_ledger(from, to, include_flows)``: the ledger export of the Amsterdam days ``from``
  to ``to`` written row by row to ``<config>/betterybalance/exports/<from>_<to>/``.
  ``include_flows`` defaults to the ``export_flows`` option; minute flows cover at most
  ``live.export.MAX_FLOW_DAYS`` days. Responds with the folder and the file names.
- ``export_results``: the anonymised live results with the daily chain heads written to
  ``<config>/betterybalance/exports/results-<today>.json``; refused unless the
  ``pool_results`` option is on (the only path of the actual result off the box, aggregated
  and opt-in).
- ``refit_site_models``: the site-model refit now (``BetteryCoordinator.async_refit``).
- ``run_backtest``: the install-time backtest (design 11). The call plans the run on the
  stored hours and the installed pack and returns at once; the run itself is the
  coordinator's background job (one at a time). Refusals: ``no_pack``, ``short_history``
  and ``no_end_soc`` (``live.backtest.plan_backtest``), ``backtest_running`` while a run
  goes and ``backtest_stopping`` while the entry stops.
- ``verify_ledger(from, to)``: exports the days ``from`` to ``to`` to a temporary folder
  (with the minute flows when the range has at most ``live.export.MAX_FLOW_DAYS`` days; they
  never leave the box), runs ``verify.check.verify_export`` on it, deletes the folder and
  posts the report lines as a persistent notification titled "BetteryBalance ledger check".
  A range without flows says why; a check that cannot run posts the translated
  ``verify_failed`` sentence instead.

Each service acts on the loaded entry named by ``config_entry_id``, or on the only loaded
entry. Every service is admin only (``run_backtest`` starts a long job; the options flow calls
it with a system context). Store reads, file writes and the verifier run in the executor.
"""

from __future__ import annotations

import datetime as dt
import logging
import tempfile
from pathlib import Path

import voluptuous as vol
from homeassistant.components import persistent_notification
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.service import async_register_admin_service
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util

from . import __version__
from .const import DOMAIN, STORE_DIRNAME
from .coordinator import BacktestRunning, BacktestStopping
from .core.fmt import eu
from .core.results import results_bytes
from .core.timegrid import ZONE
from .live import export as X
from .verify import check as V

_LOGGER = logging.getLogger(__name__)

EXPORTS_DIRNAME = "exports"
ENTRY = "config_entry_id"
VERIFY_TITLE = "BetteryBalance ledger check"

_ENTRY_SCHEMA = {vol.Optional(ENTRY): cv.string}
_RANGE_SCHEMA = {vol.Required("from"): cv.date, vol.Required("to"): cv.date}
SCHEMAS = {
    "run_backtest": vol.Schema(_ENTRY_SCHEMA),
    "export_ledger": vol.Schema(
        {**_ENTRY_SCHEMA, **_RANGE_SCHEMA, vol.Optional("include_flows"): cv.boolean}
    ),
    "export_results": vol.Schema(_ENTRY_SCHEMA),
    "refit_site_models": vol.Schema(_ENTRY_SCHEMA),
    "verify_ledger": vol.Schema({**_ENTRY_SCHEMA, **_RANGE_SCHEMA}),
}


def coordinator_for(hass: HomeAssistant, entry_id: str | None = None):
    """The coordinator of the loaded entry ``entry_id``, or of the only loaded entry.

    Raises ``ServiceValidationError`` when no such entry is loaded or several are and none
    is named.
    """
    loaded = hass.config_entries.async_loaded_entries(DOMAIN)
    if entry_id is not None:
        loaded = [e for e in loaded if e.entry_id == entry_id]
    if not loaded:
        raise ServiceValidationError(
            "BetteryBalance is not loaded", translation_domain=DOMAIN, translation_key="not_loaded"
        )
    if len(loaded) > 1:
        raise ServiceValidationError(
            f"several BetteryBalance entries are loaded; set {ENTRY}",
            translation_domain=DOMAIN,
            translation_key="several_loaded",
            translation_placeholders={"field": ENTRY},
        )
    return loaded[0].runtime_data


def _range(call: ServiceCall, include_flows: bool = False) -> tuple[int, int]:
    """``live.export.export_range`` of the call's dates; its refusal as a
    ``ServiceValidationError`` with the refusal's translation key (``from`` after ``to``, too
    many flow days, an out-of-range date)."""
    try:
        return X.export_range(call.data["from"], call.data["to"], include_flows)
    except X.RangeRefusal as err:
        raise ServiceValidationError(
            str(err),
            translation_domain=DOMAIN,
            translation_key=err.key,
            translation_placeholders=err.placeholders,
        ) from err


def _today() -> dt.date:
    return dt_util.utcnow().astimezone(ZONE).date()


def _first_day(store) -> dt.date | None:
    """The Amsterdam day of the chain's first row; None before the first row."""
    first = store.chain(1, 1)
    if not first:
        return None
    kind, row = first[0]
    ts = row["ts"] if kind == "decision" else row["quarter_ts"]
    return dt.datetime.fromtimestamp(int(ts), ZONE).date()


def _write_results(coord, path: Path, today: dt.date) -> None:
    """The live results of the first ledger day to yesterday, with the daily heads."""
    site = coord.site
    first = _first_day(site.store) or today
    start, end = X.day_range(first, today - dt.timedelta(days=1))
    d = X.export_results_live(site.store, site.lc, start, max(start, end), __version__, True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(results_bytes(d))


def _verify(site, start: int, end: int, include_flows: bool) -> V.VerifyReport:
    """Export ``[start, end)`` to a temporary folder, verify it and delete the folder."""
    with tempfile.TemporaryDirectory(prefix="betterybalance-verify-") as tmp:
        folder = Path(tmp)
        X.write_export_dir(site.store, site.lc, start, end, include_flows, folder)
        return V.verify_export(folder)


def _no_flows_line(days: int) -> str:
    """The report line for a range without minute flows (longer than ``MAX_FLOW_DAYS``)."""
    return (
        f"energies not checked (minute flows are checked for ranges up to "
        f"{eu(X.MAX_FLOW_DAYS, 0)} days, this range has {eu(days, 0)})"
    )


async def _error_text(hass: HomeAssistant, err: HomeAssistantError) -> str:
    """The translated sentence of ``err`` in HA's language (English when it lacks one)."""
    key = f"component.{DOMAIN}.exceptions.{err.translation_key}.message"
    for language in (hass.config.language, "en"):
        strings = await async_get_translations(hass, language, "exceptions", {DOMAIN})
        if key in strings:
            text = strings[key]
            for name, value in (err.translation_placeholders or {}).items():
                text = text.replace("{" + name + "}", value)
            return text
    return str(err)


def async_register_services(hass: HomeAssistant) -> None:
    """Register the services once (the integration's ``async_setup``)."""
    if hass.services.has_service(DOMAIN, "export_ledger"):
        return
    exports = Path(hass.config.path(STORE_DIRNAME, EXPORTS_DIRNAME))

    async def run_backtest(call: ServiceCall) -> None:
        coord = coordinator_for(hass, call.data.get(ENTRY))
        if coord.stopping:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="backtest_stopping"
            )
        if coord.backtest_running:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="backtest_running"
            )
        plan, pack = await coord.async_plan_backtest()
        if plan.reason == "no_pack":  # literal keys, so the translation scan finds them
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key="no_pack")
        if plan.reason == "no_end_soc":
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key="no_end_soc")
        if plan.reason is not None:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="short_history"
            )
        try:
            coord.start_backtest(plan, pack)
        except BacktestStopping as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="backtest_stopping"
            ) from err
        except BacktestRunning as err:
            raise ServiceValidationError(
                translation_domain=DOMAIN, translation_key="backtest_running"
            ) from err

    async def export_ledger(call: ServiceCall) -> ServiceResponse:
        coord = coordinator_for(hass, call.data.get(ENTRY))
        flows = call.data.get("include_flows")
        if flows is None:
            flows = bool(coord.entry.options.get("export_flows", False))
        start, end = _range(call, bool(flows))
        first, last = call.data["from"], call.data["to"]
        folder = exports / f"{first.isoformat()}_{last.isoformat()}"
        site = coord.site
        names = await hass.async_add_executor_job(
            X.write_export_dir, site.store, site.lc, start, end, bool(flows), folder
        )
        return {"path": str(folder), "files": names}

    async def export_results(call: ServiceCall) -> ServiceResponse:
        coord = coordinator_for(hass, call.data.get(ENTRY))
        if not coord.entry.options.get("pool_results", False):
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key="pooling_off")
        today = _today()
        path = exports / f"results-{today.isoformat()}.json"
        await hass.async_add_executor_job(_write_results, coord, path, today)
        return {"path": str(path)}

    async def refit_site_models(call: ServiceCall) -> None:
        coord = coordinator_for(hass, call.data.get(ENTRY))
        try:
            await coord.async_refit()
        except Exception as err:
            raise HomeAssistantError(
                f"site-model refit failed: {err}",
                translation_domain=DOMAIN,
                translation_key="refit_failed",
                translation_placeholders={"error": str(err)},
            ) from err

    async def verify_ledger(call: ServiceCall) -> None:
        coord = coordinator_for(hass, call.data.get(ENTRY))
        start, end = _range(call)
        first, last = call.data["from"], call.data["to"]
        days = (last - first).days + 1
        flows = days <= X.MAX_FLOW_DAYS
        try:
            report = await hass.async_add_executor_job(_verify, coord.site, start, end, flows)
            lines = [_no_flows_line(days) if x == V.NO_FLOWS else x for x in report.lines]
        except Exception as exc:  # the check could not run: say so in the notification
            _LOGGER.exception("ledger check of %s to %s failed", first, last)
            err = HomeAssistantError(
                f"the ledger check could not run: {exc}",
                translation_domain=DOMAIN,
                translation_key="verify_failed",
                translation_placeholders={"error": str(exc)},
            )
            lines = [await _error_text(hass, err)]
        persistent_notification.async_create(
            hass,
            "\n\n".join([f"{first.isoformat()} to {last.isoformat()}", *lines]),
            title=VERIFY_TITLE,
            notification_id=f"{DOMAIN}_ledger_check_{coord.entry.entry_id}",
        )

    # the services that write under /config or start heavy work are admin only, as HA's own
    # diagnostics and backup downloads are
    handlers = {
        "run_backtest": (run_backtest, SupportsResponse.NONE, True),
        "export_ledger": (export_ledger, SupportsResponse.OPTIONAL, True),
        "export_results": (export_results, SupportsResponse.OPTIONAL, True),
        "refit_site_models": (refit_site_models, SupportsResponse.NONE, True),
        "verify_ledger": (verify_ledger, SupportsResponse.NONE, True),
    }
    for name, (handler, response, admin) in handlers.items():
        if admin:
            async_register_admin_service(
                hass, DOMAIN, name, handler, SCHEMAS[name], supports_response=response
            )
        else:
            hass.services.async_register(
                DOMAIN, name, handler, schema=SCHEMAS[name], supports_response=response
            )
