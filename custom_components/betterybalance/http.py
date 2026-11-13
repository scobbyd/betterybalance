"""The authenticated ledger download and the ledger tail websocket command (design 10.5).

- ``LedgerDownloadView`` at ``URL_LEDGER``: ``GET ?from=YYYY-MM-DD&to=YYYY-MM-DD[&flows=1]``
  returns the export of those Amsterdam days (``live.export``) as
  ``betterybalance-ledger-<from>-<to>.zip``. ``flows`` defaults to the ``export_flows``
  option; minute flows cover at most ``live.export.MAX_FLOW_DAYS`` days (400 above). Admin
  only, as HA's diagnostics and backup downloads: the export describes the house. The zip is
  streamed row by row into a temporary file of the folder ``temp_dir()`` in the executor,
  served, then removed. A failure or a cancelled request (the client went away) removes it
  too, also when the executor job finishes after the cancel; the integration's setup and every
  download sweep the folder of files a crash left (``sweep_temp``).
- ``WS_LEDGER_TAIL`` (``limit`` default 20): the newest trading decisions, each with its
  quarter's newest settlement (``live.export.ledger_tail``), the rows the card's ledger mode
  shows.

The export and the zip run in the executor. The tail is open to any user (it carries the
shadow's decisions and the published prices only).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import tempfile
import time
from functools import partial
from http import HTTPStatus
from pathlib import Path

import voluptuous as vol
from aiohttp import web
from homeassistant.components import websocket_api
from homeassistant.components.http import HomeAssistantView, require_admin
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers.http import KEY_HASS
from homeassistant.util import dt as dt_util

from .const import DOMAIN, URL_LEDGER, WS_LEDGER_TAIL
from .live import export as X
from .services import coordinator_for

TAIL_LIMIT = 20
TAIL_MAX = 500
_TRUE = ("1", "true", "yes", "on")
TEMP_DIRNAME = "betterybalance-ledger"  # under the system temp folder, never in a backup
TEMP_PREFIX = "betterybalance-ledger-"
STALE_S = 3600  # a download sweeps temporary zips older than this


def temp_dir() -> Path:
    """The folder of the temporary ledger zips."""
    return Path(tempfile.gettempdir()) / TEMP_DIRNAME


def sweep_temp(max_age_s: float = 0.0) -> int:
    """Remove the temporary ledger zips older than ``max_age_s`` (blocking); how many."""
    folder, n = temp_dir(), 0
    if not folder.is_dir():
        return 0
    limit = time.time() - max_age_s
    for path in folder.glob(f"{TEMP_PREFIX}*.zip"):
        try:
            if path.stat().st_mtime <= limit:
                path.unlink()
                n += 1
        except OSError:
            continue
    return n


def _zip_to_temp(coord, start: int, end: int, flows: bool) -> Path:
    """The export streamed into a new temporary zip file of ``temp_dir()`` (removed on any
    failure)."""
    folder = temp_dir()
    folder.mkdir(parents=True, exist_ok=True)
    sweep_temp(STALE_S)
    fd, name = tempfile.mkstemp(prefix=TEMP_PREFIX, suffix=".zip", dir=folder)
    os.close(fd)
    path = Path(name)
    try:
        X.write_export_zip(coord.site.store, coord.site.lc, start, end, flows, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


async def build_zip(hass: HomeAssistant, coord, start: int, end: int, flows: bool) -> Path:
    """``_zip_to_temp`` in the executor. A cancel (the client went away) cannot stop the
    executor job, so the file it returns later is removed then."""
    fut = hass.async_add_executor_job(_zip_to_temp, coord, start, end, flows)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:

        def drop(done: asyncio.Future) -> None:
            if not done.cancelled() and done.exception() is None:
                hass.async_add_executor_job(partial(done.result().unlink, missing_ok=True))

        fut.add_done_callback(drop)
        raise


class _TempFileResponse(web.FileResponse):
    """A ``FileResponse`` that removes its file once aiohttp has sent it (``prepare`` sends
    the whole body), also when sending fails."""

    async def prepare(self, request: web.BaseRequest):
        try:
            return await super().prepare(request)
        finally:
            path = Path(self._path)
            await request.app[KEY_HASS].async_add_executor_job(
                partial(path.unlink, missing_ok=True)
            )


class LedgerDownloadView(HomeAssistantView):
    """The ledger zip of one entry (module docstring)."""

    url = URL_LEDGER
    name = "api:betterybalance:ledger"
    requires_auth = True

    @require_admin
    async def get(self, request: web.Request, entry_id: str) -> web.StreamResponse:
        hass = request.app[KEY_HASS]
        entry = hass.config_entries.async_get_entry(entry_id)
        if entry is None or entry.domain != DOMAIN or entry.state is not ConfigEntryState.LOADED:
            return self.json_message("no loaded BetteryBalance entry", HTTPStatus.NOT_FOUND)
        if "flows" in request.query:
            flows = request.query["flows"].strip().lower() in _TRUE
        else:
            flows = bool(entry.options.get("export_flows", False))
        try:
            first = dt.date.fromisoformat(request.query["from"])
            last = dt.date.fromisoformat(request.query["to"])
        except (KeyError, ValueError):
            return self.json_message(
                "from and to must be dates (YYYY-MM-DD)", HTTPStatus.BAD_REQUEST
            )
        try:
            start, end = X.export_range(first, last, flows)
        except ValueError as err:
            return self.json_message(str(err), HTTPStatus.BAD_REQUEST)
        path = await build_zip(hass, entry.runtime_data, start, end, flows)
        name = f"betterybalance-ledger-{first.isoformat()}-{last.isoformat()}.zip"
        return _TempFileResponse(
            path,
            headers={
                "Content-Type": "application/zip",
                "Content-Disposition": f'attachment; filename="{name}"',
            },
        )


@websocket_api.websocket_command(
    {
        vol.Required("type"): WS_LEDGER_TAIL,
        vol.Optional("limit", default=TAIL_LIMIT): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=TAIL_MAX)
        ),
        vol.Optional("entry_id"): str,
    }
)
@websocket_api.async_response
async def ws_ledger_tail(
    hass: HomeAssistant, connection: websocket_api.ActiveConnection, msg: dict
) -> None:
    """The ledger tail of the named or the only loaded entry."""
    try:
        coord = coordinator_for(hass, msg.get("entry_id"))
    except ServiceValidationError as err:
        connection.send_error(msg["id"], websocket_api.ERR_NOT_FOUND, str(err))
        return
    now = int(dt_util.utcnow().timestamp())
    rows = await hass.async_add_executor_job(
        X.ledger_tail, coord.site.store, msg["limit"], now
    )
    connection.send_result(msg["id"], rows)


def async_register_http(hass: HomeAssistant) -> None:
    """Register the download view and the websocket command (the integration's
    ``async_setup``)."""
    hass.http.register_view(LedgerDownloadView())
    websocket_api.async_register_command(hass, ws_ledger_tail)
