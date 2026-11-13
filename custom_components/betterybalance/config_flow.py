"""Config flow for BetteryBalance: four screens, and the options flow.

Screen 1: `user` detects the Solarman inverter in the entity registry, `device` asks which one
when there are several, `inverter` shows the pack derived from the inverter's settings and the
grid source. Every refusal of `core.solarman` ends the flow with its reason, the detail and the
issue link. Screen 2 `location_contract`: the rounded location and the contract terms. Screen 3
`pv`: learn the panels from the data or fill them in (`pv_fill`). Screen 4 `backtest`: the
install-time backtest checkbox. The options flow holds the knobs, the feeds key, the export
switches, the pack source, the ready-made view and a backtest run.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import callback
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import selector
from homeassistant.util import dt as dt_util

from . import detect as D
from .const import DOMAIN, ISSUE_URL, PACK_DIRNAME, STORE_DIRNAME
from .core import solarman as S
from .core.pack import PackError, load_pack
from .core.site import contract as C
from .live import copy as copy_
from .live import pack_update as PU

GRID_KEYS = ("grid_import_kwh", "grid_export_kwh")
# the sentences the flow puts into the translated texts as placeholders, per language of Home
# Assistant (``live.copy`` resolves the language; English when unknown)
COPY = {
    "en": {
        "grid_inverter": (
            "your inverter's grid measurement; a P1 reader in Home Assistant gives better "
            "export figures"
        ),
        "grid_p1": "your P1 meter's import and export counters",
        "grid_custom": "the grid meter counters you chose",
        "tariffs": (
            "Your grid meter pick leaves out these P1 tariff counters: {missing}. The grid "
            "flows then miss that tariff's energy."
        ),
        "history": (
            "the recorder holds {days} {unit} of hourly statistics for the grid counters; the "
            "backtest needs {need}"
        ),
        "day": "day",
        "days": "days",
        "no_pack": "no data pack is installed yet; set the pack source under Data pack first",
        "short_history": "the recorder holds less than 14 days of hourly grid statistics",
        "no_end_soc": (
            "the last hour of the span has no battery state of charge in the recorder; try "
            "again after the next full hour"
        ),
        "backtest_running": "a backtest is running already; wait until it is done",
        "backtest_stopping": "BetteryBalance is stopping; start the backtest after it is loaded",
        "no_service": (
            "the backtest service is not running; restart Home Assistant after the update"
        ),
        "start_pack": (
            "With the box ticked, the run starts by itself once the hourly history is read, a "
            "few minutes after this screen closes."
        ),
        "start_no_pack": (
            "A backtest needs a data pack, and none is installed yet. With the box ticked, the "
            "run starts by itself once a pack is installed (Configure, Data pack); you can "
            "also start it from the options then."
        ),
    },
    "nl": {
        "grid_inverter": (
            "de netmeting van je omvormer; een P1-lezer in Home Assistant geeft betere "
            "teruglevercijfers"
        ),
        "grid_p1": "de afname- en teruglevertellers van je P1-meter",
        "grid_custom": "de netmetertellers die je koos",
        "tariffs": (
            "Je keuze van netmetertellers laat deze P1-tarieftellers weg: {missing}. De "
            "netstromen missen dan de energie van dat tarief."
        ),
        "history": (
            "de recorder bevat {days} {unit} aan uurstatistieken van de nettellers; de backtest "
            "heeft er {need} nodig"
        ),
        "day": "dag",
        "days": "dagen",
        "no_pack": (
            "er is nog geen datapakket geïnstalleerd; stel eerst de pakketbron in onder "
            "Datapakket"
        ),
        "short_history": "de recorder bevat minder dan 14 dagen aan uurstatistieken van het net",
        "no_end_soc": (
            "het laatste uur van de periode heeft geen laadtoestand van de batterij in de "
            "recorder; probeer het opnieuw na het volgende volle uur"
        ),
        "backtest_running": "er loopt al een backtest; wacht tot die klaar is",
        "backtest_stopping": (
            "BetteryBalance stopt; start de backtest nadat de integratie geladen is"
        ),
        "no_service": "de backtestdienst draait niet; herstart Home Assistant na de update",
        "start_pack": (
            "Met het vinkje aan start de berekening vanzelf zodra de uurgeschiedenis is "
            "ingelezen, enkele minuten nadat dit scherm sluit."
        ),
        "start_no_pack": (
            "Een backtest heeft een datapakket nodig, en er is er nog geen geïnstalleerd. Met "
            "het vinkje aan start de berekening vanzelf zodra een pakket is geïnstalleerd "
            "(Configureren, Datapakket); je kunt hem dan ook starten vanuit de opties."
        ),
    },
}
INVERTER_NOTE = COPY["en"]["grid_inverter"]
P1_NOTE = COPY["en"]["grid_p1"]
CUSTOM_NOTE = COPY["en"]["grid_custom"]
GRID_NOTES = {"inverter": "grid_inverter", "p1": "grid_p1", "custom": "grid_custom"}
MAX_CAPACITY_KWH = 1000.0
MAX_FLOOR_PCT = 95
TITLE = "BetteryBalance"
LOSS_PROFILE = "deye"
EXECUTION_PROFILE = "deye_amps"  # the one-ampere-step template (design "deye")
MAX_STRINGS = 8
DEFAULT_TILT = 35
DEFAULT_AZIMUTH = 180
MAX_KWP = 100.0
DEFAULT_OPTIONS: dict[str, Any] = {
    "tennet_key": None,
    "export_flows": False,
    "pool_results": False,
    "pack_source": None,
    "fetch_ida": False,  # design 16: the Nord Pool terms are not settled for automated use
}
OPTIONS_MENU = ["advanced", "feeds", "export", "pack", "view", "backtest"]
VIEW_SNIPPET = "strategy:\n  type: custom:betterybalance"
VIEW_PATH = "custom_components/betterybalance/dashboard/view.yaml"
# the run_backtest refusals (Task 23 raises them with these translation keys) as sentences
REFUSALS = {
    k: COPY["en"][k]
    for k in ("no_pack", "short_history", "no_end_soc", "backtest_running", "backtest_stopping")
}
NO_SERVICE = COPY["en"]["no_service"]
UNIT_KNOBS = {
    "hurdle_ct_kwh": "ct/kWh",
    "wear_eur_cycle": "EUR",
    "latency_s": "s",
    "dwell_s": "s",
    "trade_floor_w": "W",
    "latest_commit_min": "min",
}
CHOICE_KNOBS = {
    "terminal_target": ("floor",),
    "impl": C.IMPLS,
    "commit_rule": C.COMMIT_RULES,
}
MAX_COMMIT_MIN = 14  # the last minute of a quarter at which a trade may start
MAX_LATENCY_S = 300  # a writer slower than 5 minutes cannot follow a quarter
MAX_DWELL_S = 900  # a hold longer than a quarter blocks every reversal of the next one
WIDEN_KNOBS = ("predicted_widen", "learning_widen")  # factors below 1 would narrow the bands
INT_KNOBS = tuple(f.name for f in fields(C.Knobs) if f.type in ("int", int))
FRACTION_KNOBS = (
    "q_take",
    "q_feed",
    "second_plane_penalty",
    "promotion_margin",
    "drift_alarm",
)


def _say(hass, key: str, **values: Any) -> str:
    """The ``COPY`` sentence ``key`` in Home Assistant's language, ``values`` filled in."""
    return COPY[copy_._lang(hass.config.language)][key].format(**values)


def _number(unit: str, low: float, high: float, step: float) -> selector.NumberSelector:
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=low,
            max=high,
            step=step,
            unit_of_measurement=unit,
            mode=selector.NumberSelectorMode.BOX,
        )
    )


def _select(
    options: list[str] | tuple[str, ...], translation_key: str | None = None
) -> selector.SelectSelector:
    """A dropdown; with ``translation_key`` its labels come from ``selector.<key>.options``."""
    config = selector.SelectSelectorConfig(
        options=list(options), mode=selector.SelectSelectorMode.DROPDOWN
    )
    if translation_key is not None:
        config["translation_key"] = translation_key
    return selector.SelectSelector(config)


def _counters() -> selector.EntitySelector:
    return selector.EntitySelector(
        selector.EntitySelectorConfig(domain="sensor", device_class="energy", multiple=True)
    )


class BetteryConfigFlow(ConfigFlow, domain=DOMAIN):
    """Set up BetteryBalance."""

    VERSION = 1

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """The options flow of an entry."""
        return BetteryOptionsFlow()

    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._device_id: str | None = None
        self._data: dict[str, Any] = {}
        self._pack: dict[str, Any] = {}
        self._p1: dict[str, list[str]] = {}

    def _refuse(self, refusal: S.Refusal) -> ConfigFlowResult:
        return self.async_abort(
            reason=refusal.reason,
            description_placeholders={"detail": refusal.detail, "issue_url": ISSUE_URL},
        )

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Find the Solarman devices with a battery. One entry per Home Assistant (the
        manifest's ``single_config_entry``: Home Assistant aborts a second flow before it
        starts; this check covers a flow started before the first entry was made)."""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")
        self._entries = D.registry_entries(self.hass)
        if not self._entries:
            return self._refuse(S.Refusal("no_solarman", "no ha-solarman entity found"))
        found = S.devices(self._entries)
        if not found:
            return self._refuse(S.Refusal("no_battery", "no Solarman device has Battery Power"))
        if len(found) == 1:
            self._device_id = found[0]
            return await self.async_step_inverter()
        return await self.async_step_device()

    async def async_step_device(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Several Solarman inverters: a dropdown of their device names."""
        found = S.devices(self._entries)
        if user_input is not None and user_input.get("device") in found:
            self._device_id = user_input["device"]
            return await self.async_step_inverter()
        devices = dr.async_get(self.hass)
        options = []
        for device_id in found:
            device = devices.async_get(device_id)
            label = (device.name_by_user or device.name) if device else None
            options.append(selector.SelectOptionDict(value=device_id, label=label or device_id))
        schema = vol.Schema(
            {
                vol.Required("device", default=found[0]): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options, mode=selector.SelectSelectorMode.DROPDOWN
                    )
                )
            }
        )
        return self.async_show_form(step_id="device", data_schema=schema)

    async def _detect(self) -> None:
        """Role mapping, signs and the derived pack of the chosen device; raises Refusal."""
        assert self._device_id is not None
        mapping = S.match_entities(self._entries, self._device_id)
        mode = await D.io_mode(self.hass, mapping)
        if mode is not None:
            mapping = S.match_entities(self._entries, self._device_id, io_mode=mode)
        settings = await D.read_settings(self.hass, mapping)
        now = dt_util.utcnow()
        voltage = await D.voltage_mean_v(self.hass, mapping["battery_voltage"], now)
        self._pack = S.derive_pack(
            settings["capacity_ah"],
            voltage,
            settings["max_charge_a"],
            settings["max_discharge_a"],
            settings["shutdown_soc"],
            settings["low_soc"],
            bms_charge_a=settings["bms_charge_a"],
            bms_discharge_a=settings["bms_discharge_a"],
        )
        self._p1 = D.p1_counters(D.dsmr_entries(self.hass))
        self._data = {
            "device_id": self._device_id,
            "entities": dict(mapping),
            "signs": await D.inferred_signs(self.hass, mapping, now),
        }

    def _inverter_schema(self) -> vol.Schema:
        pack = self._pack
        kw = S.INVERTER_KW
        return vol.Schema(
            {
                vol.Required("capacity_kwh", default=round(pack["capacity_kwh"], 2)): _number(
                    "kWh", 0.1, MAX_CAPACITY_KWH, 0.01
                ),
                vol.Required("charge_kw", default=round(pack["charge_kw"], 2)): _number(
                    "kW", 0.1, kw, 0.01
                ),
                vol.Required("discharge_kw", default=round(pack["discharge_kw"], 2)): _number(
                    "kW", 0.1, kw, 0.01
                ),
                vol.Required("soc_floor_pct", default=round(pack["soc_floor"] * 100)): _number(
                    "%", 0, 100, 1
                ),
                vol.Optional("grid_import_kwh", default=self._p1.get("grid_import_kwh", [])): (
                    _counters()
                ),
                vol.Optional("grid_export_kwh", default=self._p1.get("grid_export_kwh", [])): (
                    _counters()
                ),
            }
        )

    def _grid_source(self, grid: dict[str, list[str]]) -> str:
        """The grid source label of a choice of counters.

        `inverter` for no choice or the inverter's own totals, `p1` when every chosen entity is
        a dsmr counter, else `custom`.
        """
        chosen = [i for k in GRID_KEYS for i in grid[k]]
        entities = self._data["entities"]
        own = {
            "grid_import_kwh": [entities["grid_import_total"]],
            "grid_export_kwh": [entities["grid_export_total"]],
        }
        if not chosen or grid == own:
            return "inverter"
        if set(chosen) <= D.dsmr_counter_ids(self.hass):
            return "p1"
        return "custom"

    def _show_inverter(
        self, errors: dict[str, str] | None = None, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The inverter form; a re-shown form keeps the submitted values as suggestions."""
        schema = self._inverter_schema()
        if user_input is None:
            source = "p1" if self._p1 else "inverter"
        else:
            schema = self.add_suggested_values_to_schema(schema, user_input)
            source = self._grid_source(self._grid(user_input))
        return self.async_show_form(
            step_id="inverter",
            data_schema=schema,
            errors=errors or {},
            description_placeholders={"grid_note": _say(self.hass, GRID_NOTES[source])},
        )

    @staticmethod
    def _grid(user_input: dict[str, Any]) -> dict[str, list[str]]:
        return {k: list(user_input.get(k) or []) for k in GRID_KEYS}

    async def async_step_inverter(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The derived pack and the grid source, each changeable."""
        if user_input is None:
            assert self._device_id is not None
            await self.async_set_unique_id(self._device_id)
            self._abort_if_unique_id_configured()
            try:
                await self._detect()
            except S.Refusal as refusal:
                return self._refuse(refusal)
            return self._show_inverter()
        grid = self._grid(user_input)
        if bool(grid["grid_import_kwh"]) != bool(grid["grid_export_kwh"]):
            return self._show_inverter({"base": "grid_pair"}, user_input)
        if not 0 <= user_input["soc_floor_pct"] <= MAX_FLOOR_PCT:
            return self._show_inverter({"soc_floor_pct": "floor_range"}, user_input)
        entities = self._data["entities"]
        self._data["grid_source"] = self._grid_source(grid)
        if grid["grid_import_kwh"]:
            entities.update(grid)
        else:
            entities["grid_import_kwh"] = [entities["grid_import_total"]]
            entities["grid_export_kwh"] = [entities["grid_export_total"]]
        pack = {
            **self._pack,
            "capacity_kwh": float(user_input["capacity_kwh"]),
            "charge_kw": float(user_input["charge_kw"]),
            "discharge_kw": float(user_input["discharge_kw"]),
            "soc_floor": float(user_input["soc_floor_pct"]) / 100.0,
        }
        self._data["site"] = {"pack": pack}
        return await self._after_inverter()

    async def _after_inverter(self) -> ConfigFlowResult:
        """Screen 1 done: screen 2."""
        return await self.async_step_location_contract()

    # --- screen 2: location and contract -------------------------------------------------------

    def _grid_warning(self) -> str:
        """A sentence when the grid pick holds some of a P1 meter's tariff counters, else ''."""
        entities = self._data["entities"]
        grid = {k: entities[k] for k in GRID_KEYS}
        missing = D.missing_tariffs(D.dsmr_entries(self.hass), grid)
        if not missing:
            return ""
        return _say(self.hass, "tariffs", missing=", ".join(missing))

    def _location_schema(self) -> vol.Schema:
        lat, lon = C.round_location(self.hass.config.latitude, self.hass.config.longitude)
        terms = C.Terms()
        return vol.Schema(
            {
                vol.Required("location", default={"latitude": lat, "longitude": lon}): (
                    selector.LocationSelector(selector.LocationSelectorConfig(radius=False))
                ),
                vol.Required(
                    "fee_import_ct", default=round(terms.fee_import_eur_kwh * 100, 4)
                ): _number("ct/kWh", 0, 100, 0.01),
                vol.Required(
                    "fee_export_ct", default=round(terms.fee_export_eur_kwh * 100, 4)
                ): _number("ct/kWh", 0, 100, 0.01),
                vol.Required("dual_pricing", default=terms.dual_pricing): (
                    selector.BooleanSelector()
                ),
                vol.Required("btw_pct", default=round(terms.btw * 100)): _number("%", 0, 100, 1),
                vol.Required("tax_mode", default=terms.tax_mode): _select(C.TAX_MODES, "tax_mode"),
                vol.Optional(
                    "balans_revenue_eur_month", default=terms.balans_revenue_eur_month
                ): _number("EUR", 0, 10000, 0.01),
            }
        )

    async def async_step_location_contract(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The location the feeds see, rounded to 0,1 degree, and the contract terms."""
        if user_input is None:
            return self.async_show_form(
                step_id="location_contract",
                data_schema=self._location_schema(),
                description_placeholders={"grid_warning": self._grid_warning()},
            )
        location = user_input["location"]
        lat, lon = C.round_location(float(location["latitude"]), float(location["longitude"]))
        site = self._data["site"]
        site["latitude"], site["longitude"] = lat, lon
        site["terms"] = {
            "fee_import_eur_kwh": round(float(user_input["fee_import_ct"]) / 100.0, 6),
            "fee_export_eur_kwh": round(float(user_input["fee_export_ct"]) / 100.0, 6),
            "dual_pricing": bool(user_input["dual_pricing"]),
            "btw": round(float(user_input["btw_pct"]) / 100.0, 6),
            "tax_mode": user_input["tax_mode"],
            "balans_revenue_eur_month": float(user_input.get("balans_revenue_eur_month") or 0.0),
        }
        site["loss_profile"] = LOSS_PROFILE
        site["execution_profile"] = EXECUTION_PROFILE
        return await self.async_step_pv()

    # --- screen 3: solar panels ----------------------------------------------------------------

    async def async_step_pv(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Learn the panels from the data, or fill them in."""
        return self.async_show_menu(step_id="pv", menu_options=["pv_learn", "pv_fill"])

    async def async_step_pv_learn(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The integration fits each string; no planes stored."""
        self._data["pv_mode"] = "learn"
        self._data["site"]["pv"] = {}
        return await self.async_step_backtest()

    def _strings(self) -> list[tuple[str, str]]:
        """(row suffix, string) per string in string order: the mapped PV1 .. PV8 (PV1 without
        one), then the GEN port when it is mapped (a microinverter on it is one more string)."""
        entities = self._data["entities"]
        found = [
            (str(i), f"pv{i}") for i in range(1, MAX_STRINGS + 1) if f"pv{i}_power" in entities
        ]
        rows = found or [("1", "pv1")]
        if "gen_power" in entities:
            rows.append(("gen", "gen"))
        return rows

    def _pv_schema(self) -> vol.Schema:
        schema: dict = {}
        for i, _ in self._strings():
            schema[vol.Required(f"tilt_{i}", default=DEFAULT_TILT)] = _number("°", 0, 90, 1)
            schema[vol.Required(f"azimuth_{i}", default=DEFAULT_AZIMUTH)] = _number("°", 0, 359, 1)
            schema[vol.Required(f"kwp_{i}")] = _number("kWp", 0, MAX_KWP, 0.01)
        return vol.Schema(schema)

    async def async_step_pv_fill(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """One row per string: tilt, azimuth (compass degrees) and kWp; kWp 0 skips the string."""
        if user_input is None:
            return self.async_show_form(step_id="pv_fill", data_schema=self._pv_schema())
        planes = [
            {
                "tilt_deg": float(user_input[f"tilt_{i}"]),
                "azimuth_deg": float(user_input[f"azimuth_{i}"]),
                "kwp": float(user_input[f"kwp_{i}"]),
                "string": string,
            }
            for i, string in self._strings()
            if float(user_input[f"kwp_{i}"]) > 0
        ]
        if not planes:
            return self.async_show_form(
                step_id="pv_fill",
                data_schema=self.add_suggested_values_to_schema(self._pv_schema(), user_input),
                errors={"base": "no_planes"},
            )
        self._data["pv_mode"] = "fill"
        self._data["site"]["pv"] = {"planes": planes}
        return await self.async_step_backtest()

    # --- screen 4: the backtest ----------------------------------------------------------------

    async def _backtest_reason(self) -> str:
        """'' when the grid import counters hold HISTORY_DAYS of hourly statistics, else why not."""
        ids = list(self._data["entities"]["grid_import_kwh"])
        days = await D.history_days(self.hass, ids, dt_util.utcnow())
        if days >= D.HISTORY_DAYS:
            return ""
        whole = int(days)
        unit = _say(self.hass, "day" if whole == 1 else "days")
        return _say(self.hass, "history", days=whole, unit=unit, need=D.HISTORY_DAYS)

    async def async_step_backtest(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The install-time backtest checkbox, then the entry."""
        if user_input is None:
            reason = await self._backtest_reason()
            schema = vol.Schema(
                {vol.Required("run_backtest", default=not reason): selector.BooleanSelector()}
            )
            has_pack = await self.hass.async_add_executor_job(
                _pack_installed, self.hass.config.path(STORE_DIRNAME, PACK_DIRNAME)
            )
            start = _say(self.hass, "start_pack" if has_pack else "start_no_pack")
            return self.async_show_form(
                step_id="backtest",
                data_schema=schema,
                description_placeholders={"reason": reason, "start": start},
            )
        self._data["backtest"] = bool(user_input["run_backtest"])
        return self._create()

    def _create(self) -> ConfigFlowResult:
        site = self._data["site"]
        C.from_dict(site)  # every field above is range-checked; this keeps the entry loadable
        data = {
            "install_id": uuid.uuid4().hex,
            "device_id": self._data["device_id"],
            "entities": self._data["entities"],
            "signs": self._data["signs"],
            "grid_source": self._data["grid_source"],
            "site": site,
            "pv_mode": self._data["pv_mode"],
            "backtest": self._data["backtest"],
        }
        options = {"knobs": asdict(C.Knobs()), **DEFAULT_OPTIONS}
        return self.async_create_entry(title=TITLE, data=data, options=options)


# --- the options flow --------------------------------------------------------------------------


def _knob_selector(name: str, default: Any):
    if name in CHOICE_KNOBS:
        return _select(CHOICE_KNOBS[name])
    if isinstance(default, bool):
        return selector.BooleanSelector()
    unit = UNIT_KNOBS.get(name, "")
    if isinstance(default, int):
        return _number(unit, 0, 100000, 1)
    return selector.NumberSelector(
        selector.NumberSelectorConfig(
            min=0, step="any", unit_of_measurement=unit, mode=selector.NumberSelectorMode.BOX
        )
    )


# read by the export services at each call: a change takes effect without a reload
CALL_TIME_OPTIONS = ("export_flows", "pool_results")


def run_options(options) -> dict[str, Any]:
    """The effective options the running site starts on (``effective_options`` without
    ``CALL_TIME_OPTIONS``): the update listener reloads the entry when these change."""
    out = effective_options(options)
    for key in CALL_TIME_OPTIONS:
        out.pop(key, None)
    return out


def effective_options(options) -> dict[str, Any]:
    """The options in effect: the stored ones over ``DEFAULT_OPTIONS``, the knobs over the
    shipped defaults."""
    out = {**DEFAULT_OPTIONS, **dict(options)}
    out["knobs"] = _knobs(out.get("knobs") or {})
    return out


def _knobs(current: dict[str, Any]) -> dict[str, Any]:
    """The stored knobs over the shipped defaults (a newer release may add knobs)."""
    return {**asdict(C.Knobs()), **{k: v for k, v in current.items() if k in C.KNOB_HELP}}


def _knob_errors(knobs: dict[str, Any]) -> dict[str, str]:
    """Form errors of a knob set: ranges first, then the contract's own checks."""
    errors = {k: "knob_range" for k in FRACTION_KNOBS if not 0 <= float(knobs[k]) <= 1}
    if not 0 <= float(knobs["latest_commit_min"]) <= MAX_COMMIT_MIN:
        errors["latest_commit_min"] = "commit_minute_range"
    if not 0 <= float(knobs["latency_s"]) <= MAX_LATENCY_S:
        errors["latency_s"] = "latency_range"
    if not 0 <= float(knobs["dwell_s"]) <= MAX_DWELL_S:
        errors["dwell_s"] = "dwell_range"
    errors.update({k: "widen_range" for k in WIDEN_KNOBS if not float(knobs[k]) >= 1})
    errors.update({k: "knob_integer" for k in INT_KNOBS if not float(knobs[k]).is_integer()})
    if not errors:
        try:
            C.from_dict({"latitude": 0.0, "longitude": 0.0, "pack": _CHECK_PACK, "knobs": knobs})
        except ValueError:
            errors["base"] = "knob_invalid"
    return errors


# a valid pack, only so that from_dict checks the knobs
_CHECK_PACK = {
    "capacity_kwh": 1.0,
    "charge_kw": 1.0,
    "discharge_kw": 1.0,
    "soc_floor": 0.0,
    "soc_ceiling": 1.0,
}


class BetteryOptionsFlow(OptionsFlow):
    """Options: advanced knobs, feeds, export, pack, the view and a backtest run."""

    def _options(self) -> dict[str, Any]:
        return effective_options(self.config_entry.options)

    def _save(self, **changes: Any) -> ConfigFlowResult:
        return self.async_create_entry(data={**self._options(), **changes})

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """The options menu."""
        return self.async_show_menu(step_id="init", menu_options=OPTIONS_MENU)

    async def async_step_advanced(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Every knob, prefilled with the stored value, each with its KNOB_HELP line."""
        knobs = self._options()["knobs"]
        errors: dict[str, str] = {}
        if user_input is not None:
            types = {f.name: type(getattr(C.Knobs(), f.name)) for f in fields(C.Knobs)}
            raw = {k: user_input.get(k, knobs[k]) for k in C.KNOB_HELP}
            errors = _knob_errors(raw)
            new = {k: types[k](v) for k, v in raw.items()}
            if not errors:
                return self._save(knobs=new)
            knobs = {**knobs, **user_input}
        defaults = asdict(C.Knobs())
        schema = vol.Schema(
            {
                vol.Required(name, default=knobs[name]): _knob_selector(name, defaults[name])
                for name in C.KNOB_HELP
            }
        )
        return self.async_show_form(step_id="advanced", data_schema=schema, errors=errors)

    async def async_step_feeds(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """The optional TenneT API key for the 12-second live feed, and whether the box fetches
        the intraday auction prices from Nord Pool (off by default)."""
        if user_input is not None:
            key = (user_input.get("tennet_key") or "").strip()
            return self._save(
                tennet_key=key or None, fetch_ida=bool(user_input.get("fetch_ida", False))
            )
        options = self._options()
        schema = vol.Schema(
            {
                vol.Optional(
                    "tennet_key", description={"suggested_value": options["tennet_key"]}
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
                vol.Required(
                    "fetch_ida", default=bool(options["fetch_ida"])
                ): selector.BooleanSelector(),
            }
        )
        return self.async_show_form(step_id="feeds", data_schema=schema)

    async def async_step_export(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Flows in the ledger export, and the results file for pooling."""
        if user_input is not None:
            return self._save(
                export_flows=bool(user_input["export_flows"]),
                pool_results=bool(user_input["pool_results"]),
            )
        options = self._options()
        schema = vol.Schema(
            {
                vol.Required(k, default=bool(options[k])): selector.BooleanSelector()
                for k in ("export_flows", "pool_results")
            }
        )
        return self.async_show_form(step_id="export", data_schema=schema)

    async def async_step_pack(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """The local directory (or ``file://`` URL) the data pack releases come from, checked
        as ``live.pack_update.check_release`` reads it; empty for none (no updates)."""
        errors: dict[str, str] = {}
        current = self._options()["pack_source"]
        if user_input is not None:
            path = (user_input.get("pack_source") or "").strip()
            if not path:
                return self._save(pack_source=None)
            if await self.hass.async_add_executor_job(_pack_source_reads, path):
                return self._save(pack_source=path)
            errors["pack_source"] = "not_a_pack_source"
            current = path
        schema = vol.Schema(
            {
                vol.Optional(
                    "pack_source", description={"suggested_value": current}
                ): selector.TextSelector()
            }
        )
        return self.async_show_form(step_id="pack", data_schema=schema, errors=errors)

    async def async_step_view(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """The ready-made view: the strategy snippet (``custom:betterybalance``, the view strategy
        of the card file) and the path of ``dashboard/view.yaml`` to copy."""
        if user_input is not None:
            return self._save()
        return self.async_show_form(
            step_id="view",
            data_schema=vol.Schema({}),
            description_placeholders={"snippet": VIEW_SNIPPET, "path": VIEW_PATH},
        )

    async def async_step_backtest(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Submitting starts betterybalance.run_backtest; its refusal ends the flow with why."""
        if not self.hass.services.has_service(DOMAIN, "run_backtest"):
            return self.async_abort(
                reason="backtest_unavailable",
                description_placeholders={"reason": _say(self.hass, "no_service")},
            )
        if user_input is None:
            return self.async_show_form(step_id="backtest", data_schema=vol.Schema({}))
        try:
            await self.hass.services.async_call(DOMAIN, "run_backtest", {}, blocking=True)
        except ServiceValidationError as err:
            key = err.translation_key or ""
            reason = _say(self.hass, key) if key in REFUSALS else str(err)
            return self.async_abort(
                reason="backtest_refused", description_placeholders={"reason": reason}
            )
        except HomeAssistantError as err:
            return self.async_abort(
                reason="backtest_refused", description_placeholders={"reason": str(err)}
            )
        return self._save()


def _pack_installed(folder: str) -> bool:
    """Whether the integration's pack folder holds a version folder that loads (blocking): a
    store kept from an earlier install can have one."""
    root = Path(folder)
    if not root.is_dir():
        return False
    for sub in sorted(root.iterdir()):
        try:
            if sub.is_dir():
                load_pack(sub)
                return True
        except PackError:
            continue
    return False


def _pack_source_reads(source: str) -> bool:
    """Whether ``source`` (a directory, a ``file://`` URL or the manifest path) holds a pack
    release that ``check_release`` reads (blocking)."""
    try:
        PU.check_release(source, None)
    except PU.PackError:
        return False
    return True
