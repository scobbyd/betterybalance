"""The sensors: state, plan, results, shadow SoC, nowcasts, PV expectation and backtest.

Every value comes from the coordinator's snapshot (``coordinator.data``); the views of
``live.views`` turn it into states and attributes. Prices are ct/kWh, money EUR. The
``actual`` path never reaches a sensor: the results are the shadow's own (design 10.4).
"""

from __future__ import annotations

import math

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, UnitOfPower

from .entity import BetteryEntity
from .live import views as V
from .live.views import ACTIONS

CT_KWH = "ct/kWh"
EUR = "EUR"
BACKTEST_STATES = ("idle", "running", "done", "failed")
PLAN_ARRAYS = frozenset(
    {
        "ts", "take_q10", "take_q50", "take_q90", "feed_q10", "feed_q50", "feed_q90",
        "soc_pct", "action",
    }
)
_NOWCAST_QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _round(x, dec: int = 2) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, dec) if math.isfinite(x) else None


async def async_setup_entry(hass, entry, async_add_entities) -> None:
    coord = entry.runtime_data
    async_add_entities(
        [
            StateSensor(coord, "state"),
            PlanSensor(coord, "plan"),
            ResultTodaySensor(coord, "result_today"),
            ResultSensor(coord, "result"),
            ResultSensor(coord, "result_provisional"),
            ShadowSocSensor(coord, "shadow_soc"),
            NowcastSensor(coord, "take_nowcast"),
            NowcastSensor(coord, "feed_nowcast"),
            PvExpectedSensor(coord, "pv_expected"),
            BacktestSensor(coord, "backtest"),
        ]
    )


class BetterySensor(BetteryEntity, SensorEntity):
    _platform = "sensor"


class StateSensor(BetterySensor):
    """The newest decision: ``hold``, ``charge`` or ``discharge`` with its verdict."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = list(ACTIONS)

    def _refresh(self, data: dict) -> None:
        status, plan = data["status"], data.get("plan")
        planned = None if plan is None else plan["planned"]
        state, attrs = V.state_view(
            status, status.get("ranked"), self.hass_language(), planned
        )
        self._attr_native_value = state
        self._attr_extra_state_attributes = attrs

    def hass_language(self) -> str:
        hass = self.hass or self.coordinator.hass
        return hass.config.language


class PlanSensor(BetterySensor):
    """The expected result over the next 36 hours (EUR) and the plan arrays (unrecorded),
    from the snapshot's plan (computed in the executor once per ranking; unknown without a
    ranking or when the plan failed)."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_native_unit_of_measurement = EUR
    _unrecorded_attributes = PLAN_ARRAYS

    def _refresh(self, data: dict) -> None:
        plan = data.get("plan")
        self._attr_native_value = None if plan is None else plan["state"]
        self._attr_extra_state_attributes = {} if plan is None else plan["attrs"]


class ResultTodaySensor(BetterySensor):
    """Today's Amsterdam-day result, provisional until settled."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_native_unit_of_measurement = EUR

    def _refresh(self, data: dict) -> None:
        t = data["totals"]
        self._attr_native_value = _round(t.today_eur)
        self._attr_extra_state_attributes = {
            "settled_share": _round(t.today_settled_share, 3),
            "missing_quarters": int(t.today_missing),
        }


class ResultSensor(BetterySensor):
    """The running total of the shadow's result: settled (``result``) or with the
    provisional quarters (``result_provisional``). The bars come from the external
    statistic ``betterybalance:result``."""

    _attr_device_class = SensorDeviceClass.MONETARY
    _attr_state_class = SensorStateClass.TOTAL
    _attr_native_unit_of_measurement = EUR

    def _refresh(self, data: dict) -> None:
        t = data["totals"]
        settled = self.key == "result"
        self._attr_native_value = _round(t.settled_eur if settled else t.provisional_eur)
        if settled:
            terms = self.coordinator.site.lc.cfg.terms
            self._attr_extra_state_attributes = {
                "balans_revenue_eur_month": float(terms.balans_revenue_eur_month)
            }


class ShadowSocSensor(BetterySensor):
    """The shadow battery's state of charge."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_device_class = SensorDeviceClass.BATTERY
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _refresh(self, data: dict) -> None:
        kwh = float(data["status"]["soc_kwh"])
        cap = float(self.coordinator.site.lc.cfg.pack.capacity_kwh)
        self._attr_native_value = _round(kwh / cap * 100.0, 1) if cap > 0 else None
        self._attr_extra_state_attributes = {"kwh": _round(kwh, 3)}


class NowcastSensor(BetterySensor):
    """The running quarter's take or feed price band (ct/kWh): state the q50, attributes
    the quantiles, the horizon in quarters ahead (0, the running quarter) and the tier."""

    _attr_native_unit_of_measurement = CT_KWH
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _refresh(self, data: dict) -> None:
        ranked = data["status"].get("ranked")
        names = [f"q{round(q * 100)}" for q in _NOWCAST_QS]
        if ranked is None:
            self._attr_native_value = None
            self._attr_extra_state_attributes = dict.fromkeys([*names, "horizon", "tier"])
            return
        w = ranked.window
        band = w.take if self.key == "take_nowcast" else w.feed
        values = {n: _round(V.ct_kwh(band[i, 0])) for i, n in enumerate(names)}
        self._attr_native_value = values["q50"]
        self._attr_extra_state_attributes = {**values, "horizon": 0, "tier": w.tier[0] or None}


class PvExpectedSensor(BetterySensor):
    """The running quarter's PV expectation in kW, per string."""

    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.KILO_WATT
    _attr_state_class = SensorStateClass.MEASUREMENT

    def _refresh(self, data: dict) -> None:
        status = data["status"]
        pv = status.get("pv") or {}
        self._attr_native_value = _round(pv.get("kw"), 3)
        self._attr_extra_state_attributes = {
            "per_string": dict(pv.get("per_string") or {}),
            "model_version": pv.get("model_version"),
            "learning": bool(status.get("learning")),
            "drift": bool(status.get("drift")),
        }


class BacktestSensor(BetterySensor):
    """The install-time backtest: ``idle``, ``running`` (``progress`` 0 to 1 and ``eta_s``),
    ``done`` (``headline``: the shadow's own result, ``live.backtest.result_headline``, and
    ``report``: the report file) or ``failed`` (``error``)."""

    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = list(BACKTEST_STATES)

    def _refresh(self, data: dict) -> None:
        bt = data.get("backtest") or {}
        last = bt.get("last")
        if bt.get("running"):
            state = "running"
        elif last is not None and last.get("status") in ("done", "failed"):
            state = last["status"]
        else:
            state = "idle"
        head = None if last is None else last.get("headline")
        self._attr_native_value = state
        self._attr_extra_state_attributes = {
            "progress": bt.get("progress"),
            "eta_s": bt.get("eta_s"),
            "headline": head if state == "done" else None,
            "report": None if last is None or state != "done" else last.get("report_path"),
            "error": (head or {}).get("error") if state == "failed" else None,
        }
