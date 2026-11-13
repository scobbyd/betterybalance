"""Site contract: pack, tariff terms, PV hints, EV plan and rule knobs, with validation."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, fields, replace

from ..timegrid import ZONE

SIDES = ("dc", "ac")
TAX_MODES = ("net_exporter", "net_importer", "none")
IMPLS = ("ranking", "dp")
STRINGS = (*(f"pv{i}" for i in range(1, 9)), "gen")
COMMIT_RULES = ("margin", "expected_gain")


@dataclass(frozen=True)
class Pack:
    capacity_kwh: float
    charge_kw: float
    discharge_kw: float
    soc_floor: float
    soc_ceiling: float
    eta_rt: float = 0.88
    side: str = "dc"


@dataclass(frozen=True)
class Terms:
    fee_import_eur_kwh: float = 0.0165
    fee_export_eur_kwh: float = 0.0165
    btw: float = 0.21
    dual_pricing: bool = True
    # 2026 energiebelasting electricity, first band (0 to 10.000 kWh), excl. BTW;
    # 0,11085 incl. BTW; sources ANWB and rekenmachinepro.nl citing Belastingdienst
    # (2026-10-30). It only matters in the "none" and "net_importer" modes.
    tax_eur_kwh: float = 0.09161
    tax_mode: str = "net_exporter"
    salderen_end: str = "2027-01-01"
    balans_revenue_eur_month: float = 0.0


@dataclass(frozen=True)
class Plane:
    tilt_deg: float
    azimuth_deg: float
    kwp: float | None = None
    # the string the plane belongs to: "pv1" .. "pv8" or "gen" (the GEN port); None when the
    # plane names no string (planes written before the field, or fitted planes)
    string: str | None = None


@dataclass(frozen=True)
class PvHints:
    none: bool = False
    planes: tuple[Plane, ...] = ()
    correction: float = 1.0


@dataclass(frozen=True)
class EvPlan:
    energy_kwh: float
    by_local: str
    days: tuple[int, ...]


@dataclass(frozen=True)
class EvConfig:
    present: bool = False
    charger_kw: float = 11.0
    min_minutes: int = 45
    plan: EvPlan | None = None


@dataclass(frozen=True)
class Knobs:
    hurdle_ct_kwh: float = 3.0
    wear_eur_cycle: float = 0.5
    q_take: float = 0.25
    q_feed: float = 0.75
    predicted_widen: float = 1.5
    terminal_target: str = "floor"
    latency_s: int = 60
    dwell_s: int = 300
    trade_floor_w: float = 3000.0
    # impl and kappa: chosen by the Task 25 ladder on the owner's two windows (dp with
    # kappa 1 gains 1,70 EUR per live day over ranking with kappa 0,5, 90 % interval
    # 0,75 to 2,77, and 0,10 EUR per A/B day; docs/reports/2026-10-31-plan2-replay.md).
    # dual_penalty stays True: dp never reads it, and ranking keeps its penalty on.
    kappa: float = 1.0
    latest_commit_min: int = 10
    impl: str = "dp"
    commit_rule: str = "margin"
    dual_penalty: bool = True
    boundary_precommit: bool = True
    second_plane_penalty: float = 0.02
    promotion_margin: float = 0.03
    drift_alarm: float = 0.15
    learning_widen: float = 1.5  # widens the need; ranking only, dp ignores the need
    weekday_split: bool = False
    standby: bool = False


@dataclass(frozen=True)
class SiteConfig:
    latitude: float
    longitude: float
    pack: Pack
    terms: Terms
    pv: PvHints
    ev: EvConfig
    knobs: Knobs
    loss_profile: str = "flat"
    execution_profile: str = "ideal"


def _build(cls, data: dict, section: str):
    """Build a dataclass from a dict. Unknown keys are ignored so newer configs load."""
    known = {f.name for f in fields(cls)}
    try:
        return cls(**{k: v for k, v in data.items() if k in known})
    except TypeError as exc:
        raise ValueError(f"{section}: {exc}") from exc


def _require(d: dict, key: str, section: str = ""):
    """Read a required key, or raise a ValueError naming it."""
    if key not in d:
        raise ValueError(f"{section}: missing {key}" if section else f"missing {key}")
    return d[key]


def _choice(value: str, allowed: tuple[str, ...], key: str) -> None:
    if value not in allowed:
        raise ValueError(f"{key} must be one of {allowed}, got {value!r}")


def _plane(d: dict) -> Plane:
    plane = _build(Plane, d, "pv.planes")
    if not 0 <= plane.azimuth_deg < 360:
        raise ValueError(f"azimuth_deg must be in [0, 360), got {plane.azimuth_deg}")
    if not 0 <= plane.tilt_deg <= 90:
        raise ValueError(f"tilt_deg must be in [0, 90], got {plane.tilt_deg}")
    if plane.string is not None:
        _choice(plane.string, STRINGS, "string")
    return plane


def _ev_plan(d: dict) -> EvPlan:
    energy = d.get("energy_kwh")
    if energy is None:
        percent = _require(d, "percent", "plan")
        energy = float(percent) / 100.0 * float(_require(d, "battery_kwh", "plan"))
    return EvPlan(
        energy_kwh=float(energy),
        by_local=str(_require(d, "by", "plan")),
        days=tuple(int(x) for x in _require(d, "days", "plan")),
    )


def knobs_dict(knobs: Knobs) -> dict:
    """The knobs as a plain dict (the ``knobs`` of ``from_dict``, of a ``restart`` event and of
    an export's runs)."""
    return asdict(knobs)


def with_knobs(cfg: SiteConfig, knobs: dict) -> SiteConfig:
    """``cfg`` on the knobs of ``knobs`` (``knobs_dict`` of a run; a missing key takes the
    default, an unknown one is ignored, as ``from_dict``)."""
    out = _build(Knobs, dict(knobs), "knobs")
    _choice(out.impl, IMPLS, "impl")
    _choice(out.commit_rule, COMMIT_RULES, "commit_rule")
    return replace(cfg, knobs=out)


def from_dict(d: dict) -> SiteConfig:
    """Build and validate a SiteConfig. Every optional key has a default."""
    latitude = float(_require(d, "latitude"))
    if not -90 <= latitude <= 90:
        raise ValueError(f"latitude must be in [-90, 90], got {latitude}")
    longitude = float(_require(d, "longitude"))
    if not -180 <= longitude <= 180:
        raise ValueError(f"longitude must be in [-180, 180], got {longitude}")
    pack = _build(Pack, _require(d, "pack"), "pack")
    for key in ("capacity_kwh", "charge_kw", "discharge_kw"):
        if not float(getattr(pack, key)) > 0:
            raise ValueError(f"{key} must be above 0, got {getattr(pack, key)}")
    if not 0 < float(pack.eta_rt) <= 1:
        raise ValueError(f"eta_rt must be in (0, 1], got {pack.eta_rt}")
    if not 0 <= pack.soc_floor <= 1:
        raise ValueError(f"soc_floor must be in [0, 1], got {pack.soc_floor}")
    if not 0 <= pack.soc_ceiling <= 1:
        raise ValueError(f"soc_ceiling must be in [0, 1], got {pack.soc_ceiling}")
    if pack.soc_floor >= pack.soc_ceiling:
        raise ValueError(
            f"soc_floor must be below soc_ceiling, got {pack.soc_floor}, {pack.soc_ceiling}"
        )
    _choice(pack.side, SIDES, "side")
    terms = _build(Terms, d.get("terms", {}), "terms")
    _choice(terms.tax_mode, TAX_MODES, "tax_mode")
    pv_d = dict(d.get("pv", {}))
    pv_d["planes"] = tuple(_plane(p) for p in pv_d.get("planes", ()))
    pv = _build(PvHints, pv_d, "pv")
    ev_d = dict(d.get("ev", {}))
    if ev_d.get("plan") is not None:
        ev_d["plan"] = _ev_plan(ev_d["plan"])
    ev = _build(EvConfig, ev_d, "ev")
    if ev.charger_kw <= 0:
        raise ValueError(f"charger_kw must be above 0, got {ev.charger_kw}")
    knobs = _build(Knobs, d.get("knobs", {}), "knobs")
    _choice(knobs.impl, IMPLS, "impl")
    _choice(knobs.commit_rule, COMMIT_RULES, "commit_rule")
    return SiteConfig(
        latitude=latitude,
        longitude=longitude,
        pack=pack,
        terms=terms,
        pv=pv,
        ev=ev,
        knobs=knobs,
        loss_profile=d.get("loss_profile", "flat"),
        execution_profile=d.get("execution_profile", "ideal"),
    )


def tax_mode_at(terms: Terms, ts: int) -> str:
    """Tax mode at ts: "none" from the salderen end date (local midnight) on."""
    end = dt.date.fromisoformat(terms.salderen_end)
    cut = int(dt.datetime(end.year, end.month, end.day, tzinfo=ZONE).timestamp())
    return "none" if int(ts) >= cut else terms.tax_mode


def round_location(lat: float, lon: float) -> tuple[float, float]:
    """Round a location to 0,1 degree."""
    return round(lat, 1), round(lon, 1)


# One line per Knobs field: the options flow shows it under the field (strings.json carries the
# same text as `options.step.advanced.data_description`; a test keeps the two equal).
KNOB_HELP: dict[str, str] = {
    "hurdle_ct_kwh": "Smallest price spread in ct/kWh worth a trade, after fees, tax and losses.",
    "wear_eur_cycle": (
        "Battery wear in EUR per full cycle, charged on each kWh the battery discharges."
    ),
    "q_take": "Quantile of the take-price band that a charge must clear (lower is more careful).",
    "q_feed": (
        "Quantile of the feed-price band that a discharge must clear (higher is more careful)."
    ),
    "predicted_widen": (
        "Factor that widens the price bands of hours whose day-ahead price is not known."
    ),
    "terminal_target": "State of charge the plan aims for at the end of its window.",
    "latency_s": "Seconds between a decision and the battery following it.",
    "dwell_s": "Seconds the battery holds an action before it may reverse.",
    "trade_floor_w": "Smallest battery power in W for a trade; smaller orders become hold.",
    "kappa": (
        "Margin, in band half-widths, by which the live price must clear the threshold to commit."
    ),
    "latest_commit_min": "Last minute of a quarter at which a trade may still start.",
    "impl": "Threshold method: dp (dynamic programming) or ranking.",
    "commit_rule": "Commit test: margin against the band, or expected_gain.",
    "dual_penalty": "Ranking only: penalise quarters that may settle at two prices.",
    "boundary_precommit": (
        "At a quarter boundary, commit from the bands before the first live price."
    ),
    "second_plane_penalty": (
        "Relative fit error a second PV plane must save before the PV model takes it."
    ),
    "promotion_margin": (
        "Fraction by which a new site model must beat the current one to replace it."
    ),
    "drift_alarm": "Weekly PV forecast error that raises the drift alarm.",
    "learning_widen": "Ranking only: factor that widens the need while the site models learn.",
    "weekday_split": "Fit separate load profiles for weekdays and weekends.",
    "standby": "Count the inverter's standby drain against the shadow battery.",
}
