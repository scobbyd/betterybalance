"""Site model registry: champion and challenger PV models, cold start, holdout check, drift.

The champion is the model in use. A weekly refit on a trailing window gives a challenger.
A champion that does not pass the holdout rule (a hints-built champion without a holdout
error, or a learning-state fit that fails the rule) is replaced by the first challenger that
passes the rule, whatever its error. Between two passing models the challenger needs a
holdout error lower by the promotion margin. The state is learning until the champion passes
the rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np

from ..timegrid import Table, align
from .contract import Knobs, Pack, SiteConfig
from .flows import residual_kw
from .load import forecast_load_kw, load_median, local_fields
from .separate import PvModel, fit_pv, from_dict, to_dict
from .solar import clear_sky_ghi, clearness, daylight, solar_position
from .strings import GEN_COLUMN, fit_strings, strings_present

_DAY_S = 86400
HOLDOUT_RATIO = 0.7
STABILITY_SCALE = (0.8, 1.25)
STABILITY_AZIMUTH_MAX_DEG = 30.0
# drift quarters need a high sun: at low sun the PV is small and the load anchor error dominates
DRIFT_ZENITH_MAX_DEG = 70.0
_NEUTRAL_PACK = Pack(1.0, 1.0, 1.0, 0.0, 1.0, side="ac")


@dataclass
class SiteModelState:
    champion: PvModel | None = None
    challenger: PvModel | None = None
    learning: bool = True
    learned_days: int = 0
    drift: bool = False
    last_fit_ts: int | None = None


def _model_json(model: PvModel | None) -> dict | None:
    return None if model is None else to_dict(model)


def state_to_json(state: SiteModelState) -> str:
    """JSON text of the state. None errors and NaN stability are written as null."""
    return json.dumps(
        {
            "champion": _model_json(state.champion),
            "challenger": _model_json(state.challenger),
            "learning": bool(state.learning),
            "learned_days": int(state.learned_days),
            "drift": bool(state.drift),
            "last_fit_ts": None if state.last_fit_ts is None else int(state.last_fit_ts),
        }
    )


def state_from_json(s: str) -> SiteModelState:
    """Inverse of state_to_json."""
    d = json.loads(s)
    return SiteModelState(
        champion=None if d.get("champion") is None else from_dict(d["champion"]),
        challenger=None if d.get("challenger") is None else from_dict(d["challenger"]),
        learning=bool(d.get("learning", True)),
        learned_days=int(d.get("learned_days", 0)),
        drift=bool(d.get("drift", False)),
        last_fit_ts=d.get("last_fit_ts"),
    )


def holdout_passes(model: PvModel) -> bool:
    """Holdout error at most 0,7 of the null error, scale ratio 0,8 to 1,25, azimuth delta <= 30.

    A model without a holdout (None or non-finite errors) never passes. NaN stability means
    "not tested": fit_pv runs the stability test only when each holdout half has at least 5 lit
    days, because halves of 3 to 4 days are too noisy to compare geometry. Such a model passes
    on the error ratio alone.
    """
    if model.holdout_error is None or model.null_error is None:
        return False
    err, null = float(model.holdout_error), float(model.null_error)
    if not (np.isfinite(err) and np.isfinite(null)):
        return False
    error_ok = err <= HOLDOUT_RATIO * null
    ratio, az = model.stability
    if np.isnan(ratio) or np.isnan(az):
        return bool(error_ok)
    return bool(
        error_ok
        and STABILITY_SCALE[0] <= ratio <= STABILITY_SCALE[1]
        and az <= STABILITY_AZIMUTH_MAX_DEG
    )


def initial_state(cfg: SiteConfig) -> SiteModelState:
    """Cold start. No PV: empty champion. Planes given: usable champion. Else learning, empty."""
    if cfg.pv.none:
        champ = PvModel((), np.zeros(0), np.ones(12), None, None, None, (1.0, 0.0), 0, 0)
        return SiteModelState(champion=champ, learning=False)
    if cfg.pv.planes:
        scales = np.array([cfg.pv.correction * (p.kwp or 1.0) for p in cfg.pv.planes])
        champ = PvModel(
            tuple(cfg.pv.planes), scales, np.ones(12), None, None, None, (1.0, 0.0), 0, 0
        )
        return SiteModelState(champion=champ, learning=False)
    return SiteModelState(learning=True)


def _window(table: Table, lo: int, now: int) -> Table:
    ts = np.asarray(table["ts"], dtype=np.int64)
    keep = (ts >= lo) & (ts < now)
    return {k: np.asarray(v)[keep] for k, v in table.items()}


def _daylight_days(q: Table, cfg: SiteConfig) -> int:
    ts = np.asarray(q["ts"], dtype=np.int64)
    if len(ts) == 0:
        return 0
    zenith, _ = solar_position(ts, cfg.latitude, cfg.longitude)
    ok = daylight(zenith) & ~np.asarray(q["missing"], dtype=bool)
    _, _, day = local_fields(ts)
    return int(len(np.unique(day[ok])))


def weekly_update(
    state: SiteModelState,
    q: Table,
    irr: Table,
    cfg: SiteConfig,
    now: int,
    window_days: int = 90,
    min_daylight_days: int = 14,
) -> SiteModelState:
    """Fit a challenger on the trailing window and promote it by the registry rules.

    With cfg.pv.none the fit is skipped: the empty champion stays, learning is False.
    Otherwise the state is unchanged when fewer than min_daylight_days daylight days exist.
    The challenger comes from the supervised per-string fit (strings.fit_strings, the entered
    planes as fixed geometry) when strings.strings_present finds a measured pv{i} string in
    the window, else from plan 2's blind fit (separate.fit_pv) unchanged. The GEN port alone
    does not count: without string columns (an older profile or export) the DC strings are
    unmetered, and a fit of the GEN port alone would leave them out of the PV model. The blind
    fit learns all PV, the GEN port included, so it reads gen_kwh as 0, as its callers always
    gave it.
    A challenger replaces a missing champion. A champion that fails holdout_passes is replaced
    by a challenger that passes it, whatever the two errors: a failing first champion with a
    finite error must not block every later passing fit. Two failing models with errors keep
    the best fit so far: the challenger replaces the champion when its holdout error is at most
    (1 - promotion_margin) of the champion's. A hints-built champion (no holdout error) stays
    against a failing challenger. A passing champion is replaced only by a passing challenger
    with an error at most (1 - promotion_margin) of the champion's: the margin applies between
    two passing models. Learning ends when the champion passes the rule, except when planes
    were given or PV is none.
    """
    lo = now - window_days * _DAY_S
    qw, iw = _window(q, lo, now), _window(irr, lo, now)
    days = _daylight_days(qw, cfg)
    if cfg.pv.none:  # no PV on site: nothing to fit, the empty champion stays
        champ = state.champion or initial_state(cfg).champion
        return replace(
            state,
            champion=champ,
            challenger=None,
            learning=False,
            learned_days=days,
            last_fit_ts=int(now),
        )
    if days < min_daylight_days:
        return state
    if any(c != GEN_COLUMN for c in strings_present(qw)):
        challenger = fit_strings(qw, iw, cfg, now, planes=cfg.pv.planes or None)
    else:
        blind = dict(qw, gen_kwh=np.zeros(len(qw["ts"])))
        challenger, _, _ = fit_pv(blind, iw, cfg, now)
    champ = state.champion
    margin = cfg.knobs.promotion_margin
    beats = (
        champ is not None
        and champ.holdout_error is not None
        and challenger.holdout_error is not None
        and challenger.holdout_error <= (1.0 - margin) * champ.holdout_error
    )
    if champ is None:
        promote = True
    elif holdout_passes(champ):
        promote = holdout_passes(challenger) and beats
    elif holdout_passes(challenger):
        promote = True
    else:
        promote = beats
    if promote:
        champ = challenger
    fixed = cfg.pv.none or bool(cfg.pv.planes)
    learning = False if fixed else not holdout_passes(champ)
    return replace(
        state,
        champion=champ,
        challenger=challenger,
        learning=learning,
        learned_days=days,
        last_fit_ts=int(now),
    )


def drift_alarm(
    q: Table,
    pv_est_kw: np.ndarray,
    irr: Table,
    lat: float,
    lon: float,
    now: int,
    days: int = 7,
    kt_min: float = 0.8,
    limit: float = 0.15,
    pack: Pack | None = None,
) -> bool:
    """True when the model is biased on clear quarters over the trailing days.

    The deficit d is the night-anchored load median of the window (daylight quarters masked,
    so daylight slots take the night mean) minus the residual. On daylight quarters with
    clearness index at or above kt_min the alarm needs |mean(d - pv_est)| > limit * mean(pv_est).
    The pack only matters for the residual; the default is a pack without conversion loss.

    A diagnostic flag, not a guard for the night-anchor or the geometry risk: d uses the same
    night-anchored load as the PV scale estimator (``separate.fit_pv``), so the alarm cannot
    see the anchor's own error, and a rotated plane with a compensating scale stays quiet too.
    In plan 2 only ``SiteModelState.drift`` and the replay report read it; nothing acts on
    the flag (plan-3 work). It fired on both owner windows of the plan-2 replay.
    """
    ts = np.asarray(q["ts"], dtype=np.int64)
    pv = np.asarray(pv_est_kw, dtype=float)
    r = residual_kw(q, pack or _NEUTRAL_PACK)
    zenith, _ = solar_position(ts, lat, lon)
    lit = daylight(zenith)
    lm = load_median(r, ts, now, mask=lit, days=days)
    d = forecast_load_kw(lm, ts) - r
    ghi = align(irr, ts, ["ghi_w_m2"])["ghi_w_m2"]
    kt = clearness(ghi, clear_sky_ghi(zenith))
    with np.errstate(invalid="ignore"):
        clear = np.nan_to_num(kt, nan=-1.0) >= kt_min
    use = (
        (ts >= now - days * _DAY_S)
        & (ts < now)
        & (zenith < DRIFT_ZENITH_MAX_DEG)
        & clear
        & np.isfinite(d)
        & np.isfinite(pv)
    )
    if not use.any():
        return False
    mean_pv = float(pv[use].mean())
    if mean_pv <= 0.0:
        return False
    return bool(abs(float((d[use] - pv[use]).mean())) > limit * mean_pv)


def need_widening(state: SiteModelState, knobs: Knobs) -> float:
    """Band widening factor: knobs.learning_widen while learning, else 1,0."""
    return float(knobs.learning_widen) if state.learning else 1.0
