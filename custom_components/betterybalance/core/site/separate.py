"""Separate house load from unmetered PV: masks, robust plane fit, plane search, month factors.

The AC-side residual r = load - pv is all a house shows. A night-anchored alternating fit
learns the load profile per wall slot, fits one or two roof planes to the daytime deficit
d = L[slot] - r with a non-negative Huber fit, and checks the result on a holdout.

After the night-anchored first round, each fit subtracts the per-wall-slot mean over the fit
rows from d and from every shape column (the holdout takes the fit-row means). The per-slot
load is a nuisance term and the demeaning removes it exactly, so the fit lands on the fixed
point of the load and PV alternation. This within-slot transformation assumes the load
profile is the same per wall slot across the fit window, the same assumption the median
profile makes.

The estimator has two stages. The slot-centred rounds choose the geometry: the day-to-day
deviations of the deficit identify the plane shapes well. They attenuate the scale: the
irradiance is a forecast archive, its error is large against the day-to-day deviations, and
a centred fit has no intercept to absorb it (regression dilution: a synthetic house with
25 % irradiance noise per quarter came out at 3,27 of 5 kWp). The PV estimate then comes back
low, the load median takes up the missing PV, and month factors fitted on that deficit carry
the same bias. The dilution is a common factor on the slope, so the geometry and the
relative scales of the planes stay those of the slot-centred stage, and the second stage
fits one non-negative level multiplier of the stage-one PV estimate against a level anchor:
the night-anchored profile of round 0 (the wall-slot median of the residual over night
quarters, daytime slots at the night mean) minus the residual, on all daylight fit rows. In
the uncentred fit the daily cycle dominates the irradiance error. On the noisy synthetic
house the scale comes back at 4,93 of 5 kWp. Separate per-plane scales on the anchor were
tried: a daytime load peak above the night mean then moves energy between the planes (5,54
and 2,33 against 5,0 and 3,0 on the two-plane synthetic house).

Two narrower selections were tried and fail. An anchor over night and overcast quarters sits
low, because overcast quarters still carry PV and a noisy clearness under the threshold also
picks sunny quarters (0,52 of the truth on the noisy house). Fitting the scales on clear
quarters only selects rows by a noisy clearness, so it selects on the regressor's error and
overstates the shapes (0,85 with the night anchor).

Known risk: the anchor takes the daytime load at the night mean. A house whose daytime load
sits above its night mean gets a PV scale that is low by that excess, and one whose night
rows hold an evening peak (a late sunset) gets a scale that is high. The September synthetic
house of the acceptance test shows the second case: the night mean holds part of the evening
peak (about 0,57 kW against a daytime base of 0,30 kW), the scale comes out at 1,139 and at
half the PV the proportionality reads 0,587 against 0,5. A deep-night anchor (local 01:00 to
05:00 only) removes that case (0,94 and 0,499) but sits under the daytime base of the other
synthetic houses (scales 0,91 to 0,94 of the truth), so the all-night anchor stays. The drift
alarm (clear quarters against a night-anchored deficit) shares the anchor, so it cannot see
this error, and nothing acts on its flag in plan 2; a guard and a better two-stage estimator
are open for plan 3.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..timegrid import Table, align
from .contract import Plane, SiteConfig
from .ev import Session, detect_sessions, session_mask
from .flows import residual_kw
from .load import LoadModel, forecast_load_kw, load_median, local_fields
from .plant import plane_shape
from .shading import Shading, apply_shading
from .shading import from_dict as shading_from_dict
from .shading import to_dict as shading_to_dict
from .solar import daylight, solar_position

_DAY_S = 86400
_HOUR_S = 3600
_IRR_COLUMNS = ["ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c"]
_CD_TOL = 1e-9
_CD_SWEEPS = 200
_BEST_SINGLES = 5
_COARSE_TILTS = (20, 40, 60)
_COARSE_AZ_STEP = 30
_REFINE_TILT = 10
_REFINE_AZ = 15
_CURTAIL_SOC_PCT = 95.0
_CURTAIL_EXPORT_KW = 0.1
# a clipping day's plateau lies at least this share of the window's highest deficit
CLIP_LEVEL_SHARE = 0.9
# the unclipped model must exceed the flat deficit by this relative margin to confirm a clip
CLIP_MODEL_EXCESS = 0.08
# plateau quarters per day on which the model must exceed the deficit
CLIP_MODEL_QUARTERS = 2
# clipping days needed before an AC maximum is learned
CLIP_MIN_DAYS = 2
# relative band of the plateau detector and of the censoring at the learned AC maximum
CLIP_TOL = 0.02
# fewest holdout days that still validate (two halves of at least two days)
MIN_HOLDOUT_DAYS = 4
# fewest lit days in each holdout half for the stability test: halves of 3 to 4 days are too
# noisy to compare geometry, so a shorter holdout is tested on its error ratio alone
MIN_STABILITY_HALF_DAYS = 5
# neighbouring grid planes interpolate one off-grid roof and are not a second roof
MIN_PLANE_AZ_SEP_DEG = 45.0


@dataclass
class PvModel:
    planes: tuple[Plane, ...]
    scales_kwp: np.ndarray
    month_factor: np.ndarray  # (12,), index month - 1
    ac_max_kw: float | None
    holdout_error: float | None
    null_error: float | None
    stability: tuple[float, float]  # (scale_ratio, azimuth_delta_deg) between holdout halves
    fitted_at: int
    n_days: int
    factor_kind: str = "month"
    # the strings path (strings.fit_strings): scales fitted against measured string power
    supervised: bool = False
    # per plane, True when its scale was fitted against measured power (strings path only);
    # a plane kept at its entered hint scale is False and strings.scale_check skips it
    checked: tuple[bool, ...] = ()
    # the inverter's DC-to-AC stage loss (a, b): a * P + b * P^2 W at P kW of DC string power.
    # Set on the strings path: the planes of DC strings (string not "gen") sum on the DC side
    # and pass the stage, GEN-port planes are AC already. None: every plane counts as AC.
    dc_to_ac: tuple[float, float] | None = None
    # the shading profile (shading.Shading): factors per plane and sun-position bin on the DC
    # plane shape, on every quarter the model predicts. None: no shading.
    shading: Shading | None = None
    # the scales as fitted before the shading round (strings path with a profile): the
    # fill-in check (strings.scale_check) reads these, because the refit on the shaded shapes
    # lifts a scale to cover what the profile cuts. None: scales_kwp is that scale.
    scales_check_kwp: np.ndarray | None = None


def _json_float(x: float) -> float | None:
    """A float for JSON: NaN becomes None (null)."""
    return None if np.isnan(x) else float(x)


def _plane_dict(p: Plane) -> dict:
    """A plane as JSON; `string` only when the plane names one (older dicts stay unchanged)."""
    d = {"tilt_deg": float(p.tilt_deg), "azimuth_deg": float(p.azimuth_deg), "kwp": p.kwp}
    if p.string is not None:
        d["string"] = p.string
    return d


def to_dict(model: PvModel) -> dict:
    """Plain JSON types. None and NaN become JSON null. The strings-path fields only when the
    model is supervised (``shading`` there: the profile or null; a blind model writes it only
    when it carries one)."""
    d = {
        "planes": [_plane_dict(p) for p in model.planes],
        "scales_kwp": [float(x) for x in model.scales_kwp],
        "month_factor": [float(x) for x in model.month_factor],
        "ac_max_kw": None if model.ac_max_kw is None else float(model.ac_max_kw),
        "holdout_error": None if model.holdout_error is None else float(model.holdout_error),
        "null_error": None if model.null_error is None else float(model.null_error),
        "stability": [_json_float(x) for x in model.stability],
        "fitted_at": int(model.fitted_at),
        "n_days": int(model.n_days),
        "factor_kind": model.factor_kind,
    }
    if model.supervised:  # older dicts stay unchanged
        d["supervised"] = True
        d["checked"] = [bool(x) for x in model.checked]
        d["dc_to_ac"] = None if model.dc_to_ac is None else [float(x) for x in model.dc_to_ac]
    if model.supervised or model.shading is not None:
        d["shading"] = shading_to_dict(model.shading)
    if model.scales_check_kwp is not None:
        d["scales_check_kwp"] = [float(x) for x in model.scales_check_kwp]
    return d


def from_dict(d: dict) -> PvModel:
    """Inverse of to_dict. A missing factor_kind reads as "month", a null stability as NaN,
    missing strings-path fields as a blind model, a missing shading or scales_check_kwp as
    None."""
    return PvModel(
        planes=tuple(
            Plane(p["tilt_deg"], p["azimuth_deg"], p.get("kwp"), p.get("string"))
            for p in d.get("planes", ())
        ),
        scales_kwp=np.asarray(d.get("scales_kwp", ()), dtype=float),
        month_factor=np.asarray(d.get("month_factor", [1.0] * 12), dtype=float),
        ac_max_kw=d.get("ac_max_kw"),
        holdout_error=d.get("holdout_error"),
        null_error=d.get("null_error"),
        stability=tuple(np.nan if x is None else float(x) for x in d["stability"]),
        fitted_at=int(d["fitted_at"]),
        n_days=int(d["n_days"]),
        factor_kind=d.get("factor_kind", "month"),
        supervised=bool(d.get("supervised", False)),
        checked=tuple(bool(x) for x in d.get("checked", ())),
        dc_to_ac=None if d.get("dc_to_ac") is None else tuple(float(x) for x in d["dc_to_ac"]),
        shading=shading_from_dict(d.get("shading")),
        scales_check_kwp=(
            None
            if d.get("scales_check_kwp") is None
            else np.asarray(d["scales_check_kwp"], dtype=float)
        ),
    )


def _local_month(ts: np.ndarray) -> np.ndarray:
    """Local Europe/Amsterdam month 1..12 per timestamp."""
    _, _, day = local_fields(ts)
    return day.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12 + 1


def hour_mean(x: np.ndarray, ts: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """x with each of the rows replaced by the mean of the rows in its clock hour.

    rows defaults to every row. A NaN in an hour makes that hour's mean NaN. Clock hours are
    UTC hours, which are local hours too: the Europe/Amsterdam offsets are whole hours.
    """
    x = np.asarray(x, dtype=float)
    out = x.copy()
    rows = np.ones(len(x), dtype=bool) if rows is None else np.asarray(rows, dtype=bool)
    if not rows.any():
        return out
    _, inv = np.unique(np.asarray(ts, dtype=np.int64)[rows] // _HOUR_S, return_inverse=True)
    out[rows] = (np.bincount(inv, weights=x[rows]) / np.bincount(inv))[inv]
    return out


def hourly_origin_rows(q: Table) -> np.ndarray:
    """The rows of a quarters table that are flat splits of hourly data (`hourly_origin`)."""
    n = len(q["ts"])
    if "hourly_origin" not in q:
        return np.zeros(n, dtype=bool)
    return np.asarray(q["hourly_origin"], dtype=bool)


def dc_to_ac_kw(dc_kw: np.ndarray, stage: tuple[float, float]) -> np.ndarray:
    """AC kW out of the inverter stage for DC kW in: P - (a P + b P^2) / 1000, at least 0."""
    p = np.maximum(np.asarray(dc_kw, dtype=float), 0.0)
    a, b = stage
    return np.maximum(p - (a * p + b * p * p) / 1000.0, 0.0)


def planes_kw(
    scaled: np.ndarray, planes: tuple[Plane, ...], dc_to_ac: tuple[float, float] | None
) -> np.ndarray:
    """AC kW of the planes from the (n, planes) array of scale * shape, before month factors.

    Without dc_to_ac the columns sum. With it the DC planes (string not "gen") sum, pass
    dc_to_ac_kw, and the GEN-port planes add as they are.
    """
    scaled = np.asarray(scaled, dtype=float)
    if dc_to_ac is None:
        return scaled.sum(axis=1)
    dc = np.array([p.string != "gen" for p in planes], dtype=bool)
    return dc_to_ac_kw(scaled[:, dc].sum(axis=1), dc_to_ac) + scaled[:, ~dc].sum(axis=1)


def predict_pv_kw(
    model: PvModel, irr: Table, lat: float, lon: float, hour_mean_rows: np.ndarray | None = None
) -> np.ndarray:
    """Sum over planes of scale * plane_shape (planes_kw: DC strings through the model's
    DC-to-AC stage when it has one), times month_factor[month], clipped at ac_max_kw.

    With a shading profile the plane shapes take its factors first (shading.apply_shading:
    on the DC shape, on every quarter).

    To score data of hourly origin, hour_mean_rows (bool per irr row) names the rows whose
    plane shapes take the mean of their clock hour first, as fit_pv does on such rows.
    """
    n = len(irr["ts"])
    if not model.planes:
        return np.zeros(n)
    shapes = np.column_stack([plane_shape(irr, p, lat, lon) for p in model.planes])
    if model.shading is not None:
        shapes = apply_shading(model.shading, shapes, irr["ts"], lat, lon)
    if hour_mean_rows is not None and np.asarray(hour_mean_rows, dtype=bool).any():
        shapes = np.column_stack(
            [hour_mean(shapes[:, j], irr["ts"], hour_mean_rows) for j in range(shapes.shape[1])]
        )
    month = _local_month(irr["ts"])
    scaled = shapes * np.asarray(model.scales_kwp, dtype=float)[None, :]
    pv = planes_kw(scaled, model.planes, model.dc_to_ac) * model.month_factor[month - 1]
    if model.ac_max_kw is not None:
        pv = np.minimum(pv, model.ac_max_kw)
    return pv


def curtailment_mask(q: Table) -> np.ndarray:
    """Quarters the pack may have curtailed PV: soc_max_pct >= 95 and export under 100 W."""
    soc_max = np.asarray(q["soc_max_pct"], dtype=float)
    export_kw = np.asarray(q["grid_export_kwh"], dtype=float) * 4.0
    return (soc_max >= _CURTAIL_SOC_PCT) & (export_kw < _CURTAIL_EXPORT_KW)


def clipping_plateau(
    pv_est_kw: np.ndarray, tol: float = 0.02, min_run: int = 3
) -> tuple[np.ndarray, float | None]:
    """Inverter clipping plateau of one day.

    The quarters inside a run of at least min_run consecutive daylight quarters (value
    above 0) whose value lies within tol (relative) of the day's maximum, and the plateau
    level (mean of those values), or None when no such run exists.
    """
    v = np.asarray(pv_est_kw, dtype=float)
    lit = np.isfinite(v) & (v > 0.0)
    mask = np.zeros(len(v), dtype=bool)
    if not lit.any():
        return mask, None
    peak = float(v[lit].max())
    near = lit & (v >= peak * (1.0 - tol))
    edges = np.diff(np.r_[0, near.astype(np.int8), 0])
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if end - start >= min_run:
            mask[start:end] = True
    if not mask.any():
        return mask, None
    return mask, float(v[mask].mean())


def huber_loss(resid: np.ndarray, delta: float) -> float:
    """Mean Huber loss: r^2 / 2 inside delta, delta (|r| - delta / 2) outside."""
    a = np.abs(np.asarray(resid, dtype=float))
    if len(a) == 0:
        return 0.0
    return float(np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta)).mean())


def _nnls_cd(gram: np.ndarray, rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Non-negative least squares on the normal equations by projected coordinate descent.

    Sweeps the coefficients, each set to its exact minimiser clamped at 0, until the largest
    change in a sweep is under 1e-9 or after 200 sweeps.
    """
    x = x.copy()
    k = len(x)
    for _ in range(_CD_SWEEPS):
        largest = 0.0
        for j in range(k):
            if gram[j, j] <= 0.0:
                new = 0.0
            else:
                new = max(0.0, x[j] + (rhs[j] - gram[j] @ x) / gram[j, j])
            largest = max(largest, abs(new - x[j]))
            x[j] = new
        if largest < _CD_TOL:
            break
    return x


def huber_fit(
    shapes: np.ndarray,
    d: np.ndarray,
    delta: float,
    iters: int = 10,
    x0: np.ndarray | None = None,
) -> np.ndarray:
    """Non-negative scales minimising the Huber loss of d - shapes @ scales.

    Iteratively reweighted least squares: weight 1 inside delta, delta / |r| outside; every
    weighted step is a non-negative least squares solved by projected coordinate descent.
    x0 gives the initial scales (default zeros), clamped at 0.
    """
    x_mat = np.asarray(shapes, dtype=float)
    if x_mat.ndim == 1:
        x_mat = x_mat[:, None]
    d = np.asarray(d, dtype=float)
    if x0 is None:
        scales = np.zeros(x_mat.shape[1])
    else:
        scales = np.maximum(np.asarray(x0, dtype=float), 0.0)
    weights = np.ones(len(d))
    for _ in range(iters + 1):
        weighted = x_mat * weights[:, None]
        scales = _nnls_cd(weighted.T @ x_mat, weighted.T @ d, scales)
        resid = np.abs(d - x_mat @ scales)
        weights = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-12))
    return scales


def plane_grid(
    tilts: tuple[float, ...] = (10, 20, 30, 40, 50, 60),
    az_from: int = 60,
    az_to: int = 300,
    az_step: int = 15,
) -> list[Plane]:
    """Tilt by compass azimuth grid, east round through south to west (102 planes)."""
    return [Plane(t, a) for t in tilts for a in range(az_from, az_to + 1, az_step)]


def _score(
    planes: tuple[Plane, ...],
    d: np.ndarray,
    shapes: dict[Plane, np.ndarray],
    rows: np.ndarray,
    delta: float,
) -> tuple[float, np.ndarray]:
    """Huber fit on the rows and the Huber loss of that fit on the same rows."""
    x_mat = np.column_stack([shapes[p] for p in planes])[rows]
    scales = huber_fit(x_mat, d[rows], delta)
    return huber_loss(d[rows] - x_mat @ scales, delta), scales


def _neighbours(plane: Plane, shapes: dict[Plane, np.ndarray]) -> list[Plane]:
    """Fine-grid planes within tilt +-10 and azimuth +-15 of a plane, the plane included."""
    out = []
    for dt_ in (-_REFINE_TILT, 0, _REFINE_TILT):
        for da in (-_REFINE_AZ, 0, _REFINE_AZ):
            cand = Plane(plane.tilt_deg + dt_, plane.azimuth_deg + da)
            if cand in shapes:
                out.append(cand)
    return out


def search_planes(
    d: np.ndarray,
    shapes: dict[Plane, np.ndarray],
    fit_mask: np.ndarray,
    holdout_mask: np.ndarray,
    delta: float,
    penalty: float = 0.02,
) -> tuple[tuple[Plane, ...], np.ndarray]:
    """Grid search for one or two planes, selected on the fit rows.

    Every candidate is fitted and scored (Huber loss) on the rows of fit_mask that are not in
    holdout_mask: holdout_mask only excludes rows, it never selects, so the holdout stays
    for validation. The best five single planes are kept. Two-plane candidates: pairs among
    the best five, every pair of the coarse grid (tilt 20, 40, 60 by azimuth 60 to 300 step
    30) and the fine-grid neighbours of the best coarse pair. Pairs closer than
    MIN_PLANE_AZ_SEP_DEG in azimuth are skipped. The best pair wins only with a loss at
    least `penalty` lower in relative terms.
    """
    d = np.asarray(d, dtype=float)
    rows = np.asarray(fit_mask, dtype=bool) & ~np.asarray(holdout_mask, dtype=bool)
    grid = list(shapes)
    singles = [_score((p,), d, shapes, rows, delta) for p in grid]
    order = sorted(range(len(grid)), key=lambda i: (singles[i][0], i))
    best_loss, best_scales = singles[order[0]]
    best = ((grid[order[0]],), best_scales, best_loss)

    tried: dict[frozenset[Plane], tuple[float, tuple[Plane, ...], np.ndarray]] = {}

    def run(pairs: list[tuple[Plane, Plane]]) -> None:
        for a, b in pairs:
            key = frozenset((a, b))
            if (
                a == b
                or key in tried
                or _circular_delta(a.azimuth_deg, b.azimuth_deg) < MIN_PLANE_AZ_SEP_DEG
            ):
                continue
            loss, scales = _score((a, b), d, shapes, rows, delta)
            tried[key] = (loss, (a, b), scales)

    top = [grid[i] for i in order[:_BEST_SINGLES]]
    run(list(itertools.combinations(top, 2)))
    coarse = [
        p for p in grid if p.tilt_deg in _COARSE_TILTS and p.azimuth_deg % _COARSE_AZ_STEP == 0
    ]
    coarse_pairs = list(itertools.combinations(coarse, 2))
    run(coarse_pairs)
    scored = [tried[frozenset(pq)] for pq in coarse_pairs if frozenset(pq) in tried]
    if scored:
        _, (a, b), _ = min(scored, key=lambda s: s[0])
        run(list(itertools.product(_neighbours(a, shapes), _neighbours(b, shapes))))
    if tried:
        pair_loss, pair, pair_scales = min(tried.values(), key=lambda s: s[0])
        if pair_loss <= (1.0 - penalty) * best_loss:
            best = (pair, pair_scales, pair_loss)
    return best[0], best[1]


def month_factors(
    d: np.ndarray, pv_est: np.ndarray, months: np.ndarray, shrink: float = 0.5
) -> np.ndarray:
    """Per-month multiplier (sum d p + lam) / (sum p p + lam), a ridge towards 1.

    lam = shrink * the mean over months with data of sum p p, so a month without data is 1.
    """
    d = np.asarray(d, dtype=float)
    p = np.asarray(pv_est, dtype=float)
    idx = np.asarray(months, dtype=np.int64) - 1
    num = np.bincount(idx, weights=d * p, minlength=12)[:12]
    den = np.bincount(idx, weights=p * p, minlength=12)[:12]
    has = den > 0.0
    out = np.ones(12)
    if has.any():
        lam = shrink * float(den[has].mean())
        out[has] = (num[has] + lam) / (den[has] + lam)
    return out


def clipping_mask(
    d: np.ndarray,
    pv_prev: np.ndarray,
    ts: np.ndarray,
    daylight: np.ndarray,
    masked: np.ndarray,
    tol: float = 0.02,
    min_run: int = 3,
) -> tuple[np.ndarray, float | None]:
    """Inverter clipping in the observed deficit, confirmed by the unclipped model.

    Per local day, clipping_plateau on the daylight, unmasked deficit gives candidate
    quarters. A day counts when its plateau level is at least 90 % of the highest deficit
    and the previous round's model exceeds the deficit by more than 8 % on at least two of
    the plateau quarters. ac_max is the median level of the counted days when at least two
    days count, else None; only then are the counted plateau quarters masked.
    """
    d = np.asarray(d, dtype=float)
    pv_prev = np.asarray(pv_prev, dtype=float)
    valid = np.asarray(daylight, dtype=bool) & ~np.asarray(masked, dtype=bool) & np.isfinite(d)
    mask = np.zeros(len(d), dtype=bool)
    if not valid.any():
        return mask, None
    _, _, day = local_fields(ts)
    top = float(d[valid].max())
    levels = []
    for k in np.unique(day[valid]):
        rows = np.flatnonzero(valid & (day == k))
        day_mask, level = clipping_plateau(d[rows], tol, min_run)
        if level is None or level < CLIP_LEVEL_SHARE * top:
            continue
        plateau_rows = rows[day_mask]
        excess = pv_prev[plateau_rows] > d[plateau_rows] * (1.0 + CLIP_MODEL_EXCESS)
        if int(excess.sum()) >= CLIP_MODEL_QUARTERS:
            mask[plateau_rows] = True
            levels.append(level)
    if len(levels) < CLIP_MIN_DAYS:
        return np.zeros(len(d), dtype=bool), None
    return mask, float(np.median(levels))


def _slot_centre(x: np.ndarray, slot: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """x minus its mean per wall slot over the given rows (0 for a slot without rows).

    Centring d and the shapes per slot profiles the load out of the plane fit: the fit then
    lands on the fixed point of the load and PV alternation, not one step of it.
    """
    n = np.bincount(slot[rows], minlength=96)
    mean = np.bincount(slot[rows], weights=x[rows], minlength=96) / np.maximum(n, 1)
    return x - mean[slot]


def _circular_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _stability(
    planes: tuple[Plane, ...],
    d: np.ndarray,
    shapes: dict[Plane, np.ndarray],
    slot: np.ndarray,
    halves: tuple[np.ndarray, np.ndarray],
    delta: float,
) -> tuple[float, float]:
    """(scale_ratio, azimuth_delta_deg) between the two halves of the holdout days.

    fit_pv calls it only when each half has at least MIN_STABILITY_HALF_DAYS lit days.

    The scale ratio is the total kWp of the chosen planes fitted on the later half over the
    one fitted on the earlier half (0 when the earlier half fits no PV). The azimuth delta
    is between the best single grid planes of the two halves. Both fits are slot-centred
    on the half's own rows.
    """
    totals, best_az = [], []
    for h in halves:
        d_c = _slot_centre(d, slot, h)
        cent = {p: _slot_centre(s, slot, h) for p, s in shapes.items()}
        model = np.column_stack([cent[p] for p in planes])
        totals.append(float(huber_fit(model[h], d_c[h], delta).sum()))
        losses = [_score((p,), d_c, cent, h, delta)[0] for p in cent]
        best_az.append(float(list(cent)[int(np.argmin(losses))].azimuth_deg))
    ratio = totals[1] / totals[0] if totals[0] > 0.0 else 0.0
    return ratio, _circular_delta(best_az[0], best_az[1])


def fit_pv(
    q: Table, irr: Table, cfg: SiteConfig, now: int, holdout_days: int = 14, rounds: int = 3
) -> tuple[PvModel, LoadModel, list[Session]]:
    """Night-anchored alternating fit of the load profile and the PV planes (design 6.4).

    Round 0 learns the load per wall slot from night quarters only (daylight slots take the
    night mean) and fits the planes to that night-anchored deficit d = L[slot] - r. Later
    rounds take the slot median of r + pv_est as the load and fit slot-centred d and
    shapes, which profiles the load out of the fit (the fixed point of the alternation).
    The plane grid is searched when no planes are given; given planes start at
    correction * (kwp or 1.0). The loop stops early only from round 1 on. Masked quarters:
    missing, curtailment, EV sessions, NaN SoC, NaN irradiance and, from round 1 on, the
    clipping quarters confirmed by the previous round's model on the night-anchored deficit
    plus every fit quarter at the learned clip level (censored).

    Rows of hourly origin (`hourly_origin` True, set by hourly_to_quarters) hold flat splits
    of hourly energies. On those rows every plane shape takes the mean of the four quarter
    shapes of its clock hour before the fit, so a peaked shape is not fitted against flat
    energies (on synthetic flat-split houses the scale came out at 0,86 to 0,93 of the truth
    without it and 1,01 with it).

    After the rounds the geometry is fixed and the second stage rescales (given planes too):
    one non-negative Huber multiplier of the stage-one estimate against d_anchor = L_night - r
    on every daylight fit row, with L_night the wall-slot median of r over the unmasked night
    rows of the history before now (holdout days out, daytime slots at the night mean). The
    scales are the stage-one scales times that multiplier. pv_est, the month factors, the
    holdout errors and the stability then use those scales. Without fit rows, or with a zero
    stage-one estimate on them, the slot-centred scales stay.

    Rows per quantity. The holdout is the last n_hold = min(holdout_days, n_lit // 2)
    daylight days before now (n_lit daylight days in all), and exists only when n_hold is at
    least MIN_HOLDOUT_DAYS: a 10-day house validates on 5 days, a 14-day house on 7, 28 days
    or more on 14, a 6-day house not at all. Task 9 reads None errors as "holdout not
    passed", so a house leaves the learning state only from 8 lit days on. Fit rows are the
    daylight, unmasked, unclipped quarters of the other days. The load used in the rounds,
    the plane search (selection included), the scales, delta, the clipping mask and the
    month factors use fit rows only. holdout_error, null_error and stability use the holdout
    rows, with d formed from a load profile of the fit rows. The returned LoadModel uses
    every unmasked row. Without a holdout the errors are None, and a blind fit reports
    stability (nan, nan).

    The stability test runs only when each half of the holdout has at least
    MIN_STABILITY_HALF_DAYS (5) lit days, so from a 10-day holdout (20 lit days) on. Halves of
    3 to 4 days are too noisy to compare geometry: the scale ratio and the best azimuth of so
    few days swing with the weather, not with the roof. A shorter holdout reports stability
    (nan, nan), "not tested", and Task 9's holdout_passes then rules on the error ratio alone.
    """
    ts = np.asarray(q["ts"], dtype=np.int64)
    lat, lon = cfg.latitude, cfg.longitude
    r = residual_kw(q, cfg.pack)
    history_days = max(int(np.ceil((now - int(ts.min())) / _DAY_S)), 1) if len(ts) else 1
    slot, _, day = local_fields(ts)
    months = _local_month(ts)
    masked = (
        np.asarray(q["missing"], dtype=bool)
        | curtailment_mask(q)
        | np.isnan(np.asarray(q["soc_pct"], dtype=float))
        | np.isnan(r)
        | (ts >= now)
    )
    sessions: list[Session] = []
    if cfg.ev.present:
        first = load_median(r, ts, now, mask=masked, days=history_days)
        sessions = detect_sessions(r, ts, first, cfg.ev.charger_kw, cfg.ev.min_minutes)
        masked |= session_mask(sessions, ts)

    if cfg.pv.none:
        lm = load_median(r, ts, now, mask=masked, days=history_days)
        model = PvModel(
            (),
            np.zeros(0),
            np.ones(12),
            None,
            None,
            None,
            (1.0, 0.0),
            int(now),
            len(np.unique(day[~masked])),
        )
        return model, lm, sessions

    aligned = align(irr, ts, _IRR_COLUMNS)
    zenith, _ = solar_position(ts, lat, lon)
    lit = daylight(zenith)
    given = cfg.pv.planes
    if given:
        grid = [Plane(p.tilt_deg, p.azimuth_deg) for p in given]
    else:
        grid = plane_grid()
    shapes = {p: plane_shape(aligned, p, lat, lon) for p in grid}
    hourly_rows = hourly_origin_rows(q)
    if hourly_rows.any():  # flat-split hourly energies meet hour-mean shapes, not peaked ones
        shapes = {p: hour_mean(s, ts, hourly_rows) for p, s in shapes.items()}
    bad_irr = np.zeros(len(ts), dtype=bool)
    for s in shapes.values():
        bad_irr |= ~np.isfinite(s)
    shapes = {p: np.nan_to_num(s) for p, s in shapes.items()}
    masked |= lit & bad_irr

    days_lit = np.unique(day[lit & ~masked])
    # the last daylight days validate only, at most half of them; too few means no holdout
    n_hold = min(holdout_days, len(days_lit) // 2)
    if n_hold < MIN_HOLDOUT_DAYS:
        n_hold = 0
    hold_days = days_lit[len(days_lit) - n_hold :]
    in_hold = np.isin(day, hold_days)
    half_days = (hold_days[: (n_hold + 1) // 2], hold_days[(n_hold + 1) // 2 :])

    planes: tuple[Plane, ...] = tuple(grid) if given else ()
    scales = np.array([cfg.pv.correction * (p.kwp or 1.0) for p in given])
    pv_est = np.zeros(len(ts))
    plateau = np.zeros(len(ts), dtype=bool)
    ac_max: float | None = None
    delta: float | None = None
    for k in range(rounds):
        load_mask = masked | plateau | in_hold | (lit if k == 0 else False)
        lm_fit = load_median(r + pv_est, ts, now, mask=load_mask, days=history_days)
        d = forecast_load_kw(lm_fit, ts) - r
        if k == 0:
            d_night = d  # night-anchored deficit: flat where the observed PV is flat
        fit_mask = lit & ~masked & ~in_hold
        if k > 0:
            plateau, ac_max = clipping_mask(d_night, pv_est, ts, lit, masked | in_hold)
            if ac_max is not None:  # a quarter at the clip level is censored, in a run or not
                plateau |= fit_mask & (d_night >= (1.0 - CLIP_TOL) * ac_max)
        fit_mask &= ~plateau
        if delta is None:
            if given and all(p.kwp is not None for p in given):
                delta = sum(p.kwp for p in given) / 3.0
            else:
                delta = float(np.percentile(d[fit_mask], 90)) / 3.0 if fit_mask.any() else 1.0
            delta = max(delta, 1e-3)
        if not fit_mask.any():
            break
        if k == 0:  # night-anchored first estimate: the load level comes from the night
            d_c, centred = d, shapes
        else:  # later rounds profile the load out per slot
            d_c = _slot_centre(d, slot, fit_mask)
            centred = {p: _slot_centre(x, slot, fit_mask) for p, x in shapes.items()}
        if given:
            new_planes = planes
            x_c = np.column_stack([centred[p] for p in planes])
            new_scales = huber_fit(x_c[fit_mask], d_c[fit_mask], delta, x0=scales)
        else:
            new_planes, new_scales = search_planes(
                d_c, centred, fit_mask, in_hold, delta, cfg.knobs.second_plane_penalty
            )
        # round 0 never ends the loop: one slot-centred round with the clipping pass always runs
        converged = (
            k >= 1
            and new_planes == planes
            and bool(np.all(np.abs(new_scales - scales) < 0.01 * np.maximum(np.abs(scales), 1e-9)))
        )
        planes, scales = new_planes, new_scales
        pv_est = np.column_stack([shapes[p] for p in planes]) @ scales
        if converged:
            break

    unstable = (float("nan"), float("nan"))
    if planes:  # second stage: one level multiplier of the stage-one estimate, night anchor
        lm_night = load_median(r, ts, now, mask=masked | in_hold | lit, days=history_days)
        d_anchor = forecast_load_kw(lm_night, ts) - r
        level_rows = lit & ~masked & ~plateau & ~in_hold & np.isfinite(d_anchor)
        x_mat = np.column_stack([shapes[p] for p in planes])
        split = x_mat @ scales  # the slot-centred estimate: its plane split is kept
        if level_rows.any() and np.any(split[level_rows] > 0.0):
            level = float(huber_fit(split[level_rows], d_anchor[level_rows], delta)[0])
            scales = scales * level
            pv_est = x_mat @ scales
    if not planes:
        lm = load_median(r, ts, now, mask=masked, days=history_days)
        model = PvModel(
            (), np.zeros(0), np.ones(12), None, None, None, unstable, int(now), len(days_lit)
        )
        return model, lm, sessions

    # validation quantities: the load from fit rows only, so holdout rows stay unseen
    lm_fit = load_median(r + pv_est, ts, now, mask=masked | plateau | in_hold, days=history_days)
    d = forecast_load_kw(lm_fit, ts) - r
    if ac_max is not None:  # the level on the final deficit, free of the night-mean offset
        ac_max = float(np.median(d[plateau]))
    fit_mask = lit & ~masked & ~plateau & ~in_hold
    hold_mask = lit & ~masked & in_hold
    factor = month_factors(d[fit_mask], pv_est[fit_mask], months[fit_mask])
    pv_full = pv_est * factor[months - 1]
    if ac_max is not None:
        pv_full = np.minimum(pv_full, ac_max)
    holdout_error = null_error = None
    if hold_mask.any():
        holdout_error = huber_loss(d[hold_mask] - pv_full[hold_mask], delta)
        null_error = huber_loss(d[hold_mask], delta)
    stability = (1.0, 0.0)
    if not given:
        halves = tuple(hold_mask & np.isin(day, h) for h in half_days)
        long_enough = all(len(h) >= MIN_STABILITY_HALF_DAYS for h in half_days)
        ok = long_enough and all(h.any() for h in halves)
        stability = _stability(planes, d, shapes, slot, halves, delta) if ok else unstable
    # the operational load profile uses every row
    lm = load_median(r + pv_est, ts, now, mask=masked | plateau, days=history_days)
    model = PvModel(
        tuple(given) if given else planes,
        np.asarray(scales, dtype=float),
        factor,
        ac_max,
        holdout_error,
        null_error,
        stability,
        int(now),
        len(days_lit),
    )
    return model, lm, sessions
