"""Supervised PV fit per measured string (design 8.1) and the fill-in scale check (design 5.3).

When the inverter reports string power, each string is its own target: the measured string
power (``pv{i}_kwh * 4`` kW) against the per-kWp plane shape of plant.py. The GEN port, when it
carries production, is one more string (``gen_kwh``). The target is measured, not a residual,
so this path needs no night anchor, no level multiplier and no two-plane merge rule: each
string gets one plane from the tilt by azimuth grid (separate.plane_grid) and one non-negative
Huber scale (separate.huber_fit). With entered planes the geometry is fixed and only the scales
fit. A shared month factor and the holdout error compare the AC sum of the planes with the
measured AC sum of the strings (the DC strings through the inverter's DC-to-AC stage, the
GEN port as it is), so the registry rules of plan 2 apply unchanged.
"""

from __future__ import annotations

import numpy as np

from ..battery import pv_dc_to_ac_stage
from ..timegrid import Table, align
from .contract import Plane, SiteConfig
from .flows import STRING_COLUMNS
from .load import local_fields
from .plant import plane_shape
from .separate import (
    MIN_HOLDOUT_DAYS,
    PvModel,
    _local_month,
    curtailment_mask,
    dc_to_ac_kw,
    hour_mean,
    hourly_origin_rows,
    huber_fit,
    huber_loss,
    month_factors,
    plane_grid,
    planes_kw,
)
from .shading import N_AZ, N_EL, Shading, factors_at, fit_shading
from .solar import daylight, solar_position

MIN_STRING_DAYS = 7
SCALE_TOLERANCE = 0.25
GEN_COLUMN = "gen_kwh"
# a day counts for a string when its energy exceeds this (meter rounding is not production)
MIN_DAY_KWH = 0.01
_IRR_COLUMNS = ["ghi_w_m2", "dni_w_m2", "dhi_w_m2", "temp_c"]
_DELTA_SHARE = 1.0 / 3.0  # Huber delta: a third of the 90th percentile of the target
_DELTA_FLOOR = 1e-3
# shading profile, then the scales again on the shaded shapes, this many times (the first
# scales read unshaded shapes and come out low where shade is real). One round: the profile
# clips at 1,0, so each further round shades bins the noise or a gridded plane put below 1
# and lifts the scale to match (on the synthetic houses the scale climbs about 1 % a round
# and the holdout error does not improve after the first)
SHADING_ROUNDS = 1


def _name(column: str) -> str:
    """The Plane.string of a column: "pv1_kwh" -> "pv1", "gen_kwh" -> "gen"."""
    return column.removesuffix("_kwh")


def strings_present(q: Table) -> list[str]:
    """The ``pv{i}_kwh`` columns, then ``gen_kwh``, with positive energy on enough days.

    A column counts when its energy (NaN rows left out) sums to more than MIN_DAY_KWH on at
    least MIN_STRING_DAYS local days. String power is daytime power, so a positive day sum is
    a day with daytime production; a sensor that stays at 0 or NaN never counts.
    """
    ts = np.asarray(q["ts"], dtype=np.int64)
    if len(ts) == 0:
        return []
    _, _, day = local_fields(ts)
    out = []
    for col in (*STRING_COLUMNS, GEN_COLUMN):
        if col not in q:
            continue
        e = np.asarray(q[col], dtype=float)
        ok = np.isfinite(e)
        if not ok.any():
            continue
        _, inv = np.unique(day[ok], return_inverse=True)
        daily = np.bincount(inv, weights=e[ok])
        if int((daily > MIN_DAY_KWH).sum()) >= MIN_STRING_DAYS:
            out.append(col)
    return out


def _delta(y: np.ndarray) -> float:
    if len(y) == 0:
        return 1.0
    return max(float(np.percentile(y, 90)) * _DELTA_SHARE, _DELTA_FLOOR)


def _search(
    y: np.ndarray, grid: list[tuple[Plane, np.ndarray]], rows: np.ndarray
) -> tuple[int, float]:
    """(index, scale) of the grid plane whose non-negative Huber fit to y on rows has the
    lowest Huber loss.

    Ties go to the first plane of the grid, so the search is deterministic.
    """
    delta = _delta(y[rows])
    best: tuple[float, int, float] | None = None
    for i, (_, shape) in enumerate(grid):
        x = shape[rows]
        scale = float(huber_fit(x, y[rows], delta)[0])
        loss = huber_loss(y[rows] - x * scale, delta)
        if best is None or loss < best[0]:
            best = (loss, i, scale)
    assert best is not None
    return best[1], best[2]


def fit_strings(
    q: Table,
    irr: Table,
    cfg: SiteConfig,
    now: int,
    planes: tuple[Plane, ...] | None = None,
    holdout_days: int = 14,
    shading: bool = True,
) -> PvModel:
    """One plane per present string, fitted against the string's measured power.

    Without ``planes`` each present string (strings_present order) searches the plane grid:
    the plane whose non-negative Huber scale fits the measured power with the lowest Huber
    loss, returned as ``Plane(tilt, azimuth, string=name)``. With ``planes`` the geometry is
    fixed:
    - a plane that names a present string (Plane.string, never the position) fits its scale
      to that string;
    - a plane that names a string strings_present does not find (a GEN port with fewer than
      MIN_STRING_DAYS production days, a dead sensor) keeps its entered hint scale
      ``correction * (kwp or 1)``: there is nothing to fit it to;
    - planes that name no string (planes from before the field) fit jointly to the sum of the
      present strings no plane claims, and keep their hint scales when there is none (two
      such planes with the same geometry are collinear: their split is arbitrary);
    - present strings no plane claims search the grid when no plane without a string exists,
      so no measured production is left out.
    The model keeps the given planes in their order, then the searched strings; every plane is
    its own column, so two strings with the same geometry stay two planes. ``checked`` marks
    the planes whose scale was fitted against measured power (scale_check reads no other).

    The scales are DC kWp for the pv{i} strings (string power is DC power) and AC kWp for the
    GEN port. The measured total and the model total are AC: the DC strings sum and pass the
    loss profile's DC-to-AC stage (battery.pv_dc_to_ac_stage of cfg.loss_profile, carried as
    ``dc_to_ac``), the GEN port adds as it is.

    Rows: daylight quarters before ``now`` with a finite measured sum, not missing, not
    curtailed (separate.curtailment_mask: the inverter held the strings back) and with
    finite irradiance. Rows of hourly origin take the hour mean of every plane shape, as
    separate.fit_pv does. The holdout is the last ``min(holdout_days, n_lit // 2)`` daylight
    days (none under MIN_HOLDOUT_DAYS). The scales and a shared month factor (separate.
    month_factors on the AC total against the measured AC total) use the other days;
    holdout_error and null_error compare the measured AC total with the model and with 0 on
    the holdout rows. No AC maximum is learned. The stability is (1, 0) with fixed geometry
    and (nan, nan), "not tested", for a search, so registry.holdout_passes rules on the error
    ratio.

    Shading (``shading``, design 8.1): after the geometry, each plane whose scale was fitted
    gets a profile (shading.fit_shading) on the fit rows: the measured power of its string
    (the planes of one fitted target share it by their modelled power) against scale * shape,
    both DC for a pv{i} string. On rows of hourly origin both sides are hour means (the flat
    split against the hour-mean shape), so such rows give the profile at hour resolution:
    each quarter's sun bin takes the ratio of its hour. The scales then refit on the
    shaded shapes; profile and scales alternate SHADING_ROUNDS times. The month factor and
    the holdout error read the shaded shapes: the profile reads fit rows only, which all lie
    before the first holdout day, so the holdout sees it as a forecast would. The model
    carries the profile (``fitted_at = now``) and applies it on every quarter it predicts.
    The scales from before the shading round stay on the model as ``scales_check_kwp``, the
    figures scale_check compares with the entered kWp. Planes kept at their hint scale keep
    1,0.
    """
    ts = np.asarray(q["ts"], dtype=np.int64)
    lat, lon = cfg.latitude, cfg.longitude
    stage = pv_dc_to_ac_stage(cfg.loss_profile)
    present = strings_present(q)
    targets = {c: np.asarray(q[c], dtype=float) * 4.0 for c in present}
    if targets:
        dc_sum = sum((y for c, y in targets.items() if c != GEN_COLUMN), np.zeros(len(ts)))
        total = dc_to_ac_kw(dc_sum, stage) + targets.get(GEN_COLUMN, 0.0)
        total = np.where(np.isfinite(dc_sum), total, np.nan)
    else:
        total = np.full(len(ts), np.nan)
    aligned = align(irr, ts, _IRR_COLUMNS)
    zenith, _ = solar_position(ts, lat, lon)
    lit = daylight(zenith)
    _, _, day = local_fields(ts)
    months = _local_month(ts)
    hourly_rows = hourly_origin_rows(q)

    def hourly(s: np.ndarray) -> np.ndarray:
        if hourly_rows.any():  # flat-split hourly energies meet hour-mean shapes
            s = hour_mean(s, ts, hourly_rows)
        return s

    def raw(p: Plane) -> np.ndarray:
        return np.nan_to_num(plane_shape(aligned, p, lat, lon))

    def shape(p: Plane) -> np.ndarray:
        return hourly(plane_shape(aligned, p, lat, lon))

    bad_irr = np.zeros(len(ts), dtype=bool)
    for c in _IRR_COLUMNS:
        bad_irr |= ~np.isfinite(np.asarray(aligned[c], dtype=float))
    masked = (
        np.asarray(q["missing"], dtype=bool)
        | curtailment_mask(q)
        | ~np.isfinite(total)
        | (ts >= now)
        | (lit & bad_irr)
    )
    days_lit = np.unique(day[lit & ~masked])
    n_hold = min(holdout_days, len(days_lit) // 2)
    if n_hold < MIN_HOLDOUT_DAYS:
        n_hold = 0
    in_hold = np.isin(day, days_lit[len(days_lit) - n_hold :])
    fit_rows = lit & ~masked & ~in_hold
    hold_rows = lit & ~masked & in_hold
    total = np.nan_to_num(total)
    targets = {c: np.nan_to_num(y) for c, y in targets.items()}

    given = tuple(planes or ())
    names = {_name(c): c for c in present}
    out_planes: list[Plane] = list(given)
    columns: list[np.ndarray] = [np.nan_to_num(shape(p)) for p in given]
    raws: list[np.ndarray] = [raw(p) for p in given]
    scales = np.array([cfg.pv.correction * (p.kwp or 1.0) for p in given])
    checked = [False] * len(given)
    groups: list[tuple[list[int], np.ndarray]] = []  # (plane indexes, measured target) fitted
    claimed = {p.string for p in given if p.string in names}
    unclaimed = [c for c in present if _name(c) not in claimed]
    stringless = [i for i, p in enumerate(given) if p.string is None]
    to_search = [] if stringless else unclaimed

    if fit_rows.any():
        for name in sorted(claimed, key=lambda s: present.index(names[s])):
            idx = [i for i, p in enumerate(given) if p.string == name]
            y = targets[names[name]]
            x = np.column_stack([columns[i] for i in idx])[fit_rows]
            scales[idx] = huber_fit(x, y[fit_rows], _delta(y[fit_rows]))
            groups.append((idx, y))
            for i in idx:
                checked[i] = True
        if stringless and unclaimed:
            rest = sum((targets[c] for c in unclaimed), np.zeros(len(ts)))
            x = np.column_stack([columns[i] for i in stringless])[fit_rows]
            scales[stringless] = huber_fit(x, rest[fit_rows], _delta(rest[fit_rows]))
            groups.append((stringless, rest))
            for i in stringless:
                checked[i] = True
        if to_search:
            grid = [(p, np.nan_to_num(shape(p))) for p in plane_grid()]
            found = []
            for c in to_search:
                k, scale = _search(targets[c], grid, fit_rows)
                plane, column = grid[k]
                out_planes.append(Plane(plane.tilt_deg, plane.azimuth_deg, string=_name(c)))
                columns.append(column)
                raws.append(raw(plane))
                groups.append(([len(out_planes) - 1], targets[c]))
                found.append(scale)
                checked.append(True)
            scales = np.concatenate([scales, found])
    else:  # nothing to fit on: entered planes keep their hint scales, nothing is searched
        to_search = []

    stability = (1.0, 0.0) if given and not to_search else (float("nan"), float("nan"))
    out = tuple(out_planes)
    if not out:
        return PvModel(
            (), np.zeros(0), np.ones(12), None, None, None, stability, int(now), len(days_lit),
            supervised=True, checked=(), dc_to_ac=stage,
        )
    profile = check = None
    if shading and groups:
        check = np.asarray(scales, dtype=float).copy()
        profile, columns, scales = _fit_shading(
            np.column_stack(raws), scales, groups, fit_rows,
            ts, lat, lon, np.asarray(aligned["ghi_w_m2"], dtype=float), int(now), hourly,
        )
    pv_est = planes_kw(np.column_stack(columns) * scales[None, :], out, stage)
    factor = (
        month_factors(total[fit_rows], pv_est[fit_rows], months[fit_rows])
        if fit_rows.any()
        else np.ones(12)
    )
    holdout_error = null_error = None
    if hold_rows.any():
        delta = _delta(total[fit_rows]) if fit_rows.any() else _delta(total[hold_rows])
        pv_full = pv_est * factor[months - 1]
        holdout_error = huber_loss(total[hold_rows] - pv_full[hold_rows], delta)
        null_error = huber_loss(total[hold_rows], delta)
    return PvModel(
        out,
        np.asarray(scales, dtype=float),
        factor,
        None,
        holdout_error,
        null_error,
        stability,
        int(now),
        len(days_lit),
        supervised=True,
        checked=tuple(checked),
        dc_to_ac=stage,
        shading=profile,
        scales_check_kwp=check,
    )


def _fit_shading(
    raw: np.ndarray,
    scales: np.ndarray,
    groups: list[tuple[list[int], np.ndarray]],
    fit_rows: np.ndarray,
    ts: np.ndarray,
    lat: float,
    lon: float,
    ghi: np.ndarray,
    now: int,
    hourly,
) -> tuple[Shading, list[np.ndarray], np.ndarray]:
    """(profile, shaded columns, scales) after SHADING_ROUNDS of profile then scale refit.

    raw: (n, planes) plane shapes without the hour mean; ``hourly`` takes the hour mean on
    rows of hourly origin. The modelled power is scale * hourly(raw); the measured power of a
    group's target splits over its planes by their modelled power. Rows outside fit_rows and
    planes of no group read NaN, so the profile there stays 1,0. The shaded columns are
    hourly(raw * factors) on every row (in-sample: the profile reads fit rows only).
    """
    n, planes = raw.shape
    scales = np.asarray(scales, dtype=float).copy()
    profile = Shading(np.ones((planes, N_AZ, N_EL)), now)
    plain = np.column_stack([hourly(raw[:, j]) for j in range(planes)])
    columns = list(plain.T)
    for _ in range(SHADING_ROUNDS):
        modelled = plain * scales[None, :]
        measured = np.full((n, planes), np.nan)
        for idx, y in groups:
            total = modelled[:, idx].sum(axis=1)
            ok = fit_rows & (total > 0)
            for i in idx:
                measured[ok, i] = y[ok] * modelled[ok, i] / total[ok]
        profile = fit_shading(measured, modelled, ts, lat, lon, ghi, now)
        shaded = raw * factors_at(profile, ts, lat, lon)
        columns = [np.nan_to_num(hourly(shaded[:, j])) for j in range(planes)]
        for idx, y in groups:
            x = np.column_stack([columns[i] for i in idx])[fit_rows]
            scales[idx] = huber_fit(x, y[fit_rows], _delta(y[fit_rows]))
    return profile, columns, scales


def _match(model_planes: tuple[Plane, ...], plane: Plane, used: set[int]) -> int | None:
    """The model plane of an entered plane: by Plane.string when it names one, else equal."""
    for j, mp in enumerate(model_planes):
        if j in used:
            continue
        if (plane.string is not None and mp.string == plane.string) or (
            plane.string is None and mp == plane
        ):
            return j
    return None


def scale_check(
    model: PvModel, planes: tuple[Plane, ...], tolerance: float = SCALE_TOLERANCE
) -> list[int]:
    """Indexes into ``planes`` whose fitted kWp lies more than ``tolerance`` from the entered kWp.

    Relative: ``|fitted / entered - 1| > tolerance``, the fitted kWp being the scale from
    before the shading round (``model.scales_check_kwp``, else ``scales_kwp``). Planes
    pair with model planes by Plane.string (equal planes when no string is named). Only a
    strings-path model (``model.supervised``) is checked, and only its planes whose scale was
    fitted against measured power (``model.checked``): a blind scale carries the level
    multiplier, a hint scale is the entered figure. A plane without an entered kWp, or
    without a model plane, is not checked.
    """
    out: list[int] = []
    used: set[int] = set()
    if not model.supervised:
        return out
    fitted = model.scales_kwp if model.scales_check_kwp is None else model.scales_check_kwp
    for i, p in enumerate(planes):
        if p.kwp is None or p.kwp <= 0:
            continue
        j = _match(model.planes, p, used)
        if j is None:
            continue
        used.add(j)
        if j >= len(model.checked) or not model.checked[j]:
            continue
        if abs(float(fitted[j]) / float(p.kwp) - 1.0) > tolerance:
            out.append(i)
    return out


def string_labels(planes: tuple[Plane, ...], indexes: list[int]) -> str:
    """The repair-issue placeholder: "1, 2" for pv1 and pv2, "GEN" for the GEN port, the
    position (from 1) for a plane that names no string."""
    labels = []
    for i in indexes:
        s = planes[i].string
        if s is None:
            labels.append(str(i + 1))
        elif s == "gen":
            labels.append("GEN")
        else:
            labels.append(s.removeprefix("pv"))
    return ", ".join(labels)
