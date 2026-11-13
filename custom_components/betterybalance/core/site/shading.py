"""Shading profile per PV plane on sun-position bins (design 8.1).

A profile is one multiplicative factor per plane and sun-position bin: 36 azimuth bins of
AZ_STEP degrees (0..350) by 9 elevation bins of EL_STEP degrees (0..80). It is fitted after
the geometry, on clear quarters (clearness kt >= KT_CLEAR) where the plane's modelled power
lies above 5 % of its peak, as the median ratio of measured to modelled power, clipped to
CLIP. A bin with at least one clear quarter keeps its own median: a sparse bin (1 to
MIN_CLEAR - 1 clear quarters) is noisier than a populated one, but its own quarters say more
about its sun position than its neighbours do (a shade edge runs through bins). Only a bin
without a clear quarter takes the median of its populated neighbours (MIN_CLEAR or more
clear quarters) within one bin (azimuth wraps at 360, elevation does not), else 1,0.

The factor multiplies the plane's DC shape (kW per kWp) before the DC-to-AC stage: shading
takes irradiance off the panels. Feature honesty: the fit reads quarters before
``fitted_at`` only. The profile applies wherever the model is used, on every quarter, like
the scales fitted with it (a forecast of past quarters, the load estimate, the drift alarm,
read the scales and the profile together). Decisions stay honest because each one reads the
PV model of its moment on slots at or after that model's fit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..timegrid import Table
from .solar import clear_sky_ghi, clearness, solar_position

AZ_STEP = 10
EL_STEP = 10
KT_CLEAR = 0.8
CLIP = (0.2, 1.0)
MIN_CLEAR = 8
N_AZ = 360 // AZ_STEP
N_EL = 90 // EL_STEP
PEAK_SHARE = 0.05  # a quarter counts when the plane's modelled power exceeds this share of its peak


@dataclass
class Shading:
    factors: np.ndarray  # (planes, N_AZ, N_EL)
    fitted_at: int


def sun_bins(ts: np.ndarray, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth bin 0..35, elevation bin 0..8) of the sun at each quarter's midpoint.

    Elevation is 90 - zenith; below the horizon it falls in bin 0 (plane shapes are 0 there).
    """
    zenith, azimuth = solar_position(ts, lat, lon)
    az = (np.floor(np.asarray(azimuth, dtype=float) / AZ_STEP).astype(np.int64)) % N_AZ
    el = np.clip(np.floor((90.0 - np.asarray(zenith, dtype=float)) / EL_STEP), 0, N_EL - 1)
    return az, el.astype(np.int64)


def _inherit(factor: np.ndarray, populated: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """Empty bins take the median of their populated neighbours within one bin, else 1,0;
    every other bin keeps its own factor."""
    out = factor.copy()
    for a in range(N_AZ):
        for e in range(N_EL):
            if not empty[a, e]:
                continue
            vals = [
                factor[(a + da) % N_AZ, e + de]
                for da in (-1, 0, 1)
                for de in (-1, 0, 1)
                if (da or de) and 0 <= e + de < N_EL and populated[(a + da) % N_AZ, e + de]
            ]
            out[a, e] = float(np.median(vals)) if vals else 1.0
    return out


def fit_shading(
    measured_kw: np.ndarray,
    modelled_kw: np.ndarray,
    ts: np.ndarray,
    lat: float,
    lon: float,
    ghi: np.ndarray,
    fitted_at: int,
) -> Shading:
    """The profile of each plane (column) of measured_kw against modelled_kw, both (n, planes).

    Rows: quarters before ``fitted_at`` with kt = ghi / clear-sky ghi >= KT_CLEAR, finite
    measured and modelled power, and modelled power above PEAK_SHARE of the plane's peak (the
    highest finite modelled power before ``fitted_at``). A plane whose measured column is NaN
    throughout has no rows and keeps 1,0 everywhere. Per bin with at least one row (populated
    from MIN_CLEAR rows on, sparse below): the median of measured over modelled, clipped to
    CLIP; a sparse bin keeps its own median. A bin without a row takes the median of its
    populated neighbours within one bin, else 1,0 (see the module docstring).
    """
    ts = np.asarray(ts, dtype=np.int64)
    measured = np.asarray(measured_kw, dtype=float).reshape(len(ts), -1)
    modelled = np.asarray(modelled_kw, dtype=float).reshape(len(ts), -1)
    zenith, _ = solar_position(ts, lat, lon)
    kt = clearness(ghi, clear_sky_ghi(zenith))
    before = ts < int(fitted_at)
    clear = before & (np.nan_to_num(kt, nan=0.0) >= KT_CLEAR)
    az, el = sun_bins(ts, lat, lon)
    flat = az * N_EL + el
    planes = measured.shape[1]
    factors = np.ones((planes, N_AZ, N_EL))
    for j in range(planes):
        m, x = measured[:, j], modelled[:, j]
        finite = np.isfinite(m) & np.isfinite(x)
        if not (before & finite).any():
            continue
        peak = float(np.max(x[before & finite]))
        rows = clear & finite & (x > PEAK_SHARE * peak) & (x > 0)
        factor = np.ones(N_AZ * N_EL)
        count = np.zeros(N_AZ * N_EL, dtype=np.int64)
        if rows.any():
            ratio = m[rows] / x[rows]
            b = flat[rows]
            order = np.argsort(b, kind="stable")
            b, ratio = b[order], ratio[order]
            keys, start, n = np.unique(b, return_index=True, return_counts=True)
            for k, s, c in zip(keys, start, n, strict=True):
                count[k] = c
                factor[k] = float(np.clip(np.median(ratio[s : s + c]), *CLIP))
        count = count.reshape(N_AZ, N_EL)
        factor = factor.reshape(N_AZ, N_EL)
        factors[j] = _inherit(factor, count >= MIN_CLEAR, count == 0)
    return Shading(factors, int(fitted_at))


def factors_at(shading: Shading, ts: np.ndarray, lat: float, lon: float) -> np.ndarray:
    """(n, planes) factors of the profile at each quarter's sun position."""
    az, el = sun_bins(np.asarray(ts, dtype=np.int64), lat, lon)
    return np.asarray(shading.factors, dtype=float)[:, az, el].T


def apply_shading(
    shading: Shading, shapes: np.ndarray, ts: np.ndarray, lat: float, lon: float
) -> np.ndarray:
    """shapes (n, planes) times the profile's factors at each quarter's sun position, on
    every quarter (``fitted_at`` is the profile's fit time, not a gate)."""
    return np.asarray(shapes, dtype=float) * factors_at(shading, ts, lat, lon)


def to_dict(shading: Shading | None) -> dict | None:
    if shading is None:
        return None
    return {
        "factors": np.asarray(shading.factors, dtype=float).tolist(),
        "fitted_at": int(shading.fitted_at),
    }


def from_dict(d: dict | None) -> Shading | None:
    if d is None:
        return None
    factors = np.asarray(d["factors"], dtype=float).reshape(-1, N_AZ, N_EL)
    return Shading(factors, int(d["fitted_at"]))


def share_removed(model, irr: Table, lat: float, lon: float) -> float:
    """1 - modelled energy with the profile / without it, over every quarter of ``irr``.

    0,0 without a profile or without modelled energy.
    """
    from .separate import predict_pv_kw  # separate imports this module

    if model.shading is None or len(irr["ts"]) == 0:
        return 0.0
    plain = float(np.nansum(predict_pv_kw(dataclasses.replace(model, shading=None), irr, lat, lon)))
    if plain <= 0:
        return 0.0
    return 1.0 - float(np.nansum(predict_pv_kw(model, irr, lat, lon))) / plain


def share_by_string(model, irr: Table, lat: float, lon: float) -> dict[str, float]:
    """share_removed per Plane.string (``planeN`` for a plane without one), each on the
    model of that string's planes alone.

    The shares do not sum to (or average to) the site share of share_removed: each DC string
    passes the DC-to-AC stage alone here, and the stage is nonlinear in the DC sum.
    """
    if model.shading is None:
        return {}
    groups: dict[str, list[int]] = {}
    for i, p in enumerate(model.planes):
        groups.setdefault(p.string or f"plane{i + 1}", []).append(i)
    out = {}
    for key, idx in groups.items():
        sub = dataclasses.replace(
            model,
            planes=tuple(model.planes[i] for i in idx),
            scales_kwp=np.asarray(model.scales_kwp, dtype=float)[idx],
            checked=tuple(model.checked[i] for i in idx) if model.checked else (),
            shading=Shading(np.asarray(model.shading.factors)[idx], model.shading.fitted_at),
            scales_check_kwp=(
                None
                if model.scales_check_kwp is None
                else np.asarray(model.scales_check_kwp, dtype=float)[idx]
            ),
        )
        out[key] = share_removed(sub, irr, lat, lon)
    return out
