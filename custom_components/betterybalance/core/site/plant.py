"""Plant model: per-kWp AC shape of one roof plane from irradiance and solar geometry."""

from __future__ import annotations

import numpy as np

from ..timegrid import Table
from .contract import Plane
from .solar import solar_position

GAMMA = -0.0035
ALBEDO = 0.2
B0 = 0.05

_SOLAR_CONSTANT_WM2 = 1367.0
_COS_Z_FLOOR = float(np.cos(np.radians(89.0)))


def poa_irradiance(
    ghi: np.ndarray,
    dni: np.ndarray,
    dhi: np.ndarray,
    zenith_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    tilt_deg: float,
    plane_az_deg: float,
    albedo: float = ALBEDO,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hay-Davies transposition. Return (poa_w_m2, poa_beam_w_m2, aoi_deg).

    Beam is dni cos(aoi). Sky diffuse splits into a circumsolar part (anisotropy index
    dni / 1367, times the beam ratio) and an isotropic part. Ground reflection is
    ghi albedo (1 - cos tilt) / 2. Beam is 0 where aoi >= 90 or zenith >= 90.
    """
    ghi, dni, dhi = (np.asarray(x, dtype=float) for x in (ghi, dni, dhi))
    zenith = np.asarray(zenith_deg, dtype=float)
    sun_az = np.asarray(azimuth_deg, dtype=float)
    tilt = np.radians(tilt_deg)

    z = np.radians(zenith)
    cos_aoi = np.clip(
        np.cos(z) * np.cos(tilt)
        + np.sin(z) * np.sin(tilt) * np.cos(np.radians(sun_az - plane_az_deg)),
        -1.0,
        1.0,
    )
    aoi_deg = np.degrees(np.arccos(cos_aoi))

    facing = (cos_aoi > 0.0) & (zenith < 90.0)
    cos_aoi_up = np.where(facing, cos_aoi, 0.0)
    beam = np.where(facing, dni * cos_aoi_up, 0.0)

    beam_ratio = cos_aoi_up / np.maximum(np.cos(z), _COS_Z_FLOOR)
    anisotropy = np.clip(dni / _SOLAR_CONSTANT_WM2, 0.0, 1.0)
    sky = dhi * (anisotropy * beam_ratio + (1.0 - anisotropy) * (1.0 + np.cos(tilt)) / 2.0)
    ground = ghi * albedo * (1.0 - np.cos(tilt)) / 2.0

    poa = np.maximum(beam + sky + ground, 0.0)
    return poa, beam, aoi_deg


def iam_ashrae(aoi_deg: np.ndarray, b0: float = B0) -> np.ndarray:
    """ASHRAE incidence-angle modifier 1 - b0 (1 / cos(aoi) - 1), in [0, 1], 0 at aoi >= 90."""
    aoi = np.asarray(aoi_deg, dtype=float)
    cos_aoi = np.cos(np.radians(aoi))
    open_side = ~(aoi >= 90.0)  # NaN counts as open, so it propagates
    safe = np.where(open_side, cos_aoi, 1.0)
    iam = np.clip(1.0 - b0 * (1.0 / safe - 1.0), 0.0, 1.0)
    return np.where(open_side, iam, 0.0)


def cell_temperature(
    poa: np.ndarray, temp_c: np.ndarray, wind_m_s: float = 2.0, u0: float = 25.0, u1: float = 6.84
) -> np.ndarray:
    """Faiman cell temperature: temp_c + poa / (u0 + u1 wind)."""
    return np.asarray(temp_c, dtype=float) + np.asarray(poa, dtype=float) / (u0 + u1 * wind_m_s)


def plane_shape(
    irr: Table,
    plane: Plane,
    lat: float,
    lon: float,
    ac_max_kw_per_kwp: float | None = None,
) -> np.ndarray:
    """kW per kWp for each quarter of one plane.

    Hay-Davies POA, ASHRAE IAM on the beam part only, Faiman cell temperature, PVWatts
    temperature derating, optional AC clip. 0 where the sun is below the horizon. NaN
    inputs give NaN.
    """
    zenith, azimuth = solar_position(irr["ts"], lat, lon)
    poa, beam, aoi = poa_irradiance(
        irr["ghi_w_m2"],
        irr["dni_w_m2"],
        irr["dhi_w_m2"],
        zenith,
        azimuth,
        plane.tilt_deg,
        plane.azimuth_deg,
    )
    poa_eff = poa - beam + beam * iam_ashrae(aoi)
    t_cell = cell_temperature(poa, irr["temp_c"])
    kw = poa_eff / 1000.0 * (1.0 + GAMMA * (t_cell - 25.0))
    if ac_max_kw_per_kwp is not None:
        kw = np.minimum(kw, ac_max_kw_per_kwp)
    return np.where(zenith >= 90.0, 0.0, kw)
