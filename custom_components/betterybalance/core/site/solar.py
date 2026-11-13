"""Solar geometry: NOAA position, Haurwitz clear-sky GHI, daylight mask, clearness ratio."""

from __future__ import annotations

import numpy as np

_HALF_ISP_S = 450
_GHI_FLOOR_WM2 = 20.0


def solar_position(ts: np.ndarray, lat: float, lon: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (zenith_deg, azimuth_deg) at the midpoint ts + 450 s of each interval.

    NOAA general solar position: fractional-year series for the equation of time and
    the declination, true solar time from longitude (east positive), spherical zenith.
    No refraction correction. Azimuth is compass, 0 north, 90 east, in [0, 360).
    """
    mid = np.asarray(ts, dtype=np.int64) + _HALF_ISP_S
    utc = mid.astype("datetime64[s]")
    day = utc.astype("datetime64[D]")
    year = day.astype("datetime64[Y]")
    year_start = year.astype("datetime64[D]")
    cal_year = year.astype(np.int64) + 1970
    year_len = 365 + ((cal_year % 4 == 0) & ((cal_year % 100 != 0) | (cal_year % 400 == 0)))
    doy0 = (day - year_start).astype(np.int64)
    sec_of_day = (utc - day.astype("datetime64[s]")).astype(np.int64)
    hour = sec_of_day / 3600.0
    gamma = 2.0 * np.pi / year_len * (doy0 + (hour - 12.0) / 24.0)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    solar_minutes = (sec_of_day / 60.0 + eqtime + 4.0 * lon) % 1440.0
    hour_angle = np.radians(solar_minutes / 4.0 - 180.0)

    phi = np.radians(lat)
    cos_zenith = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    zenith = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))
    azimuth = np.degrees(
        np.arctan2(
            np.sin(hour_angle),
            np.cos(hour_angle) * np.sin(phi) - np.tan(decl) * np.cos(phi),
        )
    )
    return zenith, (azimuth + 180.0) % 360.0


def clear_sky_ghi(zenith_deg: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky GHI in W/m2: 1098 cos(z) exp(-0.059 / cos(z)) for z < 90, else 0."""
    cos_z = np.cos(np.radians(np.asarray(zenith_deg, dtype=float)))
    up = cos_z > 0.0
    safe = np.where(up, cos_z, 1.0)
    return np.where(up, 1098.0 * safe * np.exp(-0.059 / safe), 0.0)


def daylight(zenith_deg: np.ndarray, limit_deg: float = 85.0) -> np.ndarray:
    """True where the sun is higher than 90 - limit_deg above the horizon."""
    return np.asarray(zenith_deg, dtype=float) < limit_deg


def clearness(ghi: np.ndarray, ghi_cs: np.ndarray) -> np.ndarray:
    """Clearness ratio ghi / ghi_cs, NaN where the clear-sky GHI is below 20 W/m2."""
    ghi = np.asarray(ghi, dtype=float)
    ghi_cs = np.asarray(ghi_cs, dtype=float)
    ok = ghi_cs >= _GHI_FLOOR_WM2
    return np.where(ok, ghi / np.where(ok, ghi_cs, 1.0), np.nan)
