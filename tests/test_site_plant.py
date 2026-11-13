import datetime as dt

import numpy as np
import pytest
from betterybalance.core.site import plant as P
from betterybalance.core.site import solar as S
from betterybalance.core.site.contract import Plane
from betterybalance.core.timegrid import ZONE


def _day(date: str) -> np.ndarray:
    start = int(dt.datetime.fromisoformat(date + "T00:00:00").replace(tzinfo=ZONE).timestamp())
    return np.arange(start, start + 86400, 900, dtype=np.int64)


def _clear_day(ts, lat=52.1, lon=5.2):
    z, _ = S.solar_position(ts, lat, lon)
    ghi = S.clear_sky_ghi(z)
    dhi = 0.2 * ghi
    cosz = np.cos(np.radians(np.minimum(z, 89.0)))
    dni = np.where(z < 89, (ghi - dhi) / cosz, 0.0)
    temp = np.full(len(ts), 20.0)
    return {"ts": ts, "ghi_w_m2": ghi, "dni_w_m2": dni, "dhi_w_m2": dhi, "temp_c": temp}


def test_horizontal_plane_reproduces_ghi():
    ts = _day("2026-06-21")
    irr = _clear_day(ts)
    z, az = S.solar_position(ts, 52.1, 5.2)
    poa, beam, aoi = P.poa_irradiance(
        irr["ghi_w_m2"], irr["dni_w_m2"], irr["dhi_w_m2"], z, az, 0.0, 180.0
    )
    day = z < 80
    assert np.allclose(poa[day], irr["ghi_w_m2"][day], rtol=0.02)


def test_east_plane_leads_west_plane_in_the_morning_and_south_beats_horizontal_at_noon():
    ts = _day("2026-06-21")
    irr = _clear_day(ts)
    east = P.plane_shape(irr, Plane(35, 90), 52.1, 5.2)
    west = P.plane_shape(irr, Plane(35, 270), 52.1, 5.2)
    south = P.plane_shape(irr, Plane(35, 180), 52.1, 5.2)
    flat = P.plane_shape(irr, Plane(0, 180), 52.1, 5.2)
    morning, noon = 32, 54  # 08:00 and 13:30 local
    assert east[morning] > 1.5 * west[morning] and south[noon] > flat[noon]
    assert east.sum() == pytest.approx(west.sum(), rel=0.15)
    assert (south[ts < ts[0] + 4 * 3600] == 0).all()


def test_iam_and_cell_temperature_and_clipping():
    assert P.iam_ashrae(np.array([0.0, 60.0, 90.0])).tolist() == pytest.approx([1.0, 0.95, 0.0])
    assert np.isnan(P.iam_ashrae(np.array([np.nan]))[0])
    t_cell = P.cell_temperature(np.array([1000.0]), np.array([20.0]))[0]
    assert t_cell == pytest.approx(45.85, abs=0.02)
    ts = _day("2026-06-21")
    irr = _clear_day(ts)
    clipped = P.plane_shape(irr, Plane(35, 180), 52.1, 5.2, ac_max_kw_per_kwp=0.5)
    unclipped = P.plane_shape(irr, Plane(35, 180), 52.1, 5.2)
    assert clipped.max() == pytest.approx(0.5) and unclipped.max() > 0.7
