import datetime as dt

import numpy as np
import pytest
from betterybalance.core.site import solar as S
from betterybalance.core.timegrid import ZONE


def _ts(s: str) -> int:
    return int(dt.datetime.fromisoformat(s).replace(tzinfo=ZONE).timestamp())


def test_noon_elevation_on_the_june_solstice():
    # solar noon at longitude 5,2 E is about 13:39 local summer time on 2026-06-21
    z, az = S.solar_position(np.array([_ts("2026-06-21T13:31:30")]), 52.1, 5.2)  # midpoint 13:39
    assert 90 - z[0] == pytest.approx(90 - 52.1 + 23.44, abs=0.3)
    assert az[0] == pytest.approx(180.0, abs=1.5)


def test_morning_sun_is_east_and_night_is_below_horizon():
    ts = np.array([_ts("2026-06-21T07:00:00"), _ts("2026-06-21T01:00:00")])
    z, az = S.solar_position(ts, 52.1, 5.2)
    assert 60 < az[0] < 110 and z[0] < 90 and z[1] > 90


def test_afternoon_sun_is_west_and_azimuth_stays_in_range():
    ts = np.array(
        [_ts("2026-06-21T18:00:00"), _ts("2026-06-21T00:00:00"), _ts("2026-06-21T23:45:00")]
    )
    _, az = S.solar_position(ts, 52.1, 5.2)
    assert 250 < az[0] < 300
    assert np.all((az >= 0) & (az < 360))
    assert az[1] > 300 or az[1] < 60  # local midnight: sun is in the north


def test_equinox_noon_zenith_equals_latitude():
    z, _ = S.solar_position(np.array([_ts("2026-03-20T12:45:00")]), 52.1, 5.2)
    assert z[0] == pytest.approx(52.1, abs=1.0)


def test_clear_sky_and_daylight():
    z = np.array([0.0, 60.0, 89.0, 95.0])
    g = S.clear_sky_ghi(z)
    assert g[0] == pytest.approx(1098 * np.exp(-0.059)) and g[1] < g[0] and g[3] == 0.0
    assert S.daylight(z).tolist() == [True, True, False, False]
    assert np.isnan(S.clearness(np.array([10.0]), np.array([5.0]))[0])
