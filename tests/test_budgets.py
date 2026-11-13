# tests/test_budgets.py
"""Compute budgets (design 15): the minute step and the quarter setup at five times the dev-box
medians, under the design ceilings of a Pi 4 (minute 20 ms, setup 500 ms).

The fixture is a box four weeks in: ``tests.test_live_site._site`` on a file store (WAL) plus
28 days of minute rows before the start, so the view reads 40.320 minutes. The timed run is 14
quarters; the tests take the median of the first 10 setups and the first 200 minutes, so a
busy machine (the full suite beside it) moves the median far less than the 5x margin. The
nightly refit is timed and printed, not asserted.
"""

import statistics
import time

import numpy as np
import pytest
from betterybalance.core.fmt import eu
from betterybalance.core.store import Store

from tests.test_live_site import _flows, _site

# dev box medians, measured 2026-11-11 (x86-64, Python 3.12) with this fixture:
# on_minute 0,34 ms, setup_quarter 70 ms (the largest median of three runs)
MINUTE_BUDGET_S = 5 * 0.00034
QUARTER_BUDGET_S = 5 * 0.070
DESIGN_MINUTE_S = 0.020
DESIGN_QUARTER_S = 0.500
MINUTE_DAYS = 28
QUARTERS = 14
N_MINUTES = 200
N_SETUPS = 10


def _measure(path) -> dict:
    """The setup and minute times of a 14-quarter run and the refit time on a 4-week store."""
    site, store, start = _site(store=Store(path))
    for t in range(start - MINUTE_DAYS * 86400, start, 60):
        store.put_minute(dict(_flows(t), ts=t))
    setups, minutes = [], []
    for ts in range(start, start + QUARTERS * 900, 60):
        if (ts - start) % 900 == 60:
            t0 = time.perf_counter()
            site.setup_quarter(ts)
            setups.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        site.on_minute(ts, dict(_flows(ts), ts=ts))
        minutes.append(time.perf_counter() - t0)
    n = 96 * 10
    its = start - n * 900 + 900 * np.arange(n, dtype=np.int64)
    irr = {
        "ts": its, "ghi_w_m2": np.zeros(n), "dni_w_m2": np.zeros(n), "dhi_w_m2": np.zeros(n),
        "temp_c": np.full(n, 10.0),
    }
    t0 = time.perf_counter()
    site.fit(start, irr)
    refit = time.perf_counter() - t0
    store.close()
    return {"setup": setups[:N_SETUPS], "minute": minutes[:N_MINUTES], "refit": refit}


@pytest.fixture(scope="module")
def timed(tmp_path_factory):
    return _measure(tmp_path_factory.mktemp("budget") / "store.sqlite3")


def test_budgets_sit_under_the_design_ceilings():
    assert MINUTE_BUDGET_S <= DESIGN_MINUTE_S and QUARTER_BUDGET_S <= DESIGN_QUARTER_S


def test_minute_step_budget(timed):
    assert len(timed["minute"]) == N_MINUTES
    assert statistics.median(timed["minute"]) < MINUTE_BUDGET_S


def test_quarter_setup_budget(timed):
    assert len(timed["setup"]) == N_SETUPS
    assert statistics.median(timed["setup"]) < QUARTER_BUDGET_S


def test_refit_time_is_printed(timed, capsys):
    with capsys.disabled():
        print(
            f"\nrefit (fit over {MINUTE_DAYS} days of minutes): {eu(timed['refit'], 2)} s; "
            f"setup median {eu(statistics.median(timed['setup']) * 1000, 1)} ms; "
            f"minute median {eu(statistics.median(timed['minute']) * 1000, 2)} ms"
        )
    assert timed["refit"] > 0
