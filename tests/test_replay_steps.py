# tests/test_replay_steps.py
import hashlib

import numpy as np
import pytest
from betterybalance.replay import engine as E
from betterybalance.replay import steps as ST
from betterybalance.replay.report import _csv_bytes

from tests.test_replay_engine import _inputs

# sha256 of the decisions and ledger CSV bytes of ``E.replay(_inputs())``, recorded on the
# plan-2 engine before the move into ``replay.steps``
GOLDEN = "2dfcb5687b77f626dd01729d49bcc07e7ef011656499b77f12ecde20ebf9f318"


def _digest(result) -> str:
    decisions = E.decisions_table(result.decisions)
    ledger = E.ledger_table(result.virtual + result.actual)
    data = _csv_bytes(list(decisions), decisions) + _csv_bytes(list(ledger), ledger)
    return hashlib.sha256(data).hexdigest()


def test_factored_engine_is_byte_identical():
    assert _digest(E.replay(_inputs())) == GOLDEN


def test_irr_forecast_sets_the_irradiance_note():
    from betterybalance.replay.irradiance import IrrByLead

    inp = _inputs()
    assert E.replay(inp).site_report["irradiance"] == "archive (day-0 model output; optimistic)"
    inp.irr_forecast = IrrByLead({}, inp.irr, day0_lead=1)
    r = E.replay(inp)
    assert r.site_report["irradiance"] == inp.irr_forecast.note()
    assert r.site_report["irradiance"].startswith("previous runs (day 0 from lead 1")


def test_house_path_and_result_pairs():
    r = E.replay(_inputs())
    assert len(r.house) == len(r.virtual) and r.house[0].path == "house"
    total = sum(
        h.money_eur - v.money_eur
        for h, v in zip(r.house, r.virtual)
        if not (h.missing or v.missing)
    )
    assert r.result_pairs["result_eur"].sum() == pytest.approx(total)


def test_stepper_reanchor_resets_the_battery():
    inp = _inputs()
    from betterybalance.core.battery import load_profile

    s = ST.Stepper(inp.cfg, load_profile("flat", inp.cfg.pack), 24.0)
    out = s.minute(inp.start, None, 0.5, 0.0, False, False)
    assert out.decision.reason == "stale" and np.isfinite(out.net_kw)
    s.reanchor(10.0)
    assert s.battery.soc_kwh == 10.0 and s.trader._ranked_at is None
    miss = s.minute(inp.start + 60, None, float("nan"), float("nan"), True, False)
    assert miss.decision.reason == "missing" and np.isnan(miss.net_kw) and s.battery.soc_kwh == 10.0
