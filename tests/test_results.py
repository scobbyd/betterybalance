import json

from betterybalance.core import results as RS
from betterybalance.replay import engine as E

from tests.test_replay_engine import _inputs
from tests.test_shipped_model import _scan


def test_export_is_anonymous_and_binned(tmp_path):
    from betterybalance.replay import report as R

    inp = _inputs()
    r = E.replay(inp)
    names = {f.name for f in R.write_outputs(r, tmp_path, {"site": "synthetic house"}, cfg=inp.cfg)}
    assert "results.json" in names and "results.json" not in {
        f.name for f in R.write_outputs(r, tmp_path / "b", {"site": "synthetic house"})
    }
    d = RS.export_results(r, inp.cfg, "0.2.0")
    assert d["format"] == 1 and d["site_class"] == {
        "capacity_bin": ">=40",
        "power_bin": ">=10",
        "pv": "none",
        "ev": False,
    }
    assert set(d) == {
        "format",
        "version",
        "source",
        "window",
        "contract_mode",
        "site_class",
        "tiers_by_month",
        "learning_days",
        "terminal_eur",
        "interval",
        "daily",
    }
    text = json.dumps(d)
    # every EUR value is in cents, so the [lo, hi] pair never reads as a coordinate pair
    eur = [d["terminal_eur"], *d["interval"], *(x["diff_eur"] for x in d["daily"])]
    assert all(round(x, 2) == x for x in eur)
    assert f"[{d['interval'][0]!r}, {d['interval'][1]!r}]" in text
    on_disk = json.loads((tmp_path / "results.json").read_text())
    assert on_disk["interval"] == d["interval"]
    assert _scan((tmp_path / "results.json").read_text()) == []
    assert _scan(text) == [] and "lat" not in text and "price" not in text and len(d["daily"]) == 3
    assert RS.capacity_bin(9.9) == "<10" and RS.capacity_bin(20.0) == "20-40"
    assert RS.power_bin(5.0) == "5-10"


def test_results_json_is_byte_identical_and_strict(tmp_path):
    from betterybalance.replay import report as R

    inp = _inputs(days_replay=1)
    r = E.replay(inp)
    R.write_outputs(r, tmp_path / "a", {}, cfg=inp.cfg)
    R.write_outputs(r, tmp_path / "b", {}, cfg=inp.cfg)
    a = (tmp_path / "a" / "results.json").read_bytes()
    assert a == (tmp_path / "b" / "results.json").read_bytes()
    d = json.loads(a)  # strict JSON: no NaN literal
    assert d["interval"] == [None, None] or all(isinstance(x, float) for x in d["interval"])
    assert d["window"]["from"] == d["daily"][0]["day"] == d["window"]["to"]
