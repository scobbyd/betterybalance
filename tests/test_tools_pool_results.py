import json

import numpy as np

from tools import pool_results as PR


def _file(tmp_path, name, diffs, cap=">=40"):
    d = {
        "format": 1,
        "version": "0.2.0",
        "source": "replay",
        "window": {"from": "2026-09-01", "to": "2026-09-30"},
        "contract_mode": "net_exporter",
        "site_class": {"capacity_bin": cap, "power_bin": ">=10", "pv": "learned", "ev": False},
        "tiers_by_month": {"2026-09": "D"},
        "learning_days": 0,
        "terminal_eur": 0.0,
        "interval": [0.0, 1.0],
        "daily": [{"day": f"2026-09-{i + 1:02d}", "diff_eur": x} for i, x in enumerate(diffs)],
    }
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return p


def test_pool_two_houses(tmp_path):
    a = _file(tmp_path, "a.json", np.ones(30) * 1.0)
    b = _file(tmp_path, "b.json", np.ones(30) * 3.0, cap="10-20")
    pooled = PR.pool([a, b])
    assert pooled["houses"] == 2 and pooled["days"] == 60 and pooled["mean_eur_day"] == 2.0
    assert pooled["interval"][0] <= 2.0 <= pooled["interval"][1]
    assert pooled["by_class"][">=40"] == 1
    assert "2,00" in PR.render(pooled)
    assert [h["name"] for h in pooled["per_house"]] == ["a", "b"]


def test_cli_skips_other_formats(tmp_path, capsys):
    _file(tmp_path, "a.json", [1.0, 2.0])
    (tmp_path / "other.json").write_text(json.dumps({"format": 2}))
    assert PR.main([str(tmp_path)]) == 0
    assert "1,50" in capsys.readouterr().out
