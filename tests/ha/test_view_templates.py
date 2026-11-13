"""The markdown cards of the ready-made view, rendered by Home Assistant's template engine:
sensible text on a fresh install (no entities, unknown states), European numbers, never
"None" on screen."""

import json
import pathlib
import shutil
import subprocess

import pytest
from homeassistant.helpers.template import Template

CARD_JS = (
    pathlib.Path(__file__).resolve().parents[2]
    / "custom_components"
    / "betterybalance"
    / "www"
    / "betterybalance-plan-card.js"
)
NODE_VIEW = (
    "globalThis.HTMLElement = class {}; const d = {};"
    "globalThis.customElements = { define(n, c) { d[n] = c; }, get(n) { return d[n]; } };"
    "const m = await import(process.argv[1]);"
    "process.stdout.write(JSON.stringify({en: m.viewConfig('en'), nl: m.viewConfig('nl')}));"
)
STATE = "sensor.betterybalance_state"
BACKTEST = "sensor.betterybalance_backtest"
HEADLINE = {
    "result_eur": 1234.5,
    "mean_eur_day": 0.4,
    "lo_eur_day": -0.001,
    "hi_eur_day": 0.85,
    "days": 365,
    "missing_quarters": 12,
    "learning_days": 0,
    "irradiance": "previous runs",
    "indicative": "Indicative: replayed on hourly history.",
    "terminal_eur": None,
}


def _markdowns() -> dict[str, dict[str, str]]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed", allow_module_level=True)
    out = subprocess.run(
        [node, "--input-type=module", "-e", NODE_VIEW, CARD_JS.as_uri()],
        capture_output=True,
        text=True,
        check=True,
    )
    views = json.loads(out.stdout)
    found = {}
    for lang, view in views.items():
        md = [c["content"] for s in view["sections"] for c in s["cards"] if c["type"] == "markdown"]
        assert len(md) == 2
        found[lang] = {"verdict": md[0], "backtest": md[1]}
    return found


MD = _markdowns()


def _render(hass, content: str) -> str:
    out = Template(content, hass).async_render(parse_result=False)
    assert "None" not in out and "unknown" not in out.lower()
    assert out.strip()
    return out


@pytest.mark.parametrize("lang", ["en", "nl"])
async def test_fresh_install_reads_sentences_not_none(hass, lang):
    texts = [_render(hass, MD[lang][k]) for k in ("verdict", "backtest")]
    hass.states.async_set(STATE, "unknown", {})
    hass.states.async_set(BACKTEST, "unknown", {})
    assert [_render(hass, MD[lang][k]) for k in ("verdict", "backtest")] == texts
    hass.states.async_set(STATE, "hold", {"verdict": None, "status_text": []})
    hass.states.async_set(BACKTEST, "idle", {"headline": None, "progress": None})
    assert [_render(hass, MD[lang][k]) for k in ("verdict", "backtest")] == texts


async def test_verdict_and_status_sentences(hass):
    hass.states.async_set(
        STATE,
        "charge",
        {
            "verdict": "Charging 3,0 kW: the price cleared.",
            "status_text": [
                "PV forecast off by more than 15 % this week",
                "Learning the site models",
            ],
        },
    )
    out = _render(hass, MD["en"]["verdict"])
    assert "Charging 3,0 kW: the price cleared." in out
    assert "PV forecast off by more than 15 % this week" in out and "Learning the site" in out
    assert out.index("Charging") < out.index("PV forecast")


@pytest.mark.parametrize("lang", ["en", "nl"])
async def test_backtest_headline_in_european_numbers(hass, lang):
    hass.states.async_set(BACKTEST, "done", {"headline": HEADLINE})
    out = _render(hass, MD[lang]["backtest"])
    assert "1.234,50" in out and "0,40" in out and "0,85" in out and "365" in out
    assert "-0,00" not in out and "0,00" in out
    assert "1234.5" not in out and "0.85" not in out


@pytest.mark.parametrize("lang", ["en", "nl"])
async def test_backtest_running_and_failed(hass, lang):
    hass.states.async_set(BACKTEST, "running", {"progress": 0.424, "eta_s": 600})
    assert "42 %" in _render(hass, MD[lang]["backtest"])
    hass.states.async_set(BACKTEST, "running", {"progress": None})
    _render(hass, MD[lang]["backtest"])
    hass.states.async_set(BACKTEST, "failed", {"error": "no_end_soc"})
    assert "no_end_soc" in _render(hass, MD[lang]["backtest"])
    hass.states.async_set(BACKTEST, "failed", {"error": None})
    _render(hass, MD[lang]["backtest"])
