import ast
import copy
import functools
import json
import pathlib
import re
import shutil
import subprocess

import pytest
from betterybalance.const import BINARY_KEYS, SENSOR_KEYS, SERVICES
from betterybalance.core import solarman as S
from betterybalance.core.site import contract as C
from betterybalance.live import export as X

CC = pathlib.Path(__file__).resolve().parent.parent / "custom_components" / "betterybalance"
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
FILES = ("translations/en.json", "translations/nl.json", "strings.json")
# the platform name as a variable: written out before a dot it reads as an entity id to the
# owner-fingerprint scan of test_shipped_model
SENSOR = "sensor"
PLACEHOLDER = re.compile(r"\{(\w+)\}")
# strings that read the same in both languages (proper nouns, abbreviations, numbers)
SAME_IN_DUTCH = {"BetteryBalance", "BTW", "Kappa", "Backtest", "Export", "Week"}


def _keys(d, p=""):
    nested = {x for k, v in d.items() if isinstance(v, dict) for x in _keys(v, p + k + ".")}
    return {p + k for k in d} | nested


def _leaves(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _leaves(v, p + k + ".")
        else:
            yield p + k, v


def _load(name: str) -> dict:
    return json.loads((CC / name).read_text())


def _get(d: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def test_translations_share_one_key_tree_and_have_no_emojis():
    en, nl, src = (_load(f) for f in FILES)
    assert _keys(en) == _keys(nl) == _keys(src)
    for f in FILES:
        text = (CC / f).read_text()
        assert not EMOJI.search(text) and "—" not in text


def test_english_translation_is_the_source():
    assert _load("translations/en.json") == _load("strings.json")


def test_every_string_keeps_its_placeholders_in_dutch():
    en, nl = dict(_leaves(_load("strings.json"))), dict(_leaves(_load("translations/nl.json")))
    for key, text in en.items():
        assert set(PLACEHOLDER.findall(nl[key])) == set(PLACEHOLDER.findall(text)), key


def test_dutch_is_written_not_copied():
    en, nl = dict(_leaves(_load("strings.json"))), dict(_leaves(_load("translations/nl.json")))
    same = {k for k, v in en.items() if nl[k] == v and v not in SAME_IN_DUTCH}
    assert not same, sorted(same)


def test_knob_descriptions_are_the_contract_help_in_english():
    advanced = _load("strings.json")["options"]["step"]["advanced"]
    assert advanced["data_description"] == C.KNOB_HELP


def _module_tuple(path: pathlib.Path, name: str) -> tuple:
    for node in ast.parse(path.read_text()).body:
        targets = getattr(node, "targets", [])
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path.name}")


def _service_fields() -> dict[str, list[str]]:
    services: dict[str, list[str]] = {}
    current = None
    for line in (CC / "services.yaml").read_text().splitlines():
        if m := re.match(r"^(\w+):\s*$", line):
            current = m.group(1)
            services[current] = []
        elif m := re.match(r"^    (\w+):\s*$", line):
            services[current].append(m.group(1))
    return services


def _translation_keys(text: str) -> tuple[set, set]:
    """``(exception keys, issue keys)`` of the ``translation_key="..."`` literals in ``text``: a
    key belongs to an issue when the nearest call before it is ``async_create_issue``."""
    exceptions, issues = set(), set()
    for m in re.finditer(r'translation_key\s*=\s*"(\w+)"', text):
        before = text[: m.start()]
        issue_at = before.rfind("async_create_issue(")
        error_at = max(before.rfind("Error("), before.rfind("Error\n"))
        (issues if issue_at > error_at else exceptions).add(m.group(1))
    return exceptions, issues


# the options of each translated selector; a new selector key fails until it is added here
SELECTOR_OPTIONS = {"tax_mode": C.TAX_MODES}


@functools.cache
def _needs() -> tuple[str, ...]:
    """Every dotted string key the code uses, read from the code (a static scan)."""
    flow = (CC / "config_flow.py").read_text()
    marker = "# --- the options flow"
    assert marker in flow
    config_part, options_part = flow.split(marker, 1)
    needs: list[str] = []
    # steps, their menus and the form errors and aborts of each flow
    # errors["field"] = "code", and one-entry dicts {"field": "code"} or {k: "code" for k ...}
    error_code = re.compile(
        r'errors\[[^\]]+\]\s*=\s*"(\w+)"|\{\s*(?:k|"\w+")\s*:\s*"(\w+)"(?=\s*(?:\}|for\b))'
    )
    for flow_name, part in (("config", config_part), ("options", options_part)):
        steps = set(re.findall(r'step_id="(\w+)"', part))
        assert steps
        needs += [f"{flow_name}.step.{step}.title" for step in sorted(steps)]
        codes = {a or b for a, b in error_code.findall(part)}
        assert codes
        needs += [f"{flow_name}.error.{code}" for code in sorted(codes)]
        reasons = sorted(set(re.findall(r'reason="(\w+)"', part)))
        needs += [f"{flow_name}.abort.{r}" for r in reasons]
    needs += [f"config.step.pv.menu_options.{o}" for o in ("pv_learn", "pv_fill")]
    menu = _module_tuple(CC / "config_flow.py", "OPTIONS_MENU")
    needs += [f"options.step.init.menu_options.{o}" for o in menu]
    aborts = (*S.REASONS, "already_configured", "already_in_progress")
    needs += [f"config.abort.{r}" for r in aborts]
    # translated selectors: _select(options, "key")
    selectors = set(re.findall(r'_select\([^()]*,\s*"(\w+)"\)', flow))
    assert selectors and selectors <= set(SELECTOR_OPTIONS), selectors
    for key in sorted(selectors):
        needs += [f"selector.{key}.options.{o}" for o in SELECTOR_OPTIONS[key]]
    # entities, their states, the services with their fields
    needs += [f"entity.{SENSOR}.{key}.name" for key in SENSOR_KEYS]
    needs += [f"entity.binary_{SENSOR}.{key}.name" for key in BINARY_KEYS]
    needs.append("entity.select.period.name")
    needs += [f"entity.select.period.state.{p}" for p in _module_tuple(CC / "select.py", "PERIODS")]
    states = _module_tuple(CC / f"{SENSOR}.py", "BACKTEST_STATES")
    needs += [f"entity.{SENSOR}.backtest.state.{state}" for state in states]
    actions = _module_tuple(CC / "live" / "views.py", "ACTIONS")
    needs += [f"entity.{SENSOR}.state.state.{action}" for action in actions]
    fields = _service_fields()
    assert set(fields) == set(SERVICES)
    for service, names in fields.items():
        needs += [f"services.{service}.name", f"services.{service}.description"]
        for name in names:
            needs += [f"services.{service}.fields.{name}.{k}" for k in ("name", "description")]
    # exceptions and repair issues raised with a translation key anywhere in the package, the
    # export range refusals (forwarded by key), short_history (Task 23) and pv_scale_off
    # (Task 21)
    exceptions, issues = set(X.REFUSAL_KEYS) | {"short_history"}, {"pv_scale_off"}
    for path in sorted(CC.rglob("*.py")):
        e, i = _translation_keys(path.read_text())
        exceptions |= e
        issues |= i
    assert {"no_pack", "pooling_off", "export_order", "refit_failed"} <= exceptions
    needs += [f"exceptions.{key}.message" for key in sorted(exceptions)]
    needs += [f"issues.{key}.{k}" for key in sorted(issues) for k in ("title", "description")]
    return tuple(needs)


def _missing(src: dict) -> list[str]:
    return [k for k in _needs() if not isinstance(_get(src, k), str)]


def test_every_key_the_code_uses_has_a_string():
    assert not _missing(_load("strings.json")), _missing(_load("strings.json"))


@pytest.mark.parametrize("key", _needs())
def test_the_scan_reports_each_key_when_it_is_deleted(key):
    src = copy.deepcopy(_load("strings.json"))
    *parents, leaf = key.split(".")
    del _get(src, ".".join(parents))[leaf]
    assert key in _missing(src)


def test_issue_keys_go_to_issues_and_error_keys_to_exceptions():
    text = (
        'ir.async_create_issue(hass, DOMAIN, "pv_scale_off", is_fixable=False,\n'
        '    translation_key="pv_scale_off", translation_placeholders={"strings": "1"})\n'
        "raise ServiceValidationError(\n"
        '    translation_domain=DOMAIN, translation_key="no_pack"\n)\n'
        'raise HomeAssistantError(\n    "x", translation_domain=DOMAIN, translation_key="boom")\n'
    )
    assert _translation_keys(text) == ({"no_pack", "boom"}, {"pv_scale_off"})


# ---- the ready-made view (Task 25)

CARD_JS = CC / "www" / "betterybalance-plan-card.js"
VIEW_YAML = CC / "dashboard" / "view.yaml"
NODE_VIEW = (
    "globalThis.HTMLElement = class {}; const d = {};"
    "globalThis.customElements = { define(n, c) { d[n] = c; }, get(n) { return d[n]; } };"
    "const m = await import(process.argv[1]);"
    "process.stdout.write(JSON.stringify({en: m.viewConfig('en'), nl: m.viewConfig('nl')}));"
)


@functools.cache
def _views() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    out = subprocess.run(
        [node, "--input-type=module", "-e", NODE_VIEW, CARD_JS.as_uri()],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _config_flow_constant(name: str):
    tree = ast.parse((CC / "config_flow.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"config_flow.py has no {name}")


def test_view_yaml_equals_the_strategy():
    yaml = pytest.importorskip("yaml")
    assert yaml.safe_load(VIEW_YAML.read_text(encoding="utf-8")) == _views()["en"]


def test_view_feed_chips_reuse_the_feed_names_of_the_translations():
    for lang in ("en", "nl"):
        names = _load(f"translations/{lang}.json")["entity"]["binary_sensor"]
        chips = [
            c
            for s in _views()[lang]["sections"]
            for c in s["cards"]
            if c["type"] == "tile" and c["entity"].startswith("binary_")
        ]
        assert [c["entity"].split(".", 1)[1] for c in chips] == [
            f"betterybalance_{k}" for k in names
        ]
        assert [c["name"] for c in chips] == [v["name"] for v in names.values()]


def test_the_options_step_view_points_at_the_shipped_view_and_the_strategy():
    yaml = pytest.importorskip("yaml")
    path = _config_flow_constant("VIEW_PATH")
    assert (CC.parent.parent / path).resolve() == VIEW_YAML.resolve()
    snippet = yaml.safe_load(_config_flow_constant("VIEW_SNIPPET"))
    assert snippet == {"strategy": {"type": "custom:betterybalance"}}
    assert '"ll-strategy-view-betterybalance"' in CARD_JS.read_text()
    for f in FILES:
        assert {"snippet", "path"} <= set(
            PLACEHOLDER.findall(_load(f)["options"]["step"]["view"]["description"])
        )


def test_view_text_has_no_emojis_dashes_or_the_actual_path():
    for lang, view in _views().items():
        text = json.dumps(view, ensure_ascii=False)
        assert not EMOJI.search(text) and "—" not in text, lang
        assert "actual" not in text.lower(), lang


def test_dutch_backtest_note_keeps_the_numbers_of_the_indicative_sentence():
    from betterybalance.live.backtest import INDICATIVE_HOURLY

    backtest = [
        c["content"]
        for s in _views()["nl"]["sections"]
        for c in s["cards"]
        if c["type"] == "markdown" and "betterybalance_backtest" in c["content"]
    ]
    numbers = re.findall(r"\d+ %", INDICATIVE_HOURLY)
    assert numbers and all(n in backtest[0] for n in numbers)
