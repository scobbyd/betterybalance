import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent / "custom_components" / "betterybalance"
HA = re.compile(r"^\s*(from|import)\s+homeassistant\b", re.M)


def test_pure_layers_never_import_home_assistant():
    hits = [
        str(p)
        for d in ("core", "live", "replay", "verify")
        for p in (ROOT / d).rglob("*.py")
        if HA.search(p.read_text())
    ]
    assert hits == []


def test_package_init_imports_home_assistant_lazily():
    assert not re.search(
        r"^(from|import)\s+homeassistant", (ROOT / "__init__.py").read_text(), re.M
    )
