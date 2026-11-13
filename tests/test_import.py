import json
import pathlib
import sys


def test_package_imports_without_home_assistant():
    import betterybalance
    from betterybalance import core, live, replay, verify  # noqa: F401
    assert betterybalance.__version__ == "0.3.0" and "homeassistant" not in sys.modules
    manifest = pathlib.Path(betterybalance.__file__).parent / "manifest.json"
    assert json.loads(manifest.read_text())["version"] == betterybalance.__version__
