import json
import re
import subprocess
from pathlib import Path

from betterybalance.core import features as F
from betterybalance.core import pricemodel as PM

MODEL_FILE = Path(PM.__file__).resolve().parent.parent / "models" / "market_nl.json"


def test_shipped_model_loads_and_covers_grid():
    m = PM.shipped_model()
    for tier in F.TIERS:
        for h in F.HORIZONS:
            if h == 0:
                assert all((tier, 0, e) in m.models for e in F.ELAPSED_MIN)
            else:
                assert (tier, h, None) in m.models
    assert set(next(iter(m.models.values())).w_buy) == set(PM.QUANTILES)
    assert m.meta["format"] == 2 and m.tables
    raw = json.loads(MODEL_FILE.read_text())
    assert all("cov_factor" in cell for cell in raw["models"].values())
    assert {"dual_rate", "dual_spread"} <= set(raw["tables"])
    assert m.meta["train_end"] >= "2026-10-30" and "holdout_mae" in m.meta
    assert m.meta["source_report"] == "2026-10-29-october-price-study.md"


FINGERPRINTS = {
    "ipv4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # the integration's own ids and synthetic fixture ids are not owner fingerprints
    "entity id": re.compile(
        r"\b(sensor|binary_sensor|input_number|switch)\.(?!betterybalance_|synthetic_)[a-z0-9_]+"
    ),
    "hostname": re.compile(r"\.(casa|local|lan|home)\b"),
    "coordinates": re.compile(r"-?\d{1,3}\.\d{4,}\s*,\s*-?\d{1,3}\.\d{4,}"),
}
ROOT = Path(__file__).resolve().parent.parent
# fixtures may name a location only at the generic Dutch test point 52,1 / 5,2, written
# exactly and under the matching key; any other value under a location key is a hit (the
# export test's 52.149 / 5.231 lives in tests/, outside this scan)
GENERIC_POINT = {"lat": 52.1, "latitude": 52.1, "lon": 5.2, "longitude": 5.2}
LOCATION_KEY = re.compile(
    r"\b(lat|latitude|lon|longitude)\b[\"']?\s*[:=]\s*[\"']?(-?\d+(?:\.\d+)?)", re.IGNORECASE
)


def _foreign_locations(text: str) -> list[str]:
    """Location values that are not the generic point, exactly and under the matching key."""
    hits = []
    for m in LOCATION_KEY.finditer(text):
        if float(m.group(2)) != GENERIC_POINT[m.group(1).lower()]:
            hits.append(m.group(0))
    return hits


def _scan(text: str, skip: tuple[str, ...] = ()) -> list[str]:
    return [
        name for name, pattern in FINGERPRINTS.items() if name not in skip and pattern.search(text)
    ]


def test_scan_patterns_detect_what_they_name():
    assert _scan("host 192.168.1.20") == ["ipv4"]
    assert _scan("sensor.some_price") == ["entity id"]
    assert _scan("sensor.betterybalance_state") == []
    assert _scan("binary_sensor.betterybalance_feed_live") == []
    assert _scan("sensor.synthetic_pv_power") == []
    assert _scan("box.example.lan") == ["hostname"]
    assert _scan("52.123456, 4.123456") == ["coordinates"]
    assert _scan("EUR/MWh 57,91 and version 0.1.0") == []
    assert _foreign_locations('"latitude": 52.1, "longitude": 5.2') == []
    assert _foreign_locations('"lat": 52.149, "lon": 5.231') == ['lat": 52.149', 'lon": 5.231']
    assert _foreign_locations('"latitude": 51.9') == ['latitude": 51.9']
    assert _foreign_locations("lon=5.2311") == ["lon=5.2311"]
    swapped = _foreign_locations('"latitude": 5.2, "longitude": 52.1')
    assert swapped == ['latitude": 5.2', 'longitude": 52.1']


def test_shipped_file_is_small_and_has_no_owner_fingerprints():
    text = MODEL_FILE.read_text()
    assert len(text.encode()) < 2_000_000
    # weight lists are runs of decimals, so the coordinate pattern would match them
    assert _scan(text, skip=("coordinates",)) == []
    assert "NaN" not in text
    assert set(json.loads(text)["meta"]["train_start_by_tier"]) == set(F.TIERS)


def test_tracked_text_files_have_no_owner_fingerprints():
    files = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    skipped = ("docs/research/", "market_nl.json", "test_shipped_model.py")
    hits, fixtures = [], 0
    for name in files:
        if name.startswith(skipped[0]) or name.endswith(skipped[1:]):
            continue
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{name}: {kind}" for kind in _scan(text)]
        if name.startswith("tests/fixtures/"):
            fixtures += 1
            hits += [f"{name}: location {loc}" for loc in _foreign_locations(text)]
    assert hits == [] and fixtures > 0
