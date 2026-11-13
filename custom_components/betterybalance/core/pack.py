"""The data pack: the quarter table from 2024 and the weekly model series, sha-checked.

A pack is a folder with two files:

- ``betterybalance-pack-<month>.npz``: one array per ``QUARTER_COLUMNS`` entry (``ts`` int64,
  the rest float64), ``train_end`` (int64, epoch s, ascending) and ``model_000`` ..
  ``model_NNN`` (uint8, the UTF-8 bytes of ``pricemodel.to_json``). Model ``i`` trained on
  rows that end at ``train_end[i]`` (exclusive), so the replay reads it only under the
  embargo of ``replay.bands`` (``cache_from_series``).
- ``manifest.json``: format, version (the month), coverage, the sha256 of the npz, the build
  date, the integration versions it serves, the model count, the training ends, the licence
  notes per source and whether the IDA columns carry data.

Determinism: the npz is a zip written with fixed member timestamps (``np.load`` reads it as a
normal npz), so the pack bytes depend only on the inputs. The ``built`` field and each model's
``meta["fitted_at"]`` are the only wall-clock values.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .. import __version__
from .features import build_master
from .feeds.ida import AUCTIONS
from .pricemodel import PriceModel, from_json, to_json
from .timegrid import ISP_S, ZONE, Table

PACK_FORMAT = 1
QUARTER_COLUMNS = (
    "ts",
    "da_eur_mwh",
    "ida1_eur_mwh",
    "ida2_eur_mwh",
    "ida3_eur_mwh",
    "buy_eur_mwh",
    "sell_eur_mwh",
    "regulation_state",
    "buy_live_end",
    "sell_live_end",
    "buy_live_e1",
    "sell_live_e1",
    "buy_live_e5",
    "sell_live_e5",
    "buy_live_e10",
    "sell_live_e10",
)
LICENCES = {
    "da_eur_mwh": "energy-charts.info, CC BY 4.0",
    "ida*": "Nord Pool, withheld until the release gate",
    "buy_eur_mwh/sell_eur_mwh/regulation_state": "TenneT settlement publications",
    "*_live_*": "per-quarter aggregates of TenneT minute data",
}
MANIFEST = "manifest.json"
IDA_COLUMNS = ("ida1_eur_mwh", "ida2_eur_mwh", "ida3_eur_mwh")
_SETTLED = ("buy_eur_mwh", "sell_eur_mwh", "regulation_state")
_LIVE = QUARTER_COLUMNS[8:]
_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_CLAUSE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+)*)\s*$")


class PackError(Exception):
    """A pack that is malformed, damaged or not meant for this integration version."""


def npz_name(month: str) -> str:
    return f"betterybalance-pack-{month}.npz"


def _local_date(ts: int) -> str:
    return dt.datetime.fromtimestamp(int(ts), ZONE).date().isoformat()


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """``np.savez_compressed`` layout with fixed member timestamps (byte-identical reruns)."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            info = zipfile.ZipInfo(name + ".npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pack(
    out_dir: Path,
    month: str,
    quarters: Table,
    series: list[tuple[int, PriceModel]],
    built: str,
    serves: str = ">=0.3.0",
) -> dict:
    """Write the npz and manifest.json of one pack into out_dir and return the manifest.

    ``quarters`` holds every ``QUARTER_COLUMNS`` entry on one contiguous grid. ``series`` is
    ``(train_end, model)`` with strictly ascending training ends.
    """
    if not _MONTH.match(month):
        raise PackError(f"month must be YYYY-MM, got {month!r}")
    missing = [c for c in QUARTER_COLUMNS if c not in quarters]
    if missing:
        raise PackError("quarter table lacks " + ", ".join(missing))
    ts = np.asarray(quarters["ts"], dtype=np.int64)
    if len(ts) == 0:
        raise PackError("quarter table is empty")
    if any(len(quarters[c]) != len(ts) for c in QUARTER_COLUMNS):
        raise PackError("quarter columns differ in length")
    if len(ts) > 1 and (np.diff(ts) != ISP_S).any():
        raise PackError(f"quarter table is not a contiguous {ISP_S} s grid")
    ends = np.asarray([int(t) for t, _ in series], dtype=np.int64)
    if len(ends) == 0:
        raise PackError("the model series is empty")
    if len(ends) > 1 and (np.diff(ends) <= 0).any():
        raise PackError("training ends must be strictly ascending")
    _check_serves_syntax(serves)
    arrays: dict[str, np.ndarray] = {"ts": ts}
    for c in QUARTER_COLUMNS[1:]:
        arrays[c] = np.asarray(quarters[c], dtype=np.float64)
    arrays["train_end"] = ends
    for i, (_, model) in enumerate(series):
        arrays["model_%03d" % i] = np.frombuffer(to_json(model).encode("utf-8"), dtype=np.uint8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / npz_name(month)
    _write_npz(npz, arrays)
    manifest = {
        "format": PACK_FORMAT,
        "version": month,
        "coverage": {"from": _local_date(ts[0]), "to": _local_date(ts[-1])},
        "sha256": _sha256(npz),
        "built": built,
        "serves": serves,
        "n_models": len(ends),
        "train_ends": [int(t) for t in ends],
        "licences": dict(LICENCES),
        "ida_included": bool(any(np.isfinite(arrays[c]).any() for c in IDA_COLUMNS)),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / MANIFEST).write_text(text, encoding="utf-8")
    return manifest


def read_manifest(path: Path) -> dict:
    """The manifest of a pack folder, or of the manifest file itself."""
    path = Path(path)
    if path.is_dir():
        path = path / MANIFEST
    try:
        man = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PackError(f"no manifest at {path}") from None
    except json.JSONDecodeError as err:
        raise PackError(f"manifest is not JSON: {err}") from None
    if not isinstance(man, dict):
        raise PackError("manifest is not a JSON object")
    return man


def _version_tuple(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in str(text).split("."))
    except ValueError:
        raise PackError(f"version {text!r} is not a dotted list of integers") from None


def _pad(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)), b + (0,) * (n - len(b))


def _check_serves_syntax(serves: str) -> list[tuple[str, tuple[int, ...]]]:
    out = []
    for part in str(serves).split(","):
        m = _CLAUSE.match(part)
        if not m:
            raise PackError(f"pack serves clause not understood: {part.strip()!r}")
        out.append((m.group(1), _version_tuple(m.group(2))))
    return out


def serves_version(serves: str, version: str = __version__) -> bool:
    """Whether a comma list of clauses such as ``>=0.3.0,<0.5`` admits ``version``."""
    have = _version_tuple(version)
    for op, want in _check_serves_syntax(serves):
        a, b = _pad(have, want)
        ok = {
            ">=": a >= b,
            "<=": a <= b,
            "==": a == b,
            "!=": a != b,
            ">": a > b,
            "<": a < b,
        }[op]
        if not ok:
            return False
    return True


class DataPack:
    """A loaded pack: the quarter table, the training ends and the lazily parsed models."""

    def __init__(self, manifest: dict, quarters: Table, train_ends: np.ndarray, blobs: list):
        self.manifest = manifest
        self.version: str = str(manifest["version"])
        self.quarters = quarters
        self.train_ends = np.asarray(train_ends, dtype=np.int64)
        self._blobs = blobs
        self._models: dict[int, PriceModel] = {}
        self._master: Table | None = None

    def __len__(self) -> int:
        return len(self._blobs)

    def master(self) -> Table:
        """``build_master`` over the quarter table (built once)."""
        if self._master is None:
            q = self.quarters
            grid = np.asarray(q["ts"], dtype=np.int64)
            da = {"ts": grid, "da_eur_mwh": q["da_eur_mwh"]}
            settled = {"ts": grid, **{c: q[c] for c in _SETTLED}}
            ida = {a: {"ts": grid, "ida_eur_mwh": q[a.lower() + "_eur_mwh"]} for a in AUCTIONS}
            live = {"ts": grid, **{c: q[c] for c in _LIVE}}
            self._master = build_master(da, settled, ida, live, grid)
        return self._master

    def model(self, i: int) -> PriceModel:
        i = range(len(self._blobs))[i]
        if i not in self._models:
            self._models[i] = from_json(bytes(self._blobs[i]).decode("utf-8"))
        return self._models[i]

    def model_version(self, i: int) -> str:
        """``pack-<version>-<last trained local day>``: the day of the last training quarter.

        ``train_ends[i]`` is exclusive, so the name reads the local date of
        ``train_ends[i] - 1``: a weekly end at local midnight names the day before it.
        """
        i = range(len(self._blobs))[i]
        return f"pack-{self.version}-{_local_date(int(self.train_ends[i]) - 1)}"

    def newest(self) -> tuple[PriceModel, str, int]:
        """The model with the latest training end, its model_version and that training end."""
        return self.model(-1), self.model_version(-1), int(self.train_ends[-1])


def _int_list(value, what: str) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in value
    ):
        raise PackError(f"manifest {what} is not a list of integers")
    return value


def load_pack(dir: Path) -> DataPack:
    """Load and check a pack folder: format, sha256 of the npz, served versions, contents.

    Every malformed or foreign pack raises PackError, never another exception type.
    """
    dir = Path(dir)
    man = read_manifest(dir)
    if man.get("format") != PACK_FORMAT:
        raise PackError(f"pack format {man.get('format')} is not {PACK_FORMAT}")
    for key in ("version", "sha256", "serves", "n_models", "train_ends"):
        if key not in man:
            raise PackError(f"manifest lacks {key}")
    version = man["version"]
    if not isinstance(version, str) or not _MONTH.match(version):
        raise PackError(f"pack version {version!r} is not YYYY-MM")
    n_models = man["n_models"]
    if not isinstance(n_models, int) or isinstance(n_models, bool) or n_models < 1:
        raise PackError(f"manifest n_models {n_models!r} is not a positive integer")
    train_ends = _int_list(man["train_ends"], "train_ends")
    npz = dir / npz_name(version)
    if not npz.is_file():
        raise PackError(f"no {npz.name} beside the manifest")
    if _sha256(npz) != man["sha256"]:
        raise PackError("sha256 mismatch: the pack file does not match the manifest")
    if not serves_version(str(man["serves"])):
        raise PackError(f"pack serves {man['serves']}, this integration is {__version__}")
    names = ["model_%03d" % i for i in range(n_models)]
    try:
        with np.load(npz, allow_pickle=False) as z:
            missing = [c for c in (*QUARTER_COLUMNS, "train_end", *names) if c not in z.files]
            if missing:
                raise PackError("pack lacks " + ", ".join(missing[:5]))
            quarters = {c: z[c] for c in QUARTER_COLUMNS}
            ends = np.asarray(z["train_end"], dtype=np.int64)
            blobs = [z[n] for n in names]
    except (OSError, ValueError, zipfile.BadZipFile) as err:
        raise PackError(f"pack file does not read: {err}") from None
    if len(ends) != n_models:
        raise PackError("model count does not match the manifest")
    if ends.tolist() != train_ends:
        raise PackError("training ends do not match the manifest")
    ts = np.asarray(quarters["ts"], dtype=np.int64)
    if len(ts) == 0 or (len(ts) > 1 and (np.diff(ts) != ISP_S).any()):
        raise PackError(f"pack quarters are not a contiguous {ISP_S} s grid")
    if any(len(quarters[c]) != len(ts) for c in QUARTER_COLUMNS):
        raise PackError("pack quarter columns differ in length")
    return DataPack(man, quarters, ends, blobs)
