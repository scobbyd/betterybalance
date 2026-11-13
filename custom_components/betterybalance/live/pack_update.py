"""The data pack updater: a local release, its atomic install and the choice of the live model.

A release is a folder (or a ``file://`` URL of one) that holds a pack's ``manifest.json`` and
its npz (``core.pack``). ``check_release`` reads the manifest and answers a release only when
its version is newer than the installed one. ``install`` copies the two pack files into
``<dest>/.tmp-<version>``, checks the copy with ``load_pack`` (format, sha256, the served
integration versions, the contents) and renames it to ``<dest>/<version>``. A failure at any
step removes the temporary folder and re-raises, so the installed pack stays as it was. The
version installed before stays beside the new one, and so does every month in ``keep`` (the
pack of the model in use); older version folders go. An install that
a crash stopped is recovered first (``recover``): a ``.tmp-*`` folder goes, and an
``.old-<version>`` folder goes only when ``<version>`` exists and loads, else it becomes
``<version>`` again.

``choose_model`` picks the live price model: the pack's newest model only when its training
end is later than the shipped champion's and later than the model in use, else the newer of
those two. It never picks an older model. Pack series models carry a coverage factor of 1,0
in every cell (the pack builder fits no coverage); such a model takes the shipped champion's
factors cell by cell (``with_coverage``), and ``meta["cov_source"]`` names where the factors
come from: ``own``, ``shipped`` (every cell took a shipped factor) or ``mixed`` (some cells
have no shipped counterpart and keep 1,0).

``model_state`` reads the ``model_switch`` events (``live.site.LiveSite.set_model``) as
re-execution reads them, so a new run of the live loop starts on the model in effect.
A model version names its pack's month, so ``resolve`` looks in the active pack and then in
the kept previous version folder (``previous_folder``); ``models_from_dir`` maps every model
of the installed version folders for re-execution.

Every function here is blocking (file reads and copies) and runs in the executor.
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import unquote, urlparse

from ..core.pack import _MONTH, MANIFEST, DataPack, PackError, load_pack, npz_name, read_manifest
from ..core.pricemodel import PriceModel
from ..core.timegrid import ISP_S, ZONE, isp_start

TMP_PREFIX = ".tmp-"
OLD_PREFIX = ".old-"
COV_OWN = "own"
COV_SHIPPED = "shipped"
COV_MIXED = "mixed"

# (model, model_version, training end): the end is exclusive, epoch s
Entry = tuple[PriceModel, str, int]


@dataclass
class Release:
    version: str
    manifest: dict
    source: str


def source_dir(source: str) -> Path:
    """The folder a pack source names: a path, or a ``file://`` URL of one; a source that
    names the manifest file itself means its folder."""
    text = str(source).strip()
    path = Path(text)
    if text.startswith("file:"):
        url = urlparse(text)
        if url.netloc not in ("", "localhost"):
            raise PackError(f"pack source {text!r} is not a local file URL")
        path = Path(unquote(url.path))
    return path.parent if path.name == MANIFEST and path.is_file() else path


def check_release(source: str, installed: str | None) -> Release | None:
    """The release at ``source`` when its version is newer than ``installed``, else None.

    Raises ``PackError`` when the source or its manifest does not read, or when the manifest
    names no ``YYYY-MM`` version.
    """
    if not str(source or "").strip():
        raise PackError("no pack source is set")
    folder = source_dir(source)
    try:
        man = read_manifest(folder)
    except OSError as err:
        raise PackError(f"pack source {source!r} does not read: {err}") from None
    version = man.get("version")
    if not isinstance(version, str) or not _MONTH.match(version):
        raise PackError(f"release version {version!r} is not YYYY-MM")
    if installed is not None and version <= installed:
        return None
    return Release(version=version, manifest=man, source=str(source))


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def install(rel: Release, dest: Path, keep: frozenset[str] | set[str] = frozenset()) -> Path:
    """Install ``rel`` as ``dest/<version>`` (module docstring); the installed folder.

    ``keep`` names pack months whose folders the prune never removes (the model in use).

    Copies only the manifest and the npz it names. Raises what the copy or ``load_pack``
    raises (``PackError`` for a pack that does not check), after removing the temporary
    folder; the installed pack is untouched.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    recover(dest)
    folder = source_dir(rel.source)
    tmp = dest / (TMP_PREFIX + rel.version)
    final = dest / rel.version
    old = dest / (OLD_PREFIX + rel.version)
    try:
        tmp.mkdir()
        for name in (MANIFEST, npz_name(rel.version)):
            src = folder / name
            if not src.is_file():
                raise PackError(f"release {rel.version} lacks {name}")
            shutil.copyfile(src, tmp / name)
        if load_pack(tmp).version != rel.version:
            raise PackError(f"release {rel.version} changed while it was copied")
        if final.exists():  # the same version again (a record that did not reach the store)
            final.rename(old)
        try:
            tmp.rename(final)
        except BaseException:
            if old.exists() and not final.exists():
                old.rename(final)
            raise
    except BaseException:
        _remove(tmp)
        raise
    _remove(old)
    _prune(dest, rel.version, keep)
    return final


def _loads(path: Path) -> bool:
    try:
        load_pack(path)
    except PackError:
        return False
    return True


def recover(dest: Path) -> None:
    """Undo what a crashed install left in ``dest``: every ``.tmp-*`` folder goes; an
    ``.old-<version>`` folder goes only when ``<version>`` exists and loads, else it replaces
    a ``<version>`` that does not load (or is missing) and becomes ``<version>`` again."""
    dest = Path(dest)
    if not dest.is_dir():
        return
    for tmp in dest.glob(TMP_PREFIX + "*"):
        _remove(tmp)
    for old in sorted(dest.glob(OLD_PREFIX + "*")):
        final = dest / old.name[len(OLD_PREFIX):]
        if final.is_dir() and _loads(final):
            _remove(old)
            continue
        _remove(final)
        old.rename(final)


_PACK_VERSION = re.compile(r"^pack-(\d{4}-\d{2})-\d{4}-\d{2}-\d{2}$")


def version_month(model_version: str | None) -> str | None:
    """The pack month a ``pack-<month>-<day>`` model version names; None for another one."""
    m = _PACK_VERSION.match(str(model_version or ""))
    return m.group(1) if m else None


def month_folder(dest: Path, model_version: str | None) -> Path | None:
    """``dest/<month>`` of a pack model version when that folder exists, else None."""
    month = version_month(model_version)
    if month is None or not (Path(dest) / month).is_dir():
        return None
    return Path(dest) / month


def previous_folder(dest: Path, version: str) -> Path | None:
    """The newest version folder in ``dest`` before ``version`` (the kept previous pack)."""
    dest = Path(dest)
    if not dest.is_dir():
        return None
    before = sorted(
        p.name for p in dest.iterdir() if p.is_dir() and _MONTH.match(p.name) and p.name < version
    )
    return dest / before[-1] if before else None


def _prune(dest: Path, version: str, keep_months=frozenset()) -> None:
    """Keep ``version``, the newest version folder before it and the months of
    ``keep_months``; remove older ones."""
    months = sorted(p.name for p in dest.iterdir() if p.is_dir() and _MONTH.match(p.name))
    before = [m for m in months if m < version]
    keep = {version, *before[-1:], *keep_months}
    for m in months:
        if m not in keep and m < version:
            _remove(dest / m)


# -- the live model -----------------------------------------------------------------------


def cov_source(model: PriceModel) -> str:
    """Where a chosen model's coverage factors come from (``own`` unless marked)."""
    return str(model.meta.get("cov_source", COV_OWN))


def with_coverage(model: PriceModel, shipped: PriceModel) -> tuple[PriceModel, str]:
    """``model`` with the shipped champion's coverage factors when all of its own are 1,0.

    Returns the model and the source of its factors. A model with factors of its own is
    returned as it is (``own``); else a copy (the cells and meta copied, the input untouched)
    takes the shipped factor of each cell the shipped champion has: ``shipped`` when every
    cell took one, ``mixed`` when some cells have no shipped counterpart and keep 1,0.
    """
    if any(float(hm.cov_factor) != 1.0 for hm in model.models.values()):
        return model, COV_OWN
    cells = {
        k: replace(hm, cov_factor=float(shipped.models[k].cov_factor)) if k in shipped.models
        else hm
        for k, hm in model.models.items()
    }
    src = COV_SHIPPED if all(k in shipped.models for k in model.models) else COV_MIXED
    meta = {**model.meta, "cov_source": src}
    return PriceModel(models=cells, meta=meta, tables=model.tables), src


def local_midnight_after(day: str) -> int:
    """Epoch s of the Amsterdam midnight after the ISO date ``day``."""
    d = dt.date.fromisoformat(str(day)) + dt.timedelta(days=1)
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=ZONE).timestamp())


def shipped_entry(model: PriceModel) -> Entry:
    """The shipped champion as ``(model, "shipped-<train_end>", exclusive training end)``.

    ``meta["train_end"]`` is the last trained day, so the end is the midnight after it (0
    for a model without one).
    """
    day = model.meta.get("train_end")
    end = 0 if not day else local_midnight_after(day)
    return model, f"shipped-{day}", end


def choose(shipped: Entry, pack: DataPack | None, current: Entry | None = None) -> Entry:
    """``choose_model`` with the training end of the chosen model."""
    best = shipped
    if current is not None and current[2] >= shipped[2]:
        best = current
    if pack is not None and len(pack):
        model, version, end = pack.newest()
        if end > best[2]:
            return with_coverage(model, shipped[0])[0], version, end
    return best


def choose_model(
    shipped: Entry, pack: DataPack | None, current: Entry | None = None
) -> tuple[PriceModel, str]:
    """The live model and its version: the pack's newest model when its training end is later
    than the shipped one's (and than ``current``'s, the model in use), else the newer of the
    shipped model and ``current`` (an equal end keeps ``current``). Never an older model.
    A chosen pack model carries the shipped coverage factors when it has none of its own."""
    model, version, _ = choose(shipped, pack, current)
    return model, version


def resolve(
    version: str, shipped: Entry, pack: DataPack | None, previous: DataPack | None = None
) -> tuple[PriceModel, int] | None:
    """The model of ``version`` and its training end: the shipped one, else a model of the
    active ``pack``, else of the ``previous`` pack (with the shipped coverage when it has
    none); None when none of them holds it."""
    if version == shipped[1]:
        return shipped[0], shipped[2]
    for p in (pack, previous):
        if p is None:
            continue
        for i in range(len(p)):
            if p.model_version(i) == version:
                return with_coverage(p.model(i), shipped[0])[0], int(p.train_ends[i])
    return None


def models_from_dir(dest: Path, shipped: PriceModel) -> dict[str, PriceModel]:
    """Every model of the version folders in ``dest`` by its model version (with the shipped
    coverage when it has none), plus the shipped champion by its version: the ``models`` map
    of ``replay.reexec.reexecute``. A folder that does not load is skipped."""
    entry = shipped_entry(shipped)
    out = {entry[1]: shipped}
    dest = Path(dest)
    if not dest.is_dir():
        return out
    for name in sorted(p.name for p in dest.iterdir() if p.is_dir() and _MONTH.match(p.name)):
        try:
            pack = load_pack(dest / name)
        except PackError:
            continue
        for i in range(len(pack)):
            out[pack.model_version(i)] = with_coverage(pack.model(i), shipped)[0]
    return out


def _effective(ev: dict) -> int:
    at = ev["detail"].get("effective_ts")
    return int(isp_start(int(ev["ts"]))) + ISP_S if at is None else int(at)


def model_state(events: list[dict], now: int, initial: str) -> tuple[str, tuple[int, str] | None]:
    """The model version in effect at ``now`` and the pending switch ``(effective_ts,
    version)`` after it (None without one), from the ``model_switch`` events in seq order.

    As re-execution reads them: a switch is in effect from its ``effective_ts`` (else the
    quarter boundary after the event); among several, the last in seq order wins.
    """
    version, pending = initial, None
    for ev in events:
        if ev["kind"] != "model_switch" or int(ev["ts"]) > int(now):
            continue
        at = _effective(ev)
        if at <= int(now):
            version, pending = str(ev["detail"]["version"]), None
        else:
            pending = (at, str(ev["detail"]["version"]))
    return version, pending
