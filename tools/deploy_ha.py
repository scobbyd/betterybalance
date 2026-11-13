"""Copy the integration to a Home Assistant box over ssh and run ``ha core check``.

The owner supplies the target at run time: ``BB_HA_SSH`` (the ssh destination, required) and
``BB_HA_CONFIG`` (the config folder on the box, default ``/config``), from the environment or
the git-ignored ``data/owner/site.env`` (``KEY=value`` lines; the environment wins). Nothing
of the owner is written to a file.

The copy is staged: the box first refuses a config folder without ``configuration.yaml``,
extracts into ``<config>/.betterybalance-staging``, checks the file count and only then swaps
the folder in, so a dropped connection leaves the installed integration as it was and a module
removed from the integration leaves the box too. During the swap the old integration waits in
``<config>/.betterybalance-previous`` (outside the staging folder): a failed second move puts it
back, and a run that finds it there without an installed integration (a swap cut off before)
puts it back before anything else, so a retry never deletes the only old copy.

The script never restarts Home Assistant: a new version loads only after a restart, and every
restart is the owner's decision.

    python -m tools.deploy_ha --dry-run    # print the commands, run nothing
    python -m tools.deploy_ha              # copy, then check
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / "data" / "owner" / "site.env"
REQUIRED = ("BB_HA_SSH",)
DEFAULT_CONFIG = "/config"
EXCLUDES = ("__pycache__", "*.pyc")
STAGING = ".betterybalance-staging"  # under the config folder, outside custom_components
PREVIOUS = ".betterybalance-previous"  # the old copy during the swap, outside the staging folder
PIPE = ["bash", "-o", "pipefail", "-c"]  # a local tar failure fails the copy
CLOSING = "Restart needed to load the integration: ask the owner before ha core restart."


def _read_env_file(path: Path) -> dict:
    """``KEY=value`` lines of ``path`` (``export`` prefix, comments and quotes allowed)."""
    out: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key.strip()] = value
    return out


def load_env(path: Path | None = None) -> dict:
    """The ``BB_*`` settings: the environment first, then the env file (``path``, default
    ``data/owner/site.env``; a missing file adds nothing)."""
    path = ENV_FILE if path is None else Path(path)
    env = _read_env_file(path) if path.is_file() else {}
    env.update({k: v for k, v in os.environ.items() if k.startswith("BB_")})
    return env


def _local_file_count(root: Path = ROOT) -> int:
    """The regular files the copy sends: the integration folder without ``EXCLUDES``."""
    base = root / "custom_components" / "betterybalance"
    n = 0
    for folder, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in EXCLUDES)]
        for f in files:
            path = Path(folder) / f
            if path.is_file() and not path.is_symlink():
                n += 0 if any(fnmatch.fnmatch(f, p) for p in EXCLUDES) else 1
    return n


def config_path(env: dict) -> str:
    """``BB_HA_CONFIG`` (default ``/config``) without a trailing slash; refuses an empty,
    root or relative path and a path with a ``.`` or ``..`` segment with ``SystemExit``."""
    raw = env.get("BB_HA_CONFIG")
    cfg = (DEFAULT_CONFIG if raw is None else str(raw)).rstrip("/")
    segments = cfg.split("/")[1:]
    if (
        not cfg or not cfg.startswith("/") or "\n" in cfg
        or any(x in (".", "..") for x in segments)
    ):
        raise SystemExit(
            "BB_HA_CONFIG must be an absolute folder other than / without . or .. segments,"
            f" not {raw!r}"
        )
    return cfg


def remote_script(cfg: str, n_files: int) -> str:
    """The box side of the copy, under ``set -e``.

    1. Refuse a config folder without ``configuration.yaml`` (nothing changed).
    2. Put back an old copy a cut-off swap left in ``<cfg>/.betterybalance-previous`` when no
       integration is installed.
    3. Extract into the staging folder ``<cfg>/.betterybalance-staging`` (removed first) and
       check the extracted file count.
    4. Move the old ``custom_components/betterybalance`` to ``<cfg>/.betterybalance-previous``
       and the new tree in; when that second move fails, move the old copy back and fail.
    5. Remove the old copy and the staging folder.

    It touches those three paths only.
    """
    q = shlex.quote
    stage = f"{cfg}/{STAGING}"
    prev = f"{cfg}/{PREVIOUS}"
    new = f"{stage}/betterybalance"
    dest = f"{cfg}/custom_components/betterybalance"
    return "\n".join(
        [
            "set -e",
            f"if [ ! -f {q(cfg + '/configuration.yaml')} ]; then",
            f"  echo {q(f'{cfg} has no configuration.yaml: nothing changed')} >&2",
            "  exit 1",
            "fi",
            f"if [ -e {q(prev)} ] && [ ! -e {q(dest)} ]; then",
            f"  mv {q(prev)} {q(dest)}",
            "  echo 'the previous integration from a cut-off deploy is back in place' >&2",
            "fi",
            f"rm -rf {q(stage)}",
            f"mkdir -p {q(stage)} {q(cfg + '/custom_components')}",
            f"tar xzf - -C {q(stage)}",
            f'n=$(find {q(new)} -type f | wc -l | tr -d " ")',
            f'if [ "$n" -ne {int(n_files)} ]; then',
            f'  echo "staged copy has $n files, expected {int(n_files)}: nothing replaced" >&2',
            "  exit 1",
            "fi",
            f"if [ -e {q(dest)} ]; then rm -rf {q(prev)}; mv {q(dest)} {q(prev)}; fi",
            f"if ! mv {q(new)} {q(dest)}; then",
            f"  if [ -e {q(prev)} ]; then",
            f"    mv {q(prev)} {q(dest)}",
            "    echo 'the new copy did not move in: the previous integration is back in"
            " place' >&2",
            "  else",
            "    echo 'the new copy did not move in: no integration was installed before' >&2",
            "  fi",
            "  exit 1",
            "fi",
            f"rm -rf {q(prev)}",
            f"rm -rf {q(stage)}",
        ]
    )


def plan_commands(env: dict) -> list[list[str]]:
    """The staged copy (a bash pipeline with ``pipefail``: tar to ssh, ``remote_script`` on
    the box) and the ``ha core check`` over ssh.

    Raises ``SystemExit`` naming the missing setting when ``BB_HA_SSH`` is not set, and for
    a config path ``config_path`` refuses.
    """
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        raise SystemExit(
            f"set {', '.join(missing)} in the environment or in data/owner/site.env"
        )
    target = env["BB_HA_SSH"]
    cfg = config_path(env)
    excludes = " ".join(f"--exclude {shlex.quote(p)}" for p in EXCLUDES)
    remote = "sh -c " + shlex.quote(remote_script(cfg, _local_file_count()))
    copy = (
        f"tar czf - --owner=0 --group=0 --numeric-owner {excludes}"
        " -C custom_components betterybalance"
        f" | ssh {shlex.quote(target)} {shlex.quote(remote)}"
    )
    return [[*PIPE, copy], ["ssh", target, "ha core check"]]


def _show(cmd: list[str]) -> str:
    return cmd[len(PIPE)] if cmd[: len(PIPE)] == PIPE else shlex.join(cmd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    parser.add_argument("--env-file", type=Path, default=None, help="default data/owner/site.env")
    args = parser.parse_args(argv)
    env = load_env(args.env_file)
    cmds = plan_commands(env)
    for cmd in cmds:
        print(("would run: " if args.dry_run else "running: ") + _show(cmd), flush=True)
        if args.dry_run:
            if cmd[: len(PIPE)] == PIPE:
                print("  box side (sh -c):")
                script = remote_script(config_path(env), _local_file_count())
                print("\n".join("    " + x for x in script.splitlines()))
            continue
        done = subprocess.run(cmd, cwd=ROOT, check=False)
        if done.returncode != 0:
            print(f"failed with exit code {done.returncode}: {_show(cmd)}", file=sys.stderr)
            return done.returncode
    print(CLOSING)
    return 0


if __name__ == "__main__":
    sys.exit(main())
