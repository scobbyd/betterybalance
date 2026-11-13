# tests/test_deploy_ha.py
import os
import shutil
import subprocess

import pytest

from tools import deploy_ha as D


def test_plan_copies_checks_and_never_restarts():
    cmds = D.plan_commands({"BB_HA_SSH": "box", "BB_HA_CONFIG": "/config"})
    flat = " ".join(" ".join(c) for c in cmds)
    assert "tar" in flat and "ha core check" in flat and "restart" not in flat


def test_missing_ssh_target_is_named():
    with pytest.raises(SystemExit, match="BB_HA_SSH"):
        D.plan_commands({})


def test_copy_is_staged_quoted_and_runs_under_pipefail():
    copy, check = D.plan_commands({"BB_HA_SSH": "box"})
    assert copy[:4] == ["bash", "-o", "pipefail", "-c"]
    pipeline = copy[4]
    assert "--exclude __pycache__" in pipeline and "--exclude '*.pyc'" in pipeline
    assert "--owner=0 --group=0 --numeric-owner" in pipeline
    assert "-C custom_components betterybalance | ssh box" in pipeline
    script = D.remote_script("/config", 7)
    lines = script.splitlines()
    assert lines[0] == "set -e"
    assert "rm -rf /config/.betterybalance-staging" in lines
    assert "tar xzf - -C /config/.betterybalance-staging" in lines
    assert '[ "$n" -ne 7 ]' in script
    assert lines[1] == "if [ ! -f /config/configuration.yaml ]; then"
    assert (
        "if [ -e /config/custom_components/betterybalance ]; then rm -rf "
        "/config/.betterybalance-previous; mv /config/custom_components/betterybalance "
        "/config/.betterybalance-previous; fi"
    ) in lines
    assert (
        "if ! mv /config/.betterybalance-staging/betterybalance "
        "/config/custom_components/betterybalance; then"
    ) in lines
    assert lines[-2:] == [
        "rm -rf /config/.betterybalance-previous", "rm -rf /config/.betterybalance-staging"
    ]
    # every removal and move names the staging folder, the old copy or the integration only
    allowed = ("/config/.betterybalance-staging", "/config/.betterybalance-previous",
               "/config/custom_components/betterybalance")
    for x in lines:
        for part in x.replace("then ", ";").split(";"):
            part = part.strip()
            if part.startswith("if ! "):
                part = part[len("if ! "):]
            if part.startswith(("rm ", "mv ")):
                words = part.split()[2 if part.startswith("rm ") else 1:]
                assert all(w.startswith(allowed) for w in words), x
    assert check == ["ssh", "box", "ha core check"]


def test_a_config_path_with_a_space_is_quoted():
    script = D.remote_script(D.config_path({"BB_HA_CONFIG": "/srv/my ha/"}), 3)
    assert "tar xzf - -C '/srv/my ha/.betterybalance-staging'" in script
    assert "mv '/srv/my ha/.betterybalance-staging/betterybalance' " in script


@pytest.mark.parametrize(
    "bad", ["", "/", "//", "config", "relative/path", "/.", "/config/..", "/a/./config",
            "/config/../etc"],
)
def test_an_empty_root_relative_or_dotted_config_path_is_refused(bad):
    with pytest.raises(SystemExit, match="BB_HA_CONFIG"):
        D.plan_commands({"BB_HA_SSH": "box", "BB_HA_CONFIG": bad})


def _box(tmp_path, installed: str | None = "installed.py"):
    """A fake config folder with ``configuration.yaml`` and, when ``installed`` names a file,
    an installed integration holding that one file."""
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "configuration.yaml").write_text("homeassistant:\n")
    if installed is not None:
        old = cfg / "custom_components" / "betterybalance"
        old.mkdir(parents=True)
        (old / installed).write_text("x = 1\n")
    return cfg


def _local_copy(tmp_path, cfg, n_files, truncate=False, env=None):
    """The pipeline with the box side run locally (no ssh): tar, then ``remote_script``."""
    tarball = tmp_path / "copy.tgz"
    subprocess.run(
        ["tar", "czf", str(tarball), "--exclude", "__pycache__", "--exclude", "*.pyc",
         "-C", str(D.ROOT / "custom_components"), "betterybalance"],
        check=True,
    )
    data = tarball.read_bytes()
    if truncate:
        data = data[: len(data) // 2]
    return subprocess.run(
        ["sh", "-c", D.remote_script(str(cfg), n_files)], input=data, capture_output=True,
        env=env,
    )


needs_tools = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("sh", "tar", "find", "wc", "tr")), reason="no POSIX tools"
)


@needs_tools
def test_the_box_side_swaps_the_tree_and_drops_stale_modules(tmp_path):
    cfg = _box(tmp_path, "removed_module.py")
    (cfg / "custom_components" / "other").mkdir()
    n = D._local_file_count()
    done = _local_copy(tmp_path, cfg, n)
    assert done.returncode == 0, done.stderr
    new = cfg / "custom_components" / "betterybalance"
    assert not (new / "removed_module.py").exists() and (new / "manifest.json").is_file()
    assert sum(1 for p in new.rglob("*") if p.is_file()) == n
    assert not list(new.rglob("*.pyc")) and not list(new.rglob("__pycache__"))
    assert sorted(p.name for p in (cfg / "custom_components").iterdir()) == [
        "betterybalance", "other"
    ]
    assert not (cfg / D.STAGING).exists() and not (cfg / D.PREVIOUS).exists()


@needs_tools
@pytest.mark.parametrize("fault", ["truncated", "count"])
def test_a_broken_copy_leaves_the_installed_integration_alone(tmp_path, fault):
    cfg = _box(tmp_path)
    old = cfg / "custom_components" / "betterybalance"
    n = D._local_file_count() + (1 if fault == "count" else 0)
    done = _local_copy(tmp_path, cfg, n, truncate=fault == "truncated")
    assert done.returncode != 0
    assert [p.name for p in old.iterdir()] == ["installed.py"]


def _failing_mv(tmp_path, cfg):
    """A PATH whose ``mv`` fails for the move of the staged tree into place (the second
    move of the swap) and runs the real ``mv`` for everything else."""
    shim = tmp_path / "shim"
    shim.mkdir()
    real = shutil.which("mv")
    staged = f"{cfg}/{D.STAGING}/betterybalance"
    (shim / "mv").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "{staged}" ]; then echo "mv: no space left" >&2; exit 1; fi\n'
        f'exec {real} "$@"\n'
    )
    (shim / "mv").chmod(0o755)
    return {**os.environ, "PATH": f"{shim}:{os.environ['PATH']}"}


@needs_tools
def test_a_failed_second_move_puts_the_old_integration_back(tmp_path):
    cfg = _box(tmp_path)
    done = _local_copy(tmp_path, cfg, D._local_file_count(), env=_failing_mv(tmp_path, cfg))
    assert done.returncode != 0
    assert b"previous integration is back in place" in done.stderr
    old = cfg / "custom_components" / "betterybalance"
    assert [p.name for p in old.iterdir()] == ["installed.py"]
    assert not (cfg / D.PREVIOUS).exists()


@needs_tools
def test_a_retry_after_a_cut_off_swap_restores_and_never_deletes_the_old_copy(tmp_path):
    cfg = _box(tmp_path, installed=None)
    prev = cfg / D.PREVIOUS  # a swap cut off between its two moves
    prev.mkdir()
    (prev / "installed.py").write_text("x = 1\n")
    (cfg / "custom_components").mkdir()
    done = _local_copy(tmp_path, cfg, D._local_file_count() + 1)  # this run fails its count
    assert done.returncode != 0
    assert b"cut-off deploy is back in place" in done.stderr
    old = cfg / "custom_components" / "betterybalance"
    assert [p.name for p in old.iterdir()] == ["installed.py"]


@needs_tools
def test_a_config_folder_without_configuration_yaml_is_refused(tmp_path):
    cfg = tmp_path / "not-config"
    cfg.mkdir()
    done = _local_copy(tmp_path, cfg, D._local_file_count())
    assert done.returncode != 0 and b"no configuration.yaml: nothing changed" in done.stderr
    assert list(cfg.iterdir()) == []


def test_environment_wins_over_the_env_file(tmp_path, monkeypatch):
    f = tmp_path / "site.env"
    f.write_text(
        "# owner values\nexport BB_HA_SSH='file-box'\nBB_HA_CONFIG=/from/file\n\nNOT A LINE\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("BB_HA_SSH", raising=False)
    monkeypatch.delenv("BB_HA_CONFIG", raising=False)
    assert D.load_env(f)["BB_HA_SSH"] == "file-box"
    assert D.load_env(f)["BB_HA_CONFIG"] == "/from/file"
    monkeypatch.setenv("BB_HA_SSH", "env-box")
    assert D.load_env(f)["BB_HA_SSH"] == "env-box"
    assert D.load_env(tmp_path / "absent.env")["BB_HA_SSH"] == "env-box"


def test_dry_run_prints_and_runs_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BB_HA_SSH", "box")
    monkeypatch.delenv("BB_HA_CONFIG", raising=False)

    def refuse(*a, **k):
        raise AssertionError("dry run executed a command")

    monkeypatch.setattr(D.subprocess, "run", refuse)
    assert D.main(["--dry-run", "--env-file", str(tmp_path / "absent.env")]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("would run: tar czf -") and out[-1] == D.CLOSING
    assert out[-2] == "would run: ssh box 'ha core check'"
    assert "    if ! mv /config/.betterybalance-staging/betterybalance " in "\n".join(out)
    assert not any("restart" in x for x in out[:-1])


def test_a_failed_step_stops_before_the_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BB_HA_SSH", "box")
    ran = []

    class Done:
        returncode = 2

    def fake(cmd, **kw):
        ran.append(cmd)
        return Done()

    monkeypatch.setattr(D.subprocess, "run", fake)
    assert D.main(["--env-file", str(tmp_path / "absent.env")]) == 2
    assert len(ran) == 1 and ran[0][:4] == D.PIPE
    assert D.CLOSING not in capsys.readouterr().out
