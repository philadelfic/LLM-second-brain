"""Contract of scripts/build_image.sh: the OCI labels carry real values.

The 3.8.1 deploy image (llm-second-brain:latest, 2026-11-14 02:01) was built by
`docker compose up -d --build`: compose passes no build args, so the Dockerfile
defaults stayed — `docker history` shows GIT_COMMIT=unknown APP_VERSION=unknown
and `docker inspect` labels read "unknown". The script must therefore take the
revision from git and the version from the code, and must not abort when git is
absent: an honest "unknown" label beats a silently unstamped image.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_image.sh"
APP_INIT = ROOT / "app" / "__init__.py"

GIT = shutil.which("git")


def _app_version() -> str:
    match = re.search(
        r'^__version__\s*=\s*"([^"]+)"',
        APP_INIT.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None, "app/__init__.py must define __version__"
    return match.group(1)


def _fake_bin(tmp_path: Path, *, git_available: bool) -> Path:
    """A PATH with a docker stub that records argv and, optionally, a broken git."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_stub = bin_dir / "docker"
    docker_stub.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$DOCKER_ARGS_LOG"\n',
        encoding="utf-8",
    )
    docker_stub.chmod(0o755)
    if not git_available:
        git_stub = bin_dir / "git"
        git_stub.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
        git_stub.chmod(0o755)
    return bin_dir


def _run_script(
    tmp_path: Path, *, git_available: bool
) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
    bin_dir = _fake_bin(tmp_path, git_available=git_available)
    args_log = tmp_path / "docker-args.txt"
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["DOCKER_ARGS_LOG"] = str(args_log)
    proc = subprocess.run(
        ["bash", str(SCRIPT), "llm-second-brain:test"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    build_args: dict[str, str] = {}
    lines = args_log.read_text(encoding="utf-8").splitlines() if args_log.exists() else []
    for index, line in enumerate(lines[:-1]):
        value = lines[index + 1]
        if line == "--build-arg" and "=" in value:
            name, _, raw = value.partition("=")
            build_args[name] = raw
    return proc, build_args


@pytest.mark.skipif(GIT is None, reason="git is not available in the runner")
def test_revision_is_git_head_of_the_tree(tmp_path: Path) -> None:
    proc, build_args = _run_script(tmp_path, git_available=True)
    assert proc.returncode == 0, proc.stderr
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    revision = build_args["GIT_COMMIT"]
    assert revision.removesuffix("-dirty") == head
    assert len(revision) >= 40


@pytest.mark.skipif(GIT is None, reason="git is not available in the runner")
def test_version_comes_from_app_init_not_git_describe(tmp_path: Path) -> None:
    proc, build_args = _run_script(tmp_path, git_available=True)
    assert proc.returncode == 0, proc.stderr
    assert build_args["APP_VERSION"] == _app_version()
    assert not build_args["APP_VERSION"].startswith("v")


def test_missing_git_degrades_to_unknown_and_still_builds(tmp_path: Path) -> None:
    proc, build_args = _run_script(tmp_path, git_available=False)
    assert proc.returncode == 0, proc.stderr
    assert build_args["GIT_COMMIT"] == "unknown"
    assert "WARN" in proc.stderr and "revision" in proc.stderr
    assert build_args["APP_VERSION"] == _app_version()
