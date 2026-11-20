"""Version comparison and the git update used by ``mariana update``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from mariana import __version__, update
from mariana.update import (
    UpdateError,
    apply_update,
    check_for_update,
    is_newer,
    local_version,
    version_tuple,
)

RELEASE = {
    "tag": "9.9.9",
    "version": "9.9.9",
    "name": "v9.9.9",
    "url": "https://github.com/CYFARE/Mariana-Studio/releases/tag/9.9.9",
    "notes": "Release notes",
    "published_at": "",
    "prerelease": False,
}


def test_versions_compare_numerically_not_lexically():
    assert version_tuple("v2.10.0") == (2, 10, 0)
    assert version_tuple("release-2.4") == (2, 4)
    assert version_tuple("nightly") == ()
    assert is_newer("2.10.0", "2.9.9")
    assert is_newer("v2.4", "2.3.0")
    assert not is_newer("2.3.0", "2.3")
    assert not is_newer("2.2.9", "2.3.0")
    assert not is_newer("nightly", "2.3.0")


def test_local_version_prefers_the_version_file_then_the_package(tmp_path):
    assert local_version(tmp_path) == __version__
    (tmp_path / "VERSION.txt").write_text("9.9.9\n", encoding="utf-8")
    assert local_version(tmp_path) == "9.9.9"


def test_check_reports_a_newer_release(tmp_path):
    (tmp_path / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    result = check_for_update(tmp_path, fetch=lambda timeout=10: dict(RELEASE))
    assert result["ok"] is True and result["error"] is None
    assert result["update_available"] is True
    assert result["latest_version"] == "9.9.9"
    assert result["release"]["url"].endswith("9.9.9")


def test_check_keeps_network_errors_as_data(tmp_path):
    def offline(timeout=10):
        raise UpdateError("offline")

    result = check_for_update(tmp_path, fetch=offline)
    assert result["ok"] is False
    assert result["update_available"] is False
    assert "offline" in result["error"]


def test_fetch_latest_release_parses_tags_and_hides_the_v(monkeypatch):
    payload = json.dumps({
        "tag_name": "v9.9.9", "name": "v9.9.9", "html_url": RELEASE["url"],
        "body": "Notes", "published_at": "2026-01-01T00:00:00Z", "prerelease": False,
    }).encode("utf-8")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return payload

    monkeypatch.setattr(update.urllib.request, "urlopen",
                        lambda request, timeout: Response())
    release = update.fetch_latest_release()
    assert release["tag"] == "v9.9.9"
    assert release["version"] == "9.9.9"
    assert release["url"] == RELEASE["url"]
    assert release["notes"] == "Notes"


def test_apply_update_fast_forwards_a_clean_checkout(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    commands, steps = [], []

    def runner(command, on_line=None):
        commands.append(command)
        if "--porcelain" in command:
            return 0, ""
        if "--abbrev-ref" in command:
            return 0, "main\n"
        if "fetch" in command:
            return 0, "From github.com/CYFARE/Mariana-Studio"
        if "merge" in command:
            (tmp_path / "VERSION.txt").write_text("9.9.9", encoding="utf-8")
            return 0, "Fast-forward"
        raise AssertionError(command)

    outcome = apply_update(tmp_path, runner=runner, on_step=lambda index, label: steps.append(label))
    assert outcome == {
        "previous_version": "2.3.0", "version": "9.9.9", "branch": "main",
        "updated": True, "restart_required": True,
    }
    assert ["git", "fetch", "--tags", "--prune", "origin"] in commands
    assert ["git", "merge", "--ff-only", "origin/main"] in commands
    assert steps == ["Checking the checkout", "Fetching the latest release",
                     "Fast-forwarding to the latest release"]


def test_apply_update_fast_forwards_a_real_checkout(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git is not installed")
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Mariana Test", "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "Mariana Test", "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }

    def git(cwd, *arguments):
        subprocess.run(["git", *arguments], cwd=cwd, check=True, capture_output=True,
                       text=True, env=environment)

    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)], check=True,
                   capture_output=True, text=True, env=environment)
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-b", "main", str(work)], check=True,
                   capture_output=True, text=True, env=environment)
    (work / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    git(work, "add", "VERSION.txt")
    git(work, "commit", "-m", "release 2.3.0")
    git(work, "remote", "add", "origin", str(remote))
    git(work, "push", "-u", "origin", "main")

    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(remote), str(other)], check=True,
                   capture_output=True, text=True, env=environment)
    (other / "VERSION.txt").write_text("9.9.9", encoding="utf-8")
    git(other, "commit", "-am", "release 9.9.9")
    git(other, "push", "origin", "main")

    outcome = apply_update(work)
    assert outcome["previous_version"] == "2.3.0"
    assert outcome["version"] == "9.9.9"
    assert outcome["branch"] == "main" and outcome["updated"] is True


def test_apply_update_refuses_uncommitted_changes(tmp_path):
    (tmp_path / ".git").mkdir()

    def dirty(command, on_line=None):
        return 0, " M mariana/cli.py"

    with pytest.raises(UpdateError, match="uncommitted changes"):
        apply_update(tmp_path, runner=dirty)


def test_apply_update_refuses_a_detached_head(tmp_path):
    (tmp_path / ".git").mkdir()

    def detached(command, on_line=None):
        if "--porcelain" in command:
            return 0, ""
        return 0, "HEAD\n"

    with pytest.raises(UpdateError, match="detached HEAD"):
        apply_update(tmp_path, runner=detached)


def test_apply_update_requires_a_git_checkout(tmp_path):
    with pytest.raises(UpdateError, match="not a git checkout"):
        apply_update(tmp_path, runner=lambda command, on_line=None: (0, ""))


def test_cli_check_reports_without_updating(tmp_path, monkeypatch, capsys):
    (tmp_path / "VERSION.txt").write_text("2.3.0", encoding="utf-8")
    monkeypatch.setattr(update, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(update, "check_for_update", lambda *args, **kwargs: {
        "local_version": "2.3.0", "latest_version": "9.9.9", "update_available": True,
        "release": dict(RELEASE), "checked_at": 0, "error": None, "ok": True,
    })

    assert update.main(["--check"]) == 0
    output = capsys.readouterr().out
    assert "New release: 9.9.9" in output
    assert "Restart Mariana Studio" not in output
