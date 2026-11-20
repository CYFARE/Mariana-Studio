"""The cross-platform ``mariana setup`` bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mariana import setup
from tests.helpers import ROOT


def test_minor_and_exact_versions_select_the_newest_patch():
    versions = ["3.12.7", "3.13.1", "3.13.10", "3.13.2", "system", "3.13.4/envs/work"]
    assert setup.best_version(versions, "3.13") == "3.13.10"
    assert setup.best_version(versions, "3.13.2") == "3.13.2"
    assert setup.best_version(versions, "3.11") is None
    assert setup.version_matches("3.14.10", "3.14")
    assert not setup.version_matches("3.14.10", "3.1")


def test_the_default_follows_the_running_interpreter_and_falls_back():
    class Current:
        major = 3
        minor = 13

    class Old:
        major = 3
        minor = 8

    assert setup.default_python(Current()) == "3.13"
    assert setup.default_python(Old()) == setup.FALLBACK_PYTHON


def test_too_old_pythons_are_rejected_with_instructions():
    with pytest.raises(setup.SetupError, match="Python 3.10 or newer"):
        setup.require_supported("3.9.18")
    setup.require_supported("3.14.2")


def test_installed_pyenv_versions_are_reused_before_installing(monkeypatch):
    monkeypatch.setattr(setup, "installed_versions", lambda pyenv: ["3.13.2", "3.14.1"])
    monkeypatch.setattr(setup, "available_versions",
                        lambda pyenv: pytest.fail("must not query installable versions"))
    assert setup.ensure_pyenv_python(Path("/fake/pyenv"), "3.14") == "3.14.1"


def test_a_missing_pyenv_version_is_installed_at_the_newest_patch(monkeypatch):
    state = {"installed": []}
    calls = []

    def record(command, what):
        calls.append(command)
        state["installed"] = ["3.14.2"]

    monkeypatch.setattr(setup, "installed_versions", lambda pyenv: list(state["installed"]))
    monkeypatch.setattr(setup, "available_versions", lambda pyenv: ["3.14.1", "3.14.2", "3.15.0"])
    monkeypatch.setattr(setup, "_stream", record)

    assert setup.ensure_pyenv_python(Path("/fake/pyenv"), "3.14") == "3.14.2"
    assert calls and calls[0][-1] == "3.14.2" and "install" in calls[0]


def test_pyenv_listings_ignore_virtualenvs_and_non_versions(monkeypatch):
    outputs = {
        "versions": "3.12.8\n3.13.4\nsystem\n3.13.4/envs/work\n",
        "install": "Available versions:\n  3.13.9\n  3.14.0t\n",
    }

    def fake_capture(command):
        return 0, outputs["install" if "--list" in command else "versions"]

    monkeypatch.setattr(setup, "_capture", fake_capture)
    assert setup.installed_versions(Path("/fake/pyenv")) == ["3.12.8", "3.13.4"]
    assert setup.available_versions(Path("/fake/pyenv")) == ["3.13.9"]


def test_pyenv_is_found_through_pyenv_root(tmp_path, monkeypatch):
    executable = tmp_path / "bin" / "pyenv"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path))
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    assert setup.find_pyenv() == executable


def test_every_requirements_profile_exists():
    for name in list(setup.DEVICE_REQUIREMENTS.values()) + list(setup.EXTRAS.values()):
        assert (ROOT / name).is_file(), name


def test_device_profiles_and_extras_are_validated():
    assert setup.requirements_for("auto") == "requirements.txt"
    assert setup.requirements_for("cpu") == "requirements-cpu.txt"
    with pytest.raises(setup.SetupError, match="Unknown device profile"):
        setup.requirements_for("tpu")

    assert setup.parse_extras("export, data") == ["data", "export"]
    assert setup.parse_extras("all") == list(setup.EXTRAS)
    assert setup.parse_extras("") == []
    with pytest.raises(setup.SetupError, match="Unknown optional group"):
        setup.parse_extras("gpu")


def test_venv_paths_follow_the_platform(tmp_path):
    assert setup.venv_python(tmp_path / "venv", windows=False) == tmp_path / "venv" / "bin" / "python"
    assert setup.venv_python(tmp_path / "venv", windows=True) == \
        tmp_path / "venv" / "Scripts" / "python.exe"


def test_venv_version_reads_pyvenv_cfg(tmp_path):
    (tmp_path / "pyvenv.cfg").write_text(
        "home = /usr/bin\nversion_info = 3.14.10.final.0\n", encoding="utf-8",
    )
    assert setup.venv_version(tmp_path) == "3.14.10"
    (tmp_path / "pyvenv.cfg").write_text("version = 3.13.2\n", encoding="utf-8")
    assert setup.venv_version(tmp_path) == "3.13.2"


def test_an_existing_venv_for_another_python_line_needs_recreate(tmp_path):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("version = 3.10.4\n", encoding="utf-8")
    with pytest.raises(setup.SetupError, match="--recreate"):
        setup.create_venv(Path("/unused/python"), venv, windows=False,
                          expected_version="3.14.2")


def test_project_venv_honours_the_environment_override(tmp_path, monkeypatch):
    custom = tmp_path / "custom"
    python = setup.venv_python(custom)
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv("MARIANA_VENV", str(custom))
    assert setup.project_venv_python(tmp_path) == python


def test_recreate_refuses_to_delete_a_plain_directory(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep", encoding="utf-8")
    assert not setup.can_recreate(folder)
    with pytest.raises(setup.SetupError, match="will not delete it"):
        setup.create_venv(Path(sys.executable), folder, recreate=True)
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_create_venv_makes_a_working_interpreter(tmp_path):
    venv = tmp_path / "venv"
    try:
        python = setup.create_venv(Path(sys.executable), venv)
    except setup.SetupError as error:  # a Python without ensurepip
        pytest.skip(str(error))
    assert python.is_file()
    assert (venv / "pyvenv.cfg").is_file()


def test_parser_defaults_and_exclusive_pyenv_flags():
    args = setup.build_parser().parse_args([])
    assert args.device == "auto"
    assert args.venv is None and args.python is None and args.extras == ""
    with pytest.raises(SystemExit):
        setup.build_parser().parse_args(["--install-pyenv", "--no-pyenv"])


def test_pyenv_is_preferred_over_the_system_interpreter(monkeypatch):
    monkeypatch.setattr(setup, "find_pyenv", lambda: Path("/pyenv/bin/pyenv"))
    monkeypatch.setattr(setup, "ensure_pyenv_python", lambda pyenv, requested: "3.13.4")
    monkeypatch.setattr(
        setup, "pyenv_interpreter",
        lambda pyenv, version: Path("/pyenv/versions") / version / "bin" / "python",
    )
    python, version, provider = setup._choose_interpreter("3.13", False, False)
    assert python == Path("/pyenv/versions/3.13.4/bin/python")
    assert version == "3.13.4"
    assert provider == "pyenv 3.13.4"


def test_main_reports_bad_extras_without_a_traceback(capsys):
    assert setup.main(["--extras", "gpu"]) == 1
    output = capsys.readouterr().out
    assert "Setup failed" in output
    assert "data, export, finetune" in output


def test_recreate_refuses_when_running_inside_the_venv(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("", encoding="utf-8")
    (venv / "pyvenv.cfg").write_text("version = 3.14.0\n", encoding="utf-8")
    monkeypatch.setattr(setup.sys, "executable", str(venv / "bin" / "python"))
    with pytest.raises(setup.SetupError, match="deactivate"):
        setup.create_venv(Path("/unused/python"), venv, recreate=True, windows=False)


def test_setup_can_run_before_torch_is_available():
    code, output = setup._capture([
        sys.executable, "-c",
        "import sys, mariana.setup; print('torch' in sys.modules)",
    ])
    assert code == 0
    assert output.strip().splitlines()[-1] == "False"


def test_pyenv_prefix_falls_back_to_the_versions_directory(tmp_path, monkeypatch):
    pyenv = tmp_path / "pyenv-win" / "bin" / "pyenv.bat"
    pyenv.parent.mkdir(parents=True)
    pyenv.write_text("", encoding="utf-8")
    prefix = tmp_path / "pyenv-win" / "versions" / "3.14.2"
    prefix.mkdir(parents=True)
    monkeypatch.setattr(setup, "_capture", lambda command: (1, "no such command"))
    assert setup._pyenv_prefix(pyenv, "3.14.2") == prefix


def test_install_pyenv_clones_into_an_empty_target(tmp_path, monkeypatch):
    target = tmp_path / "pyenv"

    def fake_stream(command, what):
        assert command[:3] == ["git", "clone", "--depth"]
        (target / "bin").mkdir(parents=True)
        (target / "bin" / "pyenv").write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(setup, "_stream", fake_stream)
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/git")
    assert setup.install_pyenv(target, windows=False) == target / "bin" / "pyenv"


def test_install_pyenv_refuses_to_use_a_populated_directory(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    monkeypatch.setattr(setup.shutil, "which", lambda name: "/usr/bin/git")
    with pytest.raises(setup.SetupError, match="already contains files"):
        setup.install_pyenv(tmp_path, windows=False)


def test_reexec_switches_a_script_to_the_project_venv(tmp_path, monkeypatch):
    python = setup.venv_python(tmp_path / ".venv")
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    calls = []

    monkeypatch.setattr(setup, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(setup.sys, "prefix", "/system")
    monkeypatch.setattr(setup.sys, "base_prefix", "/system")
    monkeypatch.setattr(setup.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(setup.sys, "argv", ["mariana.py", "studio"])
    monkeypatch.setattr(setup.os, "execve",
                        lambda path, arguments, environment: calls.append(
                            (path, arguments, environment)))
    monkeypatch.delenv("MARIANA_VENV_ACTIVE", raising=False)
    monkeypatch.delenv("MARIANA_NO_VENV", raising=False)

    setup.reexec_in_venv()
    assert calls and calls[0][0] == str(python)
    assert calls[0][1] == [str(python), "mariana.py", "studio"]
    assert calls[0][2]["MARIANA_VENV_ACTIVE"] == "1"


def test_reexec_uses_the_module_form_for_python_m_mariana(tmp_path, monkeypatch):
    python = setup.venv_python(tmp_path / ".venv")
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    calls = []

    monkeypatch.setattr(setup, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(setup.sys, "prefix", "/system")
    monkeypatch.setattr(setup.sys, "base_prefix", "/system")
    monkeypatch.setattr(setup.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(setup.sys, "argv", ["/repo/mariana/__main__.py", "studio"])
    monkeypatch.setattr(setup.os, "execve",
                        lambda path, arguments, environment: calls.append(arguments))
    monkeypatch.delenv("MARIANA_VENV_ACTIVE", raising=False)
    monkeypatch.delenv("MARIANA_NO_VENV", raising=False)

    setup.reexec_in_venv(module=True)
    assert calls == [[str(python), "-m", "mariana", "studio"]]


def test_reexec_stays_put_inside_a_venv_or_when_disabled(tmp_path, monkeypatch):
    python = setup.venv_python(tmp_path / ".venv")
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    calls = []
    monkeypatch.setattr(setup, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(setup.os, "execve", lambda *arguments: calls.append(arguments))
    monkeypatch.setattr(setup.sys, "prefix", "/system/venv")
    monkeypatch.setattr(setup.sys, "base_prefix", "/system")
    monkeypatch.delenv("MARIANA_VENV_ACTIVE", raising=False)
    monkeypatch.delenv("MARIANA_NO_VENV", raising=False)

    setup.reexec_in_venv()
    assert calls == []

    monkeypatch.setattr(setup.sys, "prefix", "/system")
    monkeypatch.setenv("MARIANA_NO_VENV", "1")
    setup.reexec_in_venv()
    assert calls == []


def test_run_installs_the_profile_and_optional_groups(tmp_path, monkeypatch, capsys):
    calls = []

    def record_venv(interpreter, venv, recreate=False, expected_version=None):
        calls.append(("venv", interpreter, recreate, expected_version))
        return Path("/venv/python")

    monkeypatch.setattr(setup, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        setup, "_choose_interpreter",
        lambda requested, install, no_pyenv: (
            Path("/chosen/python"), "3.14.2", "pyenv 3.14.2"),
    )
    monkeypatch.setattr(setup, "create_venv", record_venv)
    monkeypatch.setattr(setup, "upgrade_pip", lambda python: calls.append(("pip", python)))
    monkeypatch.setattr(
        setup, "install_requirements",
        lambda python, path: calls.append(("requirements", path.name)),
    )
    monkeypatch.setattr(
        setup, "verify_environment",
        lambda python: {"torch": "9.9", "numpy": "1.0", "cuda": False},
    )

    args = setup.build_parser().parse_args([
        "--device", "cpu", "--extras", "export", "--recreate",
    ])
    assert setup.run(args) == 0
    installed = [call[1] for call in calls if call[0] == "requirements"]
    assert installed == ["requirements-cpu.txt", "requirements-export.txt"]
    assert ("venv", Path("/chosen/python"), True, "3.14.2") in calls
    output = capsys.readouterr().out
    assert "Verified: torch 9.9" in output
