"""Core-runtime bootstrap before CLI imports."""

from __future__ import annotations

import sys

import pytest

from mariana import bootstrap


def test_missing_core_installs_the_fixed_requirements_file(monkeypatch):
    calls = []
    states = iter([["torch"], []])
    monkeypatch.setattr(bootstrap, "_missing", lambda: next(states))
    monkeypatch.delenv("MARIANA_AUTO_INSTALL", raising=False)
    monkeypatch.delenv("MARIANA_STUDIO_AUTO_INSTALL", raising=False)
    monkeypatch.setattr(bootstrap.subprocess, "check_call", lambda command: calls.append(command))
    bootstrap.ensure_core()
    assert len(calls) == 1
    assert calls[0][:4] == [sys.executable, "-m", "pip", "install"]
    assert str(bootstrap.CORE_REQUIREMENTS) in calls[0]


def test_core_install_can_be_disabled(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "_missing", lambda: ["torch"])
    monkeypatch.setenv("MARIANA_AUTO_INSTALL", "0")
    with pytest.raises(SystemExit):
        bootstrap.ensure_core()
    assert "Mariana needs torch" in capsys.readouterr().out


def test_present_core_is_a_noop(monkeypatch):
    calls = []
    monkeypatch.setattr(bootstrap, "_missing", lambda: [])
    monkeypatch.setattr(bootstrap.subprocess, "check_call", lambda command: calls.append(command))
    bootstrap.ensure_core()
    assert calls == []
