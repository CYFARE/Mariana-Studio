"""Dependency detection and the background pip installer."""

from __future__ import annotations

import threading

import pytest

from mariana.studio.dependencies import DependencyManager
from mariana.studio.errors import BadRequestError
from mariana.studio.jobs import JobRegistry
from tests.helpers import ROOT, wait_for


@pytest.fixture()
def registry():
    instance = JobRegistry()
    yield instance
    instance.close()


def manager(registry, settings=None):
    return DependencyManager(
        ROOT, registry, settings or (lambda: {"auto_install_dependencies": False}),
        requirements_root=ROOT,
    )


def settled(registry, job_id):
    return wait_for(lambda: (lambda job: job if not job["active"] else None)(registry.get(job_id)))


def test_status_lists_groups_and_missing_packages(registry, monkeypatch):
    instance = manager(registry)
    monkeypatch.setattr(
        instance, "_installed",
        lambda name: "1.0" if name in {"torch", "numpy"} else None,
    )
    status = instance.status()
    assert [group["id"] for group in status["groups"]] == ["core", "finetune", "export", "data"]
    assert status["missing"] == ["finetune", "export", "data"]
    assert status["missing_packages"] == [
        "transformers", "safetensors", "gguf", "onnx", "onnxruntime",
        "datasets", "huggingface-hub", "pyarrow",
    ]
    assert status["installed"] is False and status["auto_install"] is False
    assert status["pip"] is True and status["busy"] is False and status["job"] is None


def test_auto_install_preference_and_environment_override(registry, monkeypatch):
    instance = manager(registry, settings=lambda: {"auto_install_dependencies": True})
    assert instance.status()["auto_install"] is True
    monkeypatch.setenv("MARIANA_STUDIO_AUTO_INSTALL", "0")
    assert instance.status()["auto_install"] is False
    monkeypatch.setenv("MARIANA_STUDIO_AUTO_INSTALL", "yes")
    assert instance.status()["auto_install"] is True


def test_install_runs_one_pip_command_per_group(registry, monkeypatch):
    instance = manager(registry)
    calls = []

    def fake_run(job, command, **kwargs):
        calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(registry, "run_process", fake_run)
    started = instance.install(["finetune", "export"])
    assert started["started"] is True and started["job"]["meta"]["groups"] == ["finetune", "export"]
    job = settled(registry, started["job"]["id"])
    assert job["status"] == "completed" and job["progress"]["percent"] == 100.0
    assert len(calls) == 2
    assert calls[0][0][-1].endswith("requirements-finetune.txt")
    assert calls[1][0][-1].endswith("requirements-export.txt")
    assert instance.status()["job"]["id"] == job["id"]


def test_failed_install_reports_the_pip_error(registry, monkeypatch):
    instance = manager(registry)

    def fake_run(job, command, **kwargs):
        job.log("ERROR: Could not find a version that satisfies the requirement transformers")
        return 1

    monkeypatch.setattr(registry, "run_process", fake_run)
    started = instance.install(["finetune"])
    job = settled(registry, started["job"]["id"])
    assert job["status"] == "failed"
    assert "pip could not install Fine-tuning" in job["error"]
    assert "Could not find a version" in job["error"]


def test_unknown_groups_are_rejected(registry):
    instance = manager(registry)
    with pytest.raises(BadRequestError, match="Unknown requirement group"):
        instance.install(["manufacturing"])
    with pytest.raises(BadRequestError, match="must be a list"):
        instance.install("finetune")


def test_a_second_install_returns_the_active_job(registry, monkeypatch):
    instance = manager(registry)
    gate = threading.Event()

    def fake_run(job, command, **kwargs):
        gate.wait(5)
        return 0

    monkeypatch.setattr(registry, "run_process", fake_run)
    first = instance.install(["finetune"])
    second = instance.install(["export"])
    assert second["started"] is False
    assert second["job"]["id"] == first["job"]["id"]
    gate.set()
    assert settled(registry, first["job"]["id"])["status"] == "completed"


def test_missing_requirement_file_is_reported(registry, tmp_path):
    instance = DependencyManager(
        ROOT, registry, lambda: {"auto_install_dependencies": False},
        requirements_root=tmp_path,
    )
    status = instance.status()
    assert all(group["error"].endswith("is missing from the Mariana repository") for group in status["groups"])
    assert status["errors"] == ["core", "finetune", "export", "data"]
    assert status["missing"] == [] and status["installed"] is False
    with pytest.raises(BadRequestError, match="is missing"):
        instance.install(["finetune"])
    with pytest.raises(BadRequestError, match="requirement files are missing"):
        instance.install()
