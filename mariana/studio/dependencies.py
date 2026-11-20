"""Detect and install the optional Python requirement groups Studio can use.

The web UI asks this service which requirement files are not satisfied and can
start one background pip job.  Only the known files below are ever installed;
callers cannot supply package names or commands.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .errors import BadRequestError, UnavailableError

REQUIREMENTS_ROOT = Path(__file__).resolve().parents[2]
_CACHE_SECONDS = 15.0
_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


@dataclass(frozen=True)
class RequirementGroup:
    """One requirements file and the feature it unlocks."""

    id: str
    title: str
    description: str
    file: str
    required: bool = False


GROUPS = (
    RequirementGroup("core", "Core runtime", "PyTorch and NumPy", "requirements.txt", required=True),
    RequirementGroup("finetune", "Fine-tuning", "Transformers chat, adapters and training", "requirements-finetune.txt"),
    RequirementGroup("export", "Export toolchain", "GGUF, ONNX and safetensors conversion", "requirements-export.txt"),
    RequirementGroup("data", "Dataset tooling", "Hugging Face downloads and Parquet corpus support", "requirements-data.txt"),
)
_BY_ID = {group.id: group for group in GROUPS}


def _packages(path: Path) -> tuple[list[dict], str]:
    """Parse one requirements file into distributions, or ``( [], error )``."""
    if not path.is_file():
        return [], f"{path.name} is missing from the Mariana repository"
    packages, seen = [], set()
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = raw.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        match = _NAME.match(text)
        if not match:
            continue
        name = match.group(1)
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        packages.append({"name": name, "spec": text, "installed": None})
    return packages, ""


class DependencyManager:
    """Requirement detection and one-at-a-time pip installation."""

    def __init__(self, root: Path, registry, settings: Callable[[], dict],
                 requirements_root: Path | None = None):
        self.root = Path(root).resolve()
        self.registry = registry
        self.settings = settings
        self.requirements_root = Path(requirements_root or REQUIREMENTS_ROOT).resolve()
        self._lock = threading.RLock()
        self._cache: tuple[float, dict] | None = None

    # -- detection --------------------------------------------------------
    def _installed(self, distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _detect(self) -> dict:
        groups, missing, missing_packages, errors = [], [], [], []
        for group in GROUPS:
            packages, error = _packages(self.requirements_root / group.file)
            absent = []
            if error:
                errors.append(group.id)
            else:
                for package in packages:
                    package["installed"] = self._installed(package["name"])
                    if not package["installed"]:
                        absent.append(package["name"])
                missing_packages.extend(absent)
                if absent:
                    missing.append(group.id)
            groups.append({
                "id": group.id, "title": group.title, "description": group.description,
                "file": group.file, "required": group.required,
                "packages": packages, "missing": absent, "error": error,
            })
        return {"groups": groups, "missing": missing, "errors": errors,
                "missing_packages": list(dict.fromkeys(missing_packages)),
                "installed": not missing and not errors}

    def _detected(self) -> dict:
        now = time.monotonic()
        with self._lock:
            if self._cache and now - self._cache[0] < _CACHE_SECONDS:
                return self._cache[1]
            value = self._detect()
            self._cache = (now, value)
            return value

    def invalidate(self) -> None:
        """Forget detected versions after pip changes the environment."""
        importlib.invalidate_caches()
        with self._lock:
            self._cache = None

    # -- installation -----------------------------------------------------
    def _auto_install(self) -> bool:
        override = os.environ.get("MARIANA_STUDIO_AUTO_INSTALL")
        if override is not None:
            return override.strip().lower() not in {"0", "false", "no", "off"}
        try:
            return bool(self.settings().get("auto_install_dependencies", True))
        except Exception:  # noqa: BLE001 - a settings failure must not block the popup
            return True

    def _select(self, groups: list[str] | None) -> list[RequirementGroup]:
        if groups is None:
            return [_BY_ID[group_id] for group_id in self._detected()["missing"]]
        if not isinstance(groups, list):
            raise BadRequestError("groups must be a list of requirement group ids")
        selected = []
        for raw in groups:
            if not isinstance(raw, str) or raw not in _BY_ID:
                raise BadRequestError(f"Unknown requirement group: {raw}")
            if raw not in [group.id for group in selected]:
                selected.append(_BY_ID[raw])
        return selected

    def _failure_detail(self, job) -> str:
        logs = [line.strip() for line in job.snapshot(logs=True).get("logs", []) if line.strip()]
        for line in reversed(logs):
            lowered = line.lower()
            if "error" in lowered or "no matching distribution" in lowered or "could not find" in lowered:
                return line
        return logs[-1] if logs else "See the job log for details."

    def _install(self, job, groups: list[RequirementGroup]) -> dict:
        total = len(groups)
        job.update(stage="Preparing pip", current=0, total=total, unit="group")
        try:
            for index, group in enumerate(groups):
                job.check_cancelled()
                job.update(stage=f"Installing {group.title}", current=index, total=total,
                           unit="group", group=group.id)
                command = [
                    sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                    "--no-input", "-r", str(self.requirements_root / group.file),
                ]
                code = self.registry.run_process(
                    job, command, cwd=self.root, env={"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
                )
                job.check_cancelled()
                if code != 0:
                    raise RuntimeError(f"pip could not install {group.title}. {self._failure_detail(job)}")
                job.update(current=index + 1)
            job.update(stage="Refreshing feature detection")
            return {"groups": [group.id for group in groups]}
        finally:
            self.invalidate()

    def install(self, groups: list[str] | None = None) -> dict:
        """Start one pip job for the selected groups (default: every missing group)."""
        with self._lock:
            active = self.registry.active("dependency")
            if active:
                return {"job": active[0], "started": False}
            selected = self._select(groups)
            if not selected:
                if groups is None and self._detected()["errors"]:
                    raise BadRequestError("Some requirement files are missing; restore them before installing")
                return {"job": None, "started": False, "installed": True}
            if importlib.util.find_spec("pip") is None:
                raise UnavailableError("pip is not available in this Python environment; install pip first")
            for group in selected:
                if not (self.requirements_root / group.file).is_file():
                    raise BadRequestError(f"{group.file} is missing from the Mariana repository")
            job = self.registry.create(
                "dependency", "Install Python dependencies",
                meta={"groups": [group.id for group in selected],
                      "files": [group.file for group in selected]},
            )
            self.registry.start(job, lambda current: self._install(current, selected))
            return {"job": job.snapshot(), "started": True}

    # -- views ------------------------------------------------------------
    def status(self) -> dict:
        """Detected groups, missing packages, the auto-install preference and the latest job."""
        detected = self._detected()
        jobs = [job for job in self.registry.list() if job.get("kind") == "dependency"]
        latest = max(jobs, key=lambda item: item.get("created") or 0, default=None)
        return {
            **detected,
            "auto_install": self._auto_install(),
            "pip": importlib.util.find_spec("pip") is not None,
            "busy": bool(self.registry.active("dependency")),
            "job": latest,
        }
