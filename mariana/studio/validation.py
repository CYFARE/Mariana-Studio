"""Boundary validation helpers shared by handlers and services.

Each helper reads one key from an untrusted mapping, returns a clean value, or
raises :class:`~mariana.studio.errors.BadRequestError` naming the key.
"""

from __future__ import annotations

import re
from pathlib import Path

from .errors import BadRequestError

DEVICES = ("auto", "cuda", "xpu", "mps", "cpu")
_NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")


def number(config: dict, key: str, default, minimum, maximum, cast=int, error=BadRequestError):
    """Return ``config[key]`` as a bounded number."""
    try:
        value = cast(config.get(key, default))
    except (TypeError, ValueError):
        raise error(f"{key} must be a number") from None
    if value != value:  # NaN never compares equal to itself
        raise error(f"{key} must be a number")
    if not minimum <= value <= maximum:
        raise error(f"{key} must be between {minimum} and {maximum}")
    return value


def choice(config: dict, key: str, default: str, allowed, label: str | None = None) -> str:
    """Return ``config[key]`` if it is one of ``allowed``."""
    value = config.get(key, default)
    if value is None or value == "":
        value = default
    if value not in allowed:
        options = ", ".join(str(item) for item in allowed)
        raise BadRequestError(f"{label or key} must be one of: {options}")
    return value


def text(config: dict, key: str, default: str = "", limit: int = 10_000) -> str:
    """Return ``config[key]`` as stripped text no longer than ``limit``."""
    value = config.get(key, default)
    if value is None:
        return default
    return str(value).strip()[:limit]


def flag(config: dict, key: str, default: bool = False) -> bool:
    """Return ``config[key]`` as a boolean, accepting common string spellings."""
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def device(config: dict, key: str = "device") -> str:
    """Return a supported torch device name."""
    value = config.get(key) or "auto"
    if value not in DEVICES:
        raise BadRequestError("Unknown device")
    return value


def safe_name(value: str, what: str = "Name") -> str:
    """Return ``value`` if it is a plain file name without path components."""
    value = str(value or "").strip()
    if not value or Path(value).name != value or not _NAME_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise BadRequestError(
            f"{what} may contain only letters, numbers, dots, dashes, and underscores"
        )
    return value


def inside(path: Path, parent: Path, what: str = "Path") -> Path:
    """Resolve ``path`` and require it to live under ``parent``."""
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(Path(parent).resolve())
    except ValueError:
        raise BadRequestError(f"{what} must be inside {parent}") from None
    return resolved
