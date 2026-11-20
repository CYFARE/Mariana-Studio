"""Ensure the core PyTorch/NumPy runtime exists before the model modules load.

CLI entry points call this; importing the package itself never installs
anything.  The optional feature groups are handled by Studio's browser
installer instead.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

CORE_REQUIREMENTS = Path(__file__).resolve().parents[1] / "requirements.txt"


def _missing() -> list[str]:
    return [name for name in ("torch", "numpy") if importlib.util.find_spec(name) is None]


def _disabled() -> bool:
    value = os.environ.get("MARIANA_AUTO_INSTALL", os.environ.get("MARIANA_STUDIO_AUTO_INSTALL"))
    return value is not None and value.strip().lower() in {"0", "false", "no", "off"}


def _hint() -> None:
    print("Install the core runtime with one of:")
    print("    python mariana.py setup                  # pyenv + venv + requirements (recommended)")
    print("    pip install -r requirements.txt          # into an existing virtual environment")
    print("    pip install -r requirements-cpu.txt      # CPU-only wheels")


def ensure_core() -> None:
    """Install ``requirements.txt`` when PyTorch or NumPy is missing."""
    missing = _missing()
    if not missing:
        return
    if _disabled():
        print(f"Mariana needs {', '.join(missing)} to run.")
        _hint()
        raise SystemExit(1)
    if not CORE_REQUIREMENTS.is_file():
        print(f"Mariana needs {', '.join(missing)}, but requirements.txt was not found.")
        _hint()
        raise SystemExit(1)
    print(f"Mariana needs {', '.join(missing)}.")
    print("Installing the core runtime from requirements.txt; this can take a few minutes…")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "-r", str(CORE_REQUIREMENTS),
        ])
    except (OSError, subprocess.CalledProcessError):
        print("The automatic install did not finish.")
        _hint()
        raise SystemExit(1) from None
    still_missing = _missing()
    if still_missing:
        print(f"Core dependencies are still unavailable: {', '.join(still_missing)}")
        _hint()
        raise SystemExit(1)
