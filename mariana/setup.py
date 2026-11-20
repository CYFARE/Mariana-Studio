"""Create the pyenv interpreter and virtual environment used by ``mariana setup``.

The module is deliberately torch-free and dependency-free: it is the first
command a fresh checkout runs, on any OS, before anything is installed.  It
never installs into the system interpreter, so distributions that mark Python
as externally managed (PEP 668) are left untouched.

The steps are: choose a CPython line with pyenv when available, create a
virtual environment, install one ``requirements*.txt`` profile inside it, then
verify that PyTorch and NumPy import from that environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
KNOWN_GOOD_MAX = (3, 15)
FALLBACK_PYTHON = "3.12"
DEFAULT_VENV = ".venv"
PYENV_REPOSITORY = "https://github.com/pyenv/pyenv.git"
PYENV_WIN_REPOSITORY = "https://github.com/pyenv-win/pyenv-win.git"
COMMAND_TIMEOUT = 120.0

DEVICE_REQUIREMENTS = {
    "auto": "requirements.txt",
    "cpu": "requirements-cpu.txt",
    "cuda": "requirements-cuda.txt",
    "xpu": "requirements-xpu.txt",
}
EXTRAS = {
    "data": "requirements-data.txt",
    "export": "requirements-export.txt",
    "finetune": "requirements-finetune.txt",
}
_VERSION = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")
_VERSION_FIELD = re.compile(r"^(\d+\.\d+(?:\.\d+)?)")
VERIFY_SCRIPT = (
    "import json\n"
    "import numpy, torch\n"
    "print(json.dumps({\n"
    "    'torch': torch.__version__,\n"
    "    'numpy': numpy.__version__,\n"
    "    'cuda': bool(torch.cuda.is_available()),\n"
    "    'xpu': bool(getattr(torch, 'xpu', None) and torch.xpu.is_available()),\n"
    "    'mps': bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()),\n"
    "}))\n"
)


class SetupError(RuntimeError):
    """Setup could not finish; the message says what to do next."""


# ---------------------------------------------------------------------------
# Versions and paths
# ---------------------------------------------------------------------------
def repository_root() -> Path:
    """The checkout that contains this package."""
    return Path(__file__).resolve().parents[1]


def version_tuple(value: object) -> tuple[int, ...] | None:
    """``"3.14"`` -> ``(3, 14)``; anything that is not a version -> ``None``."""
    match = _VERSION.match(str(value or "").strip())
    if not match:
        return None
    return tuple(int(part) for part in match.groups() if part is not None)


def version_matches(candidate: object, requested: object) -> bool:
    """True when ``candidate`` is ``requested`` or one of its patch releases."""
    left, right = version_tuple(candidate), version_tuple(requested)
    if not left or not right:
        return False
    return left[:len(right)] == right


def best_version(versions, requested: object) -> str | None:
    """The newest entry of ``versions`` that satisfies ``requested``."""
    matching = [str(item) for item in versions if version_matches(item, requested)]
    if not matching:
        return None
    return max(matching, key=lambda item: version_tuple(item) or ())


def default_python(version_info=None) -> str:
    """The CPython line to request: the running one when it is new enough."""
    info = version_info if version_info is not None else sys.version_info
    minor = getattr(info, "minor", 0)
    if getattr(info, "major", 0) == 3 and MIN_PYTHON[1] <= minor <= KNOWN_GOOD_MAX[1]:
        return f"3.{minor}"
    return FALLBACK_PYTHON


def require_supported(version: str) -> None:
    """Raise when ``version`` is too old for the package."""
    parts = version_tuple(version)
    if parts is None or parts[:2] < MIN_PYTHON:
        raise SetupError(
            f"Mariana needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer; "
            f"Python {version or '?'} does not qualify. "
            "Pass `--python 3.14` (or another stable line) and let setup use pyenv."
        )


def find_pyenv() -> Path | None:
    """Locate pyenv from ``PYENV_ROOT``, the usual install, or ``PATH``."""
    roots = []
    configured = os.environ.get("PYENV_ROOT")
    if configured:
        roots.append(Path(configured).expanduser())
    roots.append(Path.home() / ".pyenv")
    relatives = (
        "pyenv-win/bin/pyenv.bat",
        "pyenv-win/bin/pyenv.exe",
        "bin/pyenv",
        "bin/pyenv.exe",
    )
    for root in roots:
        for relative in relatives:
            candidate = root / relative
            if candidate.is_file():
                return candidate
    found = shutil.which("pyenv")
    return Path(found) if found else None


def venv_python(venv: Path, windows: bool | None = None) -> Path:
    """The interpreter path inside a virtual environment."""
    if windows is None:
        windows = os.name == "nt"
    return Path(venv) / ("Scripts/python.exe" if windows else "bin/python")


def project_venv_python(root: Path | None = None) -> Path | None:
    """The interpreter of an existing project virtual environment."""
    configured = os.environ.get("MARIANA_VENV")
    if configured:
        python = venv_python(Path(configured).expanduser())
        if python.is_file():
            return python
    base = Path(root or repository_root())
    for name in (DEFAULT_VENV, "venv"):
        python = venv_python(base / name)
        if python.is_file():
            return python
    return None


def reexec_in_venv(module: bool = False, argv: list[str] | None = None) -> None:
    """Replace this process with the project venv interpreter when appropriate.

    ``python mariana.py ...`` from the system Python would otherwise make Studio's
    dependency installer call the system pip, which PEP 668 distributions reject.
    The child is marked with ``MARIANA_VENV_ACTIVE`` so this never loops, and
    ``MARIANA_NO_VENV=1`` keeps the current interpreter.
    """
    if os.environ.get("MARIANA_VENV_ACTIVE") == "1":
        return
    disabled = os.environ.get("MARIANA_NO_VENV", "").strip().lower()
    if disabled in {"1", "true", "yes", "on"}:
        return
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return  # already inside a virtual environment; respect it

    python = project_venv_python()
    if python is None:
        return
    if Path(sys.executable).resolve() == python.resolve():
        return

    arguments = [str(python)]
    if module:
        arguments += ["-m", "mariana"]
    else:
        arguments.append(sys.argv[0])
    arguments += list(sys.argv[1:] if argv is None else argv)
    environment = {**os.environ, "MARIANA_VENV_ACTIVE": "1"}
    try:
        os.execve(str(python), arguments, environment)
    except OSError as error:
        print(f"Could not start the project virtual environment ({error}); "
              f"continuing with {sys.executable}.")


def can_recreate(venv: Path) -> bool:
    """Only delete a directory that really is a virtual environment."""
    return Path(venv).is_dir() and (Path(venv) / "pyvenv.cfg").is_file()


def venv_version(venv: Path) -> str | None:
    """The CPython version recorded in ``pyvenv.cfg``."""
    try:
        text = (Path(venv) / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for key in ("version", "version_info"):
        for line in text.splitlines():
            name, _, value = line.partition("=")
            if name.strip() != key:
                continue
            match = _VERSION_FIELD.match(value.strip())
            if match:
                return match.group(1)
    return None


def requirements_for(device: str) -> str:
    """The requirements file that matches a device profile."""
    try:
        return DEVICE_REQUIREMENTS[device]
    except KeyError:
        raise SetupError(
            f"Unknown device profile {device!r}; choose auto, cpu, cuda or xpu."
        ) from None


def parse_extras(value: str) -> list[str]:
    """Parse ``data,export`` (or ``all``) into a validated, stable order."""
    items = [part.strip().lower() for part in str(value or "").split(",") if part.strip()]
    if not items:
        return []
    if "all" in items:
        return list(EXTRAS)
    unknown = [item for item in items if item not in EXTRAS]
    if unknown:
        raise SetupError(
            f"Unknown optional group(s): {', '.join(unknown)}. "
            f"Choose from {', '.join(EXTRAS)}, or use `--extras all`."
        )
    return [name for name in EXTRAS if name in items]


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------
def _tail(output: str, limit: int = 12) -> str:
    lines = [line.strip() for line in str(output or "").splitlines() if line.strip()]
    return "\n".join(lines[-limit:])


def _capture(command: list[str]) -> tuple[int, str]:
    """Run a short command and return ``(exit_code, combined_output)``."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=COMMAND_TIMEOUT
        )
    except FileNotFoundError:
        raise SetupError(
            f"{command[0]} is not installed; install it and run setup again."
        ) from None
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SetupError(f"Could not run {command[0]}: {error}") from None
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return result.returncode, output


def _stream(command: list[str], what: str) -> None:
    """Run a long command with the user's terminal attached."""
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        raise SetupError(
            f"{what} needs {command[0]}, which is not installed; install it and try again."
        ) from None
    except subprocess.CalledProcessError as error:
        raise SetupError(
            f"{what} failed (exit code {error.returncode}); the error is printed above."
        ) from None
    except OSError as error:
        raise SetupError(f"{what} could not start: {error}") from None


# ---------------------------------------------------------------------------
# pyenv
# ---------------------------------------------------------------------------
def _pyenv_command(pyenv: Path, arguments: list[str]) -> list[str]:
    """Build a pyenv invocation, going through cmd.exe for pyenv-win."""
    if Path(pyenv).suffix.lower() in {".bat", ".cmd"}:
        return ["cmd", "/c", str(pyenv), *arguments]
    return [str(pyenv), *arguments]


def _numeric_versions(output: str) -> list[str]:
    """Keep the plain CPython versions in a pyenv listing."""
    versions = []
    for line in str(output or "").splitlines():
        value = line.strip()
        if version_tuple(value) is not None:
            versions.append(value)
    return versions


def installed_versions(pyenv: Path) -> list[str]:
    """CPython versions that pyenv already manages."""
    code, output = _capture(_pyenv_command(Path(pyenv), ["versions", "--bare"]))
    if code != 0:
        raise SetupError(
            f"Could not list the pyenv versions:\n{_tail(output)}\n"
            "Run `pyenv versions` by hand, then try setup again."
        )
    return _numeric_versions(output)


def available_versions(pyenv: Path) -> list[str]:
    """CPython versions that pyenv knows how to install."""
    code, output = _capture(_pyenv_command(Path(pyenv), ["install", "--list"]))
    if code != 0:
        raise SetupError(
            f"Could not ask pyenv which Pythons it can install:\n{_tail(output)}\n"
            "Update pyenv (git pull in its checkout) and try again."
        )
    return _numeric_versions(output)


def ensure_pyenv_python(pyenv: Path, requested: str) -> str:
    """Return an installed pyenv version, installing the newest patch if needed."""
    chosen = best_version(installed_versions(pyenv), requested)
    if chosen:
        return chosen

    candidate = best_version(available_versions(pyenv), requested)
    if candidate is None:
        raise SetupError(
            f"pyenv has no CPython {requested} to install. "
            "Update pyenv (git pull in its checkout) or pass another version "
            "such as `--python 3.13`."
        )

    print(f"CPython {candidate} is not installed; asking pyenv to install it "
          "(this compiles CPython and can take a few minutes)…")
    arguments = ["install"]
    if Path(pyenv).suffix.lower() in {".bat", ".cmd"}:
        arguments.append("-q")  # pyenv-win otherwise opens an installer wizard
    arguments.append(candidate)
    _stream(_pyenv_command(Path(pyenv), arguments),
            f"Installing CPython {candidate} with pyenv")

    chosen = best_version(installed_versions(pyenv), requested)
    if chosen is None:
        raise SetupError(
            f"pyenv reported success but CPython {requested} is still missing. "
            "Check the pyenv output above and run `pyenv install <version>` by hand."
        )
    return chosen


def _pyenv_prefix(pyenv: Path, version: str) -> Path:
    """Where pyenv placed one interpreter."""
    code, output = _capture(_pyenv_command(Path(pyenv), ["prefix", version]))
    if code == 0:
        for line in reversed(output.strip().splitlines()):
            candidate = Path(line.strip())
            if candidate.is_dir():
                return candidate

    code, output = _capture(_pyenv_command(Path(pyenv), ["root"]))
    if code == 0 and output.strip():
        root = Path(output.strip().splitlines()[-1].strip())
        candidate = root / "versions" / version
        if candidate.is_dir():
            return candidate

    roots = []
    if os.environ.get("PYENV_ROOT"):
        roots.append(Path(os.environ["PYENV_ROOT"]).expanduser())
    if Path(pyenv).parent.name == "bin":
        # Unix: <root>/bin/pyenv; pyenv-win: <root>/pyenv-win/bin/pyenv.bat.
        roots.append(Path(pyenv).parent.parent)
    for root in roots:
        candidate = root / "versions" / version
        if candidate.is_dir():
            return candidate

    raise SetupError(
        f"pyenv could not report where CPython {version} lives. "
        f"Run `pyenv prefix {version}` by hand and then re-run setup."
    )


def pyenv_interpreter(pyenv: Path, version: str, windows: bool | None = None) -> Path:
    """The ``python`` executable of a pyenv-managed version."""
    if windows is None:
        windows = os.name == "nt"
    prefix = _pyenv_prefix(Path(pyenv), version)
    candidate = prefix / ("python.exe" if windows else "bin/python")
    if not candidate.is_file():
        raise SetupError(
            f"pyenv installed CPython {version} but {candidate} is missing; "
            f"run `pyenv install {version}` again."
        )
    return candidate


def install_pyenv(target: Path | None = None, windows: bool | None = None) -> Path:
    """Clone pyenv into ``target`` (or the usual location) when it is absent."""
    if windows is None:
        windows = os.name == "nt"
    root = Path(target).expanduser() if target else Path(
        os.environ.get("PYENV_ROOT") or (Path.home() / ".pyenv")
    )
    executable = (root / "pyenv-win" / "bin" / "pyenv.bat") if windows \
        else (root / "bin" / "pyenv")
    if executable.is_file():
        return executable

    if shutil.which("git") is None:
        raise SetupError(
            "Installing pyenv needs git; install git and run "
            "`python mariana.py setup --install-pyenv` again."
        )
    if root.exists() and not root.is_dir():
        raise SetupError(
            f"{root} exists and is not a directory; move it away or set "
            "PYENV_ROOT to another path."
        )
    if root.is_dir() and any(root.iterdir()):
        raise SetupError(
            f"{root} already contains files but no pyenv. Move it away, "
            "or set PYENV_ROOT to an empty directory and try again."
        )

    print(f"Installing pyenv into {root}…")
    repository = PYENV_WIN_REPOSITORY if windows else PYENV_REPOSITORY
    _stream(["git", "clone", "--depth", "1", repository, str(root)], "Installing pyenv")
    if not executable.is_file():
        raise SetupError(
            f"pyenv was cloned into {root} but {executable} is missing. "
            "Remove the directory and try again, or install pyenv by hand."
        )
    return executable


# ---------------------------------------------------------------------------
# System interpreters and the virtual environment
# ---------------------------------------------------------------------------
def interpreter_version(python: Path) -> str | None:
    """Ask an interpreter for its ``major.minor.patch`` version."""
    code, output = _capture([
        str(python), "-c", "import sys; print('%d.%d.%d' % sys.version_info[:3])",
    ])
    if code != 0:
        return None
    for line in reversed(output.strip().splitlines()):
        if version_tuple(line.strip()) is not None:
            return line.strip()
    return None


def system_interpreter(requested: str) -> tuple[Path, str]:
    """Find a matching interpreter without pyenv, or explain how to get one."""
    parts = str(requested).split(".")
    names = []
    if len(parts) >= 2:
        names.append(f"python{parts[0]}.{parts[1]}")
    names.extend(["python3", "python"])

    candidates: list[Path] = []
    for name in names:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    if sys.executable:
        candidates.append(Path(sys.executable))

    seen: set[str] = set()
    available = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        version = interpreter_version(candidate)
        if version is None:
            continue
        available.append(version)
        if version_matches(version, requested):
            return candidate, version

    found = ", ".join(available) or "none"
    raise SetupError(
        f"Requested Python {requested}, but the available interpreters are {found}. "
        "Install a matching CPython with pyenv "
        "(`python mariana.py setup --install-pyenv --python 3.14`), "
        "or choose an installed version with `--python 3.13`."
    )


def _is_inside(path: Path, folder: Path) -> bool:
    """True when ``path`` is inside ``folder`` after resolving symlinks."""
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
    except (OSError, ValueError):
        return False
    return True


def create_venv(interpreter: Path, venv: Path,
                recreate: bool = False, windows: bool | None = None,
                expected_version: str | None = None) -> Path:
    """Create (or reuse) the project virtual environment and return its python."""
    venv = Path(venv)
    python = venv_python(venv, windows)

    if recreate and _is_inside(sys.executable, venv):
        raise SetupError(
            f"The interpreter running setup lives inside {venv}, so `--recreate` "
            "would delete the environment being used. Run setup from the system "
            "or pyenv Python (deactivate the venv first)."
        )

    if venv.exists() and recreate:
        if not can_recreate(venv):
            raise SetupError(
                f"{venv} does not look like a virtual environment, so setup will "
                "not delete it. Remove it yourself or pass another `--venv` path."
            )
        print(f"Removing the existing virtual environment at {venv}…")
        shutil.rmtree(venv)

    if venv.exists() and not venv.is_dir():
        raise SetupError(
            f"{venv} exists and is not a directory; remove it or choose another "
            "`--venv` path."
        )

    if python.is_file():
        existing = venv_version(venv)
        if expected_version and existing and \
                version_tuple(existing)[:2] != version_tuple(expected_version)[:2]:
            raise SetupError(
                f"{venv} was created with Python {existing}, but setup selected "
                f"{expected_version}. Pass `--recreate` to rebuild it with the "
                "selected interpreter."
            )
        print(f"Reusing the virtual environment at {venv} "
              "(pass `--recreate` to rebuild it).")
        return python

    if venv.is_dir() and any(venv.iterdir()):
        raise SetupError(
            f"{venv} exists but has no interpreter at {python}. "
            "Pass `--recreate` to rebuild it, or choose another `--venv` path."
        )

    print(f"Creating a virtual environment at {venv}…")
    try:
        _stream([str(interpreter), "-m", "venv", str(venv)],
                "Creating the virtual environment")
    except SetupError:
        raise SetupError(
            f"Could not create a virtual environment with {interpreter}.\n"
            "  Debian/Ubuntu: sudo apt install python3-venv\n"
            "  Or let setup manage CPython: "
            "python mariana.py setup --install-pyenv"
        ) from None

    if not python.is_file():
        raise SetupError(
            f"The virtual environment at {venv} has no interpreter at {python}; "
            "remove it and run setup again."
        )
    return python


def upgrade_pip(python: Path) -> None:
    """Best-effort pip upgrade; an old pip is still usable."""
    print("Upgrading pip inside the virtual environment…")
    try:
        _stream([
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--upgrade", "pip",
        ], "Upgrading pip")
    except SetupError as error:
        print(f"Warning: {error} Continuing with the bundled pip.")


def install_requirements(python: Path, requirements: Path) -> None:
    """Install one fixed requirements file inside the virtual environment."""
    requirements = Path(requirements)
    if not requirements.is_file():
        raise SetupError(
            f"{requirements} is missing from the checkout; re-clone the repository "
            "or restore the file, then run setup again."
        )
    print(f"Installing {requirements.name}…")
    _stream([
        str(python), "-m", "pip", "install", "--disable-pip-version-check",
        "--no-input", "-r", str(requirements),
    ], f"Installing {requirements.name}")


def verify_environment(python: Path) -> dict:
    """Import PyTorch and NumPy inside the venv and report the available devices."""
    code, output = _capture([str(python), "-c", VERIFY_SCRIPT])
    if code != 0:
        raise SetupError(
            f"The packages installed, but PyTorch could not be imported with {python}:\n"
            f"{_tail(output)}\n"
            "Check the wheel support for this platform and Python version."
        )
    result = {}
    for line in reversed(output.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                result = json.loads(stripped)
            except ValueError:
                result = {}
            break
    return result if isinstance(result, dict) else {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The ``mariana setup`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="mariana setup",
        description=(
            "Create a pyenv-managed CPython and a project virtual environment, "
            "then install the core runtime inside it. The system interpreter is "
            "never modified."
        ),
    )
    parser.add_argument(
        "--python", metavar="VERSION", default=None,
        help="CPython line or exact version, e.g. 3.14 or 3.14.2 "
             "(default: the interpreter running this command)",
    )
    parser.add_argument(
        "--venv", metavar="PATH", default=None,
        help=f"virtual environment directory (default: {DEFAULT_VENV} in the checkout)",
    )
    parser.add_argument(
        "--device", choices=tuple(DEVICE_REQUIREMENTS), default="auto",
        help="requirements profile: auto (GPU-first, CPU fallback), cpu, cuda or xpu",
    )
    parser.add_argument(
        "--extras", metavar="LIST", default="",
        help="optional groups to install: data,export,finetune (or all)",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="delete and rebuild an existing virtual environment",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--install-pyenv", action="store_true",
        help="clone pyenv into PYENV_ROOT or ~/.pyenv when it is missing",
    )
    group.add_argument(
        "--no-pyenv", action="store_true",
        help="ignore pyenv and use an installed interpreter",
    )
    return parser


def _choose_interpreter(requested: str, install_missing_pyenv: bool,
                        no_pyenv: bool) -> tuple[Path, str, str]:
    """Return ``(python, version, provider)`` following the pyenv policy."""
    pyenv = None if no_pyenv else find_pyenv()
    if pyenv is None and install_missing_pyenv:
        pyenv = install_pyenv()

    if pyenv is not None:
        version = ensure_pyenv_python(pyenv, requested)
        return pyenv_interpreter(pyenv, version), version, f"pyenv {version}"

    python, version = system_interpreter(requested)
    note = " (pyenv was not found; run setup with `--install-pyenv` for a managed interpreter)"
    return python, version, f"system Python {version}{note}"


def _print_ready(venv: Path, details: dict, failed_extras: list[str]) -> None:
    windows = os.name == "nt"
    activate = venv / ("Scripts/activate" if windows else "bin/activate")
    python = venv_python(venv, windows)
    devices = [name for name in ("cuda", "xpu", "mps") if details.get(name)]
    runtime = details.get("torch")
    numpy = details.get("numpy")

    print()
    if runtime and numpy:
        where = ", ".join(devices) if devices else "cpu"
        print(f"Verified: torch {runtime} ({where}), numpy {numpy}")
    else:
        print("Verified: PyTorch and NumPy import inside the virtual environment")
    if failed_extras:
        print("Optional groups skipped: " + ", ".join(failed_extras) +
              " (fix the error above and re-run setup with `--extras`).")

    print()
    print("Mariana is ready. The wrapper uses the project .venv automatically:")
    print("    python mariana.py studio")
    print("    python -m mariana train --steps 100")
    print()
    print("Activate it only for direct tools such as pip, pytest or pyflakes:")
    if windows:
        print(f"    {activate}       (cmd)")
        print(f"    {venv / 'Scripts/Activate.ps1'}  (PowerShell)")
    else:
        print(f"    source {activate}")
    print(f"    {python} -m pytest -q                 # without activation")


def run(args: argparse.Namespace) -> int:
    """Execute the selected setup plan; returns a process exit code."""
    root = repository_root()
    venv = Path(args.venv).expanduser() if args.venv else root / DEFAULT_VENV
    if not venv.is_absolute():
        venv = (Path.cwd() / venv).resolve()
    extras = parse_extras(args.extras)
    requested = args.python or default_python()
    require_supported(requested)

    print("Mariana setup")
    print(f"  project  {root}")
    print(f"  venv     {venv}")
    print(f"  python   {requested} (requested)")
    print(f"  profile  {requirements_for(args.device)}")

    interpreter, version, provider = _choose_interpreter(
        requested, args.install_pyenv, args.no_pyenv,
    )
    require_supported(version)
    if version_tuple(version)[:2] > KNOWN_GOOD_MAX:
        print(f"Warning: Python {version} is newer than the newest tested line "
              f"({KNOWN_GOOD_MAX[0]}.{KNOWN_GOOD_MAX[1]}); pip needs matching wheels.")
    print(f"Using {provider} at {interpreter}")

    python = create_venv(interpreter, venv, recreate=args.recreate,
                         expected_version=version)
    upgrade_pip(python)
    install_requirements(python, root / requirements_for(args.device))

    failed_extras = []
    for extra in extras:
        try:
            install_requirements(python, root / EXTRAS[extra])
        except SetupError as error:
            print(f"Warning: optional group {extra!r} was not installed: {error}")
            failed_extras.append(extra)

    details = verify_environment(python)
    _print_ready(venv, details, failed_extras)
    return 0


def main(argv: list[str] | None = None) -> int:
    """``mariana setup``; returns a process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except SetupError as error:
        print(f"Setup failed: {error}")
        return 1
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 130


if __name__ == "__main__":  # pragma: no cover - the wrappers call main()
    sys.exit(main())
