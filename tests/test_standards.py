"""Executable checks for the repository's naming and backend engineering standard."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from mariana.studio.api import router
from tests.helpers import ROOT


MARKDOWN_LIST_MARKER = re.compile(r"^ {0,3}(?:[-+*]|[0-9]{1,9}[.)])[ \t]")


def source_for_test(path: Path) -> Path | None:
    if path.name in {"__init__.py", "helpers.py", "conftest.py", "test_standards.py"}:
        return None
    match = re.fullmatch(r"test_(.+)\.py", path.name)
    if not match:
        return Path("__nonstandard__")
    parts = match.group(1).split("_")
    base = ROOT / "scripts" if parts[0] == "scripts" else ROOT / "mariana"
    if parts[0] == "scripts":
        parts = parts[1:]
    for split in range(len(parts)):
        candidate = base.joinpath(*parts[:split], "_".join(parts[split:])).with_suffix(".py")
        if candidate.is_file():
            return candidate
    for count in range(len(parts), -1, -1):
        candidate = base.joinpath(*parts[:count])
        if candidate.is_dir() and count == len(parts):
            return candidate
    return Path("__missing__")


def test_every_test_file_maps_to_one_source_path():
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = source_for_test(path)
        if source is None:
            continue
        assert source.exists(), f"{path.name} does not map to a source path"


def test_tests_use_pytest_without_path_hacks_or_script_runners():
    for path in (ROOT / "tests").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert ("sys." + "path") not in text, path.name
        assert ("__name__" + ' == "__main__"') not in text, path.name


def test_every_route_has_a_summary_and_uses_the_error_contract():
    assert len(router.routes) >= 70
    for route in router.routes:
        assert (route.handler.__doc__ or "").strip(), f"{route.method} {route.pattern}"
        assert route.pattern.startswith(("/api/", "/v1/"))
        assert re.fullmatch(r"/[a-z0-9/{}/_-]+", route.pattern), route.pattern


def test_core_model_modules_do_not_depend_on_studio_or_agents():
    for name in ("config", "data", "model", "train", "generate", "latent", "saint", "axiom"):
        path = ROOT / "mariana" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any("studio" in name.split(".") or "agents" in name.split(".") for name in names), (path.name, names)


def test_engineering_reference_covers_the_checkable_rules():
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for required in ("snake_case", "PascalCase", "StudioError", "JobRegistry", "127.0.0.1", "Content-Security-Policy", "python -m pytest", "pyflakes"):
        assert required in text
    assert "studio.validation.number" in text
    assert "tests/helpers.py" in text


def test_docs_math_blocks_avoid_markdown_list_markers():
    for path in sorted((ROOT / "docs").rglob("*.md")):
        in_math = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip() == "$$":
                in_math = not in_math
            elif in_math:
                assert not MARKDOWN_LIST_MARKER.match(line), (
                    f"{path.relative_to(ROOT)}:{number}: a line inside $$ math starts with a "
                    "Markdown list marker, which breaks GitHub's math rendering; end the "
                    "previous line with the operator instead"
                )
