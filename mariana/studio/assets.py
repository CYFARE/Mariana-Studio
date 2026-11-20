"""Static web assets: the assembled single-page document and its files.

``index.html`` is the shell; each workspace lives in ``views/<name>.html`` and
is spliced in at ``<!-- include views/<name>.html -->`` when the page is served.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".woff2": "font/woff2",
    ".md": "text/markdown; charset=utf-8",
}
# No inline scripts or styles, nothing off-origin, no framing, no plugin content.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
    "script-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
_INCLUDE = re.compile(r"[ \t]*<!--\s*include\s+([a-z0-9/_-]+\.html)\s*-->")


def render_index() -> str:
    """The complete HTML document with every view spliced in."""
    shell = (WEB / "index.html").read_text(encoding="utf-8")

    def include(match: re.Match) -> str:
        path = (WEB / match.group(1)).resolve()
        if WEB not in path.parents or not path.is_file():
            return f"<!-- missing {match.group(1)} -->"
        return path.read_text(encoding="utf-8").rstrip("\n")

    return _INCLUDE.sub(include, shell)


def read_asset(url_path: str) -> tuple[bytes, str] | None:
    """Return ``(bytes, content type)`` for a served file, or ``None`` if it is not one."""
    if url_path in {"/", "/index.html"}:
        return render_index().encode("utf-8"), CONTENT_TYPES[".html"]
    relative = url_path.lstrip("/")
    if not relative or relative.startswith("views/"):
        return None  # views are only served as part of the assembled page
    path = (WEB / relative).resolve()
    try:
        relative = path.relative_to(WEB)
    except ValueError:
        return None
    if relative.parts[0] == "views" or not path.is_file() or path.suffix not in CONTENT_TYPES:
        return None
    return path.read_bytes(), CONTENT_TYPES[path.suffix]
