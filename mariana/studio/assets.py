"""Static web assets: the assembled single-page document and its files.

``index.html`` is the shell; each workspace lives in ``views/<name>.html`` and
is spliced in at ``<!-- include views/<name>.html -->`` when the page is served.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import threading
from dataclasses import dataclass
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
    ".wasm": "application/wasm",
}
# No inline scripts or styles, nothing off-origin, no framing, no plugin content.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; connect-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
    "script-src 'self' 'wasm-unsafe-eval'; font-src 'self'; object-src 'none'; base-uri 'none'; "
    "form-action 'self'; frame-ancestors 'none'"
)
_INCLUDE = re.compile(r"[ \t]*<!--\s*include\s+([a-z0-9/_-]+\.html)\s*-->")


@dataclass(frozen=True)
class AssetEntry:
    """One immutable static response with its validators and compressed form."""

    body: bytes
    content_type: str
    etag: str
    modified: float
    gzip: bytes | None = None


_cache_lock = threading.RLock()
_file_cache: dict[str, tuple[tuple[int, int], AssetEntry]] = {}
_index_cache: tuple[tuple, str, AssetEntry] | None = None


def _index_files() -> list[Path]:
    return sorted([WEB / "index.html", *WEB.joinpath("views").rglob("*.html")])


def _index_fingerprint() -> tuple:
    fingerprint = []
    for path in _index_files():
        try:
            stat = path.stat()
        except OSError:
            continue
        fingerprint.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(fingerprint)


def _build_entry(body: bytes, content_type: str, modified: float) -> AssetEntry:
    etag = 'W/"' + hashlib.blake2b(body, digest_size=12).hexdigest() + '"'
    compressible = (
        len(body) >= 1024
        and (content_type.startswith("text/")
             or any(kind in content_type for kind in ("json", "javascript", "svg", "wasm")))
    )
    compressed = gzip.compress(body, compresslevel=6, mtime=0) if compressible else None
    return AssetEntry(body, content_type, etag, modified, compressed)


def render_index() -> str:
    """The complete HTML document with every view spliced in (cached by mtime)."""
    global _index_cache
    fingerprint = _index_fingerprint()
    cached = _index_cache
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    shell = (WEB / "index.html").read_text(encoding="utf-8")

    def include(match: re.Match) -> str:
        path = (WEB / match.group(1)).resolve()
        if WEB not in path.parents or not path.is_file():
            return f"<!-- missing {match.group(1)} -->"
        return path.read_text(encoding="utf-8").rstrip("\n")

    document = _INCLUDE.sub(include, shell)
    with _cache_lock:
        _index_cache = (fingerprint, document, _build_entry(
            document.encode("utf-8"), CONTENT_TYPES[".html"],
            max((path.stat().st_mtime for path in _index_files()
                 if path.is_file()), default=0.0),
        ))
    return document


def read_asset_entry(url_path: str) -> AssetEntry | None:
    """A revalidatable static asset, cached until the source file changes."""
    if url_path in {"/", "/index.html"}:
        render_index()  # fills _index_cache
        return _index_cache[2] if _index_cache else None
    relative = url_path.lstrip("/")
    if not relative or relative.startswith("views/"):
        return None  # views are only served as part of the assembled page
    path = (WEB / relative).resolve()
    try:
        relative = path.relative_to(WEB).as_posix()  # URL paths always use '/'
    except ValueError:
        return None
    if relative.startswith("views/") or not path.is_file() or path.suffix not in CONTENT_TYPES:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    fingerprint = (stat.st_mtime_ns, stat.st_size)
    cached = _file_cache.get(relative)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    entry = _build_entry(path.read_bytes(), CONTENT_TYPES[path.suffix], stat.st_mtime)
    with _cache_lock:
        _file_cache[relative] = (fingerprint, entry)
    return entry


def read_asset(url_path: str) -> tuple[bytes, str] | None:
    """Return ``(bytes, content type)`` for a served file, or ``None`` if it is not one."""
    entry = read_asset_entry(url_path)
    return (entry.body, entry.content_type) if entry else None


def clear_asset_cache() -> None:
    """Drop cached bytes; used by tests and after a development edit."""
    global _index_cache
    with _cache_lock:
        _file_cache.clear()
        _index_cache = None
