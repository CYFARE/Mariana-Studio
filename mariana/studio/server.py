"""The Studio HTTP server: transport, static files and request security.

Everything specific to HTTP lives here: parsing, the security checks that keep
other web pages from driving a local Studio, writing JSON / text / server-sent
events, and mapping exceptions to the one error shape.  Behaviour lives in the
services reached through ``api/``.
"""

from __future__ import annotations

import atexit
import json
import signal
import sys
import threading
import traceback
import webbrowser
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .api import router
from .app import NAME, Studio
from .assets import CONTENT_SECURITY_POLICY, read_asset_entry
from .errors import (
    STATUS_CODES,
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    StudioError,
    TooLargeError,
)
from .http import HARD_BODY_LIMIT, JSON_TYPE, EventStream, Request, Response

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def _hostname(authority: str) -> str:
    """The host part of ``host:port``, ``[v6]:port`` or a bare host."""
    authority = authority.strip().lower()
    if authority.startswith("["):
        return authority[1:authority.find("]")] if "]" in authority else authority
    return authority.rsplit(":", 1)[0] if authority.count(":") == 1 else authority


def make_handler(app: Studio, bind_host: str):
    allowed_hosts = LOOPBACK_HOSTS | {bind_host.lower()}
    exposed = bind_host in WILDCARD_HOSTS  # the user chose to listen on every interface

    class Handler(BaseHTTPRequestHandler):
        server_version = "MarianaStudio"
        sys_version = ""
        # Keep-alive amortises the connection setup across the many small
        # scripts and stylesheets the page loads.
        protocol_version = "HTTP/1.1"

        # -- security ----------------------------------------------------
        def _check_origin(self, method: str) -> None:
            """Refuse requests another web page could have triggered.

            A page on any site can make the browser send requests to
            ``127.0.0.1``.  Requiring a loopback ``Host`` defeats DNS rebinding;
            requiring a same-origin ``Origin`` and a JSON body on writes defeats
            cross-site request forgery (a JSON content type forces a preflight
            that this server never approves).
            """
            host = self.headers.get("Host")
            if host and not exposed and _hostname(host) not in allowed_hosts:
                raise ForbiddenError(
                    "This Studio only answers requests addressed to localhost"
                )
            origin = self.headers.get("Origin")
            if origin and origin != "null":
                parsed = urlparse(origin)
                if not host or parsed.netloc.lower() != host.lower():
                    raise ForbiddenError("Cross-origin requests are not allowed")
            elif origin == "null" and method in WRITE_METHODS:
                raise ForbiddenError("Cross-origin requests are not allowed")
            if method in WRITE_METHODS:
                if self.headers.get("Sec-Fetch-Site", "same-origin") == "cross-site":
                    raise ForbiddenError("Cross-origin requests are not allowed")
                length = self.headers.get("Content-Length")
                content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if length not in (None, "0") and content_type != "application/json":
                    raise BadRequestError("Send JSON with Content-Type: application/json")

        # -- reading -----------------------------------------------------
        def _read_body(self) -> bytes:
            try:
                length = int(self.headers.get("Content-Length", "0") or 0)
            except ValueError:
                raise BadRequestError("Invalid content length") from None
            if length < 0:
                raise BadRequestError("Invalid content length")
            if length > HARD_BODY_LIMIT:
                raise TooLargeError("Request is too large")
            return self.rfile.read(length) if length else b""

        # -- writing -----------------------------------------------------
        def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in {**BASE_HEADERS, **(headers or {})}.items():
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_error(self, error: StudioError) -> None:
            payload = json.dumps(error.payload(), ensure_ascii=False).encode("utf-8")
            self.close_connection = True
            self._send(error.status, payload, JSON_TYPE,
                       {"Cache-Control": "no-store", "Connection": "close"})

        def _send_not_modified(self, content_type: str, headers: dict) -> None:
            """A bodyless 304 that still carries the validators a client needs."""
            self.send_response(304)
            self.send_header("Content-Type", content_type)
            for name, value in {**BASE_HEADERS, **headers}.items():
                self.send_header(name, value)
            self.end_headers()

        @staticmethod
        def _etag_matches(header: str | None, etag: str) -> bool:
            if not header:
                return False
            values = [value.strip() for value in header.split(",")]
            if "*" in values:
                return True
            wanted = etag.removeprefix("W/")
            return any(value.removeprefix("W/") == wanted for value in values)

        def _send_stream(self, stream: EventStream) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-transform")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.close_connection = True
            for name, value in BASE_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            events = iter(stream.events)
            try:
                for event in events:
                    if event is None:
                        chunk = ": keep-alive\n\n"
                    elif isinstance(event, str):
                        chunk = f"data: {event}\n\n"
                    else:
                        chunk = f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                    self.wfile.write(chunk.encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass  # the client went away; stop producing
            finally:
                close = getattr(events, "close", None)
                if close:
                    close()

        # -- dispatch ----------------------------------------------------
        def _handle(self, method: str) -> None:
            try:
                self._check_origin(method)
                path = urlparse(self.path).path
                if method in {"GET", "HEAD"} and not path.startswith(("/api/", "/v1/")):
                    asset = read_asset_entry(path)
                    if asset is None:
                        raise NotFoundError("Not found")
                    headers = {
                        "Cache-Control": "no-cache",
                        "ETag": asset.etag,
                        "Last-Modified": formatdate(asset.modified, usegmt=True),
                        "Vary": "Accept-Encoding",
                    }
                    if asset.content_type.startswith("text/html"):
                        headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
                    if self._etag_matches(self.headers.get("If-None-Match"), asset.etag):
                        return self._send_not_modified(asset.content_type, headers)
                    compressed = (asset.gzip is not None
                                  and "gzip" in (self.headers.get("Accept-Encoding") or "").lower())
                    if compressed:
                        headers["Content-Encoding"] = "gzip"
                    return self._send(200, asset.gzip if compressed else asset.body,
                                      asset.content_type, headers)
                handler, params = router.resolve("GET" if method == "HEAD" else method, path)
                request = Request(method, self.path, self.headers, self._read_body(), params)
                result = handler(app, request)
                if isinstance(result, EventStream):
                    return self._send_stream(result)
                if not isinstance(result, Response):
                    result = Response.json(result)
                headers = {"Cache-Control": "no-store", **result.headers}
                if result.content_type.startswith("text/"):
                    headers.setdefault("Content-Security-Policy", "default-src 'none'")
                self._send(result.status, result.body, result.content_type, headers)
            except StudioError as error:
                self._send_error(error)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except (ValueError, OSError) as error:
                # Services signal bad input with ValueError; unreadable paths raise OSError.
                self._send_error(BadRequestError(str(error) or type(error).__name__))
            except Exception as error:  # noqa: BLE001 - last resort; never leak a traceback
                traceback.print_exc(file=sys.stderr)
                self._send_error(StudioError(
                    f"Internal error: {type(error).__name__}: {error}",
                    status=500, code=STATUS_CODES[500],
                ))

        def do_GET(self):  # noqa: N802 - http.server naming
            self._handle("GET")

        def do_HEAD(self):  # noqa: N802
            self._handle("HEAD")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def do_PUT(self):  # noqa: N802
            self._handle("PUT")

        def do_PATCH(self):  # noqa: N802
            self._handle("PATCH")

        def do_DELETE(self):  # noqa: N802
            self._handle("DELETE")

        def do_OPTIONS(self):  # noqa: N802
            # No CORS: a preflight is never approved, so cross-origin writes cannot proceed.
            self._send_error(ForbiddenError("Cross-origin requests are not allowed"))

        def log_message(self, format, *args):  # noqa: A002 - http.server signature
            return

    return Handler


class StudioServer(ThreadingHTTPServer):
    """A threading HTTP server that owns a :class:`Studio` application."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, app: Studio):
        self.app = app
        super().__init__(address, make_handler(app, address[0]))
        app.port = self.server_address[1]
        # Short names kept for callers and tests that reach into the services.
        self.manager = app.training
        self.gateway = app.gateway
        self.agents = app.agents

    def server_close(self):
        try:
            self.app.close()
        finally:
            super().server_close()


def make_server(host: str = "127.0.0.1", port: int = 8765, root=None) -> StudioServer:
    """Create (but do not start) a Studio server rooted at ``root``."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    return StudioServer((host, port), Studio(root))


def run(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    """Serve Studio until interrupted, releasing models and runtimes on every exit path."""
    server = make_server(host, port)
    server.app.updates.start()  # automatic release check after boot; never blocks
    url = f"http://{host if host not in WILDCARD_HOSTS else '127.0.0.1'}:{server.server_address[1]}"
    print(f"{NAME}: {url}")
    if host in WILDCARD_HOSTS:
        print("Warning: listening on every network interface. Studio has no authentication; "
              "anyone who can reach this port can run training jobs and agents.")
    print("Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    stopping = threading.Event()

    def request_stop(*_args) -> None:
        if stopping.is_set():
            return
        stopping.set()
        # shutdown() waits for serve_forever(), so it must run outside this thread.
        threading.Thread(target=server.shutdown, name="studio-shutdown", daemon=True).start()

    previous: dict[int, object] = {}
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous[number] = signal.signal(number, request_stop)
        except (ValueError, OSError):
            pass  # not running in the main thread
    # Covers interpreter exits that bypass serve_forever's finally block.
    close_server = server.server_close
    atexit.register(close_server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        atexit.unregister(close_server)
        for number, handler in previous.items():
            try:
                signal.signal(number, handler)
            except (ValueError, OSError, TypeError):
                pass
        server.shutdown()
        server.server_close()
