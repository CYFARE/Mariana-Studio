"""Transport-neutral request, response and routing primitives.

Handlers are plain functions ``handler(app, request)`` that return

* a ``dict`` or ``list``           -> ``200`` JSON,
* a :class:`Response`              -> anything else (status, text, bytes),
* an :class:`EventStream`          -> server-sent events.

They never touch sockets, so they can be called directly from tests.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from urllib.parse import parse_qs, unquote, urlparse

from .errors import BadRequestError, MethodNotAllowedError, NotFoundError, TooLargeError

MAX_BODY_BYTES = 1_000_000          # default limit for a JSON request body
HARD_BODY_LIMIT = 16_000_000        # the transport never reads more than this
JSON_TYPE = "application/json; charset=utf-8"


class Request:
    """One parsed HTTP request."""

    def __init__(self, method: str, target: str, headers=None, body: bytes | None = None,
                 params: dict | None = None):
        url = urlparse(target)
        self.method = method.upper()
        self.path = url.path or "/"
        self.query = parse_qs(url.query, keep_blank_values=True)
        self.headers = headers or {}
        self.params: dict[str, str] = params or {}
        self._body = body or b""
        self._json = None

    # -- query -----------------------------------------------------------
    def arg(self, name: str, default: str = "") -> str:
        values = self.query.get(name)
        return values[0] if values else default

    def flag(self, name: str) -> bool:
        return self.arg(name).lower() in {"1", "true", "yes", "on"}

    def int_arg(self, name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(self.arg(name, str(default)))
        except ValueError:
            return default
        return max(minimum, min(maximum, value))

    # -- body ------------------------------------------------------------
    @property
    def json(self) -> dict:
        """The JSON object sent in the body (``{}`` when the body is empty)."""
        return self.large_json(MAX_BODY_BYTES)

    def large_json(self, limit: int) -> dict:
        """Like :attr:`json`, for the few routes that accept bodies above 1 MB."""
        if self._json is None:
            if len(self._body) > limit:
                raise TooLargeError("Request is too large")
            try:
                value = json.loads(self._body or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
                raise BadRequestError("Invalid JSON") from None
            if not isinstance(value, dict):
                raise BadRequestError("Request body must be a JSON object")
            self._json = value
        return self._json

    def header(self, name: str, default: str = "") -> str:
        """A request header, matched case-insensitively."""
        getter = getattr(self.headers, "get", None)
        if getter is None:
            return default
        value = getter(name)
        if value is None and isinstance(self.headers, dict):  # plain dicts are case-sensitive
            wanted = name.lower()
            value = next((item for key, item in self.headers.items() if key.lower() == wanted), None)
        return default if value is None else str(value)


class Response:
    """An explicit HTTP response."""

    def __init__(self, body: bytes = b"", status: int = 200, content_type: str = JSON_TYPE,
                 headers: dict | None = None):
        self.body = body
        self.status = status
        self.content_type = content_type
        self.headers = headers or {}

    @classmethod
    def json(cls, value, status: int = 200, headers: dict | None = None) -> "Response":
        payload = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        return cls(payload, status, JSON_TYPE, headers)

    @classmethod
    def text(cls, value: str, content_type: str = "text/plain; charset=utf-8",
             status: int = 200, headers: dict | None = None) -> "Response":
        return cls(value.encode("utf-8"), status, content_type, headers)


class EventStream:
    """Server-sent events produced by an iterator of JSON-serialisable objects.

    The iterator is closed when the client disconnects, so producers release
    resources in a ``finally`` block.  ``heartbeat`` is the number of idle
    seconds after which the iterator is expected to yield ``None`` so the
    transport can emit a keep-alive comment and notice a dead connection.
    """

    def __init__(self, events: Iterable, heartbeat: float = 15.0):
        self.events = events
        self.heartbeat = heartbeat


def created(value) -> Response:
    """``201 Created`` with a JSON body."""
    return Response.json(value, 201)


def accepted(value) -> Response:
    """``202 Accepted`` with a JSON body, used when a job was started."""
    return Response.json(value, 202)


_PARAMETER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class Route:
    def __init__(self, method: str, pattern: str, handler: Callable):
        self.method = method
        self.pattern = pattern
        self.handler = handler
        segments = [segment for segment in pattern.strip("/").split("/") if segment]
        self.size = len(segments)
        self.literals = sum(1 for segment in segments if not _PARAMETER.fullmatch(segment))
        expression = "".join(
            "/" + (f"(?P<{match.group(1)}>[^/]+)" if (match := _PARAMETER.fullmatch(segment))
                   else re.escape(segment))
            for segment in segments
        ) or "/"
        self.regex = re.compile(expression + "/?")

    def match(self, path: str) -> dict | None:
        found = self.regex.fullmatch(path)
        if not found:
            return None
        return {key: unquote(value) for key, value in found.groupdict().items()}


class Router:
    """Method + path pattern dispatch.  ``{name}`` matches one path segment."""

    def __init__(self):
        self.routes: list[Route] = []

    def add(self, method: str, pattern: str, handler: Callable) -> Callable:
        self.routes.append(Route(method.upper(), pattern, handler))
        return handler

    def _decorator(self, method: str, pattern: str):
        return lambda handler: self.add(method, pattern, handler)

    def get(self, pattern: str):
        return self._decorator("GET", pattern)

    def post(self, pattern: str):
        return self._decorator("POST", pattern)

    def put(self, pattern: str):
        return self._decorator("PUT", pattern)

    def patch(self, pattern: str):
        return self._decorator("PATCH", pattern)

    def delete(self, pattern: str):
        return self._decorator("DELETE", pattern)

    def include(self, other: "Router") -> None:
        self.routes.extend(other.routes)

    def resolve(self, method: str, path: str) -> tuple[Callable, dict]:
        """Return ``(handler, params)``; literal segments beat ``{parameters}``."""
        best = None
        allowed: set[str] = set()
        for route in self.routes:
            params = route.match(path)
            if params is None:
                continue
            allowed.add(route.method)
            if route.method != method.upper():
                continue
            if best is None or route.literals > best[0].literals:
                best = (route, params)
        if best is not None:
            return best[0].handler, best[1]
        if allowed:
            raise MethodNotAllowedError(
                f"{method.upper()} is not supported here; use {', '.join(sorted(allowed))}"
            )
        raise NotFoundError("Not found")

    def describe(self) -> list[dict]:
        """Every registered route, for discovery and documentation tests."""
        return [
            {"method": route.method, "path": route.pattern,
             "summary": (route.handler.__doc__ or "").strip().split("\n")[0]}
            for route in sorted(self.routes, key=lambda item: (item.pattern, item.method))
        ]
