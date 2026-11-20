"""Studio error types: one hierarchy, one wire format.

Every error that should reach the browser is a :class:`StudioError`.  The HTTP
layer turns it into ``{"error": message, "code": code}`` with ``status``.
Messages are written for the person at the keyboard: they say what is wrong
and what to do next.
"""

from __future__ import annotations


class StudioError(ValueError):
    """Base class.  Subclasses only choose a default status and code."""

    status = 400
    code = "bad_request"

    def __init__(self, message: str, *, code: str | None = None, status: int | None = None,
                 detail: dict | None = None):
        super().__init__(message)
        self.message = str(message)
        if code is not None:
            self.code = code
        if status is not None:
            self.status = status
        self.detail = detail or {}

    def payload(self) -> dict:
        body = {"error": self.message, "code": self.code}
        if self.detail:
            body["detail"] = self.detail
        return body


class BadRequestError(StudioError):
    """The request is malformed or fails validation."""


class ForbiddenError(StudioError):
    """The request comes from an origin Studio does not trust."""

    status = 403
    code = "forbidden"


class NotFoundError(StudioError):
    """The addressed resource does not exist."""

    status = 404
    code = "not_found"


class MethodNotAllowedError(StudioError):
    """The path exists but not for this HTTP method."""

    status = 405
    code = "method_not_allowed"


class ConflictError(StudioError):
    """The request is valid but the current state forbids it."""

    status = 409
    code = "conflict"


class TooLargeError(StudioError):
    """The request body exceeds the accepted size."""

    status = 413
    code = "too_large"


class UpstreamError(StudioError):
    """A remote endpoint or local runtime answered with a failure."""

    status = 502
    code = "upstream_error"


class UnavailableError(StudioError):
    """A required backend or runtime is not installed or not running."""

    status = 503
    code = "unavailable"


STATUS_CODES = {
    400: "bad_request", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
    409: "conflict", 413: "too_large", 500: "server_error", 502: "upstream_error",
    503: "unavailable",
}
