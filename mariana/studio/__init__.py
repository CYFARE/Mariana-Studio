"""Mariana Studio: the local web application for training, fine-tuning, chatting with,
automating and deploying language models.

``make_server`` and ``run`` are imported lazily so that light-weight users of the
package (the fine-tune trainer subprocess, for one) do not load the whole server.
"""

from __future__ import annotations

__all__ = ["Studio", "make_server", "run"]


def __getattr__(name: str):
    if name in {"make_server", "run"}:
        from . import server

        return getattr(server, name)
    if name == "Studio":
        from .app import Studio

        return Studio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
