"""HTTP API handlers, one module per resource.

Each module owns a ``router``; this package merges them into the single router
the server dispatches on.  Handlers are ``handler(app, request)`` and return a
dict (200 JSON), a ``Response`` or an ``EventStream``.
"""

from __future__ import annotations

from ..http import Router
from . import agents, chat, corpus, dependencies, deploy, finetune, ide, jobs, models, openai_api, system, training

router = Router()
for module in (system, corpus, training, jobs, models, chat, finetune, agents, ide, deploy, dependencies, openai_api):
    router.include(module.router)

__all__ = ["router"]
