"""Corpus routes: inspect a folder and compute smart training options."""

from __future__ import annotations

from ..corpus import inspect_folder
from ..http import Router

router = Router()


@router.post("/api/corpus/inspect")
def inspect_corpus(app, request):
    """Summarise the readable text in a folder or a single file."""
    return inspect_folder(request.json.get("path", ""))


@router.post("/api/corpus/smart")
def smart_options(app, request):
    """Prepare the corpus and recommend training settings for it."""
    return app.training.smart_options(request.json)
