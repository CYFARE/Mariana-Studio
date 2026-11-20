"""Context-window metadata shared by discovery, backends and the catalog.

Providers report the same idea under many names.  These helpers keep one
definition of which fields count and always take the largest known value, so a
loaded runtime's smaller window never hides the model's advertised maximum.
"""

from __future__ import annotations

CONTEXT_KEYS = (
    "ctx_len", "context_length", "context_window", "max_context",
    "max_context_length", "max_model_len", "max_input_tokens",
    "n_ctx", "n_ctx_train", "max_position_embeddings", "n_positions",
)
_NESTED_KEYS = ("meta", "metadata", "model_info", "config", "default_generation_settings",
                 "parameters", "text_config", "llm_config", "language_config")


def positive_int(value) -> int:
    """An integer above zero, or ``0`` for anything else."""
    if isinstance(value, bool):
        return 0
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _nested_context(value: dict) -> int:
    best = 0
    for key, item in value.items():
        name = str(key).lower()
        if any(part in name for part in
               ("context_length", "context_window", "max_context", "max_model_len",
                "n_ctx", "position_embeddings", "n_positions")):
            best = max(best, positive_int(item))
    return best


def model_context(item: dict | None) -> int | None:
    """The largest context window reported by a raw provider model object."""
    if not isinstance(item, dict):
        return None
    values = [positive_int(item.get(key)) for key in CONTEXT_KEYS]
    for key, value in item.items():
        name = str(key).lower()
        if any(part in name for part in
               ("context_length", "context_window", "max_context", "max_model_len",
                "n_ctx", "position_embeddings", "n_positions")):
            values.append(positive_int(value))
    for key in _NESTED_KEYS:
        nested = item.get(key)
        if isinstance(nested, dict):
            values.append(_nested_context(nested))
        elif key == "parameters" and isinstance(nested, str):
            for line in nested.splitlines():
                name, _, value = line.partition(" ")
                if name.strip() == "num_ctx":
                    values.append(positive_int(value.strip()))
    value = max(values)
    return value or None


def entry_context(entry: dict | None) -> int:
    """The largest context window known for one catalog entry."""
    if not isinstance(entry, dict):
        return 0
    values = [model_context(entry) or 0]
    detail = entry.get("detail")
    if isinstance(detail, dict):
        values.append(model_context(detail) or 0)
        for key in ("context_length", "context_window"):
            values.append(positive_int(detail.get(key)))
    capabilities = entry.get("capabilities")
    if isinstance(capabilities, dict):
        values.append(model_context(capabilities) or 0)
    return max(values)
