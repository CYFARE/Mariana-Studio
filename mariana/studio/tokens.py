"""Token estimates for Studio context budgets.

Models in the catalog tokenize differently: Mariana checkpoints use one token per
UTF-8 byte, while GGUF, Transformers and HTTP providers use their own BPE
vocabularies.  Loading every tokenizer just to draw a meter would cost more than
the meter is worth, so this module provides one calibrated estimate and labels it
as such.  Provider ``usage`` counts stay authoritative once a request has run.
"""

from __future__ import annotations

import math


def utf8_length(text: str) -> int:
    """The byte length Mariana's byte vocabulary would use."""
    return len(str(text or "").encode("utf-8", errors="replace"))


def estimate_tokens(text: str, unit: str = "tokens") -> int:
    """Estimate a token count, exactly for byte models and heuristically otherwise.

    The heuristic treats an ASCII character as roughly a quarter token and a
    non-ASCII code point as roughly two thirds of a token.  That is close to
    common BPE vocabularies for English, source code and CJK text without
    importing a tokenizer, and it intentionally errs on the high side.
    """
    value = str(text or "")
    if not value:
        return 0
    if unit == "bytes":
        return utf8_length(value)
    ascii_chars = 0
    wide_chars = 0
    for char in value:
        if ord(char) < 128:
            ascii_chars += 1
        else:
            wide_chars += 1
    return max(1, math.ceil(ascii_chars / 4 + wide_chars / 1.6))


def message_tokens(contents, unit: str = "tokens", overhead: int = 4) -> int:
    """Estimate chat messages, including the small per-message framing cost."""
    return sum(estimate_tokens(str(content or ""), unit) + overhead for content in contents)
