"""Token estimates shared by the IDE context meter and the server budget."""

from __future__ import annotations

from mariana.studio.tokens import estimate_tokens, message_tokens, utf8_length


def test_empty_text_costs_nothing():
    assert utf8_length("") == 0
    assert estimate_tokens("") == 0
    assert message_tokens([]) == 0


def test_byte_models_count_utf8_bytes_exactly():
    assert utf8_length("hello") == 5
    assert utf8_length("héllo") == 6
    assert estimate_tokens("héllo", "bytes") == 6


def test_token_models_use_a_positive_heuristic():
    assert estimate_tokens("hello world") == 3
    assert estimate_tokens("你好世界") == 3
    assert estimate_tokens("你好世界", "bytes") == 12
    # More text never lowers the estimate.
    assert estimate_tokens("a" * 4_000) > estimate_tokens("a" * 400)


def test_message_overhead_is_counted_per_message():
    assert message_tokens(["abcd"]) == estimate_tokens("abcd") + 4
    assert message_tokens(["abcd", "efgh"]) == 10
