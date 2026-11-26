"""CRAFT context compression: anchors, density, observations and episodes."""

from __future__ import annotations

import json

from mariana.agents.compression import (
    MARKER,
    anchor_line,
    compact_turns,
    compress_observation,
    compress_text,
    extract_anchors,
)


def test_anchors_keep_paths_lines_errors_urls_and_commands():
    text = (
        "Error: failed to parse src/models/foo.cpp:42\n"
        "See https://example.com/issue/7 for details.\n"
        "$ cmake --build build/cc-debug\n"
        "`value.toInt()` returns undefined\n"
    )
    anchors = extract_anchors(text)
    joined = "\n".join(anchors)
    assert "src/models/foo.cpp:42" in joined
    assert "https://example.com/issue/7" in joined
    assert "cmake --build" in joined
    assert "value.toInt()" in joined
    assert "failed to parse" in joined


def test_text_compression_stays_inside_the_budget_and_keeps_provenance():
    text = "\n".join([
        "The parser crashed while reading src/models/foo.cpp:42.",
        *[f"Filler observation {index}: " + "x" * 80 for index in range(40)],
        "Fix: validate parts before indexing at src/models/foo.cpp:42.",
    ])
    result = compress_text(text, 500, goal="fix parser crash foo.cpp")
    assert len(result) <= 500
    assert result.startswith(MARKER)
    assert "src/models/foo.cpp:42" in result
    assert "Fix:" in result or "validate parts" in result


def test_observation_compression_preserves_error_and_match_lines():
    payload = json.dumps({
        "path": "src/models/foo.cpp",
        "content": "\n".join(f"line {index} " + "y" * 100 for index in range(30)),
        "matches": [
            {"path": "src/models/foo.cpp", "line": 42, "text": "needle.toInt()"},
            {"path": "src/models/bar.cpp", "line": 7, "text": "other match"},
        ],
        "error": "Invalid regular expression: missing )",
        "truncated": True,
    })
    result = compress_observation(f"file.search failed:\n{payload}", 420,
                                  goal="find the parser bug", tool_id="file.search")
    assert len(result) <= 420
    assert "src/models/foo.cpp:42" in result
    assert "needle.toInt()" in result
    assert "Invalid regular expression" in result


def test_episode_card_names_tools_anchors_errors_and_constraints():
    messages = [
        {"role": "assistant", "content": json.dumps({"tool": "file.read",
                                                     "input": {"path": "src/a.cpp"}})},
        {"role": "user", "content": "src/a.cpp:10 returned an error: undefined value\n"
                                    "We must keep the public API stable.\n" + "z" * 400},
    ]
    card = compact_turns(messages, 700, goal="fix the undefined value")
    assert len(card) <= 700
    assert card.startswith(MARKER)
    assert "Tools: file.read" in card
    assert "src/a.cpp:10" in card
    assert "Errors:" in card or "undefined value" in card
    assert "public API" in card


def test_small_text_is_returned_unchanged():
    text = "short finding at src/a.cpp:3"
    assert compress_text(text, 200) == text
    assert anchor_line(text) == "src/a.cpp:3"
