"""Tool-call spellings: JSON actions and DeepSeek DSML markup."""

from __future__ import annotations

import json

from mariana.agents.runtime import parse_model_action
from mariana.agents.toolcalls import (
    has_tool_markup, parse_dsml_tool_calls, usable_text,
)

DSML_READS = """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="calls">
<｜｜DSML｜｜ parameter name="input" string="false">{"path": "sharedmap.txt"}</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="tool" string="true">file.read</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="calls">
<｜｜DSML｜｜ parameter name="input" string="false">{"path": "PLA_Military_Bases.kml", "start_line": 1, "end_line": 60}</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="tool" string="true">file.read</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>"""


def test_deepseek_dsml_calls_are_parsed():
    calls = parse_dsml_tool_calls(DSML_READS)
    assert calls == [
        {"tool": "file.read", "input": {"path": "sharedmap.txt"}},
        {"tool": "file.read", "input": {
            "path": "PLA_Military_Bases.kml", "start_line": 1, "end_line": 60,
        }},
    ]
    # The line numbers keep their JSON type; a string would break file.read's range.
    assert isinstance(calls[1]["input"]["start_line"], int)


def test_standard_dsml_invoke_names_the_tool():
    text = ('<｜｜DSML｜｜ invoke name="file.read">\n'
            '<｜｜DSML｜｜ parameter name="path" string="true">notes.txt</｜｜DSML｜｜ parameter>\n'
            '<｜｜DSML｜｜ parameter name="start_line" string="false">2</｜｜DSML｜｜ parameter>\n'
            '</｜｜DSML｜｜ invoke>')
    assert parse_dsml_tool_calls(text) == [
        {"tool": "file.read", "input": {"path": "notes.txt", "start_line": 2}},
    ]


def test_explicit_string_parameters_stay_strings():
    text = ('<invoke name="file.read"><parameter name="path" string="true">123</parameter>'
            '<parameter name="start_line" string="false">2</parameter></invoke>')
    assert parse_dsml_tool_calls(text) == [
        {"tool": "file.read", "input": {"path": "123", "start_line": 2}},
    ]


def test_ascii_and_split_markers_are_tolerated():
    text = ('<||DSML|| invoke name="calc">'
            '<||DSML|| parameter name="expression" string="true">2 + 2</||DSML|| parameter>'
            '</||DSML|| invoke>')
    assert parse_dsml_tool_calls(text) == [
        {"tool": "calc", "input": {"expression": "2 + 2"}},
    ]


def test_dsml_wins_over_the_json_inside_its_parameters():
    action, calls = parse_model_action(DSML_READS)
    assert action == {"calls": calls} and len(calls) == 2
    assert action is not None and "path" not in action


def test_the_last_action_object_wins_over_quoted_json():
    text = ('The file has {"path": "quoted.txt"} in it.\n'
            '{"calls": [{"tool": "file.read", "input": {"path": "real.txt"}}]}')
    action, calls = parse_model_action(text)
    assert calls == [{"tool": "file.read", "input": {"path": "real.txt"}}]
    assert action == {"calls": calls}


def test_plain_json_actions_still_work():
    action, calls = parse_model_action(
        'Sure.\n{"calls": [{"tool": "calc", "input": {"expression": "2 + 2"}}]}')
    assert action["calls"] == calls
    assert calls == [{"tool": "calc", "input": {"expression": "2 + 2"}}]

    action, calls = parse_model_action('{"final": "done"}')
    assert action == {"final": "done"} and calls == []


def test_malformed_or_truncated_markup_is_ignored():
    assert parse_dsml_tool_calls("") == []
    assert parse_dsml_tool_calls("no tool markup here") == []
    assert parse_dsml_tool_calls('<｜｜DSML｜｜ invoke name="calls">') == []
    action, calls = parse_model_action("plain final answer")
    assert action is None and calls == []


def test_calls_are_bounded_and_entities_decoded():
    block = ('<invoke name="calls"><parameter name="tool">file.read</parameter>'
             '<parameter name="input">{"path": "a&amp;b.txt"}</parameter></invoke>')
    calls = parse_dsml_tool_calls(block * 20)
    assert len(calls) == 8
    assert calls[0]["input"]["path"] == "a&b.txt"


# The exact shape a DeepSeek run can leak: the client lost the slashes on the
# closing invoke and wrapper tags, and duplicated a parameter closer.
DSML_DAMAGED_READ = """<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="file.read">
<｜｜DSML｜｜ parameter name="end_line" string="false">300</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="path" string="true">src/main.cpp</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ parameter name="start_line" string="false">265</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ parameter>
<｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ calls>"""


def test_damaged_closing_tags_still_execute_the_call():
    action, calls = parse_model_action(DSML_DAMAGED_READ)
    assert calls == [{"tool": "file.read", "input": {
        "end_line": 300, "path": "src/main.cpp", "start_line": 265,
    }}]
    assert action == {"calls": calls}
    assert isinstance(calls[0]["input"]["start_line"], int)


def test_a_reply_cut_off_mid_stream_keeps_its_calls():
    text = ('<｜｜DSML｜｜ invoke name="file.read">'
            '<｜｜DSML｜｜ parameter name="path" string="true">a.py</｜｜DSML｜｜ parameter>'
            '<｜｜DSML｜｜ parameter name="start_line" string="false">7')
    assert parse_dsml_tool_calls(text) == [
        {"tool": "file.read", "input": {"path": "a.py", "start_line": 7}},
    ]
    assert has_tool_markup(text)


def test_several_damaged_invokes_are_all_read():
    text = ('<DSML invoke name="file.read">'
            '<parameter name="path">a.py</parameter>'
            '<DSML invoke name="file.read">'
            '<parameter name="path">b.py</parameter>'
            '<DSML invoke>')
    assert parse_dsml_tool_calls(text) == [
        {"tool": "file.read", "input": {"path": "a.py"}},
        {"tool": "file.read", "input": {"path": "b.py"}},
    ]


def test_usable_text_never_returns_tool_markup():
    assert usable_text(DSML_DAMAGED_READ) == ""
    assert usable_text('<invoke name="file.read"><parameter name="path">a.py'
                       '</parameter></invoke>') == ""
    assert usable_text("Found the bug.\n" + DSML_DAMAGED_READ) == "Found the bug."
    # Legitimate markup in an answer is not tool-call markup.
    assert usable_text("<kml><Placemark/></kml>") == "<kml><Placemark/></kml>"
    assert not has_tool_markup("<div class=\"x\">hello</div>")


def test_a_truncated_marker_counts_as_markup():
    assert has_tool_markup('<｜｜DSML｜｜ invoke name="file.read">\n<｜｜DSML')
    assert usable_text('<｜｜DSML｜｜ invoke name="file.read">\n<｜｜DSML') == ""


def test_a_final_answer_that_quotes_markup_stays_the_answer():
    text = json.dumps({"final": "Found it.\n" + DSML_DAMAGED_READ})
    action, calls = parse_model_action(text)
    assert calls == [] and "Found it." in action["final"]
    assert usable_text(action["final"]) == "Found it."


# DeepSeek also batches calls as one JSON array inside the envelope.  Treating
# only the last object as the action is what made the model keep observing
# "only one result returned?" and retry the other call.
CALLS_ARRAY_ENVELOPE = (
    '<｜｜DSML｜｜ calls>\n'
    '[{"tool": "file.search", "input": {"query": "QProcess"}}, '
    '{"tool": "file.read", "input": {"path": "src/util.cpp"}}]'
    '</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>'
)


def test_a_bare_json_batch_between_calls_tags_keeps_every_call():
    calls = parse_dsml_tool_calls(CALLS_ARRAY_ENVELOPE)
    assert [call["tool"] for call in calls] == ["file.search", "file.read"]
    assert calls[0]["input"] == {"query": "QProcess"}
    assert usable_text(CALLS_ARRAY_ENVELOPE) == ""


def test_an_invoke_named_calls_carries_a_json_batch():
    text = ('<｜｜DSML｜｜ calls>\n'
            '<｜｜DSML｜｜ invoke name="calls" string="false">'
            '[{"tool": "file.read", "input": {"path": "a.py"}}, '
            '{"tool": "file.read", "input": {"path": "b.py"}}]'
            '</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n'
            '</｜｜DSML｜｜ calls>')
    assert [call["input"]["path"] for call in parse_dsml_tool_calls(text)] == [
        "a.py", "b.py",
    ]


def test_a_plan_envelope_is_a_plan_not_a_tool_call():
    text = ('<｜｜DSML｜｜ calls>\n<｜｜DSML｜｜ invoke name="plan">'
            '[{"text": "Map surfaces", "status": "in_progress"}, '
            '{"text": "Trace sinks", "status": "pending"}]'
            '</｜｜DSML｜｜ invoke>\n</｜｜DSML｜｜ calls>')
    action, calls = parse_model_action(text)
    assert calls == []
    assert [step["text"] for step in action["plan"]] == ["Map surfaces", "Trace sinks"]


def test_nested_parameters_without_an_invoke_tag_still_name_the_tool():
    text = ('<｜｜DSML｜｜ calls>\n'
            '<｜｜DSML｜｜ parameter name="file.read">\n'
            '<｜｜DSML｜｜ parameter name="path" string="true">src/util.cpp'
            '</｜｜DSML｜｜ parameter>\n'
            '<｜｜DSML｜｜ parameter name="start_line" string="false">930'
            '</｜｜DSML｜｜ parameter>\n'
            '</｜｜DSML｜｜ parameter>\n</｜｜DSML｜｜ invoke>\n'
            '</｜｜DSML｜｜ calls>')
    assert parse_dsml_tool_calls(text) == [
        {"tool": "file.read", "input": {"path": "src/util.cpp", "start_line": 930}},
    ]
