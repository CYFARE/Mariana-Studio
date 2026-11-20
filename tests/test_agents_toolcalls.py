"""Tool-call spellings: JSON actions and DeepSeek DSML markup."""

from __future__ import annotations

from mariana.agents.runtime import parse_model_action
from mariana.agents.toolcalls import parse_dsml_tool_calls

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
