"""The bounded JSON Schema subset used by AML guardrail nodes."""

from __future__ import annotations

from mariana.agents.schema import schema_errors, validate_value


def test_schema_accepts_nested_values_and_reports_precise_paths():
    schema = {
        "type": "object", "required": ["title", "items"],
        "properties": {
            "title": {"type": "string", "minLength": 3},
            "items": {
                "type": "array", "minItems": 1, "maxItems": 3,
                "items": {
                    "type": "object", "required": ["id"],
                    "properties": {
                        "id": {"type": "integer", "minimum": 1},
                        "tag": {"enum": ["a", "b"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }
    assert validate_value({"title": "Report", "items": [{"id": 1, "tag": "a"}]}, schema) == []

    errors = validate_value({
        "title": "x",
        "items": [{"id": 0, "tag": "c", "extra": True}],
        "extra": 1,
    }, schema)
    paths = {error["path"]: error for error in errors}
    assert paths["$.title"]["rule"] == "minLength"
    assert paths["$.items[0].id"]["rule"] == "minimum"
    assert paths["$.items[0].tag"]["rule"] == "enum"
    assert paths["$.items[0]"]["rule"] == "additionalProperties"
    assert paths["$"]["rule"] == "additionalProperties"

    missing = validate_value({"title": "Report"}, schema)
    assert any("missing required property 'items'" in error["message"] for error in missing)


def test_any_of_enum_and_json_equality():
    schema = {"anyOf": [{"type": "string", "minLength": 2},
                        {"type": "integer", "minimum": 1}]}
    assert validate_value("ok", schema) == []
    assert validate_value(3, schema) == []
    errors = validate_value(0, schema)
    assert errors and errors[0]["rule"] == "anyOf"

    assert validate_value(1, {"enum": [1, 2]}) == []
    assert validate_value(True, {"enum": [1]})[0]["rule"] == "enum"
    assert validate_value({"a": 1}, {"const": {"a": 1}}) == []
    assert validate_value({"a": 2}, {"const": {"a": 1}})[0]["rule"] == "const"


def test_schema_definitions_reject_unsupported_or_unenforceable_rules():
    def errors(schema):
        return " ".join(schema_errors(schema))

    assert "type must name" in errors({"type": "thing"})
    assert "unsupported keyword" in errors({"type": "object", "$ref": "#"})
    assert "not a valid regular expression" in errors({"type": "string", "pattern": "("})
    assert "required must be a list" in errors({"type": "object", "required": "title"})
    assert "additionalProperties must be true or false" in errors(
        {"type": "object", "additionalProperties": {"type": "string"}})
    assert "properties must be a JSON object" in errors(
        {"type": "object", "properties": []})
    assert "minimum cannot be greater" in errors(
        {"type": "number", "minimum": 5, "maximum": 2})
    assert schema_errors({"type": "object", "properties": {"title": {"type": "string"}}}) == []


def test_schema_depth_is_bounded():
    schema = {"type": "object", "properties": {}}
    cursor = schema
    for _ in range(20):
        child = {"type": "object", "properties": {}}
        cursor["properties"]["next"] = child
        cursor = child

    errors = schema_errors(schema)
    assert any("nesting exceeds" in error for error in errors)
