"""A small, bounded JSON Schema subset for AML guardrail nodes.

AML deliberately does not ship a full JSON Schema implementation.  Agents need
predictable validation with precise paths, not a second configuration language,
so this module supports the common structural keywords and rejects anything it
cannot enforce.  Rejecting an unknown keyword is safer than silently ignoring a
constraint the author expected to run.
"""

from __future__ import annotations

import json
import re

MAX_SCHEMA_DEPTH = 12
MAX_SCHEMA_CHARS = 100_000
MAX_ERRORS = 50
MAX_PATTERN_CHARS = 300
MAX_ENUM_ITEMS = 100
MAX_PROPERTIES = 200
MAX_REQUIRED = 100
MAX_ANY_OF = 10
TYPE_NAMES = frozenset({
    "object", "array", "string", "number", "integer", "boolean", "null",
})
_SCHEMA_KEYWORDS = frozenset({
    "type", "required", "properties", "additionalProperties", "items",
    "enum", "const", "minimum", "maximum", "minLength", "maxLength",
    "minItems", "maxItems", "pattern", "anyOf", "title", "description",
})
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_type(value, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return _is_number(value)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    return False


def _same_json(left, right) -> bool:
    """JSON equality that does not confuse ``True`` with ``1``."""
    if isinstance(left, (dict, list)) or isinstance(right, (dict, list)):
        try:
            return (json.dumps(left, sort_keys=True, ensure_ascii=False)
                    == json.dumps(right, sort_keys=True, ensure_ascii=False))
        except (TypeError, ValueError):
            return left == right
    if _is_number(left) and _is_number(right):
        return left == right
    return type(left) is type(right) and left == right


def _child_path(path: str, key) -> str:
    text = str(key)
    if _IDENTIFIER.fullmatch(text):
        return f"{path}.{text}"
    return f"{path}[{json.dumps(text, ensure_ascii=False)}]"


def _sample(values, limit: int = 6) -> str:
    shown = [json.dumps(item, ensure_ascii=False, default=str)[:120]
             for item in values[:limit]]
    more = f" (+{len(values) - limit} more)" if len(values) > limit else ""
    return ", ".join(shown) + more


def schema_errors(schema, path: str = "$", depth: int = 0) -> list[str]:
    """Return definition errors for the supported schema subset.

    An empty list means the schema is enforceable.  Unknown keywords are errors
    on purpose: ignoring a constraint the author wrote would make validation
    quietly weaker than it looks.
    """
    if depth > MAX_SCHEMA_DEPTH:
        return [f"{path}: schema nesting exceeds {MAX_SCHEMA_DEPTH}"]
    if not isinstance(schema, dict):
        return [f"{path}: schema must be a JSON object"]
    errors: list[str] = []

    def error(message: str) -> None:
        if len(errors) < MAX_ERRORS:
            errors.append(f"{path}: {message}")

    if depth == 0:
        try:
            encoded = json.dumps(schema, ensure_ascii=False)
        except (TypeError, ValueError):
            return ["$: schema is not JSON-serialisable"]
        if len(encoded) > MAX_SCHEMA_CHARS:
            return [f"$: schema exceeds {MAX_SCHEMA_CHARS:,} characters"]

    unknown = sorted(set(schema) - _SCHEMA_KEYWORDS)
    if unknown:
        error("unsupported keyword(s): " + ", ".join(unknown))

    if "type" in schema:
        raw = schema["type"]
        names = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else None
        if (not names or any(not isinstance(name, str) or name not in TYPE_NAMES
                             for name in names)
                or len(set(names)) != len(names)):
            error("type must name one type or a list of unique types: "
                  + ", ".join(sorted(TYPE_NAMES)))

    if "required" in schema:
        required = schema["required"]
        if not isinstance(required, list):
            error("required must be a list of property names")
        elif len(required) > MAX_REQUIRED:
            error(f"required may list at most {MAX_REQUIRED} properties")
        elif any(not isinstance(name, str) or not name for name in required):
            error("every required entry must be a non-empty string")

    if "properties" in schema:
        properties = schema["properties"]
        if not isinstance(properties, dict):
            error("properties must be a JSON object")
        elif len(properties) > MAX_PROPERTIES:
            error(f"properties may define at most {MAX_PROPERTIES} keys")
        else:
            for key, child in properties.items():
                if len(errors) >= MAX_ERRORS:
                    break
                errors.extend(schema_errors(
                    child, _child_path(path, key), depth + 1,
                )[:MAX_ERRORS - len(errors)])

    if "additionalProperties" in schema and not isinstance(
            schema["additionalProperties"], bool):
        error("additionalProperties must be true or false")

    if "items" in schema:
        errors.extend(schema_errors(schema["items"], f"{path}[]", depth + 1)
                      [:MAX_ERRORS - len(errors)])

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            error("enum must be a non-empty list")
        elif len(enum) > MAX_ENUM_ITEMS:
            error(f"enum may contain at most {MAX_ENUM_ITEMS} values")

    if "anyOf" in schema:
        choices = schema["anyOf"]
        if not isinstance(choices, list) or not choices:
            error("anyOf must be a non-empty list of schemas")
        elif len(choices) > MAX_ANY_OF:
            error(f"anyOf may contain at most {MAX_ANY_OF} schemas")
        else:
            for index, child in enumerate(choices):
                if len(errors) >= MAX_ERRORS:
                    break
                errors.extend(schema_errors(
                    child, f"{path}.anyOf[{index}]", depth + 1,
                )[:MAX_ERRORS - len(errors)])

    for key in ("minimum", "maximum"):
        if key in schema and not _is_number(schema[key]):
            error(f"{key} must be a number")
    if (_is_number(schema.get("minimum")) and _is_number(schema.get("maximum"))
            and schema["minimum"] > schema["maximum"]):
        error("minimum cannot be greater than maximum")

    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema:
            value = schema[key]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                error(f"{key} must be a non-negative integer")
    for lower, upper in (("minLength", "maxLength"), ("minItems", "maxItems")):
        if (isinstance(schema.get(lower), int) and isinstance(schema.get(upper), int)
                and schema[lower] > schema[upper]):
            error(f"{lower} cannot be greater than {upper}")

    if "pattern" in schema:
        pattern = schema["pattern"]
        if not isinstance(pattern, str) or not pattern:
            error("pattern must be a non-empty string")
        elif len(pattern) > MAX_PATTERN_CHARS:
            error(f"pattern is limited to {MAX_PATTERN_CHARS} characters")
        else:
            try:
                re.compile(pattern)
            except re.error as failure:
                error(f"pattern is not a valid regular expression: {failure}")

    for key in ("title", "description"):
        if key in schema and not isinstance(schema[key], str):
            error(f"{key} must be a string")

    return errors


def validate_value(value, schema, path: str = "$", depth: int = 0) -> list[dict]:
    """Validate a JSON value and return ``{path, rule, message}`` errors."""
    if depth > MAX_SCHEMA_DEPTH:
        return [{"path": path, "rule": "schema",
                 "message": f"schema nesting exceeds {MAX_SCHEMA_DEPTH}"}]
    if not isinstance(schema, dict):
        return [{"path": path, "rule": "schema",
                 "message": "schema must be a JSON object"}]

    errors: list[dict] = []

    def add(rule: str, message: str) -> None:
        if len(errors) < MAX_ERRORS:
            errors.append({"path": path, "rule": rule, "message": message})

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        choices = schema["anyOf"][:MAX_ANY_OF]
        if not any(not validate_value(value, choice, path, depth + 1)
                   for choice in choices):
            add("anyOf", "does not match any of the allowed schemas")
            return errors

    if "type" in schema:
        raw = schema["type"]
        names = [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []
        if not any(_matches_type(value, name) for name in names):
            add("type", f"expected {' or '.join(str(name) for name in names)}, "
                        f"got {_json_type(value)}")
            return errors

    if "const" in schema and not _same_json(value, schema["const"]):
        expected = json.dumps(schema["const"], ensure_ascii=False, default=str)[:200]
        add("const", f"must equal {expected}")

    if "enum" in schema and isinstance(schema["enum"], list):
        enum = schema["enum"][:MAX_ENUM_ITEMS]
        if not any(_same_json(value, item) for item in enum):
            add("enum", "must be one of " + _sample(enum))

    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for name in required[:MAX_REQUIRED]:
                if isinstance(name, str) and name not in value:
                    add("required", f"missing required property {name!r}")
        for key, child in list(properties.items())[:MAX_PROPERTIES]:
            if key in value and len(errors) < MAX_ERRORS:
                errors.extend(validate_value(value[key], child,
                                             _child_path(path, key), depth + 1)
                              [:MAX_ERRORS - len(errors)])
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties), key=str)[:10]
            if extra:
                add("additionalProperties",
                    "unexpected propert" + ("y" if len(extra) == 1 else "ies")
                    + ": " + ", ".join(str(key) for key in extra))

    elif isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            add("minItems", f"needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            add("maxItems", f"allows at most {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                if len(errors) >= MAX_ERRORS:
                    break
                errors.extend(validate_value(item, schema["items"],
                                             f"{path}[{index}]", depth + 1)
                              [:MAX_ERRORS - len(errors)])

    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            add("minLength", f"needs at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            add("maxLength", f"allows at most {schema['maxLength']} characters")
        if isinstance(schema.get("pattern"), str):
            try:
                if re.search(schema["pattern"], value) is None:
                    add("pattern", "does not match " + json.dumps(schema["pattern"]))
            except re.error as failure:
                add("schema", f"pattern is not a valid regular expression: {failure}")

    elif _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            add("minimum", f"must be at least {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            add("maximum", f"must be at most {schema['maximum']}")

    return errors
