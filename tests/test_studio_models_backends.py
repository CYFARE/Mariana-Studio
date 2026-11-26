"""Chat backends against mock providers: OpenAI-compatible, Ollama, Transformers, Anthropic."""

from __future__ import annotations

import json
import threading

import pytest

from mariana.studio.errors import UnavailableError, UpstreamError
from mariana.studio.models import ModelGateway
from mariana.studio.models.backends import (
    ChatRequest,
    StopFilter,
    anthropic_tools,
    canonical_calls,
    native_tools,
    ollama,
    openai,
    transcript_prompt,
)
from tests.helpers import make_tiny_transformer, mock_provider, serving

OFFLINE = {"discover_local": False}


@pytest.fixture()
def gateway(tmp_path):
    instance = ModelGateway(tmp_path, settings=lambda: OFFLINE)
    yield instance
    instance.close()


def sse(*chunks) -> list[str]:
    return [f"data: {json.dumps(chunk) if not isinstance(chunk, str) else chunk}\n\n" for chunk in chunks]


def add(gateway, name, kind, base, model="test-model", **extra):
    gateway.add_endpoint({"name": name, "kind": kind, "base_url": base, "model": model, **extra})
    return next(entry for entry in gateway.catalog(force=True) if entry["name"] == name)


def test_openai_prompt_cache_payload_and_usage_normalisation():
    request = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        options={"provider": "openai", "prompt_cache": True,
                 "prompt_cache_key": "stable-prefix"},
    )
    remote = openai.build_payload("m", request, False)
    assert remote["prompt_cache_key"] == "stable-prefix"
    assert "cache_prompt" not in remote
    local = openai.build_payload("m", request, True)
    assert local["cache_prompt"] is True
    assert "prompt_cache_key" not in local
    automatic = openai.build_payload(
        "m", ChatRequest(messages=request.messages,
                         options={"provider": "groq", "prompt_cache": True,
                                  "prompt_cache_key": "ignored"}), False)
    assert "prompt_cache_key" not in automatic and "cache_prompt" not in automatic

    counts = openai.usage_counts({
        "prompt_tokens": 10, "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 7, "cache_creation_tokens": 3},
    })
    assert counts["cached_tokens"] == 7 and counts["cache_write_tokens"] == 3
    deepseek = openai.usage_counts({
        "prompt_tokens": 10, "completion_tokens": 2, "prompt_cache_hit_tokens": 6,
    })
    assert deepseek["cached_tokens"] == 6 and "cache_write_tokens" not in deepseek

    rejected = UpstreamError("no", detail={
        "status": 400, "body": "unknown parameter prompt_cache_key"})
    adapted = openai.adapt_payload(remote, rejected)
    assert adapted is not None and "prompt_cache_key" not in adapted
    rejected_local = UpstreamError("no", detail={
        "status": 400, "body": "unknown parameter cache_prompt"})
    assert "cache_prompt" not in openai.adapt_payload(local, rejected_local)


def test_openai_and_ollama_round_trips_without_streaming_support(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, {
            "choices": [{"message": {"content": "remote-openai-ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 3},
        }),
        ("POST", "/api/chat"): lambda handler, payload: (200, {
            "message": {"content": "remote-ollama-ok"}, "prompt_eval_count": 4, "eval_count": 3,
        }),
    })
    with serving(provider) as base:
        remote = add(gateway, "Local OpenAI", "openai", f"{base}/v1", api_key="secret-token")
        local = add(gateway, "Local Ollama", "ollama", base, model="llama-test")
        assert remote["available"] and local["available"]
        assert "secret-token" not in json.dumps(gateway.catalog())

        result = gateway.chat({"model": remote["id"], "messages": [{"role": "user", "content": "hi"}],
                               "temperature": 0, "max_tokens": 8})
        assert result["reply"] == "remote-openai-ok" and result["usage"]["completion_tokens"] == 3
        assert result["prompt_format"] == "openai-chat" and result["runtime"] == "openai"
        call = provider.calls[-1]
        assert call["path"] == "/v1/chat/completions"
        assert call["headers"]["authorization"] == "Bearer secret-token"
        assert call["payload"]["model"] == "test-model" and call["payload"]["stream"] is True
        assert "top_k" not in call["payload"]

        result = gateway.chat({"model": local["id"], "messages": [{"role": "user", "content": "hi"}],
                               "temperature": 0, "max_tokens": 8, "top_k": 40, "n_ctx": 2048, "seed": 7})
        assert result["reply"] == "remote-ollama-ok" and result["usage"]["prompt_tokens"] == 4
        call = provider.calls[-1]
        assert call["path"] == "/api/chat"
        assert call["payload"]["options"] == {
            "temperature": 0.0, "top_p": 0.95, "num_predict": 8, "top_k": 40, "num_ctx": 2048, "seed": 7,
        }
        assert gateway.test_endpoint(remote["id"])["reply"] == "remote-openai-ok"


def test_openai_streaming_yields_deltas_reasoning_and_usage(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, sse(
            {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
            {"choices": [{"delta": {"reasoning_content": "thinking…"}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            "[DONE]",
        )),
    })
    with serving(provider) as base:
        entry = add(gateway, "Streamer", "openai", f"{base}/v1")
        events = list(gateway.stream({"model": entry["id"], "message": "hi", "top_k": 5, "seed": "3",
                                      "stop": ["END"], "reasoning_effort": "low"}))
    assert [event["type"] for event in events] == ["start", "reasoning", "delta", "delta", "done"]
    done = events[-1]
    assert done["reply"] == "Hello" and done["reasoning"] == "thinking…"
    assert done["usage"] == {"prompt_tokens": 9, "completion_tokens": 2}
    assert done["finish_reason"] == "stop" and done["first_token_seconds"] is not None
    payload = provider.calls[-1]["payload"]
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["seed"] == 3 and payload["stop"] == ["END"] and payload["top_k"] == 5  # a local server
    assert payload["max_tokens"] == 4096  # a remote endpoint gets room for reasoning and output
    assert payload["reasoning_effort"] == "low"


def test_native_tools_use_safe_aliases_and_json_schema():
    tool = {
        "id": "file.read", "type": "file.read", "label": "Read file", "enabled": True,
        "description": "Read a UTF-8 file in line ranges.",
        "params": [
            {"key": "path", "label": "Path", "type": "template"},
            {"key": "start_line", "label": "First line", "type": "number", "min": 1},
            {"key": "mode", "label": "Mode", "type": "select",
             "options": [["text", "Text"], ["bytes", "Bytes"]]},
        ],
    }
    schemas, aliases = native_tools([tool])
    assert aliases == {"file_read": "file.read"}
    function = schemas[0]["function"]
    assert function["name"] == "file_read" and function["description"]
    properties = function["parameters"]["properties"]
    assert properties["path"]["type"] == "string"
    assert properties["start_line"] == {"type": "number", "minimum": 1,
                                        "description": "First line"}
    assert properties["mode"]["enum"] == ["text", "bytes"]
    assert function["parameters"]["required"] == ["path"]


def test_anthropic_tools_use_the_messages_api_shape():
    tool = {"id": "file.read", "type": "file.read", "description": "Read",
            "params": [{"key": "path", "type": "template"}]}
    tools, aliases = anthropic_tools([tool])
    assert aliases == {"file_read": "file.read"}
    assert tools[0]["name"] == "file_read"
    assert tools[0]["input_schema"]["properties"]["path"]["type"] == "string"


def test_canonical_calls_map_aliases_and_parse_arguments():
    calls = canonical_calls([
        {"name": "file_read", "arguments": '{"path": "notes.txt", "start_line": 2}'},
        {"name": "unknown", "arguments": "not json"},
    ], {"file_read": "file.read"})
    assert calls == [
        {"tool": "file.read", "input": {"path": "notes.txt", "start_line": 2}},
        {"tool": "unknown", "input": {"input": "not json"}},
    ]
    prefixed = canonical_calls([{"name": "functions.file_read", "arguments": {}}],
                               {"file_read": "file.read"})
    assert prefixed == [{"tool": "file.read", "input": {}}]


def test_openai_native_tool_calls_become_the_canonical_action(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, sse(
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"name": "file_read",
                                         "arguments": '{"path": "notes'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": '.txt"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        )),
    })
    tool = {"id": "file.read", "type": "file.read", "description": "Read",
            "params": [{"key": "path", "type": "template"}]}
    with serving(provider) as base:
        entry = add(gateway, "Native", "openai", f"{base}/v1")
        events = list(gateway.stream({"model": entry["id"], "message": "read",
                                      "tools": [tool]}))
    done = events[-1]
    assert done["finish_reason"] == "tool_calls"
    assert json.loads(done["reply"]) == {
        "calls": [{"tool": "file.read", "input": {"path": "notes.txt"}}],
    }
    payload = provider.calls[-1]["payload"]
    assert payload["tools"][0]["function"]["name"] == "file_read"


def test_openai_tool_call_fragments_without_an_index_are_merged(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, sse(
            {"choices": [{"delta": {"tool_calls": [{
                "function": {"name": "file_read", "arguments": '{"path": "'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "function": {"arguments": 'a.txt"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            "[DONE]",
        )),
    })
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "No index", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert json.loads(result["reply"]) == {
        "calls": [{"tool": "file.read", "input": {"path": "a.txt"}}],
    }


def test_legacy_function_call_is_accepted(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, {
            "choices": [{
                "message": {"content": None, "function_call": {
                    "name": "file_read", "arguments": '{"path": "legacy.txt"}'}},
                "finish_reason": "function_call",
            }],
        }),
    })
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "Legacy", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert json.loads(result["reply"]) == {
        "calls": [{"tool": "file.read", "input": {"path": "legacy.txt"}}],
    }


def test_openai_non_streaming_native_tool_calls_are_parsed(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, {
            "choices": [{
                "message": {"content": None, "tool_calls": [{
                    "id": "call-1", "type": "function",
                    "function": {"name": "file_read",
                                 "arguments": '{"path": "a.txt"}'},
                }]},
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }),
    })
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "Native plain", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert json.loads(result["reply"]) == {
        "calls": [{"tool": "file.read", "input": {"path": "a.txt"}}],
    }


def test_endpoints_without_native_tools_fall_back_to_the_prompted_path(gateway):
    attempts = []

    def completions(handler, payload):
        attempts.append(sorted(payload))
        if "tools" in payload:
            return 400, {"error": {"message": "tools are not supported by this model"}}
        return 200, {"choices": [{"message": {"content": '{"final": "prompted"}'},
                                   "finish_reason": "stop"}]}

    provider = mock_provider({("POST", "/v1/chat/completions"): completions})
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "No tools", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert result["reply"] == '{"final": "prompted"}'
    assert "tools" in attempts[0] and "tools" not in attempts[-1]


def test_a_reasoning_history_rejection_drops_native_tools(gateway):
    attempts = []

    def completions(handler, payload):
        attempts.append(sorted(payload))
        if "tools" in payload:
            return 400, {"error": {"message":
                                "reasoning_content is required when tools are used"}}
        return 200, {"choices": [{"message": {"content": "plain"},
                                   "finish_reason": "stop"}]}

    provider = mock_provider({("POST", "/v1/chat/completions"): completions})
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "Reasoner", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert result["reply"] == "plain"
    assert "tools" in attempts[0] and "tools" not in attempts[-1]


def test_ollama_native_tool_calls_are_parsed(gateway):
    provider = mock_provider({
        ("POST", "/api/chat"): lambda handler, payload: (200, {
            "message": {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "file_read", "arguments": {"path": "x.txt"}}},
            ]},
            "done": True, "prompt_eval_count": 3, "eval_count": 2,
        }),
    })
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "Ollama tools", "ollama", base, model="llama-test")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert json.loads(result["reply"]) == {
        "calls": [{"tool": "file.read", "input": {"path": "x.txt"}}],
    }
    payload = provider.calls[-1]["payload"]
    assert payload["tools"][0]["function"]["name"] == "file_read"


def test_deepseek_endpoints_use_the_text_tool_path(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (200, {
            "choices": [{"message": {"content": '{"final": "text path"}'},
                          "finish_reason": "stop"}],
        }),
    })
    tool = {"id": "file.read", "type": "file.read", "params": [{"key": "path"}]}
    with serving(provider) as base:
        entry = add(gateway, "DeepSeek text", "openai", f"{base}/v1",
                    model="deepseek-flash", provider="deepseek")
        result = gateway.chat({"model": entry["id"], "message": "read", "tools": [tool]})
    assert result["reply"] == '{"final": "text path"}'
    assert "tools" not in provider.calls[-1]["payload"]


def test_rejected_parameters_are_dropped_or_renamed_and_retried(gateway):
    attempts = []

    def completions(handler, payload):
        attempts.append(sorted(payload))
        if "stream_options" in payload:
            return 400, {"error": {"message": "Unrecognized request argument: stream_options"}}
        if "max_tokens" in payload:
            return 400, {"error": {"message": "Use 'max_completion_tokens' instead of 'max_tokens'."}}
        if "temperature" in payload:
            return 400, {"error": {"message": "Unsupported value: 'temperature' does not support 0.2"}}
        return 200, {"choices": [{"message": {"content": "adapted"}}]}

    provider = mock_provider({("POST", "/v1/chat/completions"): completions})
    with serving(provider) as base:
        entry = add(gateway, "Strict", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "hi", "temperature": 0.2, "max_tokens": 64})
    assert result["reply"] == "adapted" and len(attempts) == 4
    assert attempts[-1] == ["max_completion_tokens", "messages", "model", "stream", "top_p"]
    assert result["usage"]["estimated"] is True and result["usage"]["completion_tokens"] >= 1


def test_an_unsupported_reasoning_effort_is_dropped_and_retried(gateway):
    attempts = []

    def completions(handler, payload):
        attempts.append(sorted(payload))
        if "reasoning_effort" in payload:
            return 400, {"error": {"message": "Unknown parameter: reasoning_effort"}}
        return 200, {"choices": [{"message": {"content": "plain"}}]}

    provider = mock_provider({("POST", "/v1/chat/completions"): completions})
    with serving(provider) as base:
        entry = add(gateway, "Plain", "openai", f"{base}/v1")
        result = gateway.chat({"model": entry["id"], "message": "hi",
                               "max_tokens": 128, "reasoning_effort": "low"})
    assert result["reply"] == "plain" and len(attempts) == 2
    assert "reasoning_effort" in attempts[0]
    assert "reasoning_effort" not in attempts[1]


def test_provider_failures_become_actionable_errors(gateway):
    provider = mock_provider({
        ("POST", "/v1/chat/completions"): lambda handler, payload: (401, {
            "error": {"message": "Incorrect API key provided"}}),
        ("POST", "/broken/chat/completions"): lambda handler, payload: (200, sse(
            {"choices": [{"delta": {"content": "par"}}]},
            {"error": {"message": "upstream exploded"}},
        )),
        ("POST", "/odd/chat/completions"): lambda handler, payload: (200, {"unexpected": True}),
    })
    with serving(provider) as base:
        denied = add(gateway, "Denied", "openai", f"{base}/v1")
        with pytest.raises(UpstreamError, match="HTTP 401: Incorrect API key provided"):
            gateway.chat({"model": denied["id"], "message": "hi"})
        broken = add(gateway, "Broken", "openai", f"{base}/broken")
        with pytest.raises(UpstreamError, match="upstream exploded"):
            gateway.chat({"model": broken["id"], "message": "hi"})
        odd = add(gateway, "Odd", "openai", f"{base}/odd")
        with pytest.raises(UpstreamError, match="choices"):
            gateway.chat({"model": odd["id"], "message": "hi"})
    unreachable = add(gateway, "Gone", "openai", "http://127.0.0.1:9/v1")
    with pytest.raises(UnavailableError, match="Cannot reach endpoint"):
        gateway.chat({"model": unreachable["id"], "message": "hi"})


def test_ollama_streaming_and_cancellation(gateway):
    lines = [json.dumps(chunk) + "\n" for chunk in (
        {"message": {"thinking": "hmm"}, "done": False},
        {"message": {"content": "one "}, "done": False},
        {"message": {"content": "two "}, "done": False},
        {"message": {"content": "three"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "length",
         "prompt_eval_count": 5, "eval_count": 3},
    )]
    provider = mock_provider({("POST", "/api/chat"): lambda handler, payload: (200, lines)})
    with serving(provider) as base:
        entry = add(gateway, "Stream Ollama", "ollama", f"{base}/api")
        events = list(gateway.stream({"model": entry["id"], "message": "count"}))
        assert [event["type"] for event in events] == ["start", "reasoning", "delta", "delta", "delta", "done"]
        assert events[-1]["reply"] == "one two three" and events[-1]["finish_reason"] == "length"
        assert events[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}

        cancelled = threading.Event()
        seen = []
        for event in gateway.stream({"model": entry["id"], "message": "count"}, cancelled):
            seen.append(event)
            if event["type"] == "delta":
                cancelled.set()
        assert seen[-1]["finish_reason"] == "cancelled" and seen[-1]["reply"] == "one"


def test_listing_models_of_a_server(gateway):
    provider = mock_provider({
        ("GET", "/v1/models"): lambda handler, payload: (200, {"data": [
            {"id": "zeta", "context_length": 8192}, {"id": "Alpha"}, {"object": "no id"}]}),
        ("GET", "/api/tags"): lambda handler, payload: (200, {"models": [{"name": "qwen3:8b"}]}),
    })
    with serving(provider) as base:
        models = gateway.provider_models({"kind": "openai", "base_url": f"{base}/v1", "api_key": "k"})
        assert [model["id"] for model in models] == ["Alpha", "zeta"]
        assert provider.calls[-1]["headers"]["authorization"] == "Bearer k"
        assert gateway.provider_models({"kind": "ollama", "base_url": base}) == [
            {"id": "qwen3:8b", "context_length": None}]
        assert ollama.api_url(f"{base}/api/chat", "tags") == f"{base}/api/tags"
    assert openai.chat_url("https://x/v1/", "") == "https://x/v1/chat/completions"
    assert openai.chat_url("https://x/v1/chat/completions") == "https://x/v1/chat/completions"
    assert openai.chat_url("https://x", "custom") == "https://x/custom"


def test_context_length_probes_models_props_and_ollama_show():
    provider = mock_provider({
        ("GET", "/v1/models"): lambda handler, payload: (200, {"data": [
            {"id": "wide", "max_model_len": 32768},
            {"id": "local"},
        ]}),
        ("GET", "/props"): lambda handler, payload: (200, {
            "default_generation_settings": {"n_ctx": 16384}}),
        ("POST", "/api/show"): lambda handler, payload: (200, {
            "model_info": {"llama.context_length": 131072},
            "parameters": "num_ctx 4096",
        }),
    })
    with serving(provider) as base:
        assert openai.context_length(f"{base}/v1", "wide") == 32768
        assert openai.context_length(f"{base}/v1", "local") == 16384
        assert ollama.context_length(base, "qwen3:8b") == 131072
        assert openai.context_length(f"{base}/v1", "missing") == 16384


def test_gateway_context_window_uses_live_provider_metadata(gateway):
    provider = mock_provider({
        ("GET", "/v1/models"): lambda handler, payload: (200, {"data": [
            {"id": "wide", "context_length": 262144},
        ]}),
    })
    with serving(provider) as base:
        entry = add(gateway, "Context endpoint", "openai", f"{base}/v1", model="wide",
                    context_length=1000)
        resolved = gateway.context_window(entry["id"])
        assert resolved["found"] is True
        assert resolved["context_window"] == 262144
        assert resolved["source"] == "endpoint"
        refreshed = gateway.context_window(entry["id"], force=True)
        assert refreshed["context_window"] == 262144
        assert refreshed["source"] == "endpoint"
        default = gateway.context_window(None)
        assert default["found"] is True and default["context_window"] == 262144


def test_stop_filter_never_leaks_a_partial_stop_string():
    stream = StopFilter(("\nUser:",))
    shown = "".join(stream.feed(piece) for piece in ("Hello", " there\n", "Us", "er: next question"))
    assert shown + stream.flush() == "Hello there" and stream.stopped
    plain = StopFilter(("\nUser:",))
    assert "".join(plain.feed(piece) for piece in ("No stop", " here\nUse")) + plain.flush() == "No stop here\nUse"
    assert transcript_prompt([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]) == (
        "System: S\nUser: U\nAssistant:")


def test_transformers_models_chat_with_their_template_or_a_transcript(tmp_path, gateway):
    pytest.importorskip("transformers")
    import torch

    from mariana.studio.finetune.lora import LoRALinear, inject_lora, save_adapter

    base = make_tiny_transformer(tmp_path / "models" / "tiny-gpt2")
    entry = next(item for item in gateway.catalog(force=True) if item["name"] == "tiny-gpt2")
    assert entry["available"] and entry["detail"]["chat_template"] is True
    events = list(gateway.stream({"model": entry["id"], "message": "Hello there", "max_tokens": 6,
                                  "temperature": 0, "device": "cpu"}))
    assert events[1] == {"type": "status", "message": "Loading tiny-gpt2…"}
    done = events[-1]
    assert done["prompt_format"] == "chat_template" and done["usage"]["completion_tokens"] <= 6
    assert done["usage"]["prompt_tokens"] > 0 and done["runtime"] == "transformers"
    plain = gateway.chat({"model": entry["id"], "message": "Hello there", "max_tokens": 4,
                          "temperature": 0.7, "top_k": 5, "device": "cpu", "prompt_format": "plain"})
    assert plain["prompt_format"] == "transcript" and "\nUser:" not in plain["reply"]

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base)
    names = inject_lora(model, r=2, alpha=4, dropout=0.0)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B.weight, std=0.5)
    adapter = tmp_path / "models" / "tiny-adapter"
    save_adapter(model, adapter / "adapter", base_model=str(base), r=2, alpha=4, dropout=0.0,
                 target_modules=names)
    adapted = next(item for item in gateway.catalog(force=True) if item["name"] == "tiny-adapter")
    assert adapted["available"] and adapted["detail"]["adapter"] is True
    assert gateway.chat({"model": adapted["id"], "message": "Hello", "max_tokens": 3,
                         "temperature": 0, "device": "cpu"})["usage"]["completion_tokens"] <= 3


def anthropic_events(text="Hello from Claude", stop_reason="end_turn", model="claude-opus-5-5",
                     thinking=None, stop_details=None) -> list[str]:
    def event(name, data):
        return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"

    chunks = [event("message_start", {"message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 1}}})]
    index = 0
    if thinking is not None:
        chunks += [
            event("content_block_start", {"index": 0, "content_block": {
                "type": "thinking", "thinking": "", "signature": ""}}),
            event("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": thinking}}),
            event("content_block_stop", {"index": 0}),
        ]
        index = 1
    if text:
        chunks += [
            event("content_block_start", {"index": index, "content_block": {"type": "text", "text": ""}}),
            event("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": text[:5]}}),
            event("content_block_delta", {"index": index, "delta": {"type": "text_delta", "text": text[5:]}}),
            event("content_block_stop", {"index": index}),
        ]
    delta = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details:
        delta["stop_details"] = stop_details
    chunks += [event("message_delta", {"delta": delta, "usage": {"output_tokens": 7}}),
               event("message_stop", {})]
    return chunks


def test_anthropic_backend_uses_the_sdk_without_sampling_parameters(gateway):
    pytest.importorskip("anthropic")
    provider = mock_provider({
        ("POST", "/v1/messages"): lambda handler, payload: (200, anthropic_events(thinking="Let me think")),
        ("GET", "/v1/models"): lambda handler, payload: (200, {
            "data": [{"id": "claude-opus-5-5", "type": "model", "display_name": "Claude Opus 5.5",
                      "created_at": "2026-01-01T00:00:00Z", "max_input_tokens": 1000000,
                      "max_tokens": 128000}],
            "has_more": False, "first_id": "claude-opus-5-5", "last_id": "claude-opus-5-5"}),
    })
    with serving(provider) as base:
        entry = add(gateway, "Claude", "anthropic", base, model="claude-opus-5-5", api_key="test-key")
        assert entry["available"] and entry["capabilities"]["sampling"] is False
        events = list(gateway.stream({
            "model": entry["id"], "system_prompt": "Be brief.", "temperature": 0.3, "top_p": 0.5,
            "top_k": 20, "effort": "low", "show_reasoning": True,
            "messages": [{"role": "assistant", "content": "orphan turn"},
                         {"role": "user", "content": "Hi"}],
        }))
        models = gateway.provider_models({"kind": "anthropic", "base_url": base, "api_key": "test-key"})
    kinds = [event["type"] for event in events]
    assert kinds == ["start", "status", "reasoning", "delta", "delta", "done"]
    done = events[-1]
    assert done["reply"] == "Hello from Claude" and done["reasoning"] == "Let me think"
    assert done["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
    assert done["finish_reason"] == "stop" and done["prompt_format"] == "anthropic-messages"
    call = next(item for item in provider.calls if item["method"] == "POST")
    payload = call["payload"]
    assert not {"temperature", "top_p", "top_k"} & set(payload)  # current Claude models reject them
    assert payload["model"] == "claude-opus-5-5" and payload["max_tokens"] == 16000
    assert payload["system"] == "Be brief." and payload["stream"] is True
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]  # history starts with the user
    assert payload["output_config"] == {"effort": "low"}
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in call["headers"]["anthropic-beta"]
    # The SDK owns authentication; it chooses the header for the credential it was given.
    assert "test-key" in (call["headers"].get("x-api-key", "") + call["headers"].get("authorization", ""))
    assert models == [{"id": "claude-opus-5-5", "name": "Claude Opus 5.5",
                       "context_length": 1000000, "max_output_tokens": 128000}]


def test_anthropic_refusals_fallback_retry_and_errors(gateway):
    anthropic = pytest.importorskip("anthropic")
    seen = []

    def messages(handler, payload):
        seen.append(payload)
        if "fallbacks" in payload:
            return 400, {"type": "error", "error": {
                "type": "invalid_request_error",
                "message": "Unexpected value(s) `server-side-fallback-2026-07-01` for the `anthropic-beta` header."}}
        if payload["model"] == "claude-haiku-5-5":
            return 200, anthropic_events(text="", stop_reason="refusal",
                                         stop_details={"type": "refusal", "category": "cyber",
                                                       "explanation": "Declined by policy."})
        if payload["model"] == "missing-model":
            return 404, {"type": "error", "error": {"type": "not_found_error", "message": "model: missing"}}
        if payload["max_tokens"] == 5:
            return 200, anthropic_events(text="cut", stop_reason="max_tokens")
        return 200, anthropic_events(text="plain answer")

    provider = mock_provider({("POST", "/v1/messages"): messages})
    with serving(provider) as base:
        opus = add(gateway, "Opus", "anthropic", base, model="claude-opus-5-5", api_key="k")
        result = gateway.chat({"model": opus["id"], "message": "hi"})
        assert result["reply"] == "plain answer"
        assert "fallbacks" in seen[0] and "fallbacks" not in seen[1]  # retried without the beta

        seen.clear()
        off = add(gateway, "Opus no fallback", "anthropic", base, model="claude-opus-5-5",
                  api_key="k", refusal_fallback=False)
        gateway.chat({"model": off["id"], "message": "hi"})
        assert len(seen) == 1 and "fallbacks" not in seen[0]

        truncated = gateway.chat({"model": off["id"], "message": "hi", "max_tokens": 5})
        assert truncated["finish_reason"] == "length" and "Thinking counts" in truncated["notice"]

        haiku = add(gateway, "Haiku", "anthropic", base, model="claude-haiku-5-5", api_key="k")
        refused = gateway.chat({"model": haiku["id"], "message": "hi"})
        assert "fallbacks" not in seen[-1]  # this model has no server-side fallback
        assert refused["finish_reason"] == "refusal" and refused["empty"] is True
        assert refused["notice"].startswith("The model declined this request")

        missing = add(gateway, "Missing", "anthropic", base, model="missing-model", api_key="k")
        with pytest.raises(UpstreamError, match="does not recognise this model id"):
            gateway.chat({"model": missing["id"], "message": "hi"})
    assert anthropic.__version__
