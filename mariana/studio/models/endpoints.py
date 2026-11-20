"""Saved remote endpoints and the provider presets that pre-fill them.

An endpoint is one model on one server.  API keys are write-only: they are
stored in ``agent-workspace/endpoints.json`` (mode 0600, git-ignored) or read
from an environment variable, and are never returned by any API.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from pathlib import Path

from ...workspace import endpoints_path, read_json, write_json
from .catalog import KIND_LABELS, ModelError, has_module

ENDPOINT_KINDS = ("openai", "ollama", "anthropic")
ANTHROPIC_HINT = "pip install anthropic"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_ID_PATTERN = re.compile(r"endpoint:[a-z0-9][a-z0-9-]{0,80}")

# ``local`` presets need no key.  Cloud presets name the conventional environment
# variable so a key that is already exported works without being stored.  Model
# lists are fetched live from the provider ("Fetch models"); only Anthropic
# ships suggestions because its catalogue is small and stable.
PROVIDERS = [
    {"id": "ollama", "name": "Ollama", "kind": "ollama", "local": True,
     "base_url": DEFAULT_OLLAMA_URL, "key_env": "",
     "note": "Models you pulled with `ollama pull` are also found automatically."},
    {"id": "llama-cpp", "name": "llama.cpp server", "kind": "openai", "local": True,
     "base_url": "http://127.0.0.1:8080/v1", "key_env": "",
     "note": "Start it with `llama-server -m model.gguf`."},
    {"id": "lm-studio", "name": "LM Studio", "kind": "openai", "local": True,
     "base_url": "http://127.0.0.1:1234/v1", "key_env": "",
     "note": "Enable the local server in LM Studio's Developer tab."},
    {"id": "vllm", "name": "vLLM", "kind": "openai", "local": True,
     "base_url": "http://127.0.0.1:8000/v1", "key_env": ""},
    {"id": "jan", "name": "Jan", "kind": "openai", "local": True,
     "base_url": "http://127.0.0.1:1337/v1", "key_env": ""},
    {"id": "koboldcpp", "name": "KoboldCpp", "kind": "openai", "local": True,
     "base_url": "http://127.0.0.1:5001/v1", "key_env": ""},
    {"id": "text-generation-webui", "name": "text-generation-webui", "kind": "openai",
     "local": True, "base_url": "http://127.0.0.1:5000/v1", "key_env": ""},
    {"id": "anthropic", "name": "Anthropic", "kind": "anthropic", "local": False,
     "base_url": "", "key_env": "ANTHROPIC_API_KEY", "default_model": "claude-opus-5-5",
     "models": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-2"],
     "note": "Uses the official SDK. Sampling controls do not apply; set effort instead."},
    {"id": "openai", "name": "OpenAI", "kind": "openai", "local": False,
     "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
    {"id": "gemini", "name": "Google Gemini", "kind": "openai", "local": False,
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "key_env": "GEMINI_API_KEY"},
    {"id": "mistral", "name": "Mistral", "kind": "openai", "local": False,
     "base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY"},
    {"id": "groq", "name": "Groq", "kind": "openai", "local": False,
     "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    {"id": "openrouter", "name": "OpenRouter", "kind": "openai", "local": False,
     "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    {"id": "together", "name": "Together AI", "kind": "openai", "local": False,
     "base_url": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY"},
    {"id": "fireworks", "name": "Fireworks AI", "kind": "openai", "local": False,
     "base_url": "https://api.fireworks.ai/inference/v1", "key_env": "FIREWORKS_API_KEY"},
    {"id": "deepseek", "name": "DeepSeek", "kind": "openai", "local": False,
     "base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY"},
    {"id": "xai", "name": "xAI", "kind": "openai", "local": False,
     "base_url": "https://api.x.ai/v1", "key_env": "XAI_API_KEY"},
    {"id": "cerebras", "name": "Cerebras", "kind": "openai", "local": False,
     "base_url": "https://api.cerebras.ai/v1", "key_env": "CEREBRAS_API_KEY"},
    {"id": "huggingface", "name": "Hugging Face Inference", "kind": "openai", "local": False,
     "base_url": "https://router.huggingface.co/v1", "key_env": "HF_TOKEN"},
    {"id": "custom", "name": "Custom OpenAI-compatible", "kind": "openai", "local": False,
     "base_url": "", "key_env": ""},
]


def endpoint_id(name: str) -> str:
    """A stable id for an endpoint name; existing ``endpoint:…`` ids pass through."""
    value = str(name or "").strip()
    if _ID_PATTERN.fullmatch(value):
        return value
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:80] or "endpoint"
    return f"endpoint:{slug}"


def providers() -> list[dict]:
    """Provider presets annotated with whether their conventional key is exported."""
    result = []
    for preset in PROVIDERS:
        item = dict(preset)
        item["kind_label"] = KIND_LABELS[preset["kind"]]
        item["key_present"] = bool(preset.get("key_env") and os.environ.get(preset["key_env"]))
        if preset["kind"] == "anthropic":
            item["installed"] = has_module("anthropic")
            item["install_hint"] = ANTHROPIC_HINT
        result.append(item)
    return result


def normalize_base_url(kind: str, value: str) -> str:
    base_url = str(value or "").strip().rstrip("/")
    if kind == "ollama" and not base_url:
        base_url = DEFAULT_OLLAMA_URL
    if kind == "anthropic" and not base_url:
        return ""  # the SDK's default API host
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ModelError("Base URL must be a full http(s) URL, for example https://api.example.com/v1")
    return base_url


def resolve_key(endpoint: dict) -> str:
    """The API key for an endpoint: stored key first, then its environment variable."""
    key = endpoint.get("api_key")
    if not key and endpoint.get("api_key_env"):
        key = os.environ.get(endpoint["api_key_env"], "")
    return key or ""


class EndpointStore:
    """Small JSON store for remote model endpoints."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = endpoints_path(self.root)

    def load(self) -> list[dict]:
        data = read_json(self.path, [])
        return [entry for entry in data if isinstance(entry, dict)] if isinstance(data, list) else []

    def save(self, entries: list[dict]) -> None:
        write_json(self.path, entries)
        try:
            os.chmod(self.path, 0o600)  # the file may hold API keys
        except OSError:
            pass

    def get(self, entry_id: str) -> dict:
        for endpoint in self.load():
            if endpoint.get("id") == entry_id:
                return endpoint
        raise ModelError("Endpoint not found; it may have been removed")

    def upsert(self, config: dict) -> dict:
        name = str(config.get("name") or config.get("model") or "").strip()[:120]
        if not name:
            raise ModelError("Give the endpoint a name")
        kind = config.get("kind", "openai")
        if kind not in ENDPOINT_KINDS:
            raise ModelError("Endpoint kind must be openai, ollama, or anthropic")
        base_url = normalize_base_url(kind, config.get("base_url"))
        model = str(config.get("model") or "").strip()
        if not model:
            raise ModelError("Model name is required")
        headers = config.get("headers") if isinstance(config.get("headers"), dict) else {}
        entry = {
            "id": endpoint_id(config.get("id") or name),
            "kind": kind,
            "name": name,
            "provider": str(config.get("provider") or "").strip()[:40],
            "base_url": base_url,
            "model": model,
            "api_path": str(config.get("api_path") or "").strip(),
            "api_key_env": str(config.get("api_key_env") or "").strip(),
            "headers": {str(key): str(value) for key, value in headers.items()},
        }
        for key in ("context_length", "max_output_tokens"):
            try:
                value = int(config.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                entry[key] = value
        if kind == "anthropic":
            entry["refusal_fallback"] = config.get("refusal_fallback", True) is not False
        existing = self.load()
        previous = next((item for item in existing if item.get("id") == entry["id"]), None)
        key = str(config.get("api_key") or "").strip()
        if key:
            entry["api_key"] = key
        elif previous and previous.get("api_key") and not config.get("clear_api_key"):
            entry["api_key"] = previous["api_key"]
        entries = [item for item in existing if item.get("id") != entry["id"]]
        entries.append(entry)
        self.save(entries)
        return entry

    def delete(self, entry_id: str) -> bool:
        entries = self.load()
        kept = [entry for entry in entries if entry.get("id") != entry_id]
        if len(kept) == len(entries):
            return False
        self.save(kept)
        return True

    def public(self) -> list[dict]:
        result = []
        for entry in self.load():
            clean = {key: value for key, value in entry.items() if key != "api_key"}
            clean["has_api_key"] = bool(entry.get("api_key"))
            result.append(clean)
        return result


def endpoint_entry(endpoint: dict) -> dict:
    """The catalog entry for a saved endpoint."""
    kind = endpoint["kind"]
    key_env = endpoint.get("api_key_env")
    has_key = bool(endpoint.get("api_key") or (key_env and os.environ.get(key_env)))
    available, reason = True, ""
    if kind == "openai" and key_env and not has_key:
        available = False
        reason = f"API key not found; set {key_env} or store a key on this endpoint."
    if kind == "anthropic" and not has_module("anthropic"):
        available = False
        reason = f"The Anthropic SDK is not installed; {ANTHROPIC_HINT}"
    base_url = endpoint.get("base_url", "")
    host = urllib.parse.urlparse(base_url).hostname or ""
    local = host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".local")
    detail = {
        "architecture": endpoint.get("model"), "model": endpoint.get("model"),
        "base_url": base_url or ("https://api.anthropic.com" if kind == "anthropic" else ""),
        "remote": not local, "has_api_key": has_key, "provider": endpoint.get("provider") or "",
        "ctx_unit": "tokens",
    }
    if endpoint.get("context_length"):
        detail["ctx_len"] = endpoint["context_length"]
    if endpoint.get("max_output_tokens"):
        detail["max_output_tokens"] = endpoint["max_output_tokens"]
    capabilities = {"chat": True, "finetune": False, "remote": not local, "stream": True}
    if kind == "anthropic":
        # Current Claude models reject temperature/top_p/top_k; depth is set with effort.
        capabilities.update({"sampling": False, "effort": True, "reasoning": True})
        detail["refusal_fallback"] = endpoint.get("refusal_fallback", True)
    return {
        "id": endpoint["id"], "name": endpoint["name"], "kind": kind,
        "kind_label": KIND_LABELS[kind], "source": "endpoint",
        "source_label": "Saved endpoint", "runtime": kind,
        "path": base_url or "api.anthropic.com", "absolute_path": "", "bytes": 0, "modified": 0,
        "available": available, "reason": reason, "detail": detail,
        "capabilities": capabilities,
        "finetune_reason": (
            "This model runs on a remote server, so its weights are not available locally. "
            "Use the provider's own fine-tuning service, or pick a downloadable model."
        ),
    }
