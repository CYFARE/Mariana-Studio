"""GGUF header metadata: read without loading tensors, tolerate anything else."""

from __future__ import annotations

import struct

from mariana.studio.models import gguf


def _string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pair(key: str, kind: int, value: bytes) -> bytes:
    return _string(key) + struct.pack("<I", kind) + value


def write_gguf(path, pairs: list[bytes], version: int = 3) -> None:
    header = b"GGUF" + struct.pack("<I", version) + struct.pack("<Q", 0) + struct.pack("<Q", len(pairs))
    path.write_bytes(header + b"".join(pairs) + b"\0" * 64)


def test_metadata_is_summarised_and_cached(tmp_path):
    tokens = struct.pack("<I", 8) + struct.pack("<Q", 3) + b"".join(_string(t) for t in ("a", "b", "c"))
    scores = struct.pack("<I", 6) + struct.pack("<Q", 3) + struct.pack("<3f", 0.0, 1.0, 2.0)
    path = tmp_path / "tiny-Q4_K_M.gguf"
    write_gguf(path, [
        _pair("general.architecture", 8, _string("llama")),
        _pair("general.name", 8, _string("Tiny Llama")),
        _pair("general.size_label", 8, _string("1B")),
        _pair("general.file_type", 4, struct.pack("<I", 15)),
        _pair("llama.context_length", 4, struct.pack("<I", 4096)),
        _pair("llama.block_count", 4, struct.pack("<I", 12)),
        _pair("llama.embedding_length", 10, struct.pack("<Q", 768)),
        _pair("tokenizer.ggml.tokens", 9, tokens),
        _pair("tokenizer.ggml.scores", 9, scores),
        _pair("tokenizer.chat_template", 8, _string("{{ messages }}" * 400)),  # longer than kept
        _pair("general.base_model.0.repo_url", 8, _string("https://huggingface.co/org/base")),
    ])
    summary = gguf.describe(path)
    assert summary == {
        "quantization": "Q4_K_M", "architecture": "llama", "model_name": "Tiny Llama",
        "size_label": "1B", "ctx_len": 4096, "layers": 12, "dim": 768, "vocab_size": 3,
        "chat_template": True, "base_model": "https://huggingface.co/org/base",
    }
    path.write_bytes(b"changed")  # same path, new content: the cache must not serve stale data
    assert "metadata_error" in gguf.describe(path)


def test_unreadable_files_never_raise(tmp_path):
    assert gguf.describe(tmp_path / "missing.gguf") == {}
    garbage = tmp_path / "model-f16.gguf"
    garbage.write_bytes(b"not a gguf file")
    summary = gguf.describe(garbage)
    assert summary["quantization"] == "F16" and "not a GGUF file" in summary["metadata_error"]
    truncated = tmp_path / "cut.gguf"
    truncated.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 5))
    assert "metadata_error" in gguf.describe(truncated)
    old = tmp_path / "v1.gguf"
    write_gguf(old, [], version=1)
    assert "Unsupported GGUF version" in gguf.describe(old)["metadata_error"]


def test_quantization_and_auxiliary_names(tmp_path):
    assert gguf.quantization_from_name("Llama-3.2-1B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert gguf.quantization_from_name("qwen2.5-0.5b-instruct-q8_0.gguf") == "Q8_0"
    assert gguf.quantization_from_name("model-IQ3_XS.gguf") == "IQ3_XS"
    assert gguf.quantization_from_name("mariana-bf16.gguf") == "BF16"
    assert gguf.quantization_from_name("model.gguf") is None
    assert gguf.is_auxiliary(tmp_path / "ggml-vocab-llama.gguf")
    assert gguf.is_auxiliary(tmp_path / "mmproj-model-f16.gguf")
    assert not gguf.is_auxiliary(tmp_path / "model-Q4_0.gguf")
