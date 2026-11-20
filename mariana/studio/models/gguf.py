"""Read GGUF metadata without loading the model.

Only the key/value header is parsed (never the tensors), so describing a
20 GB file costs a few milliseconds.  Results are cached per file version.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

from .context import model_context

GGUF_MAGIC = b"GGUF"
_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING, _ARRAY = 8, 9
_MAX_KEYS = 4096
_MAX_HEADER_BYTES = 256 * 1024 * 1024
_KEEP_STRING_BYTES = 4096
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0",
}
_QUANT_IN_NAME = re.compile(
    r"(?<![A-Za-z0-9])((?:I?Q\d+(?:_[A-Z0-9]+)*)|BF16|F16|F32|FP16|FP32)(?![A-Za-z0-9])", re.I
)
_cache: dict[tuple, dict] = {}


class GGUFError(ValueError):
    """The file is not a readable GGUF model."""


def quantization_from_name(name: str) -> str | None:
    """Guess the quantization label from a file name such as ``model-Q4_K_M.gguf``."""
    found = _QUANT_IN_NAME.findall(Path(name).stem)
    return found[-1].upper() if found else None


def is_auxiliary(path: Path) -> bool:
    """Vocabulary-only test files and vision projectors are not chat models."""
    name = path.name.lower()
    return name.startswith("ggml-vocab-") or "mmproj" in name


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def take(self, size: int) -> bytes:
        data = self.handle.read(size)
        if len(data) != size:
            raise GGUFError("Unexpected end of GGUF header")
        return data

    def scalar(self, kind: int):
        fmt, size = _SCALARS[kind]
        return struct.unpack(fmt, self.take(size))[0]

    def string(self, keep: bool = True) -> str | None:
        length = struct.unpack("<Q", self.take(8))[0]
        if length > _MAX_HEADER_BYTES:
            raise GGUFError("Corrupt GGUF string length")
        if keep and length <= _KEEP_STRING_BYTES:
            return self.take(length).decode("utf-8", errors="replace")
        self.handle.seek(length, 1)
        return "" if keep else None

    def value(self, kind: int):
        if kind in _SCALARS:
            return self.scalar(kind)
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            element = struct.unpack("<I", self.take(4))[0]
            count = struct.unpack("<Q", self.take(8))[0]
            if element in _SCALARS:
                self.handle.seek(count * _SCALARS[element][1], 1)
            elif element == _STRING:
                for _ in range(count):
                    self.string(keep=False)
            else:
                raise GGUFError("Nested GGUF arrays are not supported")
            return {"array": count}
        raise GGUFError(f"Unknown GGUF value type {kind}")


def read_metadata(path: Path | str) -> dict:
    """Return the scalar/string metadata of a GGUF file (arrays become counts)."""
    path = Path(path)
    metadata: dict = {}
    with path.open("rb") as handle:
        reader = _Reader(handle)
        if handle.read(4) != GGUF_MAGIC:
            raise GGUFError(f"{path.name} is not a GGUF file")
        version = struct.unpack("<I", reader.take(4))[0]
        if version not in (2, 3):
            raise GGUFError(f"Unsupported GGUF version {version}")
        metadata["gguf.version"] = version
        metadata["gguf.tensors"] = struct.unpack("<Q", reader.take(8))[0]
        count = struct.unpack("<Q", reader.take(8))[0]
        for _ in range(min(count, _MAX_KEYS)):
            key = reader.string()
            kind = struct.unpack("<I", reader.take(4))[0]
            metadata[key] = reader.value(kind)
            if handle.tell() > _MAX_HEADER_BYTES:
                break
    return metadata


def describe(path: Path | str) -> dict:
    """A small, display-ready summary of a GGUF model.  Never raises."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _cache:
        return dict(_cache[key])
    summary: dict = {"quantization": quantization_from_name(path.name)}
    try:
        metadata = read_metadata(path)
    except (OSError, GGUFError, struct.error, KeyError) as error:
        summary["metadata_error"] = str(error)
    else:
        architecture = metadata.get("general.architecture")
        prefix = f"{architecture}." if architecture else ""

        def first(*names):
            for name in names:
                value = metadata.get(name)
                if value not in (None, "") and not isinstance(value, dict):
                    return value
            return None

        tokens = metadata.get("tokenizer.ggml.tokens")
        template = metadata.get("tokenizer.chat_template")
        file_type = metadata.get("general.file_type")
        summary.update({
            "architecture": architecture,
            "model_name": first("general.name", "general.basename"),
            "size_label": first("general.size_label"),
            "quantization": FILE_TYPES.get(file_type) or summary["quantization"],
            "ctx_len": model_context(metadata) or first(
                f"{prefix}context_length", "general.context_length", f"{prefix}n_ctx"),
            "layers": first(f"{prefix}block_count"),
            "dim": first(f"{prefix}embedding_length"),
            "vocab_size": tokens.get("array") if isinstance(tokens, dict) else None,
            "chat_template": template is not None,
            "license": first("general.license"),
            "base_model": first(
                "general.base_model.0.repo_url", "general.base_model.0.name",
                "general.source.repo_url", "general.source.url",
            ),
        })
    summary = {name: value for name, value in summary.items() if value is not None}
    if len(_cache) > 512:
        _cache.clear()
    _cache[key] = summary
    return dict(summary)
