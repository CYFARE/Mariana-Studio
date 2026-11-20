"""Training examples for causal-LM fine-tuning: chat rows and plain text become token ids.

Pure data plumbing.  The tokenizer is passed in -- a Transformers tokenizer, or anything
with the same ``tokenizer(text, add_special_tokens=False)["input_ids"]`` call -- so this
module needs only the standard library and torch.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import torch

IGNORE_INDEX = -100  # label of a token the loss must skip
ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}
GENERATION_PROMPT = "Assistant:"
SPEAKER_ROLES = {
    "system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
}
REPLY_KEYS = ("completion", "response", "output", "answer")
ROW_LIST_KEYS = ("data", "rows", "examples", "train")
CHAT_SUFFIXES = (".jsonl", ".ndjson", ".json")
SKIPPED_DIRECTORIES = frozenset({"venv", "node_modules", "__pycache__"})
GENERATED_CORPUS_PREFIX = "gui-corpus-"
PIECE_CHARS = 1 << 20  # text is read and tokenized about 1 MB at a time
MAX_OBJECT_BYTES = 256 << 20  # largest .json document parsed whole
MIN_BLOCK_TOKENS = 8
MAX_VALIDATION_EXAMPLES = 2000

Example = tuple[list[int], list[int]]


def transcript(messages: Iterable[dict]) -> str:
    """Render messages in the plain layout Studio's chat uses for models without a chat template.

    Chat appends ``"Assistant:"`` to this text, so it must stay identical byte for byte.
    """
    return "".join(
        f"{ROLE_LABELS.get(message['role'], 'User')}: {message['content']}\n" for message in messages
    )


def _read_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _read_content(value: object) -> str:
    """Text of a message whose content is a string or a list of OpenAI content parts."""
    if isinstance(value, list):
        parts = [
            part["text"] for part in value
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
        ]
        return "\n".join(parts).strip()
    return _read_text(value)


def _normalize_turns(turns: list) -> list[dict] | None:
    messages = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("role", turn.get("from"))
        role = SPEAKER_ROLES.get(speaker.strip().lower()) if isinstance(speaker, str) else None
        if role is None:  # tool and function turns carry no chat text to learn
            continue
        content = _read_content(turn["content"] if "content" in turn else turn.get("value"))
        if content:
            messages.append({"role": role, "content": content})
    while messages and messages[-1]["role"] != "assistant":  # nothing to learn after the last reply
        messages.pop()
    if not any(message["role"] == "user" for message in messages):
        return None
    return messages


def _normalize_pair(row: dict) -> list[dict] | None:
    prompt = _read_text(row.get("prompt"))
    reply = next((text for text in map(_read_text, map(row.get, REPLY_KEYS)) if text), "")
    if not prompt and _read_text(row.get("instruction")):  # Alpaca
        prompt = "\n\n".join(filter(None, (_read_text(row["instruction"]), _read_text(row.get("input")))))
        reply = _read_text(row.get("output")) or _read_text(row.get("response"))
    elif not prompt:
        prompt, reply = _read_text(row.get("question")), _read_text(row.get("answer"))
    if not prompt or not reply:
        return None
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]


def normalize_row(row: object) -> list[dict] | None:
    """Map one dataset row to ``[{"role", "content"}, ...]`` ending with an assistant reply.

    Understands OpenAI ``messages``, ShareGPT ``conversations``, ``prompt`` + completion,
    Alpaca ``instruction``/``input``/``output`` and ``question``/``answer`` rows, each with an
    optional top-level ``system`` text.  Returns ``None`` for anything that is not a chat example.
    """
    if not isinstance(row, dict):
        return None
    turns = next((row[key] for key in ("messages", "conversations") if isinstance(row.get(key), list)), None)
    messages = _normalize_turns(turns) if turns is not None else _normalize_pair(row)
    if not messages:
        return None
    system = _read_text(row.get("system"))
    if system and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system})
    return messages


def _iter_rows(path: Path) -> Iterator[object]:
    """Yield the rows of a JSON Lines file or a ``.json`` document; bad lines yield ``None``."""
    if path.suffix.lower() == ".json":
        try:
            if path.stat().st_size <= MAX_OBJECT_BYTES:
                # utf-8-sig: a byte-order mark would otherwise break the first JSON value.
                document = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
                if isinstance(document, dict):
                    rows = next(
                        (document[key] for key in ROW_LIST_KEYS if isinstance(document.get(key), list)),
                        [document],
                    )
                else:
                    rows = document if isinstance(document, list) else []
                yield from rows
                return
        except ValueError:
            pass  # many ".json" exports are really one object per line
    with open(path, encoding="utf-8-sig", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError:
                    yield None


def is_chat_file(path: Path, sample_rows: int = 20) -> bool:
    """Whether most of the first rows of ``path`` are chat examples."""
    valid = seen = 0
    try:
        for row in _iter_rows(path):
            seen += 1
            valid += normalize_row(row) is not None
            if seen >= sample_rows:
                break
    except OSError:
        return False
    return valid > 0 and valid * 2 >= seen


def find_chat_files(path: str | Path, sample_rows: int = 20) -> list[Path]:
    """Return the chat-format ``.jsonl``/``.json`` files in a folder (or the file itself)."""
    source = Path(path).expanduser()
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = []
        for base, directories, files in os.walk(source):
            directories[:] = sorted(
                name for name in directories
                if not name.startswith(".") and name not in SKIPPED_DIRECTORIES
            )
            candidates += [Path(base) / name for name in sorted(files)]
    else:
        return []
    return [
        candidate for candidate in candidates
        if candidate.suffix.lower() in CHAT_SUFFIXES
        and not candidate.name.startswith(GENERATED_CORPUS_PREFIX)
        and is_chat_file(candidate, sample_rows)
    ]


def load_chat_examples(files: Iterable[str | Path], max_examples: int | None = None) -> list[list[dict]]:
    """Read every chat example from ``files``; malformed rows are skipped, never fatal."""
    examples: list[list[dict]] = []
    for path in files:
        try:
            for row in _iter_rows(Path(path)):
                messages = normalize_row(row)
                if messages:
                    examples.append(messages)
                    if max_examples and len(examples) >= max_examples:
                        return examples
        except OSError:
            continue
    return examples


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _has_chat_template(tokenizer) -> bool:
    return bool(getattr(tokenizer, "chat_template", None)) and hasattr(tokenizer, "apply_chat_template")


def encode_chat(tokenizer, messages: Sequence[dict], train_on: str = "assistant",
                max_length: int | None = None) -> Example | None:
    """Tokenize one conversation into ``(input_ids, labels)``.

    With ``train_on="assistant"`` only assistant replies (and the token that ends each
    reply) carry labels; everything else is ``IGNORE_INDEX``.  The reply spans are found by
    rendering growing prefixes of the conversation; if a tokenizer does not render prefixes
    consistently the example falls back to supervising every token.  Returns ``None`` when
    truncation leaves nothing to learn.
    """
    messages = list(messages)
    if _has_chat_template(tokenizer):
        def render(turns, generation_prompt):
            return tokenizer.apply_chat_template(
                turns, tokenize=False, add_generation_prompt=generation_prompt,
            )
    else:
        def render(turns, generation_prompt):
            return transcript(turns) + (GENERATION_PROMPT if generation_prompt else "")

    input_ids = _token_ids(tokenizer, render(messages, False))
    eos = getattr(tokenizer, "eos_token_id", None)
    if not _has_chat_template(tokenizer) and eos is not None:
        input_ids.append(eos)  # teach a template-less model where a conversation stops
    labels = list(input_ids)
    if train_on == "assistant":
        labels = [IGNORE_INDEX] * len(input_ids)
        last = max(index for index, turn in enumerate(messages) if turn["role"] == "assistant")
        for index, turn in enumerate(messages):
            if turn["role"] != "assistant":
                continue
            start = _token_ids(tokenizer, render(messages[:index], True))
            end = _token_ids(tokenizer, render(messages[:index + 1], False))
            stop = len(input_ids) if index == last else len(end)
            consistent = (
                input_ids[:len(start)] == start and len(start) < stop
                and (index == last or input_ids[:len(end)] == end)
            )
            if not consistent:
                labels = list(input_ids)
                break
            labels[len(start):stop] = input_ids[len(start):stop]
    if max_length:
        input_ids, labels = input_ids[:max_length], labels[:max_length]
    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return input_ids, labels


def build_text_blocks(tokenizer, corpus_path: str | Path, block_size: int,
                      max_tokens: int = 20_000_000) -> list[Example]:
    """Tokenize a text file ~1 MB at a time and cut it into ``block_size``-token examples."""
    examples: list[Example] = []
    buffer: list[int] = []
    total = 0
    with open(corpus_path, encoding="utf-8", errors="replace") as handle:
        while total < max_tokens:
            piece = handle.read(PIECE_CHARS)
            if not piece:
                break
            piece += handle.readline()  # finish the line so no word is split between pieces
            buffer += _token_ids(tokenizer, piece)
            while len(buffer) >= block_size and total < max_tokens:
                block, buffer = buffer[:block_size], buffer[block_size:]
                examples.append((block, list(block)))
                total += block_size
    if len(buffer) >= MIN_BLOCK_TOKENS and total < max_tokens:
        examples.append((buffer, list(buffer)))
    return examples


def split_examples(examples: Sequence[Example], val_fraction: float = 0.05,
                   seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Shuffle deterministically and hold out a validation slice (never the whole set)."""
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    held_out = 0
    if len(order) >= 2:
        held_out = min(max(1, int(len(order) * val_fraction)), len(order) - 1,
                       MAX_VALIDATION_EXAMPLES)
    validation = [examples[index] for index in order[:held_out]]
    training = [examples[index] for index in order[held_out:]]
    return training, validation


def collate(batch: Sequence[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; padding is masked out of attention and of the loss."""
    width = max(len(input_ids) for input_ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (ids, row_labels) in enumerate(batch):
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[row, :len(row_labels)] = torch.tensor(row_labels, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
