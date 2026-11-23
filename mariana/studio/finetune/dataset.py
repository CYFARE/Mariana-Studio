"""Training examples for causal-LM fine-tuning: chat rows and plain text become token ids.

Pure data plumbing.  The tokenizer is passed in -- a Transformers tokenizer, or anything
with the same ``tokenizer(text, add_special_tokens=False)["input_ids"]`` call -- so this
module needs only the standard library and torch.
"""

from __future__ import annotations

import json
import os
import random
from array import array
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

Example = tuple[Sequence[int], Sequence[int]]


class ExampleStore(Sequence):
    """Tokenized examples in two flat int32 tensors plus offsets.

    A large corpus otherwise costs one Python int object per token (28 bytes or
    more).  Storing both ids and labels contiguously cuts that to 8 bytes per
    token and keeps random access and slicing cheap for the sampler.
    """

    def __init__(self, ids: torch.Tensor, labels: torch.Tensor, starts: array,
                 lengths: array, buffers: tuple) -> None:
        self._ids = ids
        self._labels = labels
        self._starts = starts
        self._lengths = lengths
        self._buffers = buffers  # frombuffer shares the array's memory
        self.tokens = int(ids.numel()) if ids.numel() else 0

    def __len__(self) -> int:
        return len(self._starts)

    def length(self, index: int) -> int:
        return int(self._lengths[index])

    @property
    def lengths(self) -> array:
        return self._lengths

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = int(self._starts[index])
        stop = (int(self._starts[index + 1]) if index + 1 < len(self._starts)
                else int(self._ids.numel()))
        return self._ids[start:stop], self._labels[start:stop]


class ExampleSubset(Sequence):
    """A view over an :class:`ExampleStore`; splitting does not copy the tokens."""

    def __init__(self, source: ExampleStore, indices: Iterable[int]) -> None:
        self.source = source
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def length(self, index: int) -> int:
        return self.source.length(self.indices[index])

    @property
    def lengths(self) -> list[int]:
        return [self.source.length(index) for index in self.indices]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.source[self.indices[item]]
                    for item in range(*index.indices(len(self)))]
        return self.source[self.indices[index]]


class ExampleBuilder:
    """Accumulate examples into compact arrays while enforcing memory caps."""

    def __init__(self, max_tokens: int = 20_000_000, max_examples: int = 500_000):
        self.max_tokens = max(0, int(max_tokens or 0))
        self.max_examples = max(0, int(max_examples or 0))
        self._ids = array("i")
        self._labels = array("i")
        self._starts = array("q")
        self._lengths = array("q")
        self.tokens = 0
        self.skipped = 0

    @property
    def full(self) -> bool:
        return bool((self.max_tokens and self.tokens >= self.max_tokens)
                    or (self.max_examples and len(self._starts) >= self.max_examples))

    def add(self, example: Example) -> bool:
        ids, labels = example
        length = len(ids)
        if length == 0 or len(labels) != length:
            return False
        if self.full or (self.max_tokens and self.tokens + length > self.max_tokens):
            self.skipped += 1
            return False
        self._starts.append(self.tokens)
        self._lengths.append(length)
        self._ids.extend(ids.tolist() if isinstance(ids, torch.Tensor) else list(ids))
        self._labels.extend(labels.tolist() if isinstance(labels, torch.Tensor) else list(labels))
        self.tokens += length
        return True

    def build(self) -> ExampleStore:
        if self._ids:
            ids = torch.frombuffer(self._ids, dtype=torch.int32)
            labels = torch.frombuffer(self._labels, dtype=torch.int32)
        else:
            ids = torch.empty(0, dtype=torch.int32)
            labels = torch.empty(0, dtype=torch.int32)
        return ExampleStore(ids, labels, self._starts, self._lengths,
                            (self._ids, self._labels))


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
    return list(iter_chat_examples(files, max_examples))


def iter_chat_examples(files: Iterable[str | Path],
                       max_examples: int | None = None) -> Iterator[list[dict]]:
    """Stream chat examples from ``files`` without materialising the whole dataset."""
    seen = 0
    for path in files:
        try:
            for row in _iter_rows(Path(path)):
                messages = normalize_row(row)
                if messages:
                    yield messages
                    seen += 1
                    if max_examples and seen >= max_examples:
                        return
        except OSError:
            continue


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


def iter_text_blocks(tokenizer, corpus_path: str | Path, block_size: int,
                     max_tokens: int = 20_000_000) -> Iterator[Example]:
    """Tokenize a text file ~1 MB at a time and yield ``block_size``-token examples."""
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
                yield block, list(block)
                total += block_size
    if len(buffer) >= MIN_BLOCK_TOKENS and total < max_tokens:
        yield buffer, list(buffer)


def build_text_blocks(tokenizer, corpus_path: str | Path, block_size: int,
                      max_tokens: int = 20_000_000) -> list[Example]:
    """Materialise :func:`iter_text_blocks`; kept for callers that need a list."""
    return list(iter_text_blocks(tokenizer, corpus_path, block_size, max_tokens))


def split_examples(examples: Sequence[Example], val_fraction: float = 0.05,
                   seed: int = 0) -> tuple[Sequence[Example], Sequence[Example]]:
    """Shuffle deterministically and hold out a validation slice (never the whole set).

    A compact :class:`ExampleStore` is returned as two views, so splitting a
    large token store does not duplicate its arrays.
    """
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    held_out = 0
    if len(order) >= 2:
        held_out = min(max(1, int(len(order) * val_fraction)), len(order) - 1,
                       MAX_VALIDATION_EXAMPLES)
    validation_indices = order[:held_out]
    training_indices = order[held_out:]
    if isinstance(examples, ExampleStore):
        return (ExampleSubset(examples, training_indices),
                ExampleSubset(examples, validation_indices))
    validation = [examples[index] for index in validation_indices]
    training = [examples[index] for index in training_indices]
    return training, validation


def example_lengths(examples: Sequence[Example]) -> Sequence[int]:
    """Lengths used to group similar examples; compact stores avoid slicing tokens."""
    lengths = getattr(examples, "lengths", None)
    if lengths is not None:
        return lengths
    return [len(ids) for ids, _ in examples]


def iter_batches(examples: Sequence[Example], batch_size: int, rng: random.Random,
                 bucket: bool = True) -> Iterator[list[int]]:
    """Endlessly yield shuffled batches, grouping similar lengths to cut padding.

    Every example is used before an epoch repeats, unlike independent random
    draws.  Sorting within a large block and shuffling the blocks keeps the
    batches both efficient and varied.
    """
    size = max(1, int(batch_size))
    lengths = example_lengths(examples)
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        if bucket and len(order) > size * 4:
            span = size * 50
            chunks = [order[start:start + span] for start in range(0, len(order), span)]
            for chunk in chunks:
                chunk.sort(key=lambda index: lengths[index])
            rng.shuffle(chunks)
            order = [index for chunk in chunks for index in chunk]
        for start in range(0, len(order), size):
            yield order[start:start + size]


def collate(batch: Sequence[Example], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; padding is masked out of attention and of the loss."""
    width = max(len(input_ids) for input_ids, _ in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, (ids, row_labels) in enumerate(batch):
        input_ids[row, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        labels[row, :len(row_labels)] = torch.as_tensor(row_labels, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
