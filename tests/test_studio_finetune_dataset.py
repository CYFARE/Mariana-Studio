"""Fine-tuning datasets: row formats, chat-file detection, label masking and batching."""

from __future__ import annotations

import json
import math
import random

from mariana.studio.finetune.dataset import (
    IGNORE_INDEX,
    ExampleBuilder,
    ExampleStore,
    build_text_blocks,
    collate,
    encode_chat,
    find_chat_files,
    iter_batches,
    iter_chat_examples,
    iter_text_blocks,
    load_chat_examples,
    normalize_row,
    split_examples,
    transcript,
)
from mariana.studio.models.backends import transcript as chat_transcript

CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "See you."},
]


class CharTokenizer:
    """One token per character; optionally renders a simple chat template."""

    eos_token_id = 0

    def __init__(self, template: bool = True):
        self.chat_template = "template" if template else None

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")


def _decode(ids):
    return "".join(chr(value) for value in ids if value > 0)


def test_transcript_matches_the_layout_chat_uses():
    assert transcript(CONVERSATION[:3]) == "System: Be brief.\nUser: Hi\nAssistant: Hello!\n"
    assert transcript(CONVERSATION) == chat_transcript(CONVERSATION)


def test_normalize_row_understands_common_dataset_formats():
    expected = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    assert normalize_row({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        {"role": "tool", "content": "ignored"},
        {"role": "assistant", "content": "A"},
    ]}) == expected
    assert normalize_row({"conversations": [
        {"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"},
    ]}) == expected
    for reply_key in ("completion", "response", "output", "answer"):
        assert normalize_row({"prompt": "Q", reply_key: "A"}) == expected
    assert normalize_row({"question": "Q", "answer": "A"}) == expected
    alpaca = normalize_row({"instruction": "Translate", "input": "chat", "output": "cat",
                            "system": "You translate."})
    assert alpaca == [
        {"role": "system", "content": "You translate."},
        {"role": "user", "content": "Translate\n\nchat"},
        {"role": "assistant", "content": "cat"},
    ]
    assert normalize_row({"messages": [{"role": "user", "content": "only a question"}]}) is None
    assert normalize_row({"text": "plain"}) is None and normalize_row("text") is None


def test_chat_files_are_found_and_plain_json_is_ignored(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / ".hidden").mkdir()
    rows = [{"prompt": f"q{i}", "completion": f"a{i}"} for i in range(4)]
    (tmp_path / "chat.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\nnot json\n", encoding="utf-8")
    (tmp_path / "nested" / "data.json").write_text(json.dumps({"data": rows}), encoding="utf-8")
    (tmp_path / "plain.json").write_text(json.dumps({"text": "no chat here"}), encoding="utf-8")
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    (tmp_path / "gui-corpus-abc.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
    (tmp_path / ".hidden" / "secret.jsonl").write_text(json.dumps(rows[0]), encoding="utf-8")
    found = find_chat_files(tmp_path)
    assert [path.name for path in found] == ["chat.jsonl", "data.json"]
    assert find_chat_files(tmp_path / "chat.jsonl") == [tmp_path / "chat.jsonl"]
    assert find_chat_files(tmp_path / "missing") == []
    examples = load_chat_examples(found)
    assert len(examples) == 8  # the malformed line is skipped, not fatal
    assert len(load_chat_examples(found, max_examples=3)) == 3


def test_only_assistant_replies_are_supervised():
    tokenizer = CharTokenizer()
    input_ids, labels = encode_chat(tokenizer, CONVERSATION)
    assert _decode(input_ids) == tokenizer.apply_chat_template(CONVERSATION)
    learned = _decode(label for label in labels if label != IGNORE_INDEX)
    assert learned == "Hello!</s>See you.</s>"  # replies and the token that closes each
    assert len(labels) == len(input_ids)
    every = encode_chat(tokenizer, CONVERSATION, train_on="all")
    assert every[1] == every[0]
    # Truncation that removes every reply leaves nothing to learn.
    assert encode_chat(tokenizer, CONVERSATION, max_length=20) is None
    kept = encode_chat(tokenizer, CONVERSATION, max_length=45)
    assert len(kept[0]) == 45 and any(label != IGNORE_INDEX for label in kept[1])


def test_models_without_a_template_train_on_the_transcript():
    tokenizer = CharTokenizer(template=False)
    input_ids, labels = encode_chat(tokenizer, CONVERSATION[1:3])
    assert _decode(input_ids) == "User: Hi\nAssistant: Hello!\n"
    assert input_ids[-1] == tokenizer.eos_token_id  # the conversation ends with EOS
    learned = [label for label in labels if label != IGNORE_INDEX]
    assert _decode(learned) == " Hello!\n" and learned[-1] == tokenizer.eos_token_id


def test_text_blocks_cover_the_corpus_across_read_boundaries(tmp_path, monkeypatch):
    from mariana.studio.finetune import dataset

    monkeypatch.setattr(dataset, "PIECE_CHARS", 64)
    text = "".join(f"line {index} of the corpus\n" for index in range(40))
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    blocks = build_text_blocks(CharTokenizer(), corpus, block_size=50)
    assert all(len(ids) == 50 for ids, _ in blocks[:-1]) and blocks[0][0] == blocks[0][1]
    assert "".join(_decode(ids) for ids, _ in blocks) == text
    limited = build_text_blocks(CharTokenizer(), corpus, block_size=50, max_tokens=100)
    assert len(limited) == 2


def test_split_is_deterministic_and_collate_pads_with_ignored_labels():
    examples = [([index] * (index + 1), [index] * (index + 1)) for index in range(40)]
    train, validation = split_examples(examples, seed=3)
    assert len(validation) == 2 and len(train) == 38
    assert split_examples(examples, seed=3) == (train, validation)
    assert split_examples(examples[:1]) == (examples[:1], [])
    assert len(split_examples(examples[:2])[1]) == 1
    batch = collate([([5, 6, 7], [IGNORE_INDEX, 6, 7]), ([9], [9])], pad_id=1)
    assert batch["input_ids"].tolist() == [[5, 6, 7], [9, 1, 1]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 6, 7], [9, IGNORE_INDEX, IGNORE_INDEX]]


def test_a_compact_store_keeps_slices_and_splits_as_views():
    builder = ExampleBuilder(max_tokens=8, max_examples=10)
    assert builder.add(([1, 2, 3], [1, 2, 3]))
    assert builder.add(([4, 5], [IGNORE_INDEX, 5]))
    assert not builder.add(([6, 7, 8, 9], [6, 7, 8, 9]))  # would pass the token cap
    assert builder.skipped == 1
    store = builder.build()
    assert isinstance(store, ExampleStore) and len(store) == 2 and store.tokens == 5
    ids, labels = store[1]
    assert ids.tolist() == [4, 5] and labels.tolist() == [IGNORE_INDEX, 5]
    assert store[-1][0].tolist() == [4, 5]
    assert [item[0].tolist() for item in store[0:2]] == [[1, 2, 3], [4, 5]]
    train, validation = split_examples(store, seed=1)
    assert len(train) + len(validation) == 2
    assert train.source is store and validation.source is store
    batch = collate([store[0], store[1]], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]


def test_batches_cover_every_example_once_per_epoch():
    examples = [([index] * (index + 2), [index] * (index + 2)) for index in range(23)]
    rng = random.Random(5)
    batches = iter_batches(examples, 4, rng)
    seen = []
    for _ in range(math.ceil(len(examples) / 4)):
        batch = next(batches)
        assert 1 <= len(batch) <= 4
        seen.extend(batch)
    assert sorted(seen) == list(range(len(examples)))


def test_text_blocks_can_be_streamed_and_match_the_list(tmp_path):
    corpus = '\n'.join(f"line {index} of a longer corpus" for index in range(30))
    path = tmp_path / "corpus.txt"
    path.write_text(corpus, encoding="utf-8")
    blocks = build_text_blocks(CharTokenizer(), path, block_size=17)
    streamed = list(iter_text_blocks(CharTokenizer(), path, block_size=17))
    assert streamed == blocks


def test_chat_rows_can_be_streamed_without_a_full_list(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text("\n".join(json.dumps({
        "messages": [{"role": "user", "content": f"q{index}"},
                     {"role": "assistant", "content": f"a{index}"}]}) for index in range(5)),
                    encoding="utf-8")
    stream = iter_chat_examples([path], max_examples=3)
    assert len(next(stream)) == 2
    assert sum(1 for _ in iter_chat_examples([path])) == 5
