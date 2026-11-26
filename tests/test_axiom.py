"""AXIOM codecs: connector algebra, dictionaries, sequences, detection, caching."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from mariana import axiom
from mariana.axiom import (
    OPERATORS,
    PROFILES,
    Codec,
    build_codec,
    canonical_text,
    decode_bytes,
    decode_text,
    decode_to_bytes,
    detect_language,
    detect_sequence,
    legacy_codec,
    prepare_corpus,
    rewrite,
    rewrite_bytes,
)
from mariana.cli import cfg_from_args, parse_args
from mariana.config import Config
from mariana.data import DEMO_TEXT, build_dataset


def code_of(name: str) -> int:
    return next(op.code for op in OPERATORS if op.name == name)


def test_alphabet_is_single_byte_and_round_trips():
    codes = [op.code for op in OPERATORS]
    assert len(codes) == len(set(codes)) == 28
    assert all(
        0x01 <= code <= 0x1F and code not in {0x09, 0x0A, 0x0D}
        for code in codes
    )
    for op in OPERATORS:
        result = rewrite(op.words[0], "compact")
        assert result.wire == bytes([op.code]), op.name
        assert result.decoded == op.words[0]


def test_longest_phrase_wins():
    cases = {
        "if and only if": "IFF",
        "is not": "IS_NOT",
        "at least": "GE",
        "greater than or equal to": "GE",
        "no less than": "GE",
        "in order to": "TO",
        "no one": "NONE",
        "as well as": "AND",
    }
    for text, name in cases.items():
        wire = rewrite(text, "compact").wire
        assert wire == bytes([code_of(name)]), (text, wire)
    # Ambiguous connectors stay with the model rather than being guessed.
    assert rewrite("such that", "compact").wire == b"such that"
    assert rewrite("since 1990", "compact").wire == b"since 1990"
    assert rewrite("then", "compact").wire == b"then"


def test_words_inside_other_words_are_untouched():
    for word in ("database", "atlas", "theatre", "android", "format", "origin"):
        assert rewrite(word, "compact").wire == word.encode(), word


def test_canonical_text_is_idempotent():
    samples = (
        "A produces B because C is in the box.",
        "The pressure is greater than 1,086 bar and it is not at least 2,000.",
        "There are twenty three cats at the station; none of them are inside.",
        "How does the value change? It increases from one to seven.",
        "We cannot use the old rule, so the new rule is simpler.",
    )
    for sample in samples:
        first, _, _, _ = canonical_text(sample, "compact", True)
        second, _, _, _ = canonical_text(first, "compact", True)
        assert second == first, sample


def test_number_words_become_numerals_without_colliding():
    assert rewrite("one hundred and five", "compact").wire == b"105"
    assert rewrite("twenty three", "compact").wire == b"23"
    assert rewrite("three hundred and five", "compact").wire == b"305"
    assert rewrite("one two", "compact").wire == b"one 2"
    assert rewrite("nineteen ninety", "compact").wire == b"19 90"
    # "one" alone is a pronoun too, so it is left alone.
    assert rewrite("one of them", "compact").wire == (
        b"one" + bytes([code_of("OF")]) + b"them"
    )
    # "and" without a preceding scale word is the boolean operator.
    assert rewrite("one and five", "compact").wire == (
        b"one" + bytes([code_of("AND")]) + b"5"
    )
    assert rewrite("no one", "compact").wire == bytes([code_of("NONE")])


def test_compact_profile_drops_only_safe_fillers():
    text = "the cat is in the box and a dog is basically asleep"
    reversible = rewrite(text, "reversible").decoded
    compact = rewrite(text, "compact").decoded
    assert reversible.count("the") == 2
    assert "the" not in compact
    assert "basically" not in compact
    # Single-letter "a" is kept because it can be a variable in technical text.
    assert "a dog" in compact


def test_decode_and_display_use_canonical_words():
    result = rewrite("A produces B because C is in the box.", "reversible")
    assert result.decoded == "A produces B because C is in the box."
    assert result.display == "A⇒B∵C≡∈the box."
    assert decode_text(chr(code_of("AND"))) == "and"
    assert decode_text("x" + chr(code_of("IS_NOT")) + "y") == "x is not y"


def test_invalid_utf8_is_identity_or_legacy():
    data = b"caf\xe9 and tea"
    wire, stats = rewrite_bytes(data, "auto")
    assert stats.codec.mode == "identity"
    assert wire == data
    assert decode_to_bytes(wire, stats.codec) == data

    # An explicit English profile keeps the legacy surrogateescape path.
    wire, stats = rewrite_bytes(data, "compact")
    assert stats.codec.legacy
    assert b"\xe9" in wire
    assert bytes([code_of("AND")]) in wire
    assert decode_to_bytes(wire, stats.codec) == data


def test_typography_and_whitespace_are_canonical():
    wire = rewrite("Café — “hello”…\r\nnext", "reversible").wire
    assert wire == 'Café - "hello"...\nnext'.encode()


def test_demo_corpus_shrinks_with_dictionary_and_connectors():
    result = rewrite(DEMO_TEXT, "auto")
    assert result.stats.codec.mode == "text"
    assert result.stats.codec.language == "en"
    assert result.stats.codec.connectors
    assert result.stats.dictionary_entries > 0
    assert result.stats.saved_percent > 20.0
    assert "Mariana" in result.decoded and "Trench" in result.decoded


def test_dictionary_compresses_repeated_text_losslessly():
    text = "The pressure at the bottom is greater than at the surface. " * 50
    result = rewrite(text, "lossless")
    assert result.stats.codec.mode == "text"
    assert result.stats.codec.connectors is False
    assert result.stats.dictionary_entries > 0
    assert result.stats.saved_percent > 50.0
    assert decode_to_bytes(result.wire, result.stats.codec) == text.encode()


def test_unicode_symbols_and_emoji_compress_losslessly():
    math = "∑ x ∈ S and ∫ f(x) dx ≈ 3 × 10^4, therefore x is at least 2. " * 20
    result = rewrite(math, "lossless")
    assert result.stats.saved_percent > 40.0
    assert decode_to_bytes(result.wire, result.stats.codec) == math.encode()

    emoji = "😀🚀😢" * 100
    result = rewrite(emoji, "lossless")
    assert result.stats.saved_percent > 50.0
    assert decode_to_bytes(result.wire, result.stats.codec) == emoji.encode()


def test_mixed_language_is_lossless_and_connectors_are_disabled():
    text = (
        "The cat is in the box. El gato está en la caja. "
        "Le chat est dans la boîte. 这是一个测试。\n"
    ) * 20
    result = rewrite(text, "auto")
    assert result.stats.codec.mode == "text"
    assert result.stats.codec.connectors is False
    assert result.stats.codec.language != "en"
    assert result.stats.saved_percent > 40.0
    assert decode_to_bytes(result.wire, result.stats.codec) == text.encode()


def test_minority_language_disables_connectors():
    text = (
        "The cat is in the box and the dog is outside.\n" * 8
        + "El gato está en la caja y el perro está fuera.\n" * 2
    ) * 2
    codec = build_codec(text, "auto")
    assert codec.language == "mixed"
    assert codec.connectors is False
    result = rewrite(text, "auto", codec=codec)
    assert decode_to_bytes(result.wire, codec) == text.encode()


def test_language_detection():
    assert detect_language(DEMO_TEXT) == "en"
    assert detect_language("这是一个测试，所有的猫都在盒子里。" * 20) == "zh"
    mixed = "The cat is in the box. El gato está en la caja. Le chat est dans la boîte. " * 8
    assert detect_language(mixed) == "mixed"


def test_dna_and_protein_are_detected_and_packed():
    random.seed(3)
    dna = "".join(random.choice("ACGT") for _ in range(4000))
    dna_text = "\n".join(dna[i:i + 70] for i in range(0, len(dna), 70)) + "\n"
    assert detect_sequence(dna_text) == "dna"
    result = rewrite(dna_text, "auto")
    assert result.stats.codec.mode == "dna"
    assert result.stats.codec.bits == 4
    assert result.stats.saved_percent > 40.0
    assert decode_to_bytes(result.wire, result.stats.codec) == dna_text.encode()

    protein = "".join(random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(4000))
    protein_text = (
        ">sp|TEST|demo\n"
        + "\n".join(protein[i:i + 60] for i in range(0, len(protein), 60))
        + "\n"
    )
    assert detect_sequence(protein_text) == "protein"
    result = rewrite(protein_text, "auto")
    assert result.stats.codec.mode == "protein"
    assert result.stats.codec.bits == 5
    assert result.stats.saved_percent > 25.0
    assert decode_to_bytes(result.wire, result.stats.codec) == protein_text.encode()


def test_random_binary_is_identity_and_never_expands():
    blob = bytes(range(256)) * 8
    wire, stats = rewrite_bytes(blob, "auto")
    assert stats.codec.mode == "identity"
    assert wire == blob
    assert stats.axiom_bytes == stats.original_bytes

    random.seed(4)
    ascii_blob = "".join(
        random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5000)
    ).encode()
    wire, stats = rewrite_bytes(ascii_blob, "auto")
    assert len(wire) <= len(ascii_blob)


def test_mixed_prose_and_fasta_stays_lossless():
    random.seed(5)
    protein = "".join(random.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(2000))
    text = (
        "Experiment notes: the sample was prepared at room temperature.\n"
        ">sp|MIXED|demo\n"
        + "\n".join(protein[i:i + 60] for i in range(0, len(protein), 60))
        + "\nA second paragraph describes the analysis and the controls.\n"
    )
    result = rewrite(text, "lossless")
    assert decode_to_bytes(result.wire, result.stats.codec) == text.encode()
    assert len(result.wire) <= len(text.encode())


def test_code_fences_inline_code_and_tables_are_protected():
    prose = "Use `and` in the box, and it is in the list."
    normal, _, _, _ = canonical_text(prose, "compact", True)
    assert "`and`" in normal
    assert normal.count(chr(code_of("AND"))) == 1

    fenced = "```\nA and B is in the box\n```\n"
    normal, _, _, _ = canonical_text(fenced, "compact", True)
    assert normal == fenced
    assert normal.count(chr(code_of("AND"))) == 0

    table = "| and | is | in |\n"
    normal, _, _, _ = canonical_text(table, "compact", True)
    assert normal == table

    code = "def f(x):\n    return x and y in z\n"
    normal, _, _, _ = canonical_text(code, "compact", True)
    assert normal == code

    math = "The relation $x \\in S$ is true and valid."
    normal, _, _, _ = canonical_text(math, "compact", True)
    assert "$x \\in S$" in normal
    assert normal.count(chr(code_of("IS"))) == 1

    links = "Visit https://example.com/and/in and email a.b@example.com; <b>is</b> inline."
    normal, _, _, _ = canonical_text(links, "compact", True)
    assert "https://example.com/and/in" in normal
    assert "a.b@example.com" in normal
    assert "<b>" in normal and "</b>" in normal
    assert normal.count(chr(code_of("AND"))) == 1


def test_codec_json_round_trips():
    samples = [
        "The pressure at the bottom is greater than at the surface. " * 20,
        "这是一个测试，所有的猫都在盒子里。" * 20,
        "😀🚀😢" * 50,
    ]
    for sample in samples:
        codec = build_codec(sample, "auto")
        assert Codec.from_json(codec.to_json()) == codec


def test_empty_and_tiny_inputs_are_safe():
    for text in ("", "a", "7", "∑"):
        result = rewrite(text, "auto")
        assert len(result.wire) <= max(1, len(text.encode())) + 1
    assert rewrite("", "auto").wire == b""


def test_code_and_latex_are_lossless_and_compress():
    code = (
        "def transform(values):\n"
        "    total = sum(values)\n"
        "    return [value / total for value in values]\n"
    ) * 20
    result = rewrite(code, "lossless")
    assert result.stats.codec.mode == "text"
    assert result.stats.saved_percent > 30.0
    assert decode_to_bytes(result.wire, result.stats.codec) == code.encode()

    latex = (
        "\\begin{equation}\n"
        "\\sum_{i=1}^{n} x_i \\approx \\int f(x)\\,dx\n"
        "\\end{equation}\n"
    ) * 20
    result = rewrite(latex, "lossless")
    assert result.stats.saved_percent > 30.0
    assert decode_to_bytes(result.wire, result.stats.codec) == latex.encode()


def test_iupac_dna_and_protein_gaps_pack_losslessly():
    dna = "ACGTNRYKMSWBDHV" * 40
    dna_text = "\n".join(dna[i:i + 60] for i in range(0, len(dna), 60)) + "\n"
    result = rewrite(dna_text, "auto")
    assert result.stats.codec.mode == "dna"
    assert decode_to_bytes(result.wire, result.stats.codec) == dna_text.encode()

    protein = "ACDE-FGHIKLMNPQRSTVWY*BXZJUO." * 20
    protein_text = (
        ">sp|GAPS|demo\n"
        + "\n".join(protein[i:i + 60] for i in range(0, len(protein), 60))
        + "\n"
    )
    result = rewrite(protein_text, "auto")
    assert result.stats.codec.mode == "protein"
    assert decode_to_bytes(result.wire, result.stats.codec) == protein_text.encode()

    # Lowercase sequence lines are outside the packer alphabet and become exact
    # literal records rather than being silently uppercased.
    lower = dna.lower() + "\n"
    result = rewrite(lower, "auto")
    assert decode_to_bytes(result.wire, result.stats.codec) == lower.encode()


def test_late_invalid_utf8_falls_back_to_identity(tmp_path):
    source = tmp_path / "late-binary.bin"
    source.write_bytes(("The cat is in the box. " * 64).encode() + b"\xff\xfe")
    destination = tmp_path / "prepared.bin"
    sample = ("The cat is in the box. " * 64).encode()
    with patch("mariana.axiom._sample_file", return_value=sample):
        stats = axiom.rewrite_file(source, destination)
    assert stats.codec.mode == "identity"
    assert destination.read_bytes() == source.read_bytes()


def test_in_memory_auto_never_expands(monkeypatch):
    monkeypatch.setattr(axiom, "SAMPLE_BYTES", 80)
    text = "abc abc def def " * 10 + "∑😀é" * 3000
    result = rewrite(text, "auto")
    assert result.stats.codec.mode == "identity"
    assert result.wire == text.encode()

    wire, stats = rewrite_bytes(text.encode(), "auto")
    assert stats.codec.mode == "identity"
    assert wire == text.encode()


def test_whole_file_expansion_falls_back_to_identity(tmp_path):
    source = tmp_path / "symbols.txt"
    source.write_text("∑😀é" * 2000, encoding="utf-8")
    sample = ("abc abc def def " * 1000).encode()
    with patch("mariana.axiom._sample_file", return_value=sample):
        path, stats = prepare_corpus(source, cache_root=tmp_path / "cache")
    assert stats.codec.mode == "identity"
    assert Path(path).read_bytes() == source.read_bytes()


def test_identity_fallback_cache_is_reused(tmp_path):
    source = tmp_path / "symbols.txt"
    source.write_text("∑😀é" * 2000, encoding="utf-8")
    sample = ("abc abc def def " * 1000).encode()
    cache = tmp_path / "cache"
    with patch("mariana.axiom._sample_file", return_value=sample):
        path, stats = prepare_corpus(source, cache_root=cache)
    assert stats.codec.mode == "identity"
    with patch("mariana.axiom._sample_file", return_value=sample), \
            patch("mariana.axiom.rewrite_file",
                  side_effect=AssertionError("identity cache miss")):
        again, cached = prepare_corpus(source, cache_root=cache)
    assert again == path
    assert cached.codec.mode == "identity"
    assert cached.axiom_bytes == stats.axiom_bytes


def test_prepare_corpus_rekeys_a_late_identity_fallback(tmp_path):
    source = tmp_path / "late-binary.bin"
    source.write_bytes(("The cat is in the box. " * 64).encode() + b"\xff\xfe")
    sample = ("The cat is in the box. " * 64).encode()
    with patch("mariana.axiom._sample_file", return_value=sample):
        path, stats = prepare_corpus(source, cache_root=tmp_path / "cache")
    assert stats.codec.mode == "identity"
    payload = json.loads(Path(path + ".json").read_text(encoding="utf-8"))
    assert payload["codec"]["mode"] == "identity"
    assert Path(path).read_bytes() == source.read_bytes()


def test_stored_sequence_codec_rejects_the_wrong_corpus(tmp_path):
    random.seed(6)
    dna = "".join(random.choice("ACGT") for _ in range(2000))
    dna_text = "\n".join(dna[i:i + 70] for i in range(0, len(dna), 70)) + "\n"
    codec = build_codec(dna_text, "auto")
    assert codec.mode == "dna"

    source = tmp_path / "prose.txt"
    source.write_text("This is prose about the experiment. " * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="expects dna"):
        prepare_corpus(source, cache_root=tmp_path / "cache", codec=codec)


def test_codec_validation_rejects_impossible_specs():
    with pytest.raises(ValueError, match="mode"):
        Codec(mode="unknown")
    with pytest.raises(ValueError, match="dictionary"):
        Codec(mode="text", dictionary=tuple("x" for _ in range(130)))
    with pytest.raises(ValueError, match="dictionary"):
        Codec(mode="text", dictionary=("a\nb",))
    with pytest.raises(ValueError, match="alphabet"):
        Codec(mode="dna", alphabet="ACGT", bits=1)
    with pytest.raises(ValueError, match="alphabet"):
        Codec(mode="dna", alphabet="AACGTUNRYKMSWBDH", bits=4)
    with pytest.raises(ValueError, match="identity"):
        Codec(mode="identity", connectors=True)
    with pytest.raises(ValueError, match="profile"):
        Codec(mode="text", profile="bogus")
    with pytest.raises(ValueError, match="dictionary"):
        Codec(mode="text", dictionary=("x" * 2000,))
    with pytest.raises(ValueError, match="legacy"):
        Codec(mode="text", legacy=True, connectors=False)
    with pytest.raises(ValueError, match="dictionary"):
        Codec(mode="text", dictionary=("\ud800",))
    with pytest.raises(ValueError, match="alphabet"):
        Codec(mode="dna", alphabet="ACGT\ud800", bits=4)
    with pytest.raises(ValueError, match="JSON"):
        Codec.from_json("not json")


def test_reserved_control_bytes_disable_semantic_rewriting():
    text = "The cat is in the box.\x01 And the dog is outside.\n" * 20
    result = rewrite(text, "auto")
    assert result.stats.codec.connectors is False
    assert result.stats.codec.mode == "text"
    assert decode_to_bytes(result.wire, result.stats.codec) == text.encode()


def test_safe_control_bytes_keep_connectors():
    # NUL and DEL are escaped by the text codec and must not cost the semantic
    # layer; only a byte that decodes as a connector is ambiguous.
    text = "The cat is in the box.\x00\x7f And the dog is outside.\n" * 20
    result = rewrite(text, "compact")
    assert result.stats.codec.connectors is True
    assert result.stats.codec.mode == "text"
    assert "\x00\x7f" in result.decoded
    lossless = rewrite(text, "lossless")
    assert decode_to_bytes(lossless.wire, lossless.stats.codec) == text.encode()


def test_literal_operator_bytes_keep_connectors_and_are_escaped():
    text = "The cat is in the box.\x01 And the dog is outside.\n" * 20
    for profile in ("compact", "reversible"):
        result = rewrite(text, profile)
        assert result.stats.codec.mode == "text", profile
        assert result.stats.codec.connectors is True, profile
        assert result.stats.operator_hits > 0, profile
        assert b"\x00\x01" in result.wire, profile
        assert b"\x01" in decode_to_bytes(result.wire, result.stats.codec), profile

    # Every reserved byte has an unambiguous escape, including 0x0B and 0x0C.
    controls = "".join(chr(op.code) for op in OPERATORS)
    text = (
        "The cat is in the box " + controls + " and the dog is outside.\n"
    ) * 5
    result = rewrite(text, "compact")
    assert result.stats.codec.connectors is True
    assert all(bytes([0, op.code]) in result.wire for op in OPERATORS)
    decoded = decode_to_bytes(result.wire, result.stats.codec)
    assert decoded.count(controls.encode()) == 5


def test_stored_connector_codec_escapes_operator_bytes():
    clean = "The cat produces a sound at dawn and the dog is at home.\n" * 100
    codec = build_codec(clean, "compact")
    assert codec.connectors
    text = clean + "late \x01 byte\n"
    result = rewrite(text, "compact", codec=codec)
    assert b"\x00\x01" in result.wire
    assert b"\x01" in decode_to_bytes(result.wire, codec)
    wire, stats = rewrite_bytes(text.encode(), "compact", codec=codec)
    assert b"\x00\x01" in wire
    assert stats.codec.connectors is True


def test_prepare_corpus_escapes_late_operator_bytes_for_a_stored_codec(tmp_path):
    clean = "The cat is in the box and produces heat.\n" * 100
    codec = build_codec(clean, "compact")
    assert codec.connectors
    source = tmp_path / "late-control.txt"
    source.write_text(clean + "late \x01 byte\n", encoding="utf-8")
    with patch("mariana.axiom._sample_file", return_value=clean.encode()):
        path, stats = prepare_corpus(
            source, cache_root=tmp_path / "cache", codec=codec
        )
    assert stats.codec.connectors is True
    wire = Path(path).read_bytes()
    assert b"\x00\x01" in wire
    assert b"\x01" in decode_to_bytes(wire, codec)


def test_prepare_corpus_keeps_connectors_for_a_late_operator_byte(tmp_path):
    clean = "The cat is in the box and produces heat.\n" * 100
    source = tmp_path / "late-control.txt"
    source.write_text(clean + "late \x01 byte\n", encoding="utf-8")
    with patch("mariana.axiom._sample_file", return_value=clean.encode()):
        path, stats = prepare_corpus(
            source, cache_root=tmp_path / "cache", profile="compact"
        )
    assert stats.codec.mode == "text"
    assert stats.codec.connectors is True
    wire = Path(path).read_bytes()
    assert b"\x00\x01" in wire
    assert b"\x01" in decode_to_bytes(wire, stats.codec)


def test_inline_sentinel_like_text_is_not_corrupted():
    text = "`code` and \ue0000\ue001 literal and `x`\n"
    decoded = rewrite(text, "reversible").decoded
    assert "\ue0000\ue001" in decoded
    assert decoded.count("`code`") == 1


def test_rewrite_file_refuses_in_place(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("The cat is in the box.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="in place"):
        axiom.rewrite_file(source, source, profile="compact")
    assert source.read_text(encoding="utf-8") == "The cat is in the box.\n"


def test_rewrite_file_refuses_a_hard_link_to_the_source(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("The cat is in the box.\n", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    os.link(source, alias)
    with pytest.raises(ValueError, match="in place"):
        axiom.rewrite_file(source, alias, profile="compact")
    assert source.read_text(encoding="utf-8") == "The cat is in the box.\n"


def test_rewrite_accepts_bytes_like_rewrite_bytes():
    data = "The cat is in the box and it produces heat.\n".encode() * 20
    result = rewrite(data, "lossless")
    assert result.stats.original_bytes == len(data)
    assert decode_to_bytes(result.wire, result.stats.codec) == data


def test_mixed_line_endings_stay_consistent_and_idempotent(tmp_path):
    # The protected line ends in a lone CR. Normalising the next line's CRLF
    # would merge the two endings and shift the protection mask.
    text = (
        "    code\r"
        "and the cat \r\n"
        "`inline code` http://example.com/a?b=1 and the dog\n"
        "the ```\n"
        "and $x+y$ \r\r\n"
        "```\n"
        "\n\n\n"
        "the "
    )
    for profile in PROFILES:
        codec = build_codec(text, profile)
        memory = rewrite(text, profile, codec=codec)
        source = tmp_path / f"mixed-{profile}.txt"
        destination = tmp_path / f"mixed-{profile}.bin"
        source.write_text(text, encoding="utf-8", newline="")
        axiom.rewrite_file(source, destination, profile=profile, codec=codec)
        assert destination.read_bytes() == memory.wire, profile
        first = canonical_text(text, codec.profile, codec.connectors)[0]
        second = canonical_text(first, codec.profile, codec.connectors)[0]
        assert second == first, profile


def test_filler_removal_does_not_create_a_fence_marker():
    text = "the ```\nand the cat is in the box\n```\n"
    first = canonical_text(text, "compact", True)[0]
    assert first.startswith("the ```")
    assert canonical_text(first, "compact", True)[0] == first


def test_unpaired_surrogates_get_a_clear_error():
    with pytest.raises(ValueError, match="surrogate"):
        rewrite("\ud800", "auto")
    # A low surrogate produced by surrogateescape still round-trips as bytes.
    result = rewrite("\udcff", "auto")
    assert result.wire == b"\xff"
    assert decode_to_bytes(result.wire, result.stats.codec) == b"\xff"


def test_invalid_profile_is_rejected_even_with_a_codec():
    codec = rewrite("a", "compact").stats.codec
    with pytest.raises(ValueError, match="profile"):
        rewrite("a", "bogus", codec=codec)
    with pytest.raises(ValueError, match="profile"):
        rewrite_bytes(b"a", "bogus", codec=codec)


def test_sample_file_respects_utf8_boundaries(tmp_path):
    source = tmp_path / "emoji.txt"
    source.write_text("😀🚀😢" * 20, encoding="utf-8")
    sample = axiom._sample_file(source, budget=11)
    sample.decode("utf-8")  # a seek must not leave a split character
    assert sample


def test_rewrite_file_preserves_fenced_code(tmp_path):
    source = tmp_path / "code.txt"
    destination = tmp_path / "code.axiom.bin"
    source.write_text(
        "```\nA and B is in the box\n```\nOutside, and in the list.\n",
        encoding="utf-8",
    )
    stats = axiom.rewrite_file(source, destination, profile="reversible")
    assert stats.operator_hits == 2  # only the prose line outside the fence
    assert decode_to_bytes(destination.read_bytes(), stats.codec) == source.read_bytes()


def test_dictionary_uses_the_full_code_range():
    words = [f"term{index:03d}" for index in range(200)]
    text = " ".join(words * 20)
    result = rewrite(text, "lossless")
    assert result.stats.dictionary_entries == 129
    assert 0xFF in result.wire
    assert decode_to_bytes(result.wire, result.stats.codec) == text.encode()


def test_memory_and_file_encoders_match(tmp_path):
    text = "```\nA and B is in the box\n```\nOutside, and in the list.\n"
    for profile in PROFILES:
        codec = build_codec(text, profile)
        memory = rewrite(text, profile, codec=codec)
        source = tmp_path / f"corpus-{profile}.txt"
        destination = tmp_path / f"corpus-{profile}.bin"
        source.write_text(text, encoding="utf-8")
        stats = axiom.rewrite_file(source, destination, profile=profile, codec=codec)
        assert destination.read_bytes() == memory.wire
        assert stats.axiom_bytes == memory.stats.axiom_bytes


def test_sequence_decode_is_strict_or_best_effort():
    random.seed(8)
    dna = "".join(random.choice("ACGT") for _ in range(2000))
    text = "\n".join(dna[i:i + 60] for i in range(0, len(dna), 60)) + "\n"
    result = rewrite(text, "auto")
    assert result.stats.codec.mode == "dna"
    with pytest.raises(ValueError):
        decode_to_bytes(result.wire[:-1], result.stats.codec)
    assert isinstance(decode_bytes(result.wire[:-1], result.stats.codec), str)


def test_rewrite_file_streams_and_reports(tmp_path):
    source = tmp_path / "corpus.txt"
    destination = tmp_path / "corpus.axiom.bin"
    source.write_text("A produces B.\nC is in the box.\n", encoding="utf-8")
    stats = axiom.rewrite_file(source, destination, profile="reversible")
    wire = destination.read_bytes()
    assert stats.operator_hits >= 2
    assert bytes([code_of("PRODUCES")]) in wire
    assert decode_to_bytes(wire, stats.codec) == source.read_bytes()


def test_prepare_corpus_is_cached(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(
        "The cat is in the box and it produces heat. " * 64, encoding="utf-8"
    )
    cache = tmp_path / "cache"
    path, stats = prepare_corpus(source, cache_root=cache, profile="compact")
    assert Path(path).is_file()
    assert stats.saved_percent > 20.0
    manifest = Path(path + ".json")
    assert manifest.is_file()

    with patch("mariana.axiom.rewrite_file", side_effect=AssertionError("cache miss")):
        again, cached = prepare_corpus(source, cache_root=cache, profile="compact")
    assert again == path
    assert cached.axiom_bytes == stats.axiom_bytes
    assert cached.codec == stats.codec


def test_truncated_cache_payload_is_rebuilt(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text(
        "The cat is in the box and it produces heat. " * 64, encoding="utf-8"
    )
    cache = tmp_path / "cache"
    path, stats = prepare_corpus(source, cache_root=cache, profile="compact")
    payload = Path(path)
    payload.write_bytes(payload.read_bytes()[:-3])
    with patch("mariana.axiom.rewrite_file", wraps=axiom.rewrite_file) as rewrite_spy:
        again, rebuilt = prepare_corpus(source, cache_root=cache, profile="compact")
    assert rewrite_spy.called
    assert again == path
    assert rebuilt.axiom_bytes == stats.axiom_bytes
    assert payload.stat().st_size == stats.axiom_bytes


def test_rewritten_corpus_survives_the_training_data_pipeline():
    source = ("the cat is in the box and it produces heat " * 200).encode("utf-8")
    prepared, stats = rewrite_bytes(source, "compact")
    assert stats.codec.mode == "text"
    train, val = build_dataset(prepared, 32)
    assert train.shape[1] == 33 and val.shape[1] == 33
    sample = bytes(int(value) for value in train[0].tolist())
    assert axiom.is_wire(sample)


def test_cli_flags_reach_the_checkpoint_config():
    args = parse_args(["train", "--axiom", "--axiom-profile", "reversible"])
    cfg = cfg_from_args(args)
    assert cfg.use_axiom and cfg.axiom_profile == "reversible"
    restored = Config.from_dict(cfg.to_dict())
    assert restored.use_axiom and restored.axiom_profile == "reversible"

    default = cfg_from_args(parse_args(["train"]))
    assert not default.use_axiom and default.axiom_profile == "auto"

    old = Config.from_dict({"dim": 64})
    assert not old.use_axiom and old.axiom_profile == "auto"
    assert old.axiom_codec == ""

    codec = rewrite(DEMO_TEXT, "auto").stats.codec
    cfg.axiom_codec = codec.to_json()
    loaded = Config.from_dict(cfg.to_dict())
    assert axiom.codec_from_settings(loaded.axiom_profile, loaded.axiom_codec) == codec


def test_operator_table_matches_the_alphabet():
    table = axiom.operator_table()
    assert len(table) == len(OPERATORS)
    assert {entry["code"] for entry in table} == {
        f"0x{op.code:02X}" for op in OPERATORS
    }
    assert all(entry["canonical"] and entry["glyph"] for entry in table)
    assert "auto" in PROFILES and "lossless" in PROFILES
    assert legacy_codec("compact").legacy
