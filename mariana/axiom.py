"""AXIOM: a domain-adaptive corpus codec for the byte-level DRV-LM.

AXIOM has two layers.

The **linguistic layer** rewrites English connectors to one-byte mathematical
operators ("A produces B" becomes ``A⇒B``), canonicalises number words, and
iterates to a fixed point. This is the semantic algebra described in the
whitepaper; it is intentionally English-only because a wrong connector mapping
in another language is worse than no mapping at all.

The **codec layer** makes AXIOM useful on any dataset:

* a lossless, corpus-derived dictionary replaces frequent words, phrases,
  Unicode symbols, emoji and k-mers with one-byte codes;
* DNA and protein corpora are detected and bit-packed (4 and 5 bits per
  residue);
* an identity codec passes through binary or incompressible data unchanged;
* a sample-based detector chooses the codec before training, and the selected
  codec is stored in the checkpoint so generation decodes with the same rules.

The model is unchanged. Every wire byte is an ordinary byte id, so Sprint-DRV,
SAINT, MoE, MTP and export paths work without new parameters or vocabulary.
This is still a lexical/data codec, not a semantic parser; the whitepaper lists
the limits, and the identity fallback means AXIOM never silently expands an
incompressible corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

AXIOM_VERSION = "2.2"
PROFILES = ("auto", "compact", "reversible", "lossless")
MODES = ("text", "dna", "protein", "identity")
DICTIONARY_LIMIT = 129
DICTIONARY_BASE = 0x7F
SAMPLE_BYTES = 8_000_000
DICTIONARY_SAMPLE_CHARS = 2_000_000
MAX_RECORD = 64 * 1024 * 1024

# C0 control bytes that C compilers, terminals and text editors leave alone.
# Tab, newline and carriage return stay whitespace; they are not operators.
_OPERATOR_CODES = (
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0x0B, 0x0C,
    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
    0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
)


@dataclass(frozen=True)
class Operator:
    """One relation in the connector algebra.

    ``words[0]`` is the canonical English word: it is what ``decode`` writes
    and what ``encode`` maps back, so a single operator survives a decode /
    encode round trip. Adjacent operators may still merge into a more canonical
    operator, as ``≡¬`` does when it decodes to "is not" and re-encodes as
    ``≢``.
    """

    name: str
    code: int
    glyph: str
    words: tuple[str, ...]


# The alphabet is deliberately small: 28 relations, one byte each. Every entry
# names a relation that survives a literal mathematical reading; words whose
# meaning depends on context (``if``, ``when``, ``about``, ``that``) are left
# to the model rather than guessed here.
OPERATORS: tuple[Operator, ...] = (
    Operator("AND", 0x01, "∧", ("and", "as well as")),
    Operator("OR", 0x02, "∨", ("or",)),
    Operator("NOT", 0x03, "¬", ("not", "never", "n't")),
    Operator("ALL", 0x04, "∀", ("all", "every", "each", "everything", "everyone")),
    Operator("EXISTS", 0x05, "∃", (
        "some", "any", "someone", "something", "there exists", "there is at least one",
    )),
    Operator("NONE", 0x06, "∄", ("none", "no", "nothing", "nobody", "no one")),
    Operator("IN", 0x07, "∈", ("in", "within")),
    Operator("AT", 0x08, "@", ("at",)),
    Operator("TO", 0x0B, "→", ("to", "into", "onto", "toward", "towards",
                               "in order to", "so as to")),
    Operator("FROM", 0x0C, "←", ("from", "out of")),
    Operator("PRODUCES", 0x0E, "⇒", (
        "produces", "produce", "produced", "producing",
        "yields", "yield", "yielded",
        "causes", "cause", "caused",
        "generates", "generate", "generated",
        "leads to", "lead to", "led to",
        "results in", "result in", "resulted in",
        "gives rise to", "gave rise to",
        "creates", "create", "created",
    )),
    Operator("THEREFORE", 0x0F, "∴", ("therefore", "thus", "hence",
                                      "consequently")),
    Operator("BECAUSE", 0x10, "∵", ("because", "because of", "due to",
                                    "as a result of")),
    Operator("IMPLIES", 0x11, "⟹", ("implies", "imply", "implied",
                                    "means that", "entails", "entail")),
    Operator("IFF", 0x12, "⟺", ("if and only if", "iff", "exactly when")),
    Operator("IS", 0x13, "≡", ("is", "are", "was", "were", "am",
                               "equals", "equal to", "is equal to")),
    Operator("IS_NOT", 0x14, "≢", (
        "is not", "are not", "was not", "were not",
        "isn't", "aren't", "wasn't", "weren't",
        "not equal to", "differs from", "differ from",
    )),
    Operator("HAS", 0x15, "∋", ("contains", "contain", "contained",
                                "includes", "include", "included",
                                "possesses")),
    Operator("OF", 0x16, "∘", ("of", "belonging to", "part of")),
    Operator("WITH", 0x17, "⊕", ("with", "together with", "combined with",
                                 "accompanied by")),
    Operator("WITHOUT", 0x18, "⊖", ("without", "lacking", "devoid of")),
    Operator("GT", 0x19, ">", ("greater than", "more than", "exceeds", "exceed",
                               "larger than", "higher than")),
    Operator("LT", 0x1A, "<", ("less than", "fewer than",
                               "smaller than", "lower than")),
    Operator("GE", 0x1B, "≥", ("at least", "greater than or equal to",
                               "no less than", "not less than")),
    Operator("LE", 0x1C, "≤", ("at most", "less than or equal to",
                               "no more than", "not more than")),
    Operator("APPROX", 0x1D, "≈", ("approximately", "roughly", "nearly",
                                   "almost", "close to")),
    Operator("QUERY", 0x1E, "⁇", ("how", "why")),
    Operator("WHILE", 0x1F, "⊙", ("while", "during", "meanwhile")),
)

assert tuple(op.code for op in OPERATORS) == _OPERATOR_CODES, (
    "the operator table must use each reserved byte exactly once"
)

_OP_BY_CHAR = {chr(op.code): op for op in OPERATORS}
_OP_BY_CODE = {op.code: op for op in OPERATORS}
_OP_CHARS = frozenset(_OP_BY_CHAR)
_WIRE_BYTES = frozenset(_OP_BY_CODE)
_DISPLAY = str.maketrans({op.code: op.glyph for op in OPERATORS})


def _phrase_pattern(phrase: str) -> str:
    return r"[ \t]+".join(re.escape(part) for part in phrase.split(" "))


# Longest phrases first: ``is not`` must win over ``is``, ``at least`` over
# ``at``, and ``if and only if`` over ``and``.
_PHRASES = sorted(
    ((word, op.code) for op in OPERATORS for word in op.words),
    key=lambda item: (-len(item[0].split()), -len(item[0])),
)
_LEXICON = re.compile(
    r"\b(?:" + "|".join(_phrase_pattern(word) for word, _ in _PHRASES) + r")\b",
    re.IGNORECASE,
)
_WORD_TO_CODE = {word.lower(): code for word, code in _PHRASES}

# ``the`` is the only article removed: single-letter variables named ``A`` and
# ``a`` are common in technical text, and deleting them would change meaning.
_FILLERS = re.compile(r"\b(?:the|basically|actually|literally|simply)\b", re.IGNORECASE)

_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
})

_CONTRACTIONS = (
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can not"),
    (re.compile(r"\bwon't\b", re.IGNORECASE), "will not"),
    (re.compile(r"\bshan't\b", re.IGNORECASE), "shall not"),
    (re.compile(r"\bain't\b", re.IGNORECASE), "is not"),
    (re.compile(r"\bcan't\b", re.IGNORECASE), "can not"),
    (re.compile(r"\b([A-Za-z]+)n't\b"), r"\1 not"),
)

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100, "thousand": 1000, "million": 10**6,
    "billion": 10**9, "trillion": 10**12,
}
_NUMBER_VALUES = {**_UNITS, **_TENS, **_SCALES}
_NUMBER_WORDS = "|".join(sorted(_NUMBER_VALUES, key=len, reverse=True))
_NUMBER_SEQUENCE = re.compile(
    rf"\b(?:{_NUMBER_WORDS})(?:[ \t-]+(?:and[ \t-]+)?(?:{_NUMBER_WORDS}))*\b",
    re.IGNORECASE,
)

# Connector-like words the lexicon deliberately leaves to the model. They are
# reported so a corpus owner can decide whether a new operator is warranted.
_RESIDUAL_WORDS = (
    "that", "this", "these", "those", "which", "who", "whose", "as",
    "when", "where", "if", "but", "so", "than", "also", "such", "between",
    "through", "over", "under", "via", "per",
)
_RESIDUAL = re.compile(r"\b(?:" + "|".join(_RESIDUAL_WORDS) + r")\b", re.IGNORECASE)

_SPACE_RUN = re.compile(r"[ \t]{2,}")
_LEADING = re.compile(r"(?m)^[ \t]+")
# A literal operator byte in source text is indistinguishable from a connector
# the encoder wrote, so a connector codec cannot represent it.
_OPERATOR_BYTES = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")
_AROUND_OPERATOR = re.compile(r"[ \t]*([\x01-\x08\x0b\x0c\x0e-\x1f])[ \t]*")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+([,.;:!?])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[]) +")
_SPACE_BEFORE_CLOSE = re.compile(r" +([)\]])")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")
_INDENT = re.compile(r"\n[ \t]+")
_IDEMPOTENT = re.compile(r"([\x01\x02])\1+")

# Code, fences, tables and inline code are data. The semantic layer preserves
# them exactly; the dictionary codec may still compress them losslessly.
_FENCE = re.compile(r"^\s*(?:```|~~~)")
_CODE_START = re.compile(
    r"^\s*(?:def|class|function|return|import|#include|#define|public|private|"
    r"protected|static|package|using|SELECT|INSERT|UPDATE|DELETE|CREATE)\b",
    re.IGNORECASE,
)
_CODE_BRANCH = re.compile(r"^\s*(?:if|elif|else|for|while|try|except|catch|finally)\b",
                          re.IGNORECASE)
_MATH_LINE = re.compile(r"^\s*(?:\$|\\\[|\\\(|\\begin\{|\\end\{|%)")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_HTML_LINE = re.compile(r"^\s*<[A-Za-z/!]")
_CODE_PUNCT = frozenset("{}[]();=<>*/\\|&%$#")
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s*$")
_INLINE_CODE = re.compile(
    r"`[^`\n]+`|\$[^$\n]+\$|https?://[^\s<>()]+|www\.[^\s<>()]+|"
    r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b|</?[A-Za-z][A-Za-z0-9-]*(?:\s[^>]*)?>"
)

# ---------------------------------------------------------------------------
# The codec specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Codec:
    """Everything needed to decode an AXIOM wire stream.

    ``mode`` selects the codec family. ``dictionary`` entries are wire-form
    strings indexed from ``DICTIONARY_BASE`` (0x7F); ``alphabet`` and ``bits``
    describe sequence packing. ``legacy`` reproduces the v1 text wire exactly,
    for checkpoints written before codecs were recorded.
    """

    mode: str
    profile: str = "auto"
    language: str = "en"
    connectors: bool = False
    dictionary: tuple[str, ...] = ()
    alphabet: str = ""
    bits: int = 0
    legacy: bool = False
    version: str = AXIOM_VERSION

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"Unknown AXIOM mode {self.mode!r}; choose {', '.join(MODES)}")
        if self.profile not in PROFILES:
            raise ValueError(
                f"Unknown AXIOM profile {self.profile!r}; choose {', '.join(PROFILES)}"
            )
        if len(self.dictionary) > DICTIONARY_LIMIT:
            raise ValueError(
                f"AXIOM dictionary has {len(self.dictionary)} entries; "
                f"the maximum is {DICTIONARY_LIMIT}"
            )
        if self.mode != "text" and self.dictionary:
            raise ValueError("only the text codec uses a dictionary")
        if self.mode in ("dna", "protein"):
            if not self.alphabet or self.bits not in (4, 5):
                raise ValueError("sequence codecs need an alphabet and 4 or 5 bits")
            if len(self.alphabet) > (1 << self.bits):
                raise ValueError("sequence alphabet does not fit in the configured bits")
            if len(set(self.alphabet)) != len(self.alphabet):
                raise ValueError("sequence alphabet contains duplicate symbols")
            if any(char.isspace() for char in self.alphabet):
                raise ValueError("sequence alphabet contains whitespace")
            if any(ord(char) < 0x20 or 0xD800 <= ord(char) <= 0xDFFF
                   for char in self.alphabet):
                raise ValueError("sequence alphabet contains invalid characters")
            if self.connectors:
                raise ValueError("sequence codecs do not use connectors")
        if self.mode == "identity" and (self.dictionary or self.connectors):
            raise ValueError("the identity codec has no dictionary or connectors")
        if self.legacy and (self.dictionary or self.mode != "text"):
            raise ValueError("the legacy codec is text-only and has no dictionary")
        if self.legacy and not self.connectors:
            raise ValueError("the legacy codec requires connectors")
        if self.mode == "text":
            for entry in self.dictionary:
                if (not isinstance(entry, str) or not entry
                        or len(entry) > 1024
                        or "\n" in entry or "\t" in entry
                        or any(ord(char) < 0x20 or ord(char) == 0x7F
                               or 0xD800 <= ord(char) <= 0xDFFF
                               for char in entry)):
                    raise ValueError(
                        "AXIOM dictionary entries must be non-empty single-line "
                        "printable text"
                    )

    @property
    def dictionary_mode(self) -> bool:
        return bool(self.dictionary)

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "mode": self.mode,
            "profile": self.profile,
            "language": self.language,
            "connectors": self.connectors,
            "dictionary": list(self.dictionary),
            "alphabet": self.alphabet,
            "bits": self.bits,
            "legacy": self.legacy,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Codec":
        if not isinstance(data, dict):
            raise ValueError("AXIOM codec must be a JSON object")
        dictionary = data.get("dictionary", [])
        if not isinstance(dictionary, (list, tuple)):
            raise ValueError("AXIOM codec dictionary must be a list")
        return cls(
            mode=str(data.get("mode", "text")),
            profile=str(data.get("profile", "auto")),
            language=str(data.get("language", "en")),
            connectors=bool(data.get("connectors", False)),
            dictionary=tuple(str(entry) for entry in dictionary),
            alphabet=str(data.get("alphabet", "")),
            bits=int(data.get("bits", 0)),
            legacy=bool(data.get("legacy", False)),
            version=str(data.get("version", AXIOM_VERSION)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "Codec":
        try:
            data = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError(f"AXIOM codec is not valid JSON: {error}") from error
        return cls.from_dict(data)


_DEFAULT_CODEC = Codec(mode="text", profile="compact", language="en", connectors=True)


def legacy_codec(profile: str = "compact") -> Codec:
    """The v1 text wire: UTF-8 bytes with C0 connector codes, no dictionary."""
    if profile not in ("compact", "reversible"):
        profile = "compact"
    return Codec(mode="text", profile=profile, language="en",
                 connectors=True, legacy=True)


def codec_from_settings(profile: str = "auto", raw_json: str = "") -> Codec:
    """Load a stored codec, or reconstruct a safe one for an older checkpoint."""
    if raw_json:
        return Codec.from_json(raw_json)
    if profile in ("compact", "reversible"):
        return legacy_codec(profile)
    # An AXIOM checkpoint without a codec predates codecs; those runs used the
    # compact text wire.
    return legacy_codec("compact")


# ---------------------------------------------------------------------------
# Linguistic layer: connector algebra
# ---------------------------------------------------------------------------


def _number_value(words: list[str]) -> int | None:
    """Read a run of number words, or ``None`` when the run is not one number."""
    total = 0
    current = 0
    saw_scale = False
    saw_tens = False
    for word in words:
        if word == "and":
            # "one hundred and five" is a number; "one and five" is a list.
            if not saw_scale:
                return None
            continue
        if word in _UNITS:
            current += _UNITS[word]
        elif word in _TENS:
            # "nineteen ninety" is a year, not 109; "one hundred twenty" is
            # still 120 because a scale word precedes the tens word.
            if 10 <= current < 100:
                return None
            current += _TENS[word]
            saw_tens = True
        elif word == "hundred":
            current = max(current, 1) * 100
            saw_scale = True
        elif word in _SCALES:
            total += max(current, 1) * _SCALES[word]
            current = 0
            saw_scale = True
        else:  # pragma: no cover - the regex only admits known words
            return None
    value = total + current
    if value > 10**15:
        return None
    if len(words) == 1 and words[0] == "one":
        # "one of them" is a pronoun, not the number 1.
        return None
    if len(words) > 1 and not (saw_tens or saw_scale):
        # "one two" is a sequence, not twelve.
        return None
    return value


def _rewrite_numbers(text: str) -> str:
    def replace(match: re.Match) -> str:
        words = [word.lower() for word in re.split(r"[ \t-]+", match.group(0))]
        value = _number_value(words)
        if value is not None:
            return str(value)
        parts = []
        for word in words:
            if word == "and" or word == "one":
                parts.append(word)
            elif word in _NUMBER_VALUES:
                parts.append(str(_NUMBER_VALUES[word]))
            else:  # pragma: no cover - the regex only admits known words
                parts.append(word)
        return " ".join(parts)

    return _NUMBER_SEQUENCE.sub(replace, text)


def _replace_phrases(text: str) -> tuple[str, Counter]:
    counts: Counter = Counter()

    def replace(match: re.Match) -> str:
        key = re.sub(r"[ \t]+", " ", match.group(0)).lower()
        code = _WORD_TO_CODE.get(key)
        if code is None:  # pragma: no cover - alternation is built from the map
            return match.group(0)
        counts[_OP_BY_CODE[code].name] += 1
        return chr(code)

    return _LEXICON.sub(replace, text), counts


def _canonical_source(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.translate(_TYPOGRAPHY)


def _is_code_line(line: str) -> bool:
    """True for code, formulas, tables and other data the semantics must not touch."""
    stripped = line.strip()
    if not stripped:
        return False
    if _MATH_LINE.match(line) or _TABLE_LINE.match(line) or _HTML_LINE.match(line):
        return True
    if stripped.startswith(("{", "[")):
        return True
    if _CODE_START.match(line):
        return True
    if _CODE_BRANCH.match(line) and any(char in "({[=:;" for char in line):
        return True
    if line.startswith(("    ", "\t")) or "\t" in line:
        return True
    letters = sum(1 for char in line if char.isalpha())
    punctuation = sum(1 for char in line if char in _CODE_PUNCT)
    if letters == 0:
        return punctuation > 0
    if _SENTENCE_END.search(line):
        return False
    if punctuation / letters > 0.22:
        return True
    if line.count(",") / letters > 0.15:
        return True
    return stripped.endswith(("{", "}", ";"))


def _mask_inline_code(line: str) -> tuple[str, list[str], str]:
    """Replace inline-code spans with sentinels that cannot occur in ``line``."""
    matches = list(_INLINE_CODE.finditer(line))
    if not matches:
        return line, [], ""
    sentinel = "\ue000"
    while any(f"{sentinel}{index}\ue001" in line for index in range(len(matches))):
        sentinel += "\ue000"
    pieces = []
    spans: list[str] = []
    position = 0
    for index, match in enumerate(matches):
        pieces.append(line[position:match.start()])
        pieces.append(f"{sentinel}{index}\ue001")
        spans.append(match.group(0))
        position = match.end()
    pieces.append(line[position:])
    return "".join(pieces), spans, sentinel


def _restore_inline_code(line: str, spans: list[str], sentinel: str) -> str:
    for index, span in enumerate(spans):
        line = line.replace(f"{sentinel}{index}\ue001", span)
    return line


def _split_ending(line: str) -> tuple[str, str]:
    """Return the line body and its exact LF, CRLF or lone-CR ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\n", "\r")):
        return line[:-1], line[-1]
    return line, ""


def _apply_prose_spans(line: str, profile: str,
                       preserve_crlf: bool = False) -> tuple[str, Counter]:
    """Rewrite prose while leaving the line ending and inline code untouched.

    Semantic profiles normalise CRLF to LF, which is the canonical form. A
    protected line ending in a lone CR must not be followed by a normalised
    CRLF: the two endings would combine into one CRLF and change the line
    count. ``preserve_crlf`` keeps the CRLF in that one case.
    """
    body, ending = _split_ending(line)
    if ending == "\r\n" and not preserve_crlf:
        ending = "\n"
    elif ending == "\r":
        ending = "\n"
    masked, spans, sentinel = _mask_inline_code(body)
    rewritten, hits = _apply_prose(masked, profile)
    if ending:
        rewritten = rewritten.rstrip(" \t")
    result = _restore_inline_code(rewritten, spans, sentinel) + ending
    if not _FENCE.match(line) and _FENCE.match(result):
        # Dropping a leading filler must not turn prose into a fence marker:
        # the next call would protect different lines and change the output.
        return line, Counter()
    return result, hits


def _apply_prose(text: str, profile: str) -> tuple[str, Counter]:
    text = _canonical_source(text)
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    # Numbers run before the lexicon: "one hundred and five" is 105, not
    # "100 ∧ 5". "one and five" stays a list, because and-without-a-scale is
    # not part of a numeral.
    text = _rewrite_numbers(text)
    text, counts = _replace_phrases(text)
    if profile == "compact":
        text = _FILLERS.sub(" ", text)
    text = _SPACE_RUN.sub(" ", text)
    text = _AROUND_OPERATOR.sub(r"\1", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_BEFORE_CLOSE.sub(r"\1", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    text = _IDEMPOTENT.sub(r"\1", text)
    if profile == "compact":
        text = _INDENT.sub("\n", text)
        text = _LEADING.sub("", text)
    return text, counts


def _protection_mask(text: str) -> list[bool]:
    """Classify each line once, before canonicalisation can change a line.

    Filler removal can turn ``the ``` `` into a fence marker. Reclassifying the
    rewritten text on the next pass would then protect different lines and the
    fixed-point loop would oscillate, so the mask is frozen for the call.
    """
    mask: list[bool] = []
    in_fence = False
    for line in _lines(text):
        if _FENCE.match(line):
            in_fence = not in_fence
            mask.append(True)
        else:
            mask.append(in_fence or _is_code_line(line))
    return mask


def _apply_once(text: str, profile: str, mask: list[bool],
                previous_line: str = "") -> tuple[str, Counter]:
    counts: Counter = Counter()
    lines = list(_lines(text))
    if len(lines) != len(mask):
        mask = _protection_mask(text)
    out = []
    previous = previous_line
    for line, protected in zip(lines, mask):
        if protected:
            rewritten = line
        else:
            preserve_crlf = previous.endswith("\r") and not previous.endswith("\r\n")
            rewritten, hits = _apply_prose_spans(line, profile, preserve_crlf)
            counts.update(hits)
        out.append(rewritten)
        previous = rewritten
    return "".join(out), counts


def canonical_text(text: str, profile: str = "compact", connectors: bool = True,
                   max_passes: int = 4, protect: bool = True,
                   previous_line: str = "") -> tuple[str, int, Counter, Counter]:
    """Return the semantic normal form, pass count, operator counts, residual.

    ``protect`` masks fenced/indented code, inline code, inline math, tables,
    URLs, emails and HTML tags. Legacy v1 wire passes ``False`` to reproduce the
    old, unprotected rewrite. ``previous_line`` is the already-emitted line in
    a streamed rewrite; it lets one CRLF stay unnormalised when the preceding
    line ends in a lone CR.
    """
    if not connectors or profile == "lossless":
        residual = Counter(word.lower() for word in _RESIDUAL.findall(text))
        return text, 1, Counter(), residual
    if profile not in ("compact", "reversible"):
        profile = "compact"
    if max_passes < 1:
        raise ValueError("max_passes must be >= 1")
    mask = _protection_mask(text) if protect else None
    counts: Counter = Counter()
    passes = 0
    for _ in range(max_passes):
        passes += 1
        if protect:
            updated, hits = _apply_once(text, profile, mask, previous_line)
        else:
            updated, hits = _apply_prose(text, profile)
        counts.update(hits)
        if updated == text:
            break
        text = updated
    residual = Counter(word.lower() for word in _RESIDUAL.findall(text))
    return text, passes, counts, residual


def _expand_connectors(text: str) -> str:
    pieces = []
    for char in text:
        operator = _OP_BY_CHAR.get(char)
        pieces.append(f" {operator.words[0]} " if operator else char)
    text = "".join(pieces)
    text = _SPACE_RUN.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _SPACE_BEFORE_CLOSE.sub(r"\1", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _LEADING.sub("", text)
    text = _INDENT.sub("\n", text)
    return text.rstrip(" \t")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "en": frozenset((
        "the", "of", "and", "to", "in", "is", "that", "for", "with", "as",
        "was", "on", "are", "it", "be", "this", "from", "or", "an", "by",
        "not", "but", "at", "which", "have", "has", "were", "they", "their",
    )),
    "es": frozenset((
        "el", "la", "los", "las", "de", "y", "que", "en", "un", "una", "es",
        "por", "con", "no", "se", "del", "al", "como", "mas", "pero", "para",
        "son", "fue", "este", "esta", "sus",
    )),
    "fr": frozenset((
        "le", "la", "les", "de", "et", "que", "en", "un", "une", "est", "pour",
        "avec", "pas", "se", "du", "au", "comme", "plus", "mais", "dans",
        "sont", "sur", "ce", "cette", "aux",
    )),
    "de": frozenset((
        "der", "die", "das", "und", "in", "zu", "von", "mit", "ist", "nicht",
        "ein", "eine", "auf", "fur", "als", "dem", "den", "sich", "auch",
        "aber", "sind", "des", "im", "werden",
    )),
    "pt": frozenset((
        "o", "a", "os", "as", "de", "e", "que", "em", "um", "uma", "e",
        "por", "com", "nao", "se", "do", "da", "ao", "como", "mais", "mas",
        "sao", "foi", "este", "esta",
    )),
    "it": frozenset((
        "il", "la", "i", "le", "di", "e", "che", "in", "un", "una", "e", "per",
        "con", "non", "si", "del", "al", "come", "piu", "ma", "sono", "era",
        "questo", "questa",
    )),
    "nl": frozenset((
        "de", "het", "een", "en", "van", "in", "is", "dat", "op", "te", "voor",
        "met", "niet", "zijn", "aan", "er", "als", "maar", "ook", "worden",
        "door", "bij",
    )),
}
_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[ \t]+")


def _script_of(code: int) -> str | None:
    if 0x3040 <= code <= 0x30FF:
        return "ja"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "zh"
    if 0xAC00 <= code <= 0xD7AF:
        return "ko"
    if 0x0400 <= code <= 0x04FF:
        return "ru"
    if 0x0600 <= code <= 0x06FF:
        return "ar"
    if 0x0900 <= code <= 0x097F:
        return "hi"
    if 0x0370 <= code <= 0x03FF:
        return "el"
    if 0x0590 <= code <= 0x05FF:
        return "he"
    if 0x0E00 <= code <= 0x0E7F:
        return "th"
    if (0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A
            or 0xC0 <= code <= 0x24F):
        return "latin"
    return None


def _scores(words: list[str]) -> dict[str, float]:
    if len(words) < 3:
        return {}
    total = len(words)
    counts: Counter = Counter()
    for word in words:
        matches = [lang for lang, stopwords in _STOPWORDS.items()
                   if word in stopwords]
        if matches:
            # A word shared by several languages is weak evidence for each of
            # them: "in" is English, Dutch, German and Italian, so counting it
            # whole made plain English look like a mixed corpus.
            weight = 1.0 / len(matches)
            for language in matches:
                counts[language] += weight
    return {language: counts[language] / total for language in _STOPWORDS}


def _ranked_scores(words: list[str]) -> list[tuple[str, float]]:
    scores = _scores(words)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _segment_language(segment: str) -> str | None:
    """A confident language vote for one long sentence, or ``None``."""
    words = [word.lower() for word in _WORD.findall(segment)]
    if len(words) < 6:
        return None
    ranked = _ranked_scores(words)
    if not ranked:
        return None
    top, top_score = ranked[0]
    second_score = ranked[1][1]
    if top_score < 0.25 or top_score < 1.5 * max(second_score, 1e-9):
        return None
    return top


def _foreign_share(sample: str, dominant: str) -> float:
    """Fraction of confident sentence votes that favour another language."""
    votes: Counter = Counter()
    for line in sample.split("\n")[:5000]:
        for segment in _SENTENCE_SPLIT.split(line):
            language = _segment_language(segment)
            if language is not None:
                votes[language] += 1
    total = sum(votes.values())
    if total < 4:
        return 0.0
    return (total - votes.get(dominant, 0)) / total


def detect_language(text: str) -> str:
    """Best-effort language tag used to decide whether connectors are safe.

    Non-Latin scripts return their script tag. Latin text is scored against a
    small function-word table over the whole sample, with ambiguous words
    shared by several languages weighted down. Long lines then vote, so a
    minority language that the aggregate hides is still noticed; a significant
    foreign vote returns "mixed", which disables the English semantic layer
    and keeps the lossless dictionary codec.
    """
    sample = text[:200_000]
    scripted = 0
    scripts: Counter = Counter()
    for char in sample:
        script = _script_of(ord(char))
        if script is not None:
            scripted += 1
            scripts[script] += 1
    if scripted < 20:
        return "unknown"
    non_latin = {key: value for key, value in scripts.items() if key != "latin"}
    if non_latin and sum(non_latin.values()) / scripted >= 0.5:
        if scripts.get("ja", 0) >= 0.1 * scripted:
            return "ja"
        return max(non_latin, key=non_latin.get)
    if scripts.get("latin", 0) / scripted < 0.5:
        return "unknown"

    words = [word.lower() for word in _WORD.findall(sample)]
    ranked = _ranked_scores(words)
    if not ranked:
        return "unknown"
    top, top_score = ranked[0]
    if top_score < 0.02:
        return "unknown"
    if _foreign_share(sample, top) >= 0.15:
        return "mixed"
    return top


# ---------------------------------------------------------------------------
# Lossless dictionary codec
# ---------------------------------------------------------------------------

_WORD_CANDIDATE = re.compile(r"[\w'’-]{2,32}", re.UNICODE)
_PHRASE_CANDIDATE = re.compile(
    r"[\w'’-]{1,32}(?:[ \t][\w'’-]{1,32}){1,2}", re.UNICODE
)
_SYMBOL_CANDIDATE = re.compile(r"[^\w\s]{1,8}", re.UNICODE)
_LATEX_CANDIDATE = re.compile(r"\\[A-Za-z]{2,24}")
_CJK_CANDIDATE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]{2,}"
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8", errors="surrogateescape"))


def _candidate_ok(candidate: str) -> bool:
    if not candidate or len(candidate) > 64:
        return False
    if "\n" in candidate or "\t" in candidate:
        return False
    if _CONTROL_CHARS.search(candidate):
        return False
    if any(0xE000 <= ord(char) <= 0xF8FF for char in candidate):
        return False
    return _utf8_len(candidate) >= 3


def _prune_counts(counts: Counter, limit: int = 200_000) -> None:
    if len(counts) > limit:
        keep = counts.most_common(limit // 4)
        counts.clear()
        counts.update(dict(keep))


def _collect_candidates(text: str, counts: Counter) -> None:
    for pattern in (_WORD_CANDIDATE, _PHRASE_CANDIDATE, _SYMBOL_CANDIDATE,
                    _LATEX_CANDIDATE):
        for match in pattern.finditer(text):
            candidate = match.group(0)
            if _candidate_ok(candidate):
                counts[candidate] += 1
                _prune_counts(counts)
    for match in _CJK_CANDIDATE.finditer(text):
        run = match.group(0)
        for size in (1, 2, 3, 4):
            for start in range(0, max(0, len(run) - size + 1)):
                candidate = run[start:start + size]
                if _candidate_ok(candidate):
                    counts[candidate] += 1
            _prune_counts(counts)


def build_dictionary(text: str, limit: int = DICTIONARY_LIMIT) -> tuple[str, ...]:
    """Choose high-value substrings for the one-byte dictionary codes.

    The score is the number of bytes saved by replacing each occurrence with a
    single code, minus the cost of storing the entry. Selection is greedy and
    overlap-aware enough for a 129-entry table; the encoder still uses longest
    match, so a subphrase that survives selection remains usable.
    """
    if limit < 1 or not text:
        return ()
    sample = text[:DICTIONARY_SAMPLE_CHARS]
    counts: Counter = Counter()
    # Line-wise counting bounds memory and keeps a pathological single line
    # from dominating the candidate table.
    for line in sample.split("\n"):
        if not line:
            continue
        _collect_candidates(line, counts)
        if len(counts) > 200_000:
            counts = Counter(dict(counts.most_common(50_000)))

    scored = []
    for candidate, count in counts.items():
        if count < 2:
            continue
        length = _utf8_len(candidate)
        score = count * (length - 1) - length
        if score > 0:
            scored.append((score, length, candidate))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

    selected: list[str] = []
    for _, _, candidate in scored:
        if len(selected) >= limit:
            break
        selected.append(candidate)
    return tuple(selected)


@lru_cache(maxsize=16)
def _dictionary_trie(entries: tuple[str, ...]) -> dict:
    root: dict = {}
    for index, entry in enumerate(entries):
        node = root
        for char in entry:
            node = node.setdefault(char, {})
        node[None] = index
    return root


def _longest_match(trie: dict, text: str, start: int) -> tuple[int, int] | None:
    node = trie
    best = None
    index = start
    while index < len(text):
        node = node.get(text[index])
        if node is None:
            break
        index += 1
        if None in node:
            best = (node[None], index)
    return best


def _utf8_end(data: bytes, start: int) -> int:
    if start >= len(data):
        return start
    first = data[start]
    if first < 0x80:
        size = 1
    elif 0xC0 <= first <= 0xDF:
        size = 2
    elif 0xE0 <= first <= 0xEF:
        size = 3
    elif 0xF0 <= first <= 0xF7:
        size = 4
    else:
        size = 1
    return min(len(data), start + size)


_ESCAPE_BASE_MIN = 0xE100
_ESCAPE_BASE_MAX = 0xF8FF
_ESCAPE_RANGE = 0x20
_PUA_RANGE = re.compile(r"[\ue100-\uf8ff]")


class _ConnectorCollisionError(ValueError):
    """A literal operator byte cannot be escaped without a free sentinel range."""


def _mask_operator_bytes(text: str) -> tuple[str, int]:
    """Hide literal operator bytes behind PUA sentinels before canonicalisation.

    A generated connector and a literal source byte would be identical on the
    wire. The mask lets the encoder write the literal as ``0x00 <byte>`` while
    keeping the connector layer. The sentinel base is absent from the source,
    so a PUA character in the encode range can only be a sentinel.
    """
    if _OPERATOR_BYTES.search(text) is None:
        return text, 0
    used = {ord(char) for char in _PUA_RANGE.findall(text)}
    base = _ESCAPE_BASE_MIN
    limit = _ESCAPE_BASE_MAX - _ESCAPE_RANGE + 1
    while base <= limit and any(base + offset in used
                                for offset in range(_ESCAPE_RANGE)):
        base += _ESCAPE_RANGE
    if base > limit:
        raise _ConnectorCollisionError(
            "The text contains too many private-use characters to escape a "
            "literal AXIOM operator byte; remove the byte or use a lossless "
            "profile."
        )
    table = {code: chr(base + code) for code in _OPERATOR_CODES}
    return text.translate(table), base


def _encode_text(text: str, codec: Codec, escape_base: int = 0) -> bytes:
    """Encode semantic text to the wire form selected by ``codec``."""
    if codec.legacy:
        return _encode_preserving(text)
    dictionary_mode = codec.dictionary_mode
    trie = _dictionary_trie(codec.dictionary) if dictionary_mode else None
    out = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        code = ord(char)
        if escape_base and escape_base < code <= escape_base + 0x1F:
            # A literal operator byte from the source, masked before rewriting.
            out.append(0x00)
            out.append(code - escape_base)
            index += 1
            continue
        if trie is not None:
            match = _longest_match(trie, text, index)
            if match is not None:
                code_index, end = match
                out.append(DICTIONARY_BASE + code_index)
                index = end
                continue
        index += 1
        if codec.connectors and 0x01 <= code <= 0x1F:
            out.append(code)
        elif 0x20 <= code <= 0x7E or code in (0x09, 0x0A, 0x0D):
            out.append(code)
        elif dictionary_mode:
            out.append(0x00)
            out.extend(_encode_preserving(char))
        elif code < 0x20 or code == 0x7F:
            out.append(0x00)
            out.extend(_encode_preserving(char))
        else:
            out.extend(_encode_preserving(char))
    return bytes(out)


def _decode_text(data: bytes, codec: Codec) -> str:
    """Decode a text wire stream to canonical text."""
    dictionary_mode = codec.dictionary_mode
    out = bytearray()
    index = 0
    while index < len(data):
        byte = data[index]
        if byte == 0x00:
            index += 1
            end = _utf8_end(data, index)
            out.extend(data[index:end])
            index = end
        elif byte in (0x09, 0x0A, 0x0D):
            out.append(byte)
            index += 1
        elif byte < 0x20:
            if codec.connectors and byte in _OP_BY_CODE:
                out.extend(f" {_OP_BY_CODE[byte].words[0]} ".encode("utf-8"))
            else:
                out.append(byte)
            index += 1
        elif dictionary_mode and byte >= DICTIONARY_BASE:
            entry = byte - DICTIONARY_BASE
            if entry < len(codec.dictionary):
                out.extend(codec.dictionary[entry].encode("utf-8"))
            index += 1
        else:
            out.append(byte)
            index += 1
    text = out.decode("utf-8", errors="replace")
    if codec.connectors:
        text = _SPACE_RUN.sub(" ", text)
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN.sub(r"\1", text)
        text = _SPACE_BEFORE_CLOSE.sub(r"\1", text)
        text = _TRAILING_SPACE.sub("\n", text)
        text = _LEADING.sub("", text)
        text = _INDENT.sub("\n", text)
        text = text.rstrip(" \t")
    return text


def _surface_text(data: bytes, codec: Codec) -> str:
    """A readable view of the wire: glyphs for connectors, entries for codes."""
    out = bytearray()
    index = 0
    while index < len(data):
        byte = data[index]
        if byte == 0x00:
            index += 1
            end = _utf8_end(data, index)
            out.extend(data[index:end])
            index = end
        elif byte in (0x09, 0x0A, 0x0D):
            out.append(byte)
            index += 1
        elif byte in _OP_BY_CODE and codec.connectors:
            out.extend(_OP_BY_CODE[byte].glyph.encode("utf-8"))
            index += 1
        elif codec.dictionary_mode and byte >= DICTIONARY_BASE:
            entry = byte - DICTIONARY_BASE
            if entry < len(codec.dictionary):
                preview = codec.dictionary[entry]
                if len(preview) > 16:
                    preview = preview[:16] + "…"
                out.extend(f"⟦{preview}⟧".encode("utf-8"))
            index += 1
        else:
            out.append(byte)
            index += 1
    return out.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Sequence codecs: DNA 4-bit and protein 5-bit packing
# ---------------------------------------------------------------------------

DNA_ALPHABET = "ACGTUNRYKMSWBDHV"
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO*-."
_SEQUENCE_MAGIC = b"\x1eA"
_DNA_SET = frozenset(DNA_ALPHABET)
_PROTEIN_SET = frozenset(PROTEIN_ALPHABET)


def detect_sequence(text: str) -> str | None:
    """Detect a DNA or protein corpus from a sample.

    Requires a long, dense, space-free alphabet. Short word lists and prose
    with spaces stay in the text codec.
    """
    total = dna = protein = spaces = 0
    lengths = []
    for raw in text.split("\n")[:5000]:
        line = raw.strip()
        if not line or line[0] in ">;@#":
            continue
        body = line
        spaces += sum(1 for char in body if char.isspace())
        letters = [char for char in body if not char.isspace()]
        if not letters:
            continue
        lengths.append(len(letters))
        for char in letters:
            total += 1
            if char in _DNA_SET:
                dna += 1
            if char in _PROTEIN_SET:
                protein += 1
    if total < 200 or not lengths:
        return None
    average = sum(lengths) / len(lengths)
    if average < 16 or spaces / max(total + spaces, 1) >= 0.02:
        return None
    if dna / total >= 0.92:
        return "dna"
    if protein / total >= 0.90:
        return "protein"
    return None


def _sequence_codec(mode: str, profile: str) -> Codec:
    alphabet = DNA_ALPHABET if mode == "dna" else PROTEIN_ALPHABET
    bits = 4 if mode == "dna" else 5
    return Codec(mode=mode, profile=profile, language="unknown",
                 alphabet=alphabet, bits=bits)


def _varint_encode(value: int) -> bytes:
    out = bytearray()
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _varint_decode(data: bytes, index: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while index < len(data):
        byte = data[index]
        index += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index
        shift += 7
        if shift > 35:
            break
    raise ValueError("AXIOM sequence record has a malformed length")


def _pack_codes(codes: list[int], bits: int) -> bytes:
    out = bytearray()
    accumulator = 0
    available = 0
    for code in codes:
        accumulator |= code << available
        available += bits
        while available >= 8:
            out.append(accumulator & 0xFF)
            accumulator >>= 8
            available -= 8
    if available:
        out.append(accumulator & 0xFF)
    return bytes(out)


def _unpack_codes(payload: bytes, count: int, bits: int) -> list[int]:
    codes = []
    accumulator = 0
    available = 0
    index = 0
    mask = (1 << bits) - 1
    for _ in range(count):
        while available < bits:
            if index >= len(payload):
                raise ValueError("AXIOM sequence record is truncated")
            accumulator |= payload[index] << available
            index += 1
            available += 8
        codes.append(accumulator & mask)
        accumulator >>= bits
        available -= bits
    return codes


def _record(kind: int, length: int, payload: bytes) -> bytes:
    return _SEQUENCE_MAGIC + bytes([kind]) + _varint_encode(length) + payload


def _encode_sequence_line(line: str, codec: Codec) -> bytes:
    if line.endswith("\r\n"):
        body, kind = line[:-2], 3
    elif line.endswith("\n"):
        body, kind = line[:-1], 1
    elif line.endswith("\r"):
        body, kind = line[:-1], 4
    else:
        body, kind = line, 2
    indices = {char: index for index, char in enumerate(codec.alphabet)}
    if len(body) >= 8 and all(char in indices for char in body):
        if len(body) > MAX_RECORD:
            raise ValueError("AXIOM cannot pack a single line larger than 64 MB")
        codes = [indices[char] for char in body]
        payload = _pack_codes(codes, codec.bits)
        return _record(kind, len(codes), payload)
    payload = _encode_preserving(line)
    if len(payload) > MAX_RECORD:
        raise ValueError("AXIOM cannot store a single literal line larger than 64 MB")
    return _record(0, len(payload), payload)


def _encode_sequence(text: str, codec: Codec) -> bytes:
    out = bytearray()
    for line in _lines(text):
        out.extend(_encode_sequence_line(line, codec))
    return bytes(out)


def _decode_sequence_record(data: bytes, index: int, codec: Codec) -> tuple[bytes, int]:
    if data[index:index + len(_SEQUENCE_MAGIC)] != _SEQUENCE_MAGIC:
        raise ValueError("AXIOM sequence record does not start with its magic")
    index += len(_SEQUENCE_MAGIC)
    if index >= len(data):
        raise ValueError("AXIOM sequence record is truncated")
    kind = data[index]
    index += 1
    if kind not in (0, 1, 2, 3, 4):
        raise ValueError("AXIOM sequence record has an unknown line type")
    length, index = _varint_decode(data, index)
    if length > MAX_RECORD:
        raise ValueError("AXIOM sequence record is larger than the safety limit")
    if kind == 0:
        end = index + length
        if end > len(data):
            raise ValueError("AXIOM literal record is truncated")
        return data[index:end], end
    payload_length = (length * codec.bits + 7) // 8
    end = index + payload_length
    if end > len(data):
        raise ValueError("AXIOM packed record is truncated")
    codes = _unpack_codes(data[index:end], length, codec.bits)
    try:
        body = "".join(codec.alphabet[code] for code in codes)
    except IndexError as error:
        raise ValueError("AXIOM packed record uses a code outside its alphabet") from error
    ending = {1: "\n", 2: "", 3: "\r\n", 4: "\r"}[kind]
    return (body + ending).encode("utf-8"), end


def _decode_sequence(data: bytes, codec: Codec, strict: bool = True) -> bytes:
    out = bytearray()
    index = 0
    while index < len(data):
        if data[index:index + len(_SEQUENCE_MAGIC)] != _SEQUENCE_MAGIC:
            if strict:
                raise ValueError(
                    f"AXIOM sequence stream is not aligned at byte {index}"
                )
            index = data.find(_SEQUENCE_MAGIC, index + 1)
            if index < 0:
                break
        try:
            chunk, index = _decode_sequence_record(data, index, codec)
        except ValueError:
            if strict:
                raise
            index = data.find(_SEQUENCE_MAGIC, index + 1)
            if index < 0:
                break
            continue
        out.extend(chunk)
    return bytes(out)


# ---------------------------------------------------------------------------
# Rewrite results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteStats:
    """What one rewrite (or one streamed corpus) saved."""

    profile: str
    original_bytes: int
    axiom_bytes: int
    passes: int
    operators: tuple[tuple[str, int], ...] = ()
    residual: tuple[tuple[str, int], ...] = ()
    codec: Codec = _DEFAULT_CODEC
    version: str = AXIOM_VERSION

    @property
    def saved_bytes(self) -> int:
        return max(0, self.original_bytes - self.axiom_bytes)

    @property
    def ratio(self) -> float:
        return self.axiom_bytes / self.original_bytes if self.original_bytes else 0.0

    @property
    def saved_percent(self) -> float:
        return 100.0 * (1.0 - self.ratio) if self.original_bytes else 0.0

    @property
    def operator_hits(self) -> int:
        return sum(count for _, count in self.operators)

    @property
    def mode(self) -> str:
        return self.codec.mode

    @property
    def language(self) -> str:
        return self.codec.language

    @property
    def dictionary_entries(self) -> int:
        return len(self.codec.dictionary)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "profile": self.profile,
            "original_bytes": self.original_bytes,
            "axiom_bytes": self.axiom_bytes,
            "saved_bytes": self.saved_bytes,
            "saved_percent": round(self.saved_percent, 2),
            "passes": self.passes,
            "operator_hits": self.operator_hits,
            "operators": dict(self.operators),
            "residual": dict(self.residual),
            "codec": self.codec.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RewriteStats":
        codec = (
            Codec.from_dict(data["codec"])
            if data.get("codec")
            else Codec(mode="text", profile=str(data.get("profile", "compact")),
                       language="en", connectors=True, legacy=True)
        )
        return cls(
            profile=str(data.get("profile", "compact")),
            original_bytes=int(data.get("original_bytes", 0)),
            axiom_bytes=int(data.get("axiom_bytes", 0)),
            passes=int(data.get("passes", 1)),
            operators=tuple(sorted(
                (str(k), int(v)) for k, v in data.get("operators", {}).items()
            )),
            residual=tuple(sorted(
                (str(k), int(v)) for k, v in data.get("residual", {}).items()
            )),
            codec=codec,
            version=str(data.get("version", AXIOM_VERSION)),
        )


@dataclass(frozen=True)
class Rewrite:
    """The AXIOM wire bytes plus their measured savings."""

    wire: bytes
    stats: RewriteStats

    def to_bytes(self) -> bytes:
        return self.wire

    @property
    def display(self) -> str:
        return display_bytes(self.wire, self.stats.codec)

    @property
    def decoded(self) -> str:
        return decode_bytes(self.wire, self.stats.codec)

    def __str__(self) -> str:
        return self.display


# ---------------------------------------------------------------------------
# Codec selection
# ---------------------------------------------------------------------------


def _sample_text(sample: bytes | str) -> tuple[str | None, bool]:
    """Return decoded text and whether the sample is valid UTF-8."""
    if isinstance(sample, str):
        if any(0xD800 <= ord(char) <= 0xDFFF for char in sample):
            return None, False
        return sample, True
    try:
        return sample.decode("utf-8"), True
    except UnicodeDecodeError:
        return None, False


def _text_codec(text: str, connectors: bool, effective: str,
                language: str, max_passes: int) -> tuple[int, Codec]:
    escape_base = 0
    if connectors:
        try:
            text, escape_base = _mask_operator_bytes(text)
        except _ConnectorCollisionError:
            # No free sentinel range in this sample; measure the lossless codec
            # rather than emit an ambiguous connector wire.
            return _text_codec(text, False, "lossless", language, max_passes)
    normal, _, _, _ = canonical_text(text, effective, connectors, max_passes)
    dictionary = build_dictionary(normal)
    base = Codec(mode="text", profile=effective, language=language,
                 connectors=connectors)
    if dictionary:
        with_dictionary = replace(base, dictionary=dictionary)
        dictionary_size = len(_encode_text(normal, with_dictionary, escape_base))
        utf8_size = len(_encode_text(normal, base, escape_base))
        if dictionary_size < utf8_size:
            return dictionary_size, with_dictionary
        return utf8_size, base
    return len(_encode_text(normal, base, escape_base)), base


def build_codec(sample: bytes | str, profile: str = "auto",
                max_passes: int = 4) -> Codec:
    """Choose the smallest lossless codec for a corpus sample.

    ``auto`` detects sequence corpora and non-English text. ``compact`` and
    ``reversible`` force the English semantic layer. ``lossless`` keeps text
    exactly, uses sequence packing when it helps, and otherwise uses the
    dictionary codec.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown AXIOM profile {profile!r}; choose {', '.join(PROFILES)}")
    text, valid = _sample_text(sample)
    if not valid or text is None:
        if profile in ("compact", "reversible"):
            return legacy_codec(profile)
        return Codec(mode="identity", profile=profile)

    sample_text = text[:SAMPLE_BYTES]
    raw_size = len(sample_text.encode("utf-8", errors="surrogateescape"))

    if profile in ("auto", "lossless"):
        mode = detect_sequence(sample_text)
        if mode is not None:
            sequence = _sequence_codec(mode, profile)
            encoded = len(_encode_sequence(sample_text, sequence))
            if encoded < raw_size * 0.98:
                return sequence

    language = detect_language(sample_text)
    candidates: list[tuple[int, Codec]] = []
    if profile == "compact":
        candidates.append(_text_codec(sample_text, True, "compact", language,
                                      max_passes))
    elif profile == "reversible":
        candidates.append(_text_codec(sample_text, True, "reversible", language,
                                      max_passes))
    elif profile == "lossless":
        candidates.append(_text_codec(sample_text, False, "lossless", language,
                                      max_passes))
    else:
        lossless = _text_codec(sample_text, False, "lossless", language,
                               max_passes)
        candidates.append(lossless)
        if language == "en":
            semantic = _text_codec(sample_text, True, "compact", language,
                                   max_passes)
            # Prefer the connector algebra unless dropping it saves more than
            # five percent; the semantic layer is the point of AXIOM, but it
            # must not cost meaningful compression on symbol-heavy corpora.
            candidates.append(semantic)
            if semantic[0] <= lossless[0] * 1.05:
                candidates.remove(lossless)

    best_size, best = min(candidates, key=lambda item: item[0])
    if best_size > raw_size * 1.005:
        return Codec(mode="identity", profile=profile)
    return best


# ---------------------------------------------------------------------------
# Public rewrite API
# ---------------------------------------------------------------------------


def _lines(text: str):
    """Split like Python's universal newlines, preserving every ending.

    LF, CRLF and lone CR each end a line exactly once; the ending is kept so a
    lossless codec can reproduce the source byte for byte. Matching the text
    reader's line splitting keeps file and in-memory rewrites identical.
    """
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\n":
            yield text[start:index + 1]
            start = index + 1
        elif char == "\r":
            if index + 1 < length and text[index + 1] == "\n":
                index += 1
            yield text[start:index + 1]
            start = index + 1
        index += 1
    if start < length:
        yield text[start:]


def _encode_preserving(text: str) -> bytes:
    try:
        return text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError as error:
        raise ValueError(
            "AXIOM text contains an unpaired surrogate; pass the original "
            "bytes to rewrite_bytes instead"
        ) from error


def _stats_for(text: str, wire: bytes, profile: str, passes: int,
               counts: Counter, residual: Counter, codec: Codec) -> RewriteStats:
    original = len(_encode_preserving(text))
    return RewriteStats(
        profile=profile,
        original_bytes=original,
        axiom_bytes=len(wire),
        passes=passes,
        operators=tuple(sorted(counts.items())),
        residual=tuple(sorted(residual.items())[:20]),
        codec=codec,
    )


def _has_operator_bytes(text: str) -> bool:
    """True when text contains a byte a connector wire would decode as a word."""
    return _OPERATOR_BYTES.search(text) is not None


def _check_profile(profile: str) -> None:
    if profile not in PROFILES:
        raise ValueError(f"Unknown AXIOM profile {profile!r}; choose {', '.join(PROFILES)}")


def _identity_rewrite(data: bytes, profile: str) -> Rewrite:
    codec = Codec(mode="identity", profile=profile)
    return Rewrite(
        wire=data,
        stats=RewriteStats(
            profile=profile,
            original_bytes=len(data),
            axiom_bytes=len(data),
            passes=1,
            codec=codec,
        ),
    )


def _auto_expanded(result: Rewrite, profile: str, auto_selected: bool) -> bool:
    return (
        auto_selected
        and profile in ("auto", "lossless")
        and result.stats.codec.mode != "identity"
        and result.stats.axiom_bytes > result.stats.original_bytes * 1.005
    )


def rewrite(text: str | bytes, profile: str = "auto", max_passes: int = 4,
            codec: Codec | None = None) -> Rewrite:
    """Rewrite one text (or raw bytes) to an AXIOM wire form."""
    if isinstance(text, bytes):
        wire, stats = rewrite_bytes(text, profile, max_passes, codec)
        return Rewrite(wire=wire, stats=stats)
    _check_profile(profile)
    auto_selected = codec is None
    if codec is None:
        codec = build_codec(text, profile, max_passes)
    if codec.mode == "identity":
        return _identity_rewrite(
            _encode_preserving(text), profile
        )
    if codec.mode in ("dna", "protein"):
        wire = _encode_sequence(text, codec)
        result = Rewrite(
            wire=wire,
            stats=_stats_for(text, wire, profile, 1, Counter(), Counter(), codec),
        )
    else:
        prepared = text
        escape_base = 0
        if codec.connectors and not codec.legacy:
            try:
                prepared, escape_base = _mask_operator_bytes(text)
            except _ConnectorCollisionError:
                if auto_selected:
                    return _identity_rewrite(_encode_preserving(text), profile)
                raise
        if not codec.legacy and codec.connectors is False and codec.profile == "lossless":
            normal, passes, counts, residual = prepared, 1, Counter(), Counter()
        else:
            normal, passes, counts, residual = canonical_text(
                prepared, codec.profile, codec.connectors, max_passes,
                protect=not codec.legacy,
            )
        wire = _encode_text(normal, codec, escape_base)
        result = Rewrite(
            wire=wire,
            stats=_stats_for(text, wire, profile, passes, counts, residual, codec),
        )
    if _auto_expanded(result, profile, auto_selected):
        return _identity_rewrite(
            _encode_preserving(text), profile
        )
    return result


def rewrite_bytes(data: bytes, profile: str = "auto", max_passes: int = 4,
                  codec: Codec | None = None) -> tuple[bytes, RewriteStats]:
    """Rewrite raw bytes; the codec decides whether text, packing or identity."""
    _check_profile(profile)
    auto_selected = codec is None
    if codec is None:
        codec = build_codec(data, profile, max_passes)
    if codec.mode == "identity":
        return data, _identity_rewrite(data, profile).stats
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        if codec.legacy:
            text = data.decode("utf-8", errors="surrogateescape")
        else:
            raise ValueError(
                "The AXIOM codec is a text/sequence codec but the corpus is "
                "not valid UTF-8; retrain with --axiom-profile auto or lossless "
                "to use the identity codec."
            ) from error
    result = rewrite(text, profile=profile, max_passes=max_passes, codec=codec)
    if _auto_expanded(result, profile, auto_selected):
        return data, _identity_rewrite(data, profile).stats
    return result.wire, result.stats


def encode_bytes(data: bytes, profile: str = "auto", max_passes: int = 4,
                 codec: Codec | None = None) -> bytes:
    """Convenience wrapper returning only the wire bytes."""
    return rewrite_bytes(data, profile=profile, max_passes=max_passes,
                         codec=codec)[0]


def decode_bytes(data: bytes, codec: Codec | None = None,
                 strict: bool = False) -> str:
    """Decode a wire stream to readable text."""
    codec = codec or legacy_codec("compact")
    if codec.mode == "identity":
        return data.decode("utf-8", errors="replace")
    if codec.mode in ("dna", "protein"):
        return _decode_sequence(data, codec, strict=strict).decode(
            "utf-8", errors="replace"
        )
    if codec.legacy:
        return _expand_connectors(
            data.decode("utf-8", errors="replace")
        )
    return _decode_text(data, codec)


def decode_to_bytes(data: bytes, codec: Codec | None = None,
                    strict: bool = True) -> bytes:
    """Reconstruct source bytes for lossless codecs and canonical text for semantic ones."""
    codec = codec or legacy_codec("compact")
    if codec.mode == "identity":
        return data
    if codec.mode in ("dna", "protein"):
        return _decode_sequence(data, codec, strict=strict)
    if codec.legacy:
        return _expand_connectors(
            data.decode("utf-8", errors="surrogateescape")
        ).encode("utf-8", errors="surrogateescape")
    return _decode_text(data, codec).encode("utf-8")


def display_bytes(data: bytes, codec: Codec | None = None) -> str:
    """A human-readable view: glyphs, dictionary previews, or decoded sequence."""
    codec = codec or legacy_codec("compact")
    if codec.mode == "identity":
        return data.decode("utf-8", errors="replace")
    if codec.mode in ("dna", "protein"):
        return f"[{codec.mode}, {codec.bits}-bit packed]\n" + decode_bytes(
            data, codec, strict=False
        )
    if codec.legacy:
        return display_text(data, codec)
    return _surface_text(data, codec)


def decode_text(wire: str | bytes, codec: Codec | None = None) -> str:
    """Decode a text wire given as bytes or as the legacy string form."""
    codec = codec or legacy_codec("compact")
    data = wire.encode("utf-8", errors="surrogateescape") if isinstance(wire, str) else wire
    return decode_bytes(data, codec)


def display_text(wire: str | bytes, codec: Codec | None = None) -> str:
    """Display a text wire given as bytes or as the legacy string form."""
    codec = codec or legacy_codec("compact")
    if isinstance(wire, str):
        text = wire
    else:
        if codec.legacy:
            text = wire.decode("utf-8", errors="replace")
        else:
            return _surface_text(wire, codec)
    pieces = []
    for char in text:
        operator = _OP_BY_CHAR.get(char)
        pieces.append(operator.glyph if operator else char)
    return "".join(pieces)


def is_wire(data: bytes | str) -> bool:
    """True when ``data`` already contains AXIOM connector or sequence framing."""
    if isinstance(data, str):
        return any(char in _OP_CHARS for char in data)
    return data.startswith(_SEQUENCE_MAGIC) or any(
        byte in _WIRE_BYTES or byte >= DICTIONARY_BASE for byte in data
    )


def summary(stats: RewriteStats) -> str:
    """One line for training logs."""
    multiplier = stats.original_bytes / stats.axiom_bytes if stats.axiom_bytes else 1.0
    codec = stats.codec
    detail = codec.mode
    if codec.mode == "text":
        detail = f"text/{codec.language}"
        if codec.connectors:
            detail += "+connectors"
        if codec.dictionary:
            detail += f", {len(codec.dictionary)} entries"
    elif codec.mode in ("dna", "protein"):
        detail = f"{codec.mode}, {codec.bits}-bit"
    hits = f"{stats.operator_hits:,} operator hits, " if stats.operator_hits else ""
    return (
        f"AXIOM {stats.profile} [{detail}]: {stats.original_bytes:,} -> "
        f"{stats.axiom_bytes:,} bytes ({stats.saved_percent:.1f}% smaller, "
        f"{multiplier:.2f}x more source per context byte), "
        f"{hits}{stats.passes} pass(es), version {stats.version}"
    )


def operator_table() -> list[dict]:
    """The alphabet as plain dicts, for tooling and documentation checks."""
    return [
        {
            "name": op.name,
            "code": f"0x{op.code:02X}",
            "glyph": op.glyph,
            "canonical": op.words[0],
            "variants": list(op.words[1:]),
        }
        for op in OPERATORS
    ]


# ---------------------------------------------------------------------------
# Files and caching
# ---------------------------------------------------------------------------


def _encode_stream(reader, writer, codec: Codec, profile: str,
                   max_passes: int) -> RewriteStats:
    counts: Counter = Counter()
    residual: Counter = Counter()
    original = 0
    reduced = 0
    passes = 0
    in_fence = False
    previous_line = ""
    for line in reader:
        original += len(line.encode("utf-8", errors="surrogateescape"))
        prepared = line
        escape_base = 0
        if codec.mode == "text" and codec.connectors and not codec.legacy:
            prepared, escape_base = _mask_operator_bytes(line)
        if codec.mode in ("dna", "protein"):
            encoded = _encode_sequence_line(line, codec)
        elif codec.mode == "identity":
            encoded = line.encode("utf-8", errors="surrogateescape")
        elif codec.legacy:
            normal, line_passes, hits, line_residual = canonical_text(
                line, codec.profile, codec.connectors, max_passes, protect=False
            )
            encoded = normal.encode("utf-8", errors="surrogateescape")
            counts.update(hits)
            residual.update(line_residual)
            passes = max(passes, line_passes)
            previous_line = normal
        else:
            if _FENCE.match(line):
                in_fence = not in_fence
                encoded = _encode_text(prepared, codec, escape_base)
                passes = max(passes, 1)
            elif in_fence:
                encoded = _encode_text(prepared, codec, escape_base)
                passes = max(passes, 1)
            else:
                normal, line_passes, hits, line_residual = canonical_text(
                    prepared, codec.profile, codec.connectors, max_passes,
                    previous_line=previous_line,
                )
                encoded = _encode_text(normal, codec, escape_base)
                counts.update(hits)
                residual.update(line_residual)
                passes = max(passes, line_passes)
            previous_line = line if (in_fence or _FENCE.match(line)) else normal
        writer.write(encoded)
        reduced += len(encoded)
    return RewriteStats(
        profile=profile,
        original_bytes=original,
        axiom_bytes=reduced,
        passes=max(1, passes),
        operators=tuple(sorted(counts.items())),
        residual=tuple(sorted(residual.items())[:20]),
        codec=codec,
    )


class _NullWriter:
    def write(self, _data: bytes) -> None:
        return None


def _identity_copy(source_path: Path, destination: str | os.PathLike | None,
                   profile: str) -> RewriteStats:
    size = source_path.stat().st_size
    if destination is not None:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    return RewriteStats(
        profile=profile,
        original_bytes=size,
        axiom_bytes=size,
        passes=1,
        codec=Codec(mode="identity", profile=profile),
    )


def _open_text(source: str | os.PathLike, codec: Codec):
    errors = "surrogateescape" if codec.legacy else "strict"
    # Universal newlines with untranslated endings matches ``_lines`` exactly,
    # so a file and the same text in memory produce the same wire.
    return open(source, "r", encoding="utf-8", errors=errors, newline="")


def rewrite_file(source: str | os.PathLike, destination: str | os.PathLike | None,
                 profile: str = "auto", max_passes: int = 4,
                 codec: Codec | None = None,
                 allow_fallback: bool = False) -> RewriteStats:
    """Stream a corpus through AXIOM.

    With ``destination=None`` the rewrite is measured and discarded. The codec
    is chosen from a spread sample when it is not supplied. ``allow_fallback``
    lets a caller that auto-selected ``codec`` fall back to identity when the
    full file turns out to be binary or would grow.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown AXIOM profile {profile!r}; choose {', '.join(PROFILES)}")
    auto_selected = codec is None or allow_fallback
    if codec is None:
        codec = build_codec(_sample_file(source), profile, max_passes)
    source_path = Path(source)
    if destination is not None:
        target_path = Path(destination)
        same = target_path.resolve() == source_path.resolve()
        if not same and target_path.exists():
            try:
                same = os.path.samefile(source_path, target_path)
            except OSError:
                same = False
        if same:
            raise ValueError(
                "AXIOM cannot rewrite a corpus in place; choose a different "
                "destination file."
            )
    if codec.mode == "identity":
        return _identity_copy(source_path, destination, profile)

    try:
        if destination is None:
            with _open_text(source, codec) as reader:
                stats = _encode_stream(reader, _NullWriter(), codec, profile, max_passes)
            if (auto_selected and profile in ("auto", "lossless")
                    and codec.mode != "identity"
                    and stats.axiom_bytes > stats.original_bytes * 1.005):
                return _identity_copy(source_path, None, profile)
            return stats

        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            with _open_text(source, codec) as reader, open(temporary, "wb") as writer:
                stats = _encode_stream(reader, writer, codec, profile, max_passes)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        if (auto_selected and profile in ("auto", "lossless")
                and codec.mode != "identity"
                and stats.axiom_bytes > stats.original_bytes * 1.005):
            return _identity_copy(source_path, destination, profile)
        return stats
    except UnicodeDecodeError as error:
        if auto_selected and not codec.legacy:
            return _identity_copy(source_path, destination, profile)
        raise ValueError(
            f"{source_path.name} is not valid UTF-8, but the selected AXIOM "
            f"codec is {codec.mode}; retrain with --axiom-profile auto to let "
            "AXIOM choose the identity codec for binary data."
        ) from error
    except _ConnectorCollisionError:
        if auto_selected:
            return _identity_copy(source_path, destination, profile)
        raise


def _sample_block(block: bytes) -> bytes:
    """Trim a byte-positioned sample to valid UTF-8 unless it is binary."""
    try:
        block.decode("utf-8")
        return block
    except UnicodeDecodeError:
        decoded = block.decode("utf-8", errors="replace")
        bad = decoded.count("\ufffd")
        # A seek can split one multi-byte character at each end. A short tail or
        # a handful of replacement characters is boundary noise, not binary
        # data; only a long, densely invalid chunk is treated as binary.
        if bad > 4 and len(decoded) >= 64 and bad / len(decoded) > 0.02:
            return block
        return decoded.replace("\ufffd", "").encode("utf-8")


def _sample_file(source: str | os.PathLike, budget: int = SAMPLE_BYTES) -> bytes:
    path = Path(source)
    size = path.stat().st_size
    if size <= budget:
        return path.read_bytes()
    parts = []
    chunk = max(1, budget // 3)
    with open(path, "rb") as handle:
        for position in (0, max(0, size // 2 - chunk // 2), max(0, size - chunk)):
            handle.seek(position)
            block = handle.read(chunk + 4096)
            first = block.find(b"\n")
            last = block.rfind(b"\n")
            if first >= 0 and last > first:
                block = block[first + 1:last + 1]
            parts.append(_sample_block(block))
    return b"\n".join(parts)


def _validate_codec_for_source(source: Path, codec: Codec) -> None:
    """Refuse a stored codec that cannot represent the source losslessly."""
    sample = _sample_file(source)
    if codec.mode == "identity":
        return
    if not codec.legacy:
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "The checkpoint's AXIOM codec expects UTF-8 text, but the "
                f"corpus {source.name} is not valid UTF-8; start a fresh "
                "checkpoint with --axiom-profile auto."
            ) from error
    if codec.mode in ("dna", "protein"):
        text = sample.decode("utf-8", errors="ignore")
        detected = detect_sequence(text)
        if detected != codec.mode:
            raise ValueError(
                f"The checkpoint's AXIOM codec expects {codec.mode} data, but "
                f"{source.name} does not look like {codec.mode}; start a fresh "
                "checkpoint or use the corpus it was trained on."
            )


def _cached_result(source: Path, codec: Codec,
                   cache: Path) -> tuple[str, RewriteStats] | None:
    key = _cache_key(source, codec)
    destination = cache / f"{_safe_stem(source)}-{key}.bin"
    manifest = destination.with_name(destination.name + ".json")
    if not (destination.is_file() and manifest.is_file()):
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("version") != AXIOM_VERSION:
            return None
        stored = Codec.from_dict(data.get("codec", {}))
        if stored.digest != codec.digest:
            return None
        stats = RewriteStats.from_dict(data.get("stats", {}))
        if stats.codec.digest != codec.digest:
            return None
        if destination.stat().st_size != stats.axiom_bytes:
            # A crash or an external edit left a truncated or stale payload;
            # never hand a half-written corpus to the trainer.
            return None
        return str(destination), stats
    except (OSError, ValueError, TypeError):
        return None


def prepare_corpus(source: str | os.PathLike, cache_root: str | os.PathLike = Path(
        "trained-model", "axiom"), profile: str = "auto",
        max_passes: int = 4, codec: Codec | None = None) -> tuple[str, RewriteStats]:
    """Encode ``source`` once and cache the prepared corpus and its codec.

    A supplied codec is the checkpoint's codec; it is validated against the
    source before use. Otherwise the codec is selected from a spread sample.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown AXIOM profile {profile!r}; choose {', '.join(PROFILES)}")
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"AXIOM corpus not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"AXIOM corpus is empty: {path}")

    if codec is None:
        codec = build_codec(_sample_file(path), profile, max_passes)
        auto_selected = True
    else:
        auto_selected = False
        _validate_codec_for_source(path, codec)

    cache = Path(cache_root)
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except (FileExistsError, NotADirectoryError) as error:
        raise ValueError(
            f"AXIOM cache path is not a directory: {cache}"
        ) from error
    cached = _cached_result(path, codec, cache)
    if cached is not None:
        return cached
    if auto_selected and profile in ("auto", "lossless") and codec.mode != "identity":
        # A previous run may already have stored the identity fallback. Check it
        # before paying for another full encode that will fall back again.
        identity = Codec(mode="identity", profile=profile)
        cached = _cached_result(path, identity, cache)
        if cached is not None:
            return cached

    key = _cache_key(path, codec)
    destination = cache / f"{_safe_stem(path)}-{key}.bin"
    manifest = destination.with_name(destination.name + ".json")
    try:
        stats = rewrite_file(path, destination, profile=profile,
                             max_passes=max_passes, codec=codec,
                             allow_fallback=auto_selected)
    except ValueError as error:
        if not auto_selected or "not valid UTF-8" not in str(error):
            raise
        # The sample looked like text but a later byte is binary. Use the
        # identity codec rather than failing the preparation.
        stats = rewrite_file(path, destination, profile=profile,
                             max_passes=max_passes,
                             codec=Codec(mode="identity", profile=profile))
    if stats.codec.digest != codec.digest:
        # The whole-file pass found binary data outside the sample and fell
        # back to identity; re-key the cached file so the manifest and the
        # checkpoint agree on the codec.
        codec = stats.codec
        final_key = _cache_key(path, codec)
        final_destination = cache / f"{_safe_stem(path)}-{final_key}.bin"
        if final_destination != destination:
            os.replace(destination, final_destination)
            destination = final_destination
            manifest = destination.with_name(destination.name + ".json")
    payload = {
        "version": AXIOM_VERSION,
        "profile": profile,
        "source": str(path),
        "source_bytes": path.stat().st_size,
        "codec": codec.to_dict(),
        "stats": stats.to_dict(),
    }
    temporary = manifest.with_name(f"{manifest.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest)
    return str(destination), stats


def _cache_key(source: Path, codec: Codec) -> str:
    stat = source.stat()
    payload = "\0".join((
        AXIOM_VERSION, codec.digest, str(source.resolve()),
        str(stat.st_size), str(stat.st_mtime_ns),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _safe_stem(source: Path) -> str:
    """Keep cache names inside the cache directory for any source name."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem)[:64]
    return stem or "corpus"
