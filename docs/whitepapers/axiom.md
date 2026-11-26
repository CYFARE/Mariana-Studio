# AXIOM: a connector algebra and domain-adaptive corpus codec

**Whitepaper · Mariana v2.3.3 · AXIOM 2.2 · 2026-11-24**

| | |
| --- | --- |
| Status | Implemented; capability effect not yet established |
| Implementation | `mariana/axiom.py`, `mariana/train.py`, `mariana/cli.py` |
| Runbook | [axiom.md](../axiom.md) |
| Related | [Architecture spec](../../architecture.txt), [Sprint-DRV paper](sprint-drv.md), [SAINT paper](saint.md) |

## Abstract

AXIOM (**A**lgebraic e**X**pression **I**ntermediate **O**perator **M**apping)
is a training-time corpus codec for a byte-level language model. It has two
layers.

The **semantic layer** rewrites English connectors to a fixed alphabet of 28
one-byte mathematical operators, canonicalises number words and typography, and
iterates to a fixed point. "A produces B" becomes `A⇒B`.

The **codec layer** makes the same technique safe and useful on any corpus. A
sample-based detector chooses one of four codecs: a lossless dictionary codec
for text (frequent words, phrases, Unicode symbols, emoji, CJK n-grams and
LaTeX commands become one-byte codes), a 4-bit DNA packer, a 5-bit protein
packer, or an identity codec that passes binary and incompressible data through
unchanged. Non-English and mixed-language corpora keep the dictionary but
disable the English semantic layer, so an unknown language is never
mistranslated.

The model does not change. Every wire byte is an ordinary byte id, so
Sprint-DRV, SAINT, MoE, MTP and every export path work without new parameters
or vocabulary. The checkpoint records the exact codec, so generation rewrites
its prompt and decodes its output with the same rules.

Measured selection: the built-in demo corpus shrinks 26.3%, a realistic
technical whitepaper in this repository shrinks 37.5% with connectors enabled,
repeated Chinese text shrinks 92.6% losslessly, repeated mixed-language text
80.5%, repeated emoji 80.6%, a DNA corpus 45.0% and a protein corpus 30.9%.
Random binary data is passed through byte-for-byte. These are codec
measurements, not a capability result.

## 1. Why one lexicon is not enough

The first AXIOM design was an English connector lexicon. It exposed three
failure modes immediately.

1. **Languages.** "and", "of" and "is" have different spellings and different
   grammar in every language. A hand-written mapping for one language cannot be
   applied to another without risking a wrong relation. Mixed-language corpora
   make the problem worse: a line-level choice is itself ambiguous.
2. **Symbols.** A mathematical corpus already contains `∑`, `∫`, `≈`, `∈` and
   `×`; rewriting ASCII words does nothing for a 3- or 4-byte glyph. Emoji have
   the same problem at a larger scale.
3. **Non-text domains.** A protein or DNA corpus has no words. It is a
   single-alphabet sequence; the useful compression is bit packing, not
   grammar.

A codec has to decide what kind of data it is looking at before it chooses a
transform. AXIOM 2 does that on a spread sample, records the decision in the
checkpoint, and includes a lossless fallback for every branch. The design
principle is: **a codec may fail to compress, but it must never silently
corrupt or expand the corpus.**

## 2. The semantic layer: connector algebra

The semantic layer defines 28 relations. Each relation has a canonical English
word, a mathematical glyph for reading, and a reserved one-byte code in the C0
control range (`0x01`–`0x1F`, excluding tab, newline and carriage return).

| Code | Glyph | Operator | Canonical form | Intended reading |
| --- | --- | --- | --- | --- |
| `0x01` | ∧ | `AND` | and | conjunction |
| `0x02` | ∨ | `OR` | or | disjunction |
| `0x03` | ¬ | `NOT` | not | negation |
| `0x04` | ∀ | `ALL` | all | universal quantifier |
| `0x05` | ∃ | `EXISTS` | some | existential quantifier |
| `0x06` | ∄ | `NONE` | none | empty quantifier |
| `0x07` | ∈ | `IN` | in | membership / location |
| `0x08` | @ | `AT` | at | point anchor |
| `0x0B` | → | `TO` | to | direction / mapping |
| `0x0C` | ← | `FROM` | from | source / inverse direction |
| `0x0E` | ⇒ | `PRODUCES` | produces | causal implication |
| `0x0F` | ∴ | `THEREFORE` | therefore | consequence |
| `0x10` | ∵ | `BECAUSE` | because | cause |
| `0x11` | ⟹ | `IMPLIES` | implies | implication |
| `0x12` | ⟺ | `IFF` | if and only if | equivalence |
| `0x13` | ≡ | `IS` | is | identity / copula |
| `0x14` | ≢ | `IS_NOT` | is not | distinctness |
| `0x15` | ∋ | `HAS` | contains | containment / possession |
| `0x16` | ∘ | `OF` | of | composition / possession |
| `0x17` | ⊕ | `WITH` | with | combination |
| `0x18` | ⊖ | `WITHOUT` | without | removal |
| `0x19` | > | `GT` | greater than | order |
| `0x1A` | < | `LT` | less than | order |
| `0x1B` | ≥ | `GE` | at least | weak order |
| `0x1C` | ≤ | `LE` | at most | weak order |
| `0x1D` | ≈ | `APPROX` | approximately | approximation |
| `0x1E` | ⁇ | `QUERY` | how | interrogative relation |
| `0x1F` | ⊙ | `WHILE` | while | concurrency / duration |

The alphabet is deliberately small, and it is not a formal language with proven
algebraic laws. It is a **relation alphabet**: glyphs whose intended readings
are stable enough for a human to read and for a model to learn one operator
instead of many surface forms.

Reserved C0 bytes have three practical advantages:

1. an operator is exactly one byte, so it is one token for a byte-level model;
2. natural-language UTF-8 never contains them, so the wire form is unambiguous
   and the ordinary 256-byte vocabulary needs no new embedding rows;
3. no tokenizer exists to migrate: the same checkpoint and export path work.

The display glyphs are a view. The model and checkpoint use the wire bytes.

### 2.1 Normal form, not an encoding

Let `x` be the source text and `R` one rewrite pass. AXIOM applies `R` until
`R(w) = w`, or until a pass limit is reached:

```text
w0 = x
w1 = R(w0)
w2 = R(w1)
...
w* = the first wk with R(wk) = wk
```

Iterating matters: removing "the" can join fragments, contraction expansion
can expose "is not", and number canonicalisation can expose a connector. The
implemented system is deterministic (longest phrase first, fixed stage order)
and idempotent: `R(w*) = w*`. `tests/test_axiom.py` asserts this on
representative prose.

### 2.2 Rewrite stages

1. **Source canonicalisation.** CRLF becomes LF; curly quotes, dashes,
   ellipses and non-breaking spaces become ASCII. A CRLF stays CRLF when
   normalising it would merge into a preceding protected lone CR and shift the
   line protection mask.
2. **Contraction expansion.** "cannot", "won't", "can't" and the regular
   `…n't` family expand to auxiliary plus "not".
3. **Numeric canonicalisation.** A run of number words becomes one numeral when
   unambiguous: "twenty three" becomes `23`, "one hundred and five" becomes
   `105`. "one" alone stays because it is also a pronoun; "one and five" stays
   a list; "nineteen ninety" becomes `19 90` rather than the wrong `109`.
4. **Connector substitution.** A single alternation sorted by word count and
   then length replaces the longest matching phrase. "if and only if" beats
   "and"; "is not" beats "is"; "at least" beats "at".
5. **Filler removal (compact only).** "the", "basically", "actually",
   "literally" and "simply" are removed. "a"/"an" are kept because single-letter
   variables named `A` are common in technical text.
6. **Whitespace canonicalisation.** Spaces and tabs adjacent to an operator
   disappear; "A produces B" becomes `A⇒B`.
7. **Idempotence guard.** Repeated ∧ and ∨ collapse, three or more newlines
   become two, and compact mode removes indentation.

Profiles `compact` and `reversible` force this layer. Profile `auto` enables it
only when the sample looks English, and then only if its total wire is not more
than 5% larger than the best lossless dictionary candidate. Profile `lossless`
never applies the semantic layer.

The semantic layer does not touch data. Fenced and indented code blocks, inline
code, `$...$` math, Markdown tables, URLs, email addresses and HTML tags are
masked out before connector substitution. Code and formulas therefore survive
exactly, while the surrounding prose is normalised. The dictionary codec can
still compress the protected spans losslessly.

## 3. The codec layer

### 3.1 Codec selection

AXIOM samples the corpus before choosing a codec. `SAMPLE_BYTES` is 8 MB,
spread over the start, middle and end of the file and trimmed to line
boundaries.

```text
valid UTF-8?
  no  -> identity (byte-for-byte copy)
  yes -> sequence candidate?
           DNA fraction >= 92% and >= 200 symbols and long, space-free lines
           protein fraction >= 90% and >= 200 symbols and long, space-free lines
         -> measure packing; accept if it saves >= 2%
         otherwise text:
           detect language / script
           literal operator bytes -> 0x00 <byte> escape in the connector wire
           build a dictionary on the rewritten sample
           compare semantic, lossless-dictionary and utf8 wire sizes
           choose the smallest; identity if every text wire would grow
```

`auto` prefers the semantic codec unless dropping connectors saves more than
5%; this keeps the connector algebra on prose, where it wins, without paying for
it on emoji- or formula-heavy corpora. `compact` and `reversible` always force
the semantic layer. `lossless` never applies it.

The selected codec is stored twice:

- in the prepared-corpus manifest (`trained-model/axiom/*.json`);
- as a JSON string in the checkpoint config (`axiom_codec`).

A resumed or fine-tuned checkpoint uses its stored codec. The source is
validated against it; a DNA checkpoint refuses a prose corpus with a clear
error instead of silently encoding it as literal records.

### 3.2 Text codecs and the one-byte dictionary

The text codec reserves the byte space as follows.

| Byte | Meaning |
| --- | --- |
| `0x00` | escape: one literal UTF-8 code point follows |
| `0x01`–`0x1F` | connector operators (tab, LF and CR stay whitespace) |
| `0x20`–`0x7E` | literal printable ASCII |
| `0x7F`–`0xFF` | 129 dictionary codes, when a dictionary is present |

Two text variants exist:

- **utf8**: no dictionary. Non-ASCII text is encoded as raw UTF-8, so a
  non-English or emoji corpus never expands because of the code space. Only
  control bytes are escaped.
- **dictionary**: the 129 codes index a corpus-derived table. Non-ASCII code
  points that do not match a table entry are escaped as `0x00` + UTF-8, which
  costs one extra byte; the dictionary has to earn that back. Encoding uses
  longest match over a trie, so a code can cover a word, a phrase, an emoji
  sequence, a CJK n-gram, a LaTeX command or a math glyph.

Dictionary candidates are counted on the rewritten sample: words, two- and
three-word phrases, symbol runs, LaTeX commands, and 1–4 character CJK n-grams.
Each candidate is scored by `count × (utf8_length − 1) − storage_cost`. The top
129 positive-score entries are used. A single `∑` is a valid candidate because
it is three UTF-8 bytes; a single `😀` is four.

Decoding is exact for `lossless` text. For compact or reversible text it is
canonical, not byte-identical: "are" decodes as "is", and `≡¬` re-encodes as
the single operator `≢`. "Reversible" means every operator has a canonical
word and the semantic rewrite is idempotent, not that the original surface is
recovered.

### 3.3 DNA and protein packing

A sequence corpus is split into records, each starting with the two-byte magic
`0x1E 0x41`:

| Field | Meaning |
| --- | --- |
| magic | `0x1E 0x41` |
| type | 0 literal line, 1 packed LF, 2 packed no newline, 3 packed CRLF, 4 packed CR |
| length | unsigned varint: symbols for packed, bytes for literal |
| payload | bit-packed symbols, or the exact UTF-8 line bytes |

DNA uses a 4-bit alphabet of 16 IUPAC symbols (`ACGTUNRYKMSWBDHV`): two bases
per byte, a 50% raw reduction before record overhead. Protein uses a 5-bit
alphabet (`ACDEFGHIKLMNPQRSTVWYBXZJUO*-.`): eight residues in five bytes, a
37.5% raw reduction. Lines that contain lowercase, spaces or characters outside
the alphabet are stored as literal records, so FASTA headers, comments and
spaced alignments survive exactly.

The magic lets a decoder recover later records from a generated suffix; a
strict decoder used for verification rejects a misaligned stream.

### 3.4 Identity and the no-expansion guarantee

Binary or invalid-UTF-8 data, and any text whose best wire would grow, uses the
identity codec: the prepared file is a byte-for-byte copy. Random data is
information-theoretically incompressible; AXIOM's contract is that it leaves
such data alone.

## 4. Languages, symbols and mixed corpora

**English** uses the semantic layer plus the dictionary. **Other languages**
use the dictionary alone: a frequent Spanish, Chinese or Arabic word becomes a
one-byte code, and decoding restores the exact word. This is deliberate. A
hand-written connector map for one language is a semantic claim; an exact
dictionary entry is not.

Language detection combines script ranges (CJK, kana, Hangul, Cyrillic, Arabic,
Devanagari, Greek, Hebrew, Thai) with a small function-word table for English,
Spanish, French, German, Portuguese, Italian and Dutch. A word shared by several
tables ("in" is English, Dutch, German and Italian) counts only fractionally,
so shared function words do not make monolingual text look mixed. Long sentences
then vote: if at least 15% of confident votes favour another language, the tag
is `mixed`. `mixed` and every non-English tag disable the English semantic
layer. This keeps connectors off an 80/20 English-Spanish corpus while still
allowing the lossless dictionary to compress it.

**Math and symbols** are text. A frequent Unicode glyph becomes a one-byte
dictionary code; an infrequent one stays raw UTF-8. LaTeX commands such as
`\begin` or `\sum` are dictionary candidates. Code fences, indented code,
inline code, inline math, tables, URLs, emails and HTML tags are protected from
the semantic layer. The connector layer still applies to English prose inside a
math paper, but a formula-only sample is detected as non-English or as having
no useful dictionary and falls back to utf8 or identity.

**Emoji** are Unicode text. Frequent emoji and emoji sequences become codes;
if the dictionary does not beat raw UTF-8, utf8 mode passes them through
unchanged.

**Mixed datasets** are handled losslessly. A sample that mixes prose, formulas
and sequences usually fails the sequence thresholds and becomes a text codec;
the dictionary can still capture repeated sequence motifs, and every line
decodes exactly. A corpus that is dominantly one sequence alphabet uses the
packer and stores other lines as literal records.

## 5. Measured compression

Two kinds of numbers are reported: realistic prose and deliberately repeated
samples. The repeated samples show the upper bound of a corpus-specific
dictionary; they are not predictions for a different corpus.

| Corpus | Source bytes | Wire bytes | Saving | Codec |
| --- | ---: | ---: | ---: | --- |
| Built-in demo corpus | 1,512 | 1,114 | 26.3% | text/en, 13 entries, connectors |
| `docs/whitepapers/saint.md` | 20,824 | 13,018 | 37.5% | text/en, 129 entries, connectors |
| Same paper, lossless profile | 20,824 | 14,384 | 30.9% | text/en, 129 entries |
| Repeated Chinese sentence | 324 | 24 | 92.6% | text/zh, 76 entries |
| Repeated mixed-language text | 308 | 60 | 80.5% | text/mixed, 17 entries |
| Repeated emoji text | 496 | 96 | 80.6% | text/en, 12 entries |
| Repeated math/symbol text | 1,400 | 740 | 47.1% | text/en, 10 entries |
| DNA, 4,058 bytes in 70-base lines | 4,058 | 2,232 | 45.0% | dna, 4-bit |
| Protein FASTA, 4,081 bytes | 4,081 | 2,819 | 30.9% | protein, 5-bit |
| Random 4,000-byte binary blob | 4,000 | 4,000 | 0.0% | identity |

The realistic prose result is the relevant one: a 129-entry dictionary plus
English connectors compresses a technical paper by 37.5%. The demo corpus is
short and lexically varied, so its dictionary has only 13 useful entries.
Repeated samples compress dramatically because the dictionary can cover most of
the text; that is a property of the sample, not of AXIOM.

`python scripts/axiom_corpus.py --data corpus.txt --report` produces the same
measurements, the operator histogram, the dictionary, and the "left to the
model" residual list for any corpus.

## 6. Why this can be faster

For a byte-level model, one token is one byte. If the source has `N` bytes and
the wire has `M` bytes, `c = 1 − M/N` and three consequences follow with no
assumption about model quality:

- **Context multiplier.** A `T`-byte window holds `1/(1−c)` more source
  content. At 26.3% it holds 1.36× more; at 37.5%, 1.60×; at 45%, 1.82×.
- **Training steps.** A corpus pass needs roughly `(1−c)` as many rows. At a
  fixed step count, the model sees more distinct source content per step; at a
  fixed source-byte target, fewer steps are needed.
- **Inference wall-clock.** Producing one unit of source content needs roughly
  `(1−c)` as many decode steps. Tokens per second is unchanged; seconds per
  document fall.

The qualitative arguments are weaker: a relation is one token; operators are
case-folded and shared; numbers have one canonical spelling; and the lossless
dictionary gives non-English corpora the same shorter-token benefit without a
linguistic mapping.

None of this proves a capability gain. A smaller sequence is cheaper, not
automatically smarter. Section 7 defines the measurement that decides.

**Sequence caveat.** A model trained on DNA or protein records generates
records, not prose. Free-form `gen` decoding of a sequence checkpoint is
best-effort: it decodes whole records and skips a partial first record. Use
sequence checkpoints for training, scoring or embedding work, or accept the
record framing when sampling.

## 7. How to evaluate AXIOM honestly

### 7.1 Perplexity is not comparable across encodings

A model trained on wire bytes predicts wire bytes. Its cross-entropy is per
wire byte, and one wire byte represents more than one source byte. Compare
**bits per original byte**:

`BPB = (sum of wire-token NLL in nats) / (ln 2 × original source bytes)`.

The manifest records source bytes for every prepared corpus, so validation rows
can carry their source length. Raw loss is not a valid cross-codec comparison.

### 7.2 Required ablations

1. **Matched steps.** Plain bytes and each AXIOM codec, identical model, batch,
   context, seed and step count. Report validation BPB and wall-clock.
2. **Matched source bytes.** Scale steps so both runs see the same total source
   bytes. Isolate the compute saving.
3. **Matched wire tokens.** Scale steps so both runs see the same number of
   training tokens. AXIOM should cover more source; if it does not, the codec
   is not helping.
4. **Codec ablation.** text/utf8, text/dictionary, semantic, DNA/protein packing
   and identity on the same domain. A codec that wins only by deleting "the" or
   by packing a corpus that was never a sequence is not evidence for the
   technique.
5. **Operator ablation.** Replace every connector operator with one symbol,
   then with randomly permuted codes. If collapsing all operators costs
   nothing, the semantic distinctions were not used.
6. **Re-encode ablation.** Train on wire and evaluate on plain text, and the
   reverse. A large penalty means the model learned an encoding rather than a
   language.
7. **Corpus families.** Prose, dialogue, technical text, code, math,
   multilingual text, emoji, DNA and protein. A technique that only helps one
   genre is a preprocessing trick, not a general codec.
8. **Dictionary leakage.** Build the dictionary on the training split only and
   apply it to validation; compare with a corpus-wide dictionary. The current
   implementation uses a corpus-wide sample, which is standard tokenizer
   practice but must be stated.

No completed study in this repository establishes a capability gain. The
implementation and protocol exist so the next run can answer the question.

## 8. Implementation

`mariana/axiom.py` is torch-free and standard-library only. Its public surface:

- `rewrite`, `rewrite_bytes`, `encode_bytes` — encode with an explicit or
  automatically selected codec;
- `build_codec`, `detect_language`, `detect_sequence` — selection diagnostics;
- `decode_bytes`, `decode_to_bytes`, `display_bytes` — readable view and
  lossless reconstruction;
- `prepare_corpus`, `rewrite_file` — sampled selection, streaming encoding and
  the content-addressed cache;
- `Codec`, `legacy_codec`, `codec_from_settings` — checkpoint serialization and
  v1 compatibility;
- `operator_table` — the alphabet for tooling and documentation.

Training integration is one branch in `mariana/train.py`. The prepared path is
an ordinary `.bin` file, so small-corpus repetition and the large-file memmap
path both work unchanged. The codec JSON is written into `Config.axiom_codec`;
`gen` loads it, rewrites prompts and decodes outputs. A resumed checkpoint uses
its stored codec and validates the new corpus against it.

Studio exposes AXIOM in the advanced training fields with four profiles. It is
a DRV-LM corpus codec, not an NCM option, and the backend rejects the
combination.

## 9. Honest limits

**The semantic layer is English-only.** Other languages use the lossless
dictionary. This is a deliberate trade: exact compression over guessed grammar.
A future multilingual semantic layer must be evaluated language by language.

**It is not a parser.** Within English, `to`, `of`, `is`, `while`, `how`, `at`
and `with` have non-connector senses that are still rewritten. "if", "when",
"where", "that", "which", "since", "then", "about", "above", "below",
"inside", the auxiliaries "have/has/had" and "be/been/being" are deliberately
left alone. The residual report counts them.

**Inflection is collapsed** in semantic profiles: "is/are/was/were" all become
≡; "produces/produce/produced" all become ⇒. Lossless profiles do not do this.

**The dictionary is corpus-specific.** It lives in the checkpoint and manifest.
A model trained with one dictionary must not decode a corpus prepared with
another; AXIOM checks the codec digest and refuses a mismatched resume.

**Sequence packing is for sequence data.** DNA and protein packing assume the
declared alphabet. Lines outside it are stored literally, which preserves
correctness but gives up compression. Spaced alignments and lowercase sequences
are literal. Free-form generation from a sequence checkpoint is best-effort.

**Detection is heuristic.** Sequence thresholds and language scores can
misclassify an unusual corpus. The fallbacks are lossless or identity, so the
failure mode is "less compression", not corruption. Use `compact`, `reversible`
or `lossless` to override detection.

**Literal operator bytes are escaped.** A source byte in `0x01`–`0x08`,
`0x0B`, `0x0C` or `0x0E`–`0x1F` cannot be distinguished from a connector on a
connector wire. AXIOM masks it before rewriting and writes it as `0x00 <byte>`,
which the decoder restores exactly. A few stray control bytes therefore cost
one extra byte each, not the semantic layer. NUL, TAB, LF, CR and DEL are
escaped as before.

**Incompressible data stays incompressible.** Random bytes use the identity
codec and do not shrink. No codec can beat information theory.

**No capability result.** The measured effect is on corpus length. Whether the
shorter, denser representation improves reasoning, sample efficiency or final
BPB is the subject of the ablations in Section 7.

## 10. Relationship to the rest of Mariana

- **Sprint-DRV** trains AR, diffusion, verifier and MTP in one pass. AXIOM
  changes only the bytes those objectives see.
- **SAINT** writes a private thought memory over the context. A shorter, denser
  context leaves more room for content; SAINT does not require AXIOM.
- **MoE and MTP** are unchanged; every wire byte is ordinary vocabulary.
- **Export** is unchanged. A GGUF/ONNX/safetensors model exported from an AXIOM
  checkpoint expects AXIOM wire bytes at inference; the codec is in the
  checkpoint, not in the GGUF file.
- **NCM** is not integrated. Its tokenizer learns concepts over raw bytes;
  applying AXIOM before that stage is a separate experiment.

The closest external relatives are function-word normalisation in information
retrieval, reversible text compression, byte-pair tokenization, and semantic
parsing. AXIOM differs from the first by being model-facing, from the second by
adding an English semantic layer, from the third by using reserved byte codes
instead of a new embedding vocabulary, and from the fourth by refusing to
resolve ambiguity. It is a representation change with a measurable cost and an
unmeasured benefit.

## 11. Reproduce

```bash
# Inspect the alphabet and any corpus.
python scripts/axiom_corpus.py --lexicon
python scripts/axiom_corpus.py --data corpus.txt --report
python scripts/axiom_corpus.py --data corpus.txt \
    --out trained-model/corpus.axiom.bin --profile auto --display 400

# Train; AXIOM selects and records the codec.
python -m mariana train --axiom --axiom-profile auto \
    --data corpus.txt --checkpoint trained-model/mariana-axiom.pt \
    --steps 5000 --fast --no-gen --device cuda

# Text generation rewrites its prompt and decodes its output.
python -m mariana gen --checkpoint trained-model/mariana-axiom.pt \
    --prompt "The pressure at the bottom is" --gen-len 120 --show-axiom

# Force the English semantic layer regardless of dictionary economics.
python -m mariana train --axiom --axiom-profile compact ...
```

The test contract is `tests/test_axiom.py`: the alphabet, longest-phrase
precedence, number parsing, semantic idempotence, the lossless dictionary,
Unicode symbols and emoji, mixed-language safety, DNA and protein round trips,
identity on random bytes, codec JSON round trips, the training data pipeline,
cache reuse and the CLI/config round trip.
