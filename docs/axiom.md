# AXIOM: the corpus codec

AXIOM rewrites a training corpus into a shorter wire form before the model sees
it. It has two layers:

- an **English semantic layer** that turns connectors into one-byte
  mathematical operators (`A produces B` becomes `A⇒B`);
- a **domain-adaptive codec layer** that builds a lossless one-byte dictionary
  for text (any language, math symbols, emoji, LaTeX), bit-packs DNA and
  protein corpora, or passes binary data through unchanged.

The model is unchanged. The codec is recorded in the checkpoint, so generation
rewrites its prompt and decodes its output with the same rules.

Design, measurements and the evaluation protocol:
[whitepapers/axiom.md](whitepapers/axiom.md).

## Quick start

```bash
# Auto-detect the corpus and pick the smallest codec.
python -m mariana train --axiom --data corpus.txt \
    --checkpoint trained-model/mariana-axiom.pt \
    --steps 5000 --fast --no-gen --device cuda

# Generate: the prompt is encoded and the output decoded automatically.
python -m mariana gen --checkpoint trained-model/mariana-axiom.pt \
    --prompt "The pressure at the bottom is" --gen-len 120

# Show the codec's surface view (glyphs / dictionary previews / packing label).
python -m mariana gen --checkpoint trained-model/mariana-axiom.pt \
    --prompt "The pressure at the bottom is" --gen-len 120 --show-axiom
```

The training log prints the selected codec and the measured shrink before the
first step:

```text
AXIOM corpus codec: selecting from the corpus (profile auto)
AXIOM: 28 connector operators plus adaptive text/dictionary/sequence codecs, version 2.2
AXIOM auto [text/en+connectors, 129 entries]: 20,824 -> 13,018 bytes (37.5% smaller, 1.60x more source per context byte), ...
AXIOM effective context: 256 wire bytes -> 430 source bytes
```

## Profiles

| Profile | Behaviour | Use |
| --- | --- | --- |
| `auto` (default) | Detects text/language/DNA/protein, compares the semantic and lossless dictionary codecs, chooses the smallest, and falls back to identity | Normal training |
| `compact` | Forces the English semantic layer and drops "the" plus four discourse fillers; adds the dictionary | English prose, maximum semantic normalisation |
| `reversible` | Forces the English semantic layer but keeps every word; adds the dictionary | English where fillers matter |
| `lossless` | No connector or number rewriting; dictionary for text, packing for sequences | Exact text corpora, non-English, evaluation |

`auto` prefers the semantic codec unless dropping connectors saves more than
5%. On a formula- or emoji-heavy corpus it may choose the lossless dictionary
instead. Use `compact` if you require connector rewriting regardless of the
compression result.

## What gets selected

| Corpus | Codec | What it does |
| --- | --- | --- |
| English prose | `text/en`, dictionary, connectors | 28 operators + 129 dictionary entries |
| Non-English or mixed text | `text/<lang>` or `text/mixed`, dictionary | exact one-byte codes for frequent words, phrases and characters |
| Math / symbols / emoji | `text/...`, dictionary | frequent glyphs and emoji sequences get one-byte codes; rare ones stay UTF-8 |
| DNA | `dna`, 4-bit | 16-symbol IUPAC alphabet, two bases per byte |
| Protein | `protein`, 5-bit | 29-symbol alphabet, eight residues in five bytes |
| Binary / random | `identity` | byte-for-byte copy; never expands |

Lines that do not fit a sequence alphabet (FASTA headers, comments, spaced
alignments, lowercase sequences) are stored as literal records, so sequence
codecs remain lossless on messy files.

## Prepare or inspect a corpus

Training prepares and caches the corpus automatically. Use the script to inspect,
ship or pre-convert:

```bash
# Measure without writing.
python scripts/axiom_corpus.py --data corpus.txt --report

# Write the prepared corpus and show the surface view.
python scripts/axiom_corpus.py --data corpus.txt \
    --out trained-model/corpus.axiom.bin --display 400

# Force a codec family/profile.
python scripts/axiom_corpus.py --data dna.fasta --profile lossless

# Print the operator alphabet.
python scripts/axiom_corpus.py --lexicon
```

The report contains the codec, source and wire byte counts, saving, passes,
operator histogram, dictionary, and the "left to the model" residual list of
connector-like words the semantic layer deliberately does not touch.

## Checkpoints, cache and resumes

- Prepared corpora and manifests live under `trained-model/axiom/` (git-ignored).
  The cache key includes the source path, size, mtime, profile and codec digest,
  so editing the corpus or changing the codec produces a new entry.
- The checkpoint stores the full codec JSON in `axiom_codec` (plus
  `use_axiom`/`axiom_profile`). This is a few kilobytes for a 129-entry
  dictionary.
- A resumed or fine-tuned checkpoint uses its stored codec. The source is
  validated against it: a DNA codec refuses a prose corpus, and a text codec
  refuses invalid UTF-8, with an actionable error.
- Changing the profile on resume does not change the stored codec. Start a fresh
  checkpoint to change encoding.
- Export is unchanged. A GGUF/ONNX/safetensors file exported from an AXIOM model
  expects AXIOM wire bytes at inference; it does not carry the codec. Keep the
  codec JSON or use the checkpoint for decoding.

## Generation

Text codecs are the normal generation path: the prompt is encoded and the
output decoded automatically. `--show-axiom` prints:

- connector glyphs and `⟦dictionary entry⟧` previews for text;
- a `[dna, 4-bit packed]` or `[protein, 5-bit packed]` label for sequences.

Sequence checkpoints are trained on record-framed packed lines. Free-form
generation is best-effort: whole records decode, and a partial first record is
skipped. Use sequence checkpoints for training, scoring or embeddings unless you
intend to sample the record format itself.

`--gen-len` counts wire bytes. A given amount of source content needs fewer of
them than with a plain-byte checkpoint.

## Studio

Train → Advanced settings has two DRV-LM controls:

- **AXIOM corpus codec** — enable the codec for a fresh run.
- **AXIOM codec profile** — Auto, Compact, Reversible or Lossless.

The NCM technique hides the controls and the backend rejects AXIOM with NCM.
The generated command includes `--axiom --axiom-profile …`, so the same cache,
log line and checkpoint metadata are used.

## Reading the result

Do not compare AXIOM cross-entropy with plain-byte cross-entropy. A wire byte
represents more than one source byte, so raw loss rewards the shorter encoding
for being shorter.

Use **bits per original byte**:

`BPB = wire NLL in nats / (ln 2 × source bytes)`.

The manifest records source bytes for each prepared corpus. At a fixed model,
batch, context, seed and step count, lower BPB is better. The whitepaper lists
the matched-step, matched-source-byte, matched-token, codec, operator, re-encode
and corpus-family ablations required before claiming a capability gain.

## Limits and troubleshooting

- **The semantic layer is English-only.** Non-English and mixed corpora use the
  lossless dictionary instead of guessed connector mappings. The dictionary still
  gives one-byte codes for frequent words and characters.
- **The semantic layer is lexical.** "to", "of", "at", "while" and "how" are
  mapped in their connector sense; other senses are mapped too. "if", "when",
  "where", "that", "which", "since", "then", "about", "above", "below",
  "inside", "have/has/had" and "be/been/being" are deliberately left alone.
- **Code and formulas are protected.** Fenced and indented code blocks, inline
  code, `$...$` math, Markdown tables, URLs, emails and HTML tags are not
  rewritten by the semantic layer; they are still compressed losslessly by the
  dictionary.
- **Literal operator bytes are escaped.** A source byte in `0x01`–`0x08`,
  `0x0B`, `0x0C` or `0x0E`–`0x1F` is written as `0x00 <byte>` and restored
  exactly, so a few stray control bytes do not disable the connector layer.
  NUL, TAB, LF, CR and DEL are escaped by the text codec as before.
- **Compact is lossy.** "the" and four discourse fillers disappear. Use
  `lossless` for exact text.
- **The dictionary is corpus-specific.** It lives in the checkpoint; a different
  corpus needs a fresh codec or the original prepared corpus.
- **Sequence packing assumes the declared alphabet.** Lowercase, spaced or
  non-standard lines become literal records. Detection is heuristic; override it
  with a profile.
- **Random data stays random.** Identity is chosen when compression would grow
  the corpus. No codec can shrink incompressible data.
- **A resumed run uses its stored codec.** This is intentional. Pass a new
  checkpoint path to change encoding. A stored connector codec escapes literal
  operator bytes exactly, so a resumed run and a fresh one agree.

## Definition of done for a change

```bash
python -m pytest -q tests/test_axiom.py tests/test_studio_training.py
python -m pyflakes mariana tests
```

If the alphabet, a profile, the dictionary format or a sequence alphabet changes,
update `AXIOM_VERSION` in `mariana/axiom.py`, the tables in the whitepaper, and
the cache/codec expectations in `tests/test_axiom.py` in the same change.
