"""Corpus ingestion: turn a folder (or one file) of documents into training text.

``inspect_folder`` summarises what would be read; ``prepare_corpus`` extracts
the text into one UTF-8 file under ``trained-model/`` and reports per-file skip
reasons.  Generated ``gui-corpus-*`` artifacts are never re-ingested.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import uuid
from html.parser import HTMLParser
from pathlib import Path

SKIPPED_DIRECTORIES = {
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", "node_modules",
    "venv", ".venv", "dist", "build", "agent-workspace",
}
MARKUP_EXTENSIONS = {".html", ".htm", ".xhtml", ".xml", ".svg", ".rss", ".atom"}
JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
PARQUET_EXTENSIONS = {".parquet", ".pq"}
CORPUS_PREFIX = "gui-corpus-"


def _parquet_available():
    try:
        return importlib.util.find_spec("pyarrow") is not None
    except ModuleNotFoundError:
        return False


def _is_parquet(path: Path):
    if path.suffix.lower() not in PARQUET_EXTENSIONS:
        return False
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                return False
            handle.seek(-4, os.SEEK_END)
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def _decode(data: bytes):
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return data.decode("utf-32")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    sample = data[:8192]
    if b"\x00" in sample:
        return None
    controls = sum(byte < 32 and byte not in (9, 10, 13) for byte in sample)
    if sample and controls / len(sample) > 0.02:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError:
            return None


def _looks_text(path: Path):
    try:
        with path.open("rb") as handle:
            sample = handle.read(8192)
        if not sample:
            return True
        return _decode(sample) is not None
    except OSError:
        return False


def _is_generated(name: str) -> bool:
    return name.startswith(CORPUS_PREFIX) and name.endswith((".txt", ".tmp"))


def iter_corpus_files(folder: Path):
    """Yield the readable candidates under ``folder`` (or ``folder`` itself if it is a file)."""
    if folder.is_file():
        if not _is_generated(folder.name):
            yield folder
        return
    for base, directories, files in os.walk(folder, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in SKIPPED_DIRECTORIES
        )
        for name in sorted(files):
            if _is_generated(name):
                continue
            path = Path(base) / name
            if not path.is_symlink() and path.is_file():
                yield path


def _display_root(source: Path) -> Path:
    """The directory that skip reports and fingerprints are relative to."""
    return source.parent if source.is_file() else source


def inspect_folder(folder):
    """Summarise the readable text under a folder, or in a single file."""
    source = Path(folder).expanduser().resolve()
    if not str(folder).strip() or not (source.is_dir() or source.is_file()):
        raise ValueError(f"Corpus folder does not exist: {source}")
    folder = _display_root(source)
    files, skipped, total_bytes, formats = 0, 0, 0, {}
    skipped_files, skipped_reasons = [], {}
    fingerprint = hashlib.sha256()

    def skip(path, reason):
        nonlocal skipped
        skipped += 1
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        if len(skipped_files) < 100:
            skipped_files.append({"path": str(path.relative_to(folder)), "reason": reason})

    for path in iter_corpus_files(source):
        if path.suffix.lower() in PARQUET_EXTENSIONS:
            if not _is_parquet(path):
                skip(path, "Invalid Parquet header or footer")
                continue
            if not _parquet_available():
                skip(path, "Parquet support requires pyarrow")
                continue
        elif not _looks_text(path):
            skip(path, "Binary or undecodable content")
            continue
        files += 1
        stat = path.stat()
        total_bytes += stat.st_size
        fingerprint.update(f"{path.relative_to(folder)}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        extension = path.suffix.lower() or "(no extension)"
        formats[extension] = formats.get(extension, 0) + 1
    return {
        "path": str(source), "source": "file" if source.is_file() else "folder",
        "files": files, "skipped": skipped,
        "bytes": total_bytes, "signature": fingerprint.hexdigest()[:20],
        "formats": sorted(formats.items(), key=lambda item: -item[1])[:8],
        "skipped_files": skipped_files, "skipped_reasons": sorted(skipped_reasons.items()),
    }


class _TextMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _json_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _json_strings(child)


def extract_text(path: Path, text: str, mode="smart"):
    if mode == "raw":
        return text
    extension = path.suffix.lower()
    try:
        if extension == ".json":
            value = json.loads(text)
            extracted = "\n".join(_json_strings(value))
            return extracted if extracted.strip() else text
        if extension in {".jsonl", ".ndjson"}:
            rows = []
            for line in text.splitlines():
                if line.strip():
                    rows.extend(_json_strings(json.loads(line)))
            return "\n".join(rows) if rows else text
        if extension in MARKUP_EXTENSIONS:
            parser = _TextMarkupParser()
            parser.feed(text)
            return "\n".join(parser.parts) if parser.parts else text
    except (ValueError, TypeError):
        return text
    return text


def _stream_jsonl(path, output):
    written, lengths = 0, []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                parts = list(_json_strings(json.loads(line)))
                text = "\n".join(parts) if parts else line
            except (ValueError, TypeError):
                text = line
            text = text.strip()
            if not text:
                continue
            size = len(text.encode("utf-8")) + 2
            output.write(text + "\n")
            written += size - 1
            lengths.append(size)
    if written:
        output.write("\n")
        written += 1
    return written, lengths


def _parquet_row_text(row, mode):
    if mode == "raw":
        return json.dumps(row, ensure_ascii=False, default=str)
    parts = list(_json_strings(row))
    return "\n".join(parts) if parts else json.dumps(row, ensure_ascii=False, default=str)


def _stream_parquet(path, output, mode):
    import pyarrow.parquet as pq

    written, lengths = 0, []
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                text = _parquet_row_text(row, mode).strip()
                if not text:
                    continue
                size = len(text.encode("utf-8")) + 2
                output.write(text + "\n")
                written += size - 1
                lengths.append(size)
    if written:
        output.write("\n")
        written += 1
    return written, lengths


def prepare_corpus(folder, trained_model_dir: Path, mode="smart"):
    if mode not in {"smart", "raw"}:
        raise ValueError("Corpus mode must be smart or raw")
    summary = inspect_folder(folder)
    source = Path(summary["path"])
    folder = _display_root(source)
    digest = hashlib.sha256(f"{source}\0{mode}".encode()).hexdigest()[:16]
    destination = Path(trained_model_dir) / f"{CORPUS_PREFIX}{digest}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.tmp")
    files, skipped, output_bytes = 0, 0, 0
    lengths, skipped_files, skipped_reasons = [], [], {}

    def skip(path, reason):
        nonlocal skipped
        skipped += 1
        skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
        if len(skipped_files) < 100:
            skipped_files.append({"path": str(path.relative_to(folder)), "reason": reason})

    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for path in iter_corpus_files(source):
            extension = path.suffix.lower()
            if extension in PARQUET_EXTENSIONS:
                if not _is_parquet(path):
                    skip(path, "Invalid Parquet header or footer")
                    continue
                if not _parquet_available():
                    skip(path, "Parquet support requires pyarrow")
                    continue
                try:
                    written, row_lengths = _stream_parquet(path, output, mode)
                except MemoryError:
                    raise
                except Exception as error:
                    skip(path, f"Parquet read failed: {error}")
                    continue
                if not written:
                    skip(path, "No extractable text rows")
                    continue
                files += 1
                output_bytes += written
                lengths.extend(row_lengths)
                continue
            if mode == "smart" and extension in {".jsonl", ".ndjson"} and _looks_text(path):
                try:
                    written, row_lengths = _stream_jsonl(path, output)
                except (OSError, UnicodeError):
                    written, row_lengths = 0, []
                if written:
                    files += 1
                    output_bytes += written
                    lengths.extend(row_lengths)
                    continue
            try:
                text = _decode(path.read_bytes())
            except OSError:
                text = None
            if text is None:
                skip(path, "Binary or undecodable content")
                continue
            text = extract_text(path, text, mode).strip()
            if not text:
                skip(path, "No extractable text")
                continue
            output.write(text)
            output.write("\n\n")
            files += 1
            size = len(text.encode("utf-8")) + 2
            output_bytes += size
            lengths.append(size)
    if not files:
        temporary.unlink(missing_ok=True)
        raise ValueError("No readable text files were found in this folder")
    os.replace(temporary, destination)
    lengths.sort()
    summary.update({
        "output": str(destination), "files": files, "skipped": skipped,
        "output_bytes": output_bytes, "mode": mode, "documents": len(lengths),
        "median_document_bytes": lengths[len(lengths) // 2] if lengths else 0,
        "p90_document_bytes": lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))] if lengths else 0,
        "skipped_files": skipped_files, "skipped_reasons": sorted(skipped_reasons.items()),
    })
    return summary
