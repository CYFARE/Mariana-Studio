"""Corpus ingestion: smart and raw extraction, Parquet, atomic writes, single files."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from mariana.studio.corpus import inspect_folder, prepare_corpus
from tests.helpers import make_corpus


def test_corpus_ingestion_smart_and_raw():
    with tempfile.TemporaryDirectory() as directory:
        folder = make_corpus(Path(directory) / "input")
        summary = inspect_folder(folder)
        assert summary["files"] == 5
        assert summary["skipped"] == 1
        prepared = prepare_corpus(folder, Path(directory) / "output")
        text = Path(prepared["output"]).read_text(encoding="utf-8")
        assert "JSON text\nMore text" in text
        assert "First row\nSecond row" in text
        assert "XML text\nAnother value" in text
        assert "<root>" not in text
        assert "Text without an extension" in text
        raw = prepare_corpus(folder, Path(directory) / "raw", mode="raw")
        assert "<root>" in Path(raw["output"]).read_text(encoding="utf-8")


def test_generated_corpus_is_not_reingested():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "notes.txt").write_text("original corpus text", encoding="utf-8")
        results = [prepare_corpus(root, root / "trained-model") for _ in range(3)]
        assert {result["files"] for result in results} == {1}
        assert len({result["output_bytes"] for result in results}) == 1
        assert len({result["signature"] for result in results}) == 1


def test_concurrent_corpus_preparation_is_atomic():
    from concurrent.futures import ThreadPoolExecutor

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        for index in range(12):
            (source / f"{index}.txt").write_text(
                f"document {index} abcdefghij\n" * 200, encoding="utf-8"
            )

        def prepare(_):
            return prepare_corpus(source, root / "output")

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(prepare, range(24)))
        assert {result["files"] for result in results} == {12}
        sizes = {result["output_bytes"] for result in results}
        assert len(sizes) == 1
        output = Path(results[0]["output"])
        payload = output.read_bytes()
        assert len(payload) == sizes.pop() and b"\x00" not in payload
        assert not list(output.parent.glob("*.tmp"))


def test_parquet_ingestion_smart_raw_and_invalid():
    import pytest
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    with tempfile.TemporaryDirectory() as directory:
        folder = make_corpus(Path(directory) / "input")
        table = pa.table({
            "conversation": [[
                {"role": "user", "content": "Parquet question"},
                {"role": "assistant", "content": "Parquet answer"},
            ]],
        })
        pq.write_table(table, folder / "nested" / "table.parquet")
        pq.write_table(pa.table({"value": [42]}), folder / "numeric.parquet")
        (folder / "invalid.parquet").write_bytes(b"not a parquet file")

        summary = inspect_folder(folder)
        assert summary["files"] == 7
        assert summary["skipped"] == 2
        assert dict(summary["formats"])[".parquet"] == 2
        assert any(item["path"] == "invalid.parquet" for item in summary["skipped_files"])

        prepared = prepare_corpus(folder, Path(directory) / "output")
        text = Path(prepared["output"]).read_text(encoding="utf-8")
        assert "user\nParquet question\nassistant\nParquet answer" in text
        assert '"value": 42' in text
        assert prepared["skipped"] == 2
        raw = prepare_corpus(folder, Path(directory) / "raw", mode="raw")
        raw_text = Path(raw["output"]).read_text(encoding="utf-8")
        assert '"role": "user"' in raw_text
        assert '"value": 42' in raw_text


def test_parquet_missing_dependency_has_actionable_reason():
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory) / "input"
        folder.mkdir()
        (folder / "table.parquet").write_bytes(b"PAR1dataPAR1")
        with patch("mariana.studio.corpus._parquet_available", return_value=False):
            summary = inspect_folder(folder)
        assert summary["files"] == 0 and summary["skipped"] == 1
        assert summary["skipped_files"][0]["reason"] == "Parquet support requires pyarrow"


def test_a_single_file_is_a_valid_corpus(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("One file is enough to train on.\n" * 4, encoding="utf-8")
    summary = inspect_folder(source)
    assert summary["source"] == "file" and summary["files"] == 1 and summary["skipped"] == 0
    prepared = prepare_corpus(source, tmp_path / "out")
    assert Path(prepared["output"]).read_text(encoding="utf-8").startswith("One file is enough")
    assert prepared["files"] == 1
    binary = tmp_path / "weights.bin"
    binary.write_bytes(b"\x00\x01\x02")
    assert inspect_folder(binary)["skipped_files"] == [
        {"path": "weights.bin", "reason": "Binary or undecodable content"},
    ]
    with pytest.raises(ValueError, match="does not exist"):
        inspect_folder(tmp_path / "missing")
    with pytest.raises(ValueError, match="does not exist"):
        inspect_folder("")
