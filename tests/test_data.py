"""Tests for the byte-level data pipeline and the offline demo corpus."""

from __future__ import annotations

from mariana.data import DEMO_TEXT, build_dataset, fetch_corpus


def test_fetch_corpus_returns_the_builtin_demo_text():
    assert fetch_corpus() == DEMO_TEXT.encode("utf-8")


def test_fetch_corpus_reads_an_explicit_file(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"a local corpus")
    assert fetch_corpus(str(corpus)) == b"a local corpus"


def test_demo_corpus_supports_the_default_context():
    train, val = build_dataset(fetch_corpus(), ctx=256)
    assert train.shape[1] == 257
    assert val.shape[1] == 257
    assert train.numel() > 0
    assert val.numel() > 0


def test_demo_corpus_supports_the_largest_default_drv_context():
    train, val = build_dataset(fetch_corpus(), ctx=512)
    assert train.shape[1] == 513
    assert val.shape[1] == 513
    assert train.numel() > 0
    assert val.numel() > 0
