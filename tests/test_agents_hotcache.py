"""Process-wide hot cache: bounded LRU, signatures and disk invalidation."""

from __future__ import annotations

import os
import time
from pathlib import Path

from mariana.agents.hotcache import HotCache, read_cached


def test_hot_cache_is_an_lru_with_signature_checks():
    cache = HotCache(budget=120, ttl=60)
    assert cache.put("a", "1", b"a" * 50)
    assert cache.get("a", "1") == b"a" * 50
    assert cache.get("a", "2") is None  # the file changed on disk
    assert cache.put("a", "2", b"a" * 50)
    assert cache.get("a", "2") == b"a" * 50

    assert cache.put("b", "1", b"b" * 50)
    assert cache.put("c", "1", b"c" * 50)
    assert cache.get("a", "1") is None  # the oldest entry was evicted
    assert cache.get("c", "1") == b"c" * 50
    assert cache.stats()["evictions"] >= 1

    assert cache.put("huge", "1", b"x" * 121) is False  # rejected, never stored
    assert cache.invalidate("b") == 1
    assert cache.stats()["entries"] == 1


def test_read_cached_reuses_bytes_until_the_file_changes(tmp_path: Path):
    cache = HotCache(budget=1_000, ttl=60)
    path = tmp_path / "data.txt"
    path.write_text("one", encoding="utf-8")

    first, _ = read_cached(path, 1_000, cache=cache)
    second, _ = read_cached(path, 1_000, cache=cache)
    assert first == second == b"one"
    assert cache.stats()["hits"] == 1

    path.write_text("two", encoding="utf-8")
    # Ensure the nanosecond signature differs even on a coarse filesystem.
    now = time.time() + 2
    os.utime(path, (now, now))
    changed, _ = read_cached(path, 1_000, cache=cache)
    assert changed == b"two"


def test_read_cached_does_not_store_rejected_bytes(tmp_path: Path):
    cache = HotCache(budget=1_000, ttl=60)
    path = tmp_path / "binary.bin"
    path.write_bytes(b"a\x00b")
    data, _ = read_cached(path, 1_000, cache=cache, accept=lambda value: b"\x00" not in value)
    assert data is None
    assert cache.stats()["entries"] == 0
