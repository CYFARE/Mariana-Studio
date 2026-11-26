"""A process-wide, memory-aware hot cache for workspace text.

The swarm reads the same files repeatedly: duplicate member reads, follow-up
searches, re-checks after a review.  A run-scoped tool cache already removes
duplicate executor calls, but every new run and every different tool call still
hits the disk.  This cache sits below the tools and keeps decoded file bytes in
system RAM, keyed by the absolute path and the file's ``mtime_ns``/size, so a
change on disk is never served from a stale entry.

The budget adapts to the machine instead of being a fixed constant: a fifth of
the currently available RAM, bounded to 64 MiB–2 GiB.  ``MARIANA_HOT_CACHE_MB``
overrides it.  Large files and binary data are not cached.  The cache is
in-memory only and never written to disk.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

MIN_BUDGET = 64 * 1024 * 1024
MAX_BUDGET = 2 * 1024 * 1024 * 1024
DEFAULT_BUDGET = 256 * 1024 * 1024
MAX_ENTRY_BYTES = 8 * 1024 * 1024
DEFAULT_TTL = 600.0


def _available_memory() -> int:
    """Best-effort free RAM in bytes, without a third-party dependency."""
    try:
        with open("/proc/meminfo", encoding="ascii", errors="ignore") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0


def cache_budget() -> int:
    """The cache budget: an explicit env override, else a share of free RAM."""
    override = os.environ.get("MARIANA_HOT_CACHE_MB", "").strip()
    if override:
        try:
            value = int(float(override) * 1024 * 1024)
        except ValueError:
            value = 0
        if value > 0:
            return max(MIN_BUDGET, min(MAX_BUDGET, value))
    available = _available_memory()
    if available <= 0:
        return DEFAULT_BUDGET
    return max(MIN_BUDGET, min(MAX_BUDGET, available // 5))


class HotCache:
    """A bounded LRU of ``key -> (signature, bytes, expires_at)``."""

    def __init__(self, budget: int | None = None, ttl: float = DEFAULT_TTL):
        self.budget = max(0, int(budget if budget is not None else cache_budget()))
        self.ttl = max(1.0, float(ttl))
        self._entries: OrderedDict[str, tuple[str, bytes, float]] = OrderedDict()
        self._chars = 0
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "stores": 0}
        self._lock = threading.Lock()

    def get(self, key: str, signature: str) -> bytes | None:
        """Return cached bytes for an unchanged signature, else ``None``."""
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            stored_signature, value, expires = entry
            if stored_signature != signature or expires <= now:
                self._entries.pop(key, None)
                self._chars -= len(value)
                self._stats["misses"] += 1
                if expires <= now:
                    self._stats["evictions"] += 1
                return None
            self._entries.move_to_end(key)
            self._stats["hits"] += 1
            return value

    def put(self, key: str, signature: str, value: bytes,
            ttl: float | None = None) -> bool:
        """Store bytes when they fit the per-entry and total budgets."""
        if self.budget <= 0 or len(value) > MAX_ENTRY_BYTES or len(value) > self.budget:
            return False
        expires = time.monotonic() + (self.ttl if ttl is None else max(1.0, float(ttl)))
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._chars -= len(old[1])
            self._entries[key] = (signature, value, expires)
            self._chars += len(value)
            self._stats["stores"] += 1
            while self._chars > self.budget and self._entries:
                _, dropped = self._entries.popitem(last=False)
                self._chars -= len(dropped[1])
                self._stats["evictions"] += 1
        return True

    def invalidate(self, prefix: str = "") -> int:
        """Drop one key, or every key beginning with ``prefix``."""
        with self._lock:
            if not prefix:
                count = len(self._entries)
                self._entries.clear()
                self._chars = 0
                return count
            doomed = [key for key in self._entries if key.startswith(prefix)]
            for key in doomed:
                _, value, _ = self._entries.pop(key)
                self._chars -= len(value)
            return len(doomed)

    def clear(self) -> None:
        self.invalidate()

    def stats(self) -> dict:
        with self._lock:
            return {
                **self._stats, "entries": len(self._entries), "chars": self._chars,
                "budget": self.budget,
            }


HOT_CACHE = HotCache()


def read_cached(path, max_bytes: int, cache: HotCache | None = None, accept=None):
    """Read a small file through the hot cache.

    Returns ``(data, stat)`` or ``(None, stat)`` when the path is missing, too
    large, unreadable, or rejected by ``accept(data)``.  Rejected bytes are not
    cached, so binary files never occupy the budget.
    """
    cache = cache or HOT_CACHE
    try:
        info = os.stat(path)
    except OSError:
        return None, None
    if not os.path.isfile(path) or info.st_size > max_bytes:
        return None, info
    signature = f"{info.st_mtime_ns}:{info.st_size}"
    key = str(path)
    data = cache.get(key, signature)
    if data is None:
        try:
            with open(path, "rb") as handle:
                data = handle.read(max_bytes + 1)
        except OSError:
            return None, info
        if len(data) > max_bytes or (accept is not None and not accept(data)):
            return None, info
        cache.put(key, signature, data)
    return data, info


__all__ = [
    "DEFAULT_BUDGET", "HOT_CACHE", "HotCache", "MAX_ENTRY_BYTES",
    "MIN_BUDGET", "cache_budget", "read_cached",
]
