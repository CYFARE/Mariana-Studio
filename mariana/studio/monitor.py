"""Live hardware sampling for the Studio dashboard.

One short-lived cache feeds every consumer: the dashboard's event stream and the
``/api/system`` overview share the same probe, so any number of open pages cost
at most one memory/GPU/disk reading per interval.  Streams stop as soon as the
client disconnects or Studio closes.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from ..device import pick_device

DEFAULT_INTERVAL = 1.0
CACHE_SECONDS = 0.5


def _memory_usage() -> dict:
    """Total and available host memory in bytes (``None`` where unsupported)."""
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        return {
            "total": os.sysconf("SC_PHYS_PAGES") * page,
            "available": os.sysconf("SC_AVPHYS_PAGES") * page,
        }
    except (ValueError, OSError, AttributeError):
        return {"total": None, "available": None}


def _gpu_name(device: torch.device) -> str:
    try:
        if device.type == "cuda":
            return torch.cuda.get_device_properties(device).name
        if device.type == "xpu" and hasattr(torch, "xpu"):
            return torch.xpu.get_device_properties(device).name
    except Exception:  # noqa: BLE001 - a missing device must not break the dashboard
        pass
    return str(device)


def _gpu_usage(device: torch.device, name: str) -> dict | None:
    """Driver-level used/free memory for the selected accelerator, if it has one."""
    try:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            free, total = torch.xpu.mem_get_info(device)
        else:
            return None
        return {"name": name, "total": int(total), "used": int(total - free), "free": int(free)}
    except Exception:  # noqa: BLE001 - a busy or lost device must not break the dashboard
        return None


def _disk_usage(root: Path) -> dict:
    try:
        usage = shutil.disk_usage(root)
        return {"total": usage.total, "free": usage.free}
    except OSError:
        return {"total": None, "free": None}


class SystemMonitor:
    """Samples hardware usage at most once per interval and shares the result."""

    def __init__(self, root: Path, started: float | None = None,
                 interval: float = DEFAULT_INTERVAL):
        self.root = Path(root).resolve()
        self.started = started if started is not None else time.time()
        self.interval = max(0.05, float(interval))
        self._lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._closed = threading.Event()
        self._snapshot: dict | None = None
        self._sampled_at = 0.0
        self._device: torch.device | None = None
        self._device_name = ""

    def snapshot(self, max_age: float | None = None) -> dict:
        """The newest hardware snapshot, probing only when the cache is stale."""
        age = CACHE_SECONDS if max_age is None else max(0.0, float(max_age))
        with self._lock:
            if self._snapshot is not None and time.monotonic() - self._sampled_at < age:
                return self._snapshot
        with self._probe_lock:
            with self._lock:
                if self._snapshot is not None and time.monotonic() - self._sampled_at < age:
                    return self._snapshot
            value = self._probe()
            with self._lock:
                self._snapshot = value
                self._sampled_at = time.monotonic()
            return value

    def events(self) -> Iterator[dict]:
        """Yield one ``stats`` event per interval until the client leaves or Studio closes."""
        while not self._closed.is_set():
            yield {"type": "stats", **self.snapshot(max_age=self.interval * 0.9)}
            self._closed.wait(self.interval)

    def close(self) -> None:
        """Wake every open stream so it ends."""
        self._closed.set()

    # -- internals -------------------------------------------------------
    def _probe(self) -> dict:
        now = time.time()
        if self._device is None:
            self._device = pick_device("auto")
            self._device_name = _gpu_name(self._device)
        return {
            "sampled_at": now,
            "uptime": max(0.0, now - self.started),
            "memory": _memory_usage(),
            "gpu": _gpu_usage(self._device, self._device_name),
            "disk": _disk_usage(self.root),
        }
