"""Live hardware sampling: shared caching, snapshots and stream shutdown."""

from __future__ import annotations

import time

from mariana.studio.monitor import SystemMonitor


def test_snapshot_reports_memory_disk_and_uptime(tmp_path):
    monitor = SystemMonitor(tmp_path, started=time.time() - 5)
    try:
        stats = monitor.snapshot()
        assert stats["memory"].keys() == {"total", "available"}
        assert stats["disk"]["total"] > 0 and stats["disk"]["free"] > 0
        assert stats["uptime"] >= 4.9 and stats["sampled_at"] > 0
        assert stats["gpu"] is None or set(stats["gpu"]) == {"name", "total", "used", "free"}
    finally:
        monitor.close()


def test_snapshot_probes_once_for_every_consumer(tmp_path, monkeypatch):
    monitor = SystemMonitor(tmp_path, interval=60)
    calls = []

    def probe():
        calls.append(True)
        return {"sampled_at": 1.0, "uptime": 1.0, "memory": {}, "gpu": None, "disk": {}}

    monkeypatch.setattr(monitor, "_probe", probe)
    try:
        first = monitor.snapshot(max_age=60)
        second = monitor.snapshot(max_age=60)
        assert first is second and len(calls) == 1
        monitor.snapshot(max_age=0)
        assert len(calls) == 2
    finally:
        monitor.close()


def test_streams_share_a_sample_and_stop_when_the_monitor_closes(tmp_path, monkeypatch):
    monitor = SystemMonitor(tmp_path, interval=60)
    calls = []

    def probe():
        calls.append(True)
        return {"sampled_at": 1.0, "uptime": 1.0, "memory": {}, "gpu": None, "disk": {}}

    monkeypatch.setattr(monitor, "_probe", probe)
    left, right = monitor.events(), monitor.events()
    try:
        assert next(left)["type"] == "stats"
        assert next(right)["type"] == "stats"
        assert len(calls) == 1
        monitor.close()
        assert next(left, None) is None and next(right, None) is None
    finally:
        monitor.close()
