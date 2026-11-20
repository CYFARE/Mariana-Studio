"""The Studio service container: shutdown releases models exactly once."""

from __future__ import annotations

from mariana.studio.app import Studio


def test_close_stops_the_monitor_and_gateway_once(tmp_path, monkeypatch):
    app = Studio(tmp_path)
    calls = []

    monitor_close, gateway_close = app.monitor.close, app.gateway.close

    def close_monitor():
        calls.append("monitor")
        monitor_close()

    def close_gateway():
        calls.append("gateway")
        gateway_close()

    monkeypatch.setattr(app.monitor, "close", close_monitor)
    monkeypatch.setattr(app.gateway, "close", close_gateway)
    app.close()
    app.close()
    assert calls == ["monitor", "gateway"]
