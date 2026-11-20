"""Local IDE voice input: backend discovery, validation and transcription jobs."""

from __future__ import annotations

import base64
import os

import pytest

from mariana.studio import voice as voice_module
from mariana.studio.errors import BadRequestError, UnavailableError
from mariana.studio.jobs import JobRegistry
from mariana.studio.voice import VoiceService
from tests.helpers import wait_for


def no_backend(service, monkeypatch):
    monkeypatch.setattr(service, "_has_faster_whisper", lambda: False)
    monkeypatch.setattr(voice_module.shutil, "which", lambda name: None)
    monkeypatch.delenv("MARIANA_WHISPER_CPP", raising=False)
    monkeypatch.delenv("MARIANA_WHISPER_CPP_MODEL", raising=False)


def audio() -> str:
    return base64.b64encode(b"RIFF" * 100).decode()


def test_voice_status_explains_how_to_enable_local_transcription(tmp_path, monkeypatch):
    registry = JobRegistry()
    service = VoiceService(tmp_path, registry)
    no_backend(service, monkeypatch)
    status = service.status()
    assert status["available"] is False
    assert "pip install faster-whisper" in status["install"]
    assert status["machine_local"] is False
    assert status["max_bytes"] > 0 and status["models"]
    with pytest.raises(UnavailableError):
        service.transcribe({"audio": audio(), "format": "webm"})
    service.close()
    registry.close()


def test_voice_transcription_rejects_bad_recordings(tmp_path, monkeypatch):
    registry = JobRegistry()
    service = VoiceService(tmp_path, registry)
    monkeypatch.setattr(service, "_backend", lambda: {
        "id": "test", "label": "Test", "kind": "openai_whisper", "command": "/bin/true",
    })
    with pytest.raises(BadRequestError):
        service.transcribe({})
    with pytest.raises(BadRequestError):
        service.transcribe({"audio": "not base64", "format": "webm"})
    with pytest.raises(BadRequestError):
        service.transcribe({"audio": audio(), "format": "exe"})
    with pytest.raises(BadRequestError):
        service.transcribe({"audio": audio(), "format": "webm", "model": "huge"})
    with pytest.raises(BadRequestError):
        service.transcribe({"audio": audio(), "format": "webm", "language": "english!"})
    service.close()
    registry.close()


@pytest.mark.skipif(os.name != "posix", reason="the fake backend is a POSIX executable")
def test_voice_transcription_runs_a_local_backend_as_a_job(tmp_path, monkeypatch):
    fake = tmp_path / "whisper"
    fake.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "folder = pathlib.Path(args[args.index('--output_dir') + 1])\n"
        "(folder / 'voice.txt').write_text('hello from a fake whisper\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    registry = JobRegistry()
    service = VoiceService(tmp_path, registry)
    monkeypatch.setattr(service, "_has_faster_whisper", lambda: False)
    monkeypatch.setattr(voice_module.shutil, "which",
                        lambda name: str(fake) if name == "whisper" else None)
    monkeypatch.delenv("MARIANA_WHISPER_CPP", raising=False)
    monkeypatch.delenv("MARIANA_WHISPER_CPP_MODEL", raising=False)

    started = service.transcribe({"audio": audio(), "format": "webm", "language": "en-US"})
    assert started["started"] is True
    job = wait_for(lambda: (lambda snapshot: snapshot if snapshot["status"] in {
        "completed", "failed", "cancelled",
    } else None)(registry.get(started["job"]["id"])), timeout=10)
    assert job["status"] == "completed", job.get("error")
    assert job["result"]["text"] == "hello from a fake whisper"
    assert job["result"]["backend"] == "whisper-cli"
    assert job["result"]["language"] == "en"
    assert list((tmp_path / "agent-workspace" / "voice").iterdir()) == []
    service.close()
    registry.close()
