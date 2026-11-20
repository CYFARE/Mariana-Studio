"""Local speech-to-text for the IDE coding agent.

The browser can dictate through its own speech service, and speech synthesis is
built into the browser.  This service adds the private path: a short recording is
transcribed by a local Whisper backend when one is installed.  No backend is
required; the status carries the exact install hint and the browser remains a
fallback.  Audio is decoded, bounded, written to a temporary folder and deleted
with the job.
"""

from __future__ import annotations

import base64
import binascii
import importlib.util
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from .errors import BadRequestError, ConflictError, UnavailableError
from .jobs import JobRegistry

MAX_AUDIO_BYTES = 8_000_000
MAX_TRANSCRIPT_CHARS = 20_000
MAX_RECORDING_SECONDS = 180
MODEL_CHOICES = ("tiny", "base", "small")
DEFAULT_MODEL = "base"
ALLOWED_FORMATS = frozenset({"webm", "ogg", "oga", "wav", "mp3", "m4a", "mp4", "flac"})
_LANGUAGE = re.compile(r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})?$")
_FASTER_WHISPER = """import sys
from faster_whisper import WhisperModel
audio, model_name, language, destination = sys.argv[1:5]
model = WhisperModel(model_name, device="cpu", compute_type="int8")
segments, _info = model.transcribe(audio, language=language or None, vad_filter=True)
text = " ".join(segment.text.strip() for segment in segments).strip()
with open(destination, "w", encoding="utf-8") as handle:
    handle.write(text)
"""


class VoiceService:
    """Discover a local Whisper backend and transcribe recordings as jobs."""

    def __init__(self, root: Path, registry: JobRegistry):
        self.root = Path(root).resolve()
        self.registry = registry
        self._lock = threading.Lock()
        self._closed = False

    # -- discovery --------------------------------------------------------
    def _model_name(self) -> str:
        requested = os.environ.get("MARIANA_WHISPER_MODEL", DEFAULT_MODEL).strip().lower()
        return requested if requested in MODEL_CHOICES else DEFAULT_MODEL

    def _install_hint(self) -> str:
        return (
            "Install a local transcriber with `pip install faster-whisper`, or point "
            "MARIANA_WHISPER_CPP and MARIANA_WHISPER_CPP_MODEL at a whisper.cpp binary "
            "and model. Browser dictation still works where the browser supports it."
        )

    def _whisper_cpp(self) -> dict | None:
        binary = os.environ.get("MARIANA_WHISPER_CPP", "").strip()
        model = os.environ.get("MARIANA_WHISPER_CPP_MODEL", "").strip()
        if not binary:
            binary = shutil.which("whisper-cli") or shutil.which("whisper-cpp") or ""
        if not binary or not model:
            return None
        if not Path(binary).is_file():
            binary = shutil.which(binary) or ""
        if not binary or not Path(model).is_file():
            return None
        return {
            "id": "whisper-cpp", "label": "whisper.cpp", "kind": "whisper_cpp",
            "command": binary, "model": model,
        }

    def _has_faster_whisper(self) -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _backend(self) -> dict | None:
        if self._has_faster_whisper():
            return {"id": "faster-whisper", "label": "faster-whisper (CPU)", "kind": "faster_whisper"}
        cpp = self._whisper_cpp()
        if cpp is not None:
            return cpp
        binary = shutil.which("whisper")
        if binary:
            return {"id": "whisper-cli", "label": "OpenAI Whisper CLI", "kind": "openai_whisper",
                    "command": binary}
        return None

    def status(self) -> dict:
        """The available backend, its install hint and the accepted recording limits."""
        backend = self._backend()
        active = self.registry.active("voice")
        return {
            "available": backend is not None,
            "backend": backend["id"] if backend else "",
            "label": backend["label"] if backend else "",
            "reason": "" if backend else "No local speech-to-text backend is installed.",
            "install": "" if backend else self._install_hint(),
            "machine_local": backend is not None,
            "busy": bool(active),
            "max_bytes": MAX_AUDIO_BYTES,
            "max_seconds": MAX_RECORDING_SECONDS,
            "formats": sorted(ALLOWED_FORMATS),
            "models": list(MODEL_CHOICES),
            "default_model": self._model_name(),
            "privacy": (
                "Recordings are transcribed by a process on this machine and deleted "
                "when the job finishes." if backend else
                "Install a local backend to keep recordings on this machine; browser "
                "dictation is handled by the browser and may use a remote service."
            ),
        }

    # -- transcription ----------------------------------------------------
    def _decode(self, payload: dict) -> tuple[bytes, str, str, str]:
        raw = payload.get("audio")
        if not isinstance(raw, str) or not raw.strip():
            raise BadRequestError("Record a short voice message first, then try again")
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        try:
            audio = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            raise BadRequestError("The recording was not valid base64; record it again") from None
        if not audio:
            raise BadRequestError("The recording was empty; hold the microphone a little longer")
        if len(audio) > MAX_AUDIO_BYTES:
            raise BadRequestError("The recording is too large; keep voice input under three minutes")
        value = str(payload.get("format") or "webm").strip().lower().lstrip(".")
        if value not in ALLOWED_FORMATS:
            raise BadRequestError(
                "Unsupported recording format; use webm, ogg, wav, mp3, m4a, mp4 or flac"
            )
        model = str(payload.get("model") or self._model_name()).strip().lower()
        if model not in MODEL_CHOICES:
            raise BadRequestError("Choose a Whisper model from tiny, base or small")
        language = str(payload.get("language") or "").strip()
        if language and not _LANGUAGE.match(language):
            raise BadRequestError("Language must look like en, en-US or pt-BR")
        return audio, value, model, language

    def transcribe(self, payload: dict | None = None) -> dict:
        """Validate a recording and start one background transcription job."""
        if self._closed:
            raise UnavailableError("Studio is shutting down; voice input is unavailable")
        payload = payload if isinstance(payload, dict) else {}
        backend = self._backend()
        if backend is None:
            raise UnavailableError(self._install_hint())
        audio, fmt, model, language = self._decode(payload)
        with self._lock:
            active = self.registry.active("voice")
            if active:
                raise ConflictError("A voice transcription is already running; wait for it to finish")
            job = self.registry.create(
                "voice", "Transcribe voice input",
                meta={"view": "ide", "backend": backend["id"], "model": model, "format": fmt},
            )
            self.registry.start(job, lambda current: self._run_job(
                current, backend, audio, fmt, model, language,
            ))
        return {"job": job.snapshot(), "started": True}

    def _temp_root(self) -> Path:
        path = self.root / "agent-workspace" / "voice"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _run_job(self, job, backend: dict, audio: bytes, fmt: str, model: str,
                 language: str) -> dict:
        started = time.time()
        language_code = language.split("-", 1)[0] if language else ""
        job.update(stage=f"Transcribing with {backend['label']}")
        with tempfile.TemporaryDirectory(dir=self._temp_root(), prefix="record-") as folder:
            folder_path = Path(folder)
            audio_path = folder_path / f"voice.{fmt}"
            audio_path.write_bytes(audio)
            if backend["kind"] == "faster_whisper":
                script = folder_path / "transcribe.py"
                script.write_text(_FASTER_WHISPER, encoding="utf-8")
                destination = folder_path / "transcript.txt"
                command = [
                    sys.executable, str(script), str(audio_path), model, language_code,
                    str(destination),
                ]
            elif backend["kind"] == "whisper_cpp":
                destination = folder_path / "transcript.txt"
                command = [
                    backend["command"], "-m", backend["model"], "-f", str(audio_path),
                    "-otxt", "-of", str(folder_path / "transcript"),
                    "-l", language_code or "auto",
                ]
            else:
                destination = folder_path / "voice.txt"
                command = [
                    backend["command"], str(audio_path), "--model", model,
                    "--output_format", "txt", "--output_dir", folder,
                ]
                if language_code:
                    command.extend(["--language", language_code])
            code = self.registry.run_process(job, command, cwd=folder_path)
            job.check_cancelled()
            if code != 0:
                raise RuntimeError(
                    f"{backend['label']} failed with exit code {code}. "
                    "Read the job log, or try the tiny model for a faster run."
                )
            try:
                text = destination.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raise RuntimeError(
                    "The transcriber finished but wrote no transcript. "
                    "Try a shorter recording or reinstall the local backend."
                ) from None
        text = " ".join(text.split()).strip()[:MAX_TRANSCRIPT_CHARS]
        if not text:
            raise RuntimeError(
                "No speech was recognised. Move closer to the microphone and record again."
            )
        job.update(stage="Transcript ready")
        return {
            "text": text,
            "backend": backend["id"],
            "model": model,
            "language": language_code,
            "seconds": round(time.time() - started, 2),
        }

    def close(self) -> None:
        """Refuse new work. Active transcription processes belong to the shared job registry."""
        self._closed = True
