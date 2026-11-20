"""Deployment: export Mariana checkpoints and register them with Ollama.

Exports wrap the existing ``mariana.py export`` command as a job, so the
GGUF / ONNX / safetensors converters stay the single implementation.  Serving
through the OpenAI-compatible API lives in ``api/openai_api.py``.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from .errors import BadRequestError, ConflictError, NotFoundError, UnavailableError
from .training import launcher
from .validation import choice

FORMATS = ("gguf", "onnx", "safetensors", "all")
DTYPES = ("f16", "f32", "bf16")
EXPORT_HINT = "pip install -r requirements-export.txt"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,60}")
_OLLAMA_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,60}(:[a-z0-9._-]{1,40})?")
_ARTIFACT_SUFFIXES = (".gguf", ".onnx", ".safetensors")


class DeployService:
    """Export jobs and the list of produced artifacts."""

    def __init__(self, root: Path, registry, gateway):
        self.root = Path(root).resolve()
        self.registry = registry
        self.gateway = gateway
        self.folder = self.root / "exports"

    def exports(self) -> list[dict]:
        """Files in ``exports/`` that can be served, shared or loaded elsewhere."""
        if not self.folder.is_dir():
            return []
        modelfile = self.folder / "Modelfile"
        items = []
        for path in sorted(self.folder.iterdir()):
            if path.is_file() and path.suffix in _ARTIFACT_SUFFIXES:
                stat = path.stat()
                items.append({
                    "name": path.name, "path": str(path.relative_to(self.root)),
                    "format": path.suffix[1:], "bytes": stat.st_size, "modified": stat.st_mtime,
                    "modelfile": modelfile.is_file() and path.suffix == ".gguf",
                })
        return sorted(items, key=lambda item: -item["modified"])

    def exportable(self) -> list[dict]:
        """Mariana checkpoints, each with whether it can be exported and why not."""
        result = []
        for entry in self.gateway.catalog():
            if entry["kind"] != "mariana-drv" or not entry["available"]:
                continue
            moe = bool(entry["detail"].get("moe"))
            result.append({
                "id": entry["id"], "name": entry["name"], "bytes": entry["bytes"],
                "exportable": not moe,
                "reason": "Sparse-MoE checkpoints cannot be exported; train a dense model." if moe else "",
                "latent": bool(entry["detail"].get("latent")),
            })
        return result

    def start_export(self, config: dict) -> dict:
        entry = self.gateway.get(str(config.get("model") or ""))
        if entry["kind"] != "mariana-drv":
            raise BadRequestError(
                "Only Mariana DRV-LM checkpoints are exported here. Third-party models are "
                "already in a portable format; NCM checkpoints have no export path yet."
            )
        if entry["detail"].get("moe"):
            raise BadRequestError("Sparse-MoE checkpoints cannot be exported; train a dense model")
        model_format = choice(config, "format", "gguf", FORMATS, "Format")
        dtype = choice(config, "dtype", "f16", DTYPES, "Precision")
        name = str(config.get("name") or Path(entry["name"]).stem).strip()
        if not _NAME.fullmatch(name):
            raise BadRequestError("Export name may contain only letters, numbers, dots, dashes, and underscores")
        if self.registry.active("export"):
            raise ConflictError("An export is already running")
        command = [
            sys.executable, launcher(self.root), "export",
            "--checkpoint", entry["absolute_path"], "--format", model_format,
            "--out", str(self.folder), "--dtype", dtype, "--name", name,
        ]
        job = self.registry.create(
            "export", f"Export {entry['name']} → {model_format.upper()}",
            meta={"model": entry["id"], "format": model_format, "name": name, "view": "deploy"},
        )

        def run(job):
            job.update(stage="Converting")
            written: list[str] = []

            def collect(line: str) -> None:
                if line.startswith("Wrote "):
                    written.append(line[6:].strip())

            code = self.registry.run_process(job, command, cwd=self.root, on_line=collect)
            if code != 0:
                tail = job.snapshot(logs=True)["logs"][-6:]
                missing = any("is required" in line or "No module named" in line for line in tail)
                reason = next((line for line in reversed(tail) if line.strip()), "")
                raise RuntimeError(
                    f"The export toolchain is not installed; {EXPORT_HINT}" if missing
                    else f"Export failed: {reason or f'exit code {code}'}"
                )
            return {"files": [str(Path(path).name) for path in written]}

        self.registry.start(job, run)
        return job.snapshot()

    def start_ollama(self, config: dict) -> dict:
        """Register an exported GGUF with Ollama: ``ollama create <name> -f Modelfile``."""
        artifact = (self.folder / str(config.get("file") or "")).resolve()
        if artifact.parent != self.folder or artifact.suffix != ".gguf" or not artifact.is_file():
            raise NotFoundError("Choose a GGUF file from exports/")
        name = str(config.get("name") or artifact.stem).strip().lower()
        if not _OLLAMA_NAME.fullmatch(name):
            raise BadRequestError("Ollama model names use lowercase letters, numbers, dots and dashes")
        binary = shutil.which("ollama")
        if not binary:
            raise UnavailableError("Ollama is not installed; get it from https://ollama.com/download")
        if not self.gateway.ollama.running():
            raise UnavailableError("Ollama is not running. Start it from Models → Runtimes first.")
        modelfile = self.folder / f"{artifact.stem}.Modelfile"
        template = self.folder / "Modelfile"
        if template.is_file():
            lines = template.read_text(encoding="utf-8").splitlines()
            lines = [f"FROM ./{artifact.name}" if line.startswith("FROM ") else line for line in lines]
        else:
            lines = [f"FROM ./{artifact.name}"]
        modelfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        job = self.registry.create(
            "deploy", f"Create Ollama model {name}",
            meta={"file": artifact.name, "name": name, "view": "deploy"},
        )

        def run(job):
            job.update(stage="Registering with Ollama")
            code = self.registry.run_process(
                job, [binary, "create", name, "-f", modelfile.name], cwd=self.folder,
            )
            if code != 0:
                raise RuntimeError("`ollama create` failed; see the log for details")
            self.gateway.invalidate()
            return {"model": name, "model_id": f"ollama:{name if ':' in name else name + ':latest'}"}

        self.registry.start(job, run)
        return job.snapshot()
