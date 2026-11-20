"""Training jobs: command builders and the single-slot training manager.

A training run is a subprocess (``mariana.py train …`` or the Hugging Face
fine-tune trainer) whose stdout is parsed for progress.  Only one run may be
active at a time because runs compete for the same GPU.  The manager also acts
as a job provider so runs appear in the unified job list.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

from ..workspace import jobs_path, write_json
from .corpus import inspect_folder, prepare_corpus
from .errors import ConflictError, NotFoundError
from .smart import recommend_training_options
from .validation import number

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
MAX_LOG_LINES = 5000
LOG_TAIL = 250
MAX_METRIC_POINTS = 1200

TECHNIQUES = {
    "fast": {
        "name": "Sprint-DRV",
        "description": "Fast hybrid AR + diffusion training with one shared backbone pass.",
        "badge": "Recommended",
    },
    "ncm": {
        "name": "Next-Concept Modeling",
        "description": "Two-stage coarse-to-fine residual VQ and parallel concept generation.",
        "badge": "Parallel",
    },
    "legacy": {
        "name": "Legacy DRV-LM",
        "description": "Two-pass hybrid objective with full-sequence AR alignment.",
        "badge": "Alignment",
    },
}


def launcher(root: Path) -> str:
    """The ``mariana.py`` entry script: the workspace's own, else the installed package's."""
    local = Path(root) / "mariana.py"
    return str(local if local.is_file() else PACKAGE_PARENT / "mariana.py")


def _checkpoint_name(value, technique):
    value = (value or f"mariana-{technique}-gui.pt").strip()
    if Path(value).name != value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("Model name may contain only letters, numbers, dots, dashes, and underscores")
    return value if value.endswith(".pt") else value + ".pt"


NCM_MODES = {"fresh", "continue", "update", "fresh_generator", "update_tokenizer", "fresh_tokenizer", "finetune"}
NCM_TOKENIZER_MODES = {"fresh", "update", "update_tokenizer", "fresh_tokenizer"}
NCM_TOKENIZER_ONLY_MODES = {"update_tokenizer", "fresh_tokenizer"}
DRV_MODES = {"fresh", "continue", "finetune"}


def _checkpoint_step(path, expected_stage=None):
    if not path.exists():
        raise ValueError(f"Required checkpoint does not exist: {path.name}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    stage = state.get("stage")
    if expected_stage == "drv" and stage is not None:
        raise ValueError(f"{path.name} must be a Mariana DRV-LM checkpoint")
    if expected_stage not in {None, "drv"} and stage != expected_stage:
        raise ValueError(f"{path.name} must be an NCM {expected_stage} checkpoint")
    return int(state.get("step", 0))


def _integrated_training_command(config, corpus_path, checkpoint, root: Path):
    technique = config.get("technique", "fast")
    if technique not in TECHNIQUES:
        raise ValueError("Unknown training technique")
    mode = config.get("mode", "fresh")
    steps = number(config, "steps", 1000, 1, 10_000_000)
    if mode == "continue":
        steps += int(config.get("_generator_start_step", 0))
    command = [
        sys.executable, launcher(root), "train",
        "--data", str(corpus_path), "--checkpoint", str(checkpoint),
        "--steps", str(steps),
        "--batch", str(number(config, "batch", 16, 1, 4096)),
        "--grad-accum", str(number(config, "grad_accum", 1, 1, 4096)),
        "--ctx", str(number(config, "ctx", 128, 8, 32768)),
        "--dim", str(number(config, "dim", 256, 8, 8192)),
        "--heads", str(number(config, "heads", 4, 1, 128)),
        "--lr", str(number(config, "lr", 3e-3, 1e-8, 1.0, float)),
        "--warmup", str(number(config, "warmup", 50, 0, 1_000_000)),
        "--eval-every", str(number(config, "eval_every", 200, 1, 10_000_000)),
        "--device", config.get("device", "auto"), "--no-gen",
    ]
    if config.get("device", "auto") not in {"auto", "cuda", "xpu", "mps", "cpu"}:
        raise ValueError("Unknown device")
    if config.get("pause_file"):
        command += ["--pause-file", str(config["pause_file"])]
    if config.get("init_from"):
        command += ["--init-from", str(config["init_from"])]
    if int(config.get("freeze_layers", 0) or 0) > 0:
        command += ["--freeze-layers", str(number(config, "freeze_layers", 0, 0, 10000))]
    if config.get("freeze_embeddings"):
        command.append("--freeze-embeddings")
    if technique == "ncm":
        tokenizer = checkpoint.with_name(checkpoint.stem + "-tokenizer.pt")
        command += [
            "--ncm", "--ncm-tokenizer", str(tokenizer),
            "--ncm-tokenizer-steps", str(number(config, "tokenizer_steps", 1000, 1, 10_000_000)),
            "--ncm-generator-lr", str(number(config, "generator_lr", 3e-4, 1e-8, 1.0, float)),
            "--ncm-prefix", str(number(config, "prefix", 64, 1, 16384)),
        ]
        if config.get("tokenizer_batch") is not None:
            command += ["--ncm-tokenizer-batch", str(number(config, "tokenizer_batch", 1, 1, 4096))]
        if config.get("tokenizer_grad_accum") is not None:
            command += ["--ncm-tokenizer-grad-accum", str(number(config, "tokenizer_grad_accum", 1, 1, 4096))]
        levels = config.get("levels", "").strip()
        if levels:
            if not re.fullmatch(r"\d+(,\d+)*", levels):
                raise ValueError("Levels must be comma-separated positive integers")
            command += ["--ncm-levels", levels]
        if config.get("compile"):
            command.append("--compile")
        if config.get("flex"):
            command.append("--flex")
        if config.get("cache_codes"):
            command.append("--ncm-cache-codes")
    else:
        command += ["--legacy" if technique == "legacy" else "--fast"]
        if config.get("compile"):
            command.append("--compile")
        command += [
            "--n-pre", str(number(config, "n_pre", 2, 0, 128)),
            "--n-core", str(number(config, "n_core", 2, 0, 128)),
            "--n-post", str(number(config, "n_post", 2, 0, 128)),
            "--loops", str(number(config, "loops", 2, 1, 128)),
            "--block-size", str(number(config, "block_size", 32, 1, 8192)),
        ]
        if config.get("moe"):
            command.append("--moe")
            if not config.get("grouped_moe", True):
                command.append("--no-grouped-moe")
        latent_mode = config.get("latent_mode", "none")
        if latent_mode not in {"none", "saint", "legacy"}:
            raise ValueError("Latent mode must be none, saint, or legacy")
        if latent_mode != "none":
            command.append("--latent")
            if latent_mode == "legacy":
                command.append("--latent-legacy")
            command += [
                "--thoughts", str(number(config, "thoughts", 8, 1, 1024)),
                "--latent-symbols", str(number(config, "latent_symbols", 256, 2, 65536)),
                "--latent-window", str(number(config, "latent_window", 32, 1, 32768)),
            ]
            if latent_mode == "saint":
                command += [
                    "--latent-min-thoughts", str(number(config, "latent_min_thoughts", 2, 1, 1024)),
                    "--latent-block-size", str(number(config, "latent_block_size", 2, 1, 1024)),
                    "--latent-continuous-slots", str(number(config, "latent_continuous_slots", 2, 1, 64)),
                    "--latent-refine-steps", str(number(config, "latent_refine_steps", 2, 0, 32)),
                    "--latent-grammar-prob", str(number(config, "latent_grammar_prob", 0.25, 0.0, 1.0, float)),
                    "--w-latent-rl", str(number(config, "w_latent_rl", 0.1, 0.0, 10.0, float)),
                ]
    if mode == "continue":
        command.append("--resume")
    return command


def _advanced_ncm_commands(config, corpus_path, checkpoint, root: Path):
    mode = config["mode"]
    tokenizer = checkpoint.with_name(checkpoint.stem + "-tokenizer.pt")
    device = config.get("device", "auto")
    if device not in {"auto", "cuda", "xpu", "mps", "cpu"}:
        raise ValueError("Unknown device")

    def stage_command(stage, steps):
        command = [
            sys.executable, launcher(root), "ncm", stage,
            "--data", str(corpus_path), "--steps", str(steps), "--device", device,
            "--warmup", str(number(config, "warmup", 50, 0, 1_000_000)),
            "--eval-every", str(number(config, "eval_every", 200, 1, 10_000_000)),
        ]
        if config.get("pause_file"):
            command += ["--pause-file", str(config["pause_file"])]
        return command

    def tokenizer_command():
        steps = number(config, "tokenizer_steps", 1000, 1, 10_000_000)
        if mode in {"update", "update_tokenizer"}:
            steps += int(config.get("_tokenizer_start_step", 0))
        command = stage_command("tokenizer", steps) + [
            "--checkpoint", str(tokenizer),
            "--batch", str(number(config, "tokenizer_batch", config.get("batch", 16), 1, 4096)),
            "--grad-accum", str(number(config, "tokenizer_grad_accum", config.get("grad_accum", 1), 1, 4096)),
            "--lr", str(number(config, "lr", 3e-3, 1e-8, 1.0, float)),
            "--ctx", str(number(config, "ctx", 128, 8, 32768)),
            "--dim", str(number(config, "dim", 256, 8, 8192)),
            "--heads", str(number(config, "heads", 4, 1, 128)),
            "--prefix", str(number(config, "prefix", 64, 1, 16384)),
        ]
        if mode in {"update", "update_tokenizer"}:
            command.append("--resume")
        levels = config.get("levels", "").strip()
        if levels:
            if not re.fullmatch(r"\d+(,\d+)*", levels):
                raise ValueError("Levels must be comma-separated positive integers")
            command += ["--levels", levels]
        return command

    def generator_command():
        command = stage_command("generator", number(config, "steps", 1000, 1, 10_000_000)) + [
            "--tokenizer", str(tokenizer), "--checkpoint", str(checkpoint),
            "--batch", str(number(config, "batch", 16, 1, 4096)),
            "--grad-accum", str(number(config, "grad_accum", 1, 1, 4096)),
            "--lr", str(number(config, "generator_lr", 3e-4, 1e-8, 1.0, float)),
        ]
        if config.get("compile"):
            command.append("--compile")
        if config.get("flex"):
            command.append("--flex")
        if config.get("cache_codes"):
            command.append("--cache-codes")
        return command

    if mode == "update":
        return [tokenizer_command(), generator_command()]
    if mode == "fresh_generator":
        return [generator_command()]
    return [tokenizer_command()]


def _ncm_finetune_command(config, corpus_path, checkpoint, root: Path):
    """Continue an NCM generator's weights without resuming its optimizer."""
    init_from = config.get("init_from")
    if not init_from:
        raise ValueError("NCM fine-tuning needs a base checkpoint")
    tokenizer = checkpoint.with_name(checkpoint.stem + "-tokenizer.pt")
    command = [
        sys.executable, launcher(root), "ncm", "generator",
        "--data", str(corpus_path), "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer), "--init-from", str(init_from),
        "--steps", str(number(config, "steps", 1000, 1, 10_000_000)),
        "--batch", str(number(config, "batch", 16, 1, 4096)),
        "--grad-accum", str(number(config, "grad_accum", 1, 1, 4096)),
        "--lr", str(number(config, "lr", 3e-4, 1e-8, 1.0, float)),
        "--warmup", str(number(config, "warmup", 50, 0, 1_000_000)),
        "--eval-every", str(number(config, "eval_every", 200, 1, 10_000_000)),
        "--device", config.get("device", "auto"),
    ]
    if config.get("pause_file"):
        command += ["--pause-file", str(config["pause_file"])]
    if int(config.get("freeze_layers", 0) or 0) > 0:
        command += ["--freeze-layers", str(number(config, "freeze_layers", 0, 0, 10000))]
    if config.get("freeze_embeddings"):
        command.append("--freeze-embeddings")
    if config.get("compile"):
        command.append("--compile")
    if config.get("flex"):
        command.append("--flex")
    if config.get("cache_codes"):
        command.append("--cache-codes")
    return command


def build_training_commands(config, corpus_path, checkpoint, root: Path):
    technique = config.get("technique", "fast")
    mode = config.get("mode", "fresh")
    if technique == "ncm":
        if mode not in NCM_MODES:
            raise ValueError(f"Unknown NCM training mode: {mode}")
        if mode == "finetune":
            return [_ncm_finetune_command(config, corpus_path, checkpoint, root)]
        if mode in {"fresh", "continue"}:
            return [_integrated_training_command(config, corpus_path, checkpoint, root)]
        return _advanced_ncm_commands(config, corpus_path, checkpoint, root)
    if mode not in DRV_MODES:
        raise ValueError("DRV training supports fresh, continue, and finetune modes")
    return [_integrated_training_command(config, corpus_path, checkpoint, root)]


def build_training_command(config, corpus_path, checkpoint, root: Path):
    commands = build_training_commands(config, corpus_path, checkpoint, root)
    if len(commands) != 1:
        raise ValueError("This training mode requires multiple commands")
    return commands[0]


_STEP_LINE = re.compile(r"^step\s+(\d+)/(\d+)\s*\|(.+)$", re.I)
_VAL_LINE = re.compile(r"^VAL(?:\s+step\s+(\d+))?\s*:?\s*(.+)$")
_RATE_KEYS = ("tok/s", "pos/s")


def _parse_metrics(text: str) -> dict:
    metrics = {}
    for part in text.split("|"):
        pieces = part.strip().split()
        if len(pieces) != 2:
            continue
        try:
            value = float(pieces[1])
        except ValueError:
            continue
        if value == value and value not in (float("inf"), float("-inf")):
            metrics[pieces[0].lower()] = value
    return metrics


class MetricsHistory:
    """A bounded, evenly thinned record of the loss curve for one run."""

    def __init__(self, limit: int = MAX_METRIC_POINTS):
        self.limit = limit
        self.stride = 1
        self.count = 0
        self.points: list[list] = []     # [stage, step, total loss]
        self.validation: list[list] = []  # [stage, step, total loss]
        self.latest: dict = {}
        self.best: float | None = None
        self.stage = "model"
        self.last_step = 0

    def feed(self, line: str) -> None:
        line = line.strip()
        if "Tokenizer parameters:" in line:
            self.stage = "tokenizer"
            return
        if "Generator trainable parameters:" in line:
            self.stage = "generator"
            return
        match = _STEP_LINE.match(line)
        if match:
            metrics = _parse_metrics(match.group(3))
            step = int(match.group(1))
            self.last_step = step
            rate = next((key for key in _RATE_KEYS if key in metrics), None)
            self.latest = {
                "stage": self.stage, "step": step, "total_steps": int(match.group(2)),
                "metrics": {key: value for key, value in metrics.items() if key not in _RATE_KEYS},
                "throughput": metrics.get(rate) if rate else None, "throughput_unit": rate or "",
            }
            loss = metrics.get("total")
            if loss is None:
                return
            self.best = loss if self.best is None else min(self.best, loss)
            if self.points and self.points[-1][0] == self.stage and self.points[-1][1] == step:
                return
            self.count += 1
            if self.count % self.stride == 0:
                self.points.append([self.stage, step, loss])
                if len(self.points) > self.limit:
                    self.points = self.points[::2]
                    self.stride *= 2
            return
        match = _VAL_LINE.match(line)
        if match:
            metrics = _parse_metrics(match.group(2))
            if "total" in metrics:
                step = int(match.group(1)) if match.group(1) else self.last_step
                self.validation.append([self.stage, step, metrics["total"]])
                del self.validation[:-400]

    def snapshot(self) -> dict:
        return {
            "points": list(self.points), "validation": list(self.validation),
            "latest": dict(self.latest), "best": self.best,
        }


class TrainingManager:
    """Runs one training or fine-tuning subprocess at a time."""

    _INTERNAL = frozenset({
        "process", "logs", "cancel_requested", "pause_requested", "pause_file",
        "process_stage", "generator_start_step", "tokenizer_start_step", "history",
        "log_count",
    })

    def __init__(self, root: Path, registry=None):
        self.root = Path(root).resolve()
        self.condition = threading.Condition()
        self.job = None
        self.version = 0
        self.corpus_lock = threading.Lock()
        self.corpus_cache = {}
        self.registry = registry
        self.before_start: list = []  # callables run before a subprocess claims the GPU

    # -- corpus & smart options ------------------------------------------
    def _corpus(self, folder, mode):
        with self.corpus_lock:
            current = inspect_folder(folder)
            key = (current["path"], mode)
            cached = self.corpus_cache.get(key)
            if (cached and cached.get("signature") == current.get("signature")
                    and Path(cached["output"]).exists()):
                return cached
            prepared = prepare_corpus(current["path"], self.root / "trained-model", mode)
            self.corpus_cache[key] = prepared
            return prepared

    def smart_options(self, config):
        technique = config.get("technique", "fast")
        corpus = self._corpus(config.get("folder", ""), config.get("corpus_mode", "smart"))
        result = recommend_training_options(
            corpus, technique, config.get("device", "auto"), config
        )
        result["corpus"] = corpus
        return result

    # -- state -----------------------------------------------------------
    def _changed_locked(self) -> None:
        self.version += 1
        self.condition.notify_all()

    def _notify_registry(self) -> None:
        if self.registry is not None:
            self.registry.touch()

    def _state_locked(self):
        if self.job is None:
            return None
        result = {k: v for k, v in self.job.items() if k not in self._INTERNAL}
        result["logs"] = self.job["logs"][-LOG_TAIL:]
        result["log_count"] = self.job["log_count"]
        result["metrics"] = self.job["history"].snapshot()
        result["active"] = self.job["status"] in {"preparing", "running"}
        result["cancel_requested"] = self.job["cancel_requested"]
        result["pause_requested"] = self.job["pause_requested"]
        return result

    def state(self):
        with self.condition:
            return self._state_locked()

    @property
    def active(self) -> bool:
        with self.condition:
            return bool(self.job and self.job["status"] in {"preparing", "running"})

    def _append(self, line, job=None):
        """Record one output line on ``job`` (the current job by default).

        Workers pass their own job: a worker that outlived a cancel must never
        write into the run that replaced it.
        """
        with self.condition:
            job = self.job if job is None else job
            if job is None or job is not self.job:
                return
            logs = job["logs"]
            logs.append(line.rstrip())
            job["log_count"] += 1
            if len(logs) > MAX_LOG_LINES:
                del logs[:len(logs) - MAX_LOG_LINES]
            job["history"].feed(line)
            if "Tokenizer parameters:" in line:
                job["stage"] = "Tokenizer (stage 1/2)"
                job["process_stage"] = "tokenizer"
            elif "Generator trainable parameters:" in line:
                job["stage"] = "Next-concept generator (stage 2/2)"
                job["process_stage"] = "generator"
            match = re.search(r"step\s+(\d+)/(\d+)", line)
            if match:
                stage = job.get("process_stage", "generator")
                start = job.get(f"{stage}_start_step", 0)
                completed = max(0, int(match.group(1)) - start)
                offset = job["tokenizer_steps"] if stage == "generator" else 0
                job["current_step"] = offset + completed
            self._changed_locked()
        self._notify_registry()

    def _set_steps(self, config, job=None):
        technique = config.get("technique", "fast")
        mode = config.get("mode", "fresh")
        tokenizer_steps = (
            number(config, "tokenizer_steps", 1000, 1, 10_000_000)
            if technique == "ncm" and mode in NCM_TOKENIZER_MODES else 0
        )
        steps = 0 if technique == "ncm" and mode in NCM_TOKENIZER_ONLY_MODES else number(
            config, "steps", 1000, 1, 10_000_000
        )
        with self.condition:
            job = self.job if job is None else job
            job["tokenizer_steps"] = tokenizer_steps
            job["total_steps"] = tokenizer_steps + steps

    def _backup(self, checkpoint, tokenizer, technique, mode, enabled):
        if not enabled:
            return None
        if technique != "ncm" or mode in {"continue", "fresh_generator"}:
            paths = [checkpoint]
        elif mode in NCM_TOKENIZER_ONLY_MODES:
            paths = [tokenizer]
        else:
            paths = [checkpoint, tokenizer]
        existing = [path for path in paths if path.exists()]
        if not existing:
            return None
        destination = self.root / "backups" / checkpoint.stem / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        destination.mkdir(parents=True, exist_ok=False)
        for path in existing:
            shutil.copy2(path, destination / path.name)
        return str(destination.relative_to(self.root))

    def _new_job(self, **fields) -> dict:
        job = {
            "id": uuid.uuid4().hex[:12], "status": "preparing", "technique": "fast",
            "mode": "fresh", "finetune": False, "base": None, "title": "Training",
            "created": time.time(), "started": None, "ended": None,
            "stage": "Preparing corpus", "logs": [], "log_count": 0, "process": None,
            "history": MetricsHistory(),
            "cancel_requested": False, "pause_requested": False, "pause_file": None,
            "current_step": 0, "total_steps": 0, "tokenizer_steps": 0,
            "process_stage": "generator", "generator_start_step": 0,
            "tokenizer_start_step": 0, "checkpoint": None, "checkpoint_ready": False,
            "backup": None, "command": None, "corpus": None, "returncode": None,
            "error": None,
        }
        job.update(fields)
        return job

    def _require_idle_locked(self) -> None:
        if self.job and self.job["status"] in {"preparing", "running"}:
            raise ConflictError("A training run is already active")

    def _release_gpu(self) -> None:
        for callback in self.before_start:
            try:
                callback()
            except Exception:  # noqa: BLE001 - freeing memory is best effort
                pass

    # -- Mariana DRV-LM / NCM runs ---------------------------------------
    def start(self, config):
        config = dict(config)
        if config.get("engine") == "hf":
            return self.start_external(config)
        with self.condition:
            self._require_idle_locked()
            technique = config.get("technique", "fast")
            mode = config.get("mode", "fresh")
            if technique not in TECHNIQUES:
                raise ValueError("Unknown training technique")
            if technique == "ncm" and mode not in NCM_MODES:
                raise ValueError(f"Unknown NCM training mode: {mode}")
            if technique != "ncm" and mode not in DRV_MODES:
                raise ValueError("DRV training supports fresh, continue, and finetune modes")
            name = _checkpoint_name(config.get("name"), technique)
            checkpoint = self.root / "trained-model" / name
            tokenizer = checkpoint.with_name(checkpoint.stem + "-tokenizer.pt")
            if technique != "ncm" and mode == "fresh" and checkpoint.exists():
                raise ValueError(
                    "A checkpoint with this name already exists; choose continue or a new name"
                )
            base_tokenizer = None
            if mode == "finetune":
                base_value = config.get("base")
                if not base_value:
                    raise ValueError("Choose a base checkpoint to fine-tune")
                base = Path(base_value)
                if not base.is_absolute():
                    base = self.root / base
                base = base.resolve()
                if not base.is_file():
                    raise ValueError("Base checkpoint does not exist")
                try:
                    base.relative_to(self.root)
                except ValueError:
                    raise ValueError("Base checkpoint must be inside the workspace") from None
                if base == checkpoint:
                    raise ValueError("Choose a new output name so the base checkpoint is preserved")
                base_state = torch.load(base, map_location="cpu", weights_only=True)
                stage = base_state.get("stage")
                if technique == "ncm":
                    if stage != "generator":
                        raise ValueError("NCM fine-tuning requires an NCM generator checkpoint")
                    base_tokenizer = base.with_name(base.stem + "-tokenizer.pt")
                    if not base_tokenizer.exists():
                        raise ValueError("The base NCM checkpoint is missing its tokenizer file")
                elif stage not in {None, "drv"}:
                    raise ValueError("DRV fine-tuning requires a DRV-LM checkpoint")
                config["init_from"] = str(base)
            generator_start = _checkpoint_step(
                checkpoint, "generator" if technique == "ncm" else "drv"
            ) if mode == "continue" else 0
            tokenizer_start = _checkpoint_step(
                tokenizer, "tokenizer"
            ) if technique == "ncm" and mode in {"update", "update_tokenizer"} else 0
            if technique == "ncm" and mode == "fresh_generator":
                _checkpoint_step(tokenizer, "tokenizer")
            folder = Path(config.get("folder", "")).expanduser()
            if not str(config.get("folder", "")).strip() or not folder.exists():
                raise ValueError("Select a valid corpus folder")
            if config.get("corpus_mode", "smart") not in {"smart", "raw"}:
                raise ValueError("Unknown corpus extraction mode")
            pause_file = checkpoint.with_name(checkpoint.name + ".pause")
            pause_file.parent.mkdir(parents=True, exist_ok=True)
            pause_file.unlink(missing_ok=True)
            backup = self._backup(
                checkpoint, tokenizer, technique, mode, config.get("backup", True)
            )
            if base_tokenizer is not None and base_tokenizer != tokenizer:
                tokenizer.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(base_tokenizer, tokenizer)
            config.update({
                "mode": mode, "pause_file": str(pause_file),
                "_generator_start_step": generator_start,
                "_tokenizer_start_step": tokenizer_start,
            })
            finetune = mode == "finetune"
            title = (
                f"Fine-tune · {Path(config['init_from']).name}" if finetune
                else f"{TECHNIQUES[technique]['name']} · {mode.replace('_', ' ')}"
            )
            job = self.job = self._new_job(
                technique=technique, mode=mode, finetune=finetune,
                base=config.get("init_from"), title=title, pause_file=str(pause_file),
                process_stage=(
                    "tokenizer" if technique == "ncm" and mode in NCM_TOKENIZER_MODES
                    else "generator"
                ),
                generator_start_step=generator_start, tokenizer_start_step=tokenizer_start,
                checkpoint=str(checkpoint.relative_to(self.root)), backup=backup,
            )
            self._set_steps(config, job)
            thread = threading.Thread(
                target=self._run, args=(job, config, folder, checkpoint),
                name="training-run", daemon=True,
            )
            thread.start()
            self._changed_locked()
            state = self._state_locked()
        self._notify_registry()
        return state

    def _run(self, job, config, folder, checkpoint):
        pause_file = Path(config["pause_file"])
        try:
            if job.get("backup"):
                self._append(f"Backup: {job['backup']}", job)
            self._append(f"Scanning {folder}", job)
            corpus = self._corpus(folder, config.get("corpus_mode", "smart"))
            with self.condition:
                job["corpus"] = corpus
            self._append(
                f"Prepared {corpus['files']} text files ({corpus['output_bytes'] / 1e6:.2f} MB); "
                f"skipped {corpus['skipped']} binary/unreadable files", job,
            )
            if config.get("smart"):
                smart = recommend_training_options(
                    corpus, config.get("technique", "fast"), config.get("device", "auto"), config
                )
                config.update(smart["config"])
                with self.condition:
                    job["smart_analysis"] = smart["analysis"]
                self._set_steps(config, job)
                self._append(
                    f"Smart options: {corpus['output_bytes']:,} byte-tokens | ctx {config['ctx']} | "
                    f"dim {config['dim']} | batch {config['batch']} x {config['grad_accum']} | "
                    f"steps {config['steps']}", job,
                )
            commands = build_training_commands(config, corpus["output"], checkpoint, self.root)
            returncode = self._execute(job, commands)
            tokenizer_only = job["technique"] == "ncm" and job["mode"] in NCM_TOKENIZER_ONLY_MODES
            self._settle(job, returncode, pause_file, not tokenizer_only and checkpoint.exists())
        except Exception as error:  # noqa: BLE001 - reported on the job
            self._fail(job, error, pause_file)

    def _execute(self, job, commands, env: dict | None = None) -> int:
        with self.condition:
            job["command"] = " && ".join(shlex.join(command) for command in commands)
        self._release_gpu()
        returncode = 0
        for command in commands:
            with self.condition:
                if job is not self.job or job["cancel_requested"] or job["pause_requested"]:
                    break
                job["status"] = "running"
                if job["started"] is None:
                    job["started"] = time.time()
                job["stage"] = "Training"
            self._append("$ " + shlex.join(command), job)
            process = subprocess.Popen(
                command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", **(env or {})},
            )
            with self.condition:
                job["process"] = process
                if job["cancel_requested"] and process.poll() is None:
                    process.terminate()
            for line in process.stdout:
                self._append(line, job)
            returncode = process.wait()
            with self.condition:
                job["process"] = None
            if returncode != 0:
                break
        return returncode

    def _settle(self, job, returncode: int, pause_file: Path, ready: bool) -> None:
        with self.condition:
            if job is not self.job:
                return  # a newer run owns the manager (and possibly this pause file)
            job["returncode"] = returncode
            job["ended"] = time.time()
            if job["cancel_requested"]:
                job["status"] = "cancelled"
                job["stage"] = "Cancelled"
            elif job["pause_requested"]:
                job["status"] = "paused"
                job["stage"] = "Paused at checkpoint"
            elif returncode == 0:
                job["status"] = "completed"
                job["stage"] = "Complete"
            else:
                job["status"] = "failed"
                job["stage"] = "Failed"
                job["error"] = _last_error(job["logs"], returncode)
            job["checkpoint_ready"] = bool(ready)
            pause_file.unlink(missing_ok=True)
            self._changed_locked()
        self._notify_registry()

    def _fail(self, job, error: Exception, pause_file: Path) -> None:
        self._append(f"ERROR: {error}", job)
        with self.condition:
            if job is not self.job:
                return
            if job["status"] != "cancelled":
                job["status"] = "failed"
                job["stage"] = "Failed"
                job["error"] = str(error)
            job["ended"] = job["ended"] or time.time()
            pause_file.unlink(missing_ok=True)
            self._changed_locked()
        self._notify_registry()

    # -- external (Hugging Face) fine-tunes ------------------------------
    def start_external(self, plan: dict) -> dict:
        """Launch a planned fine-tune of a third-party model (see ``finetune.hf``)."""
        plan = dict(plan)
        output = Path(plan["output"]).resolve()
        models = (self.root / "models").resolve()
        try:
            output.relative_to(models)
        except ValueError:
            raise ValueError("Fine-tuned models are written under models/") from None
        folder = Path(str(plan.get("folder", ""))).expanduser()
        if not str(plan.get("folder", "")).strip() or not folder.exists():
            raise ValueError("Select a valid fine-tuning data folder")
        with self.condition:
            self._require_idle_locked()
            job_id = uuid.uuid4().hex[:12]
            pause_file = jobs_path(self.root) / f"{job_id}.pause"
            pause_file.unlink(missing_ok=True)
            backup = None
            if output.exists():
                if not plan.get("backup", True):
                    raise ValueError("Output model exists; choose a new name or enable backups")
                destination = (
                    self.root / "backups" / output.name
                    / f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
                )
                destination.mkdir(parents=True, exist_ok=False)
                shutil.move(str(output), str(destination / output.name))
                backup = str(destination.relative_to(self.root))
            job = self.job = self._new_job(
                id=job_id, technique="hf", mode="finetune", finetune=True,
                base=plan.get("base_name") or plan.get("base"),
                title=f"Fine-tune · {plan.get('base_name') or Path(str(plan.get('base'))).name}",
                pause_file=str(pause_file), checkpoint=str(output.relative_to(self.root)),
                backup=backup, total_steps=int(plan.get("steps", 0)),
                stage="Preparing data",
            )
            thread = threading.Thread(
                target=self._run_external, args=(job, plan, folder, output, pause_file),
                name="finetune-run", daemon=True,
            )
            thread.start()
            self._changed_locked()
            state = self._state_locked()
        self._notify_registry()
        return state

    def _run_external(self, job, plan, folder, output, pause_file):
        try:
            if job.get("backup"):
                self._append(f"Backup: {job['backup']}", job)
            data = {"corpus": None, "chat_files": []}
            if plan.get("data_mode") == "chat":
                data["chat_files"] = [str(path) for path in plan.get("chat_files", [])]
                self._append(
                    f"Using {len(data['chat_files'])} chat-format file(s) from {folder}", job,
                )
            else:
                self._append(f"Scanning {folder}", job)
                corpus = self._corpus(folder, plan.get("corpus_mode", "smart"))
                with self.condition:
                    job["corpus"] = corpus
                self._append(
                    f"Prepared {corpus['files']} text files ({corpus['output_bytes'] / 1e6:.2f} MB); "
                    f"skipped {corpus['skipped']} binary/unreadable files", job,
                )
                data["corpus"] = corpus["output"]
            job_file = jobs_path(self.root) / f"{job['id']}.json"
            write_json(job_file, {
                **plan.get("trainer", {}), "base": plan["base"], "output": str(output),
                "data": data, "pause_file": str(pause_file),
            })
            command = [
                sys.executable, "-m", "mariana.studio.finetune.trainer",
                "--config", str(job_file),
            ]
            inherited = os.environ.get("PYTHONPATH")
            python_path = str(PACKAGE_PARENT) + (os.pathsep + inherited if inherited else "")
            returncode = self._execute(job, [command], env={"PYTHONPATH": python_path})
            ready = (output / "config.json").exists() or (
                output / "adapter" / "adapter_config.json"
            ).exists()
            self._settle(job, returncode, pause_file, ready)
        except Exception as error:  # noqa: BLE001 - reported on the job
            self._fail(job, error, pause_file)

    # -- control ---------------------------------------------------------
    def pause(self, job_id):
        with self.condition:
            if not self.job or self.job["id"] != job_id:
                raise NotFoundError("Training job not found")
            if self.job["status"] not in {"preparing", "running"}:
                raise ConflictError("Training job is not active")
            if not self.job["pause_requested"]:
                self.job["pause_requested"] = True
                Path(self.job["pause_file"]).write_text("pause\n", encoding="utf-8")
                process = self.job["process"]
                if process is None:
                    self.job["stage"] = "Pause requested before training"
                else:
                    self.job["stage"] = "Saving checkpoint before pause"
                self._changed_locked()
            state = self._state_locked()
        self._notify_registry()
        return state

    def cancel(self, job_id):
        with self.condition:
            if not self.job or self.job["id"] != job_id:
                raise NotFoundError("Training job not found")
            if self.job["status"] not in {"preparing", "running"}:
                raise ConflictError("Training job is not active")
            process = self.job["process"]
            self.job["cancel_requested"] = True
            if process is None:
                self.job["status"] = "cancelled"
                self.job["stage"] = "Cancelled"
                self.job["ended"] = time.time()
            elif process.poll() is None:
                process.terminate()
                threading.Timer(5.0, lambda: process.poll() is None and process.kill()).start()
            self._changed_locked()
            state = self._state_locked()
        self._notify_registry()
        return state

    def wait(self, version):
        with self.condition:
            self.condition.wait_for(lambda: self.version != version, timeout=15)
            return self.version, self._state_locked()

    def close(self) -> None:
        with self.condition:
            job = self.job
            if job and job["status"] in {"preparing", "running"}:
                job["cancel_requested"] = True
                process = job["process"]
                if process is not None and process.poll() is None:
                    process.terminate()
            self._changed_locked()

    def checkpoints(self):
        """Every Mariana ``.pt`` checkpoint in the workspace, newest first."""
        folder = self.root / "trained-model"
        paths = set(folder.glob("**/*.pt")) if folder.exists() else set()
        paths.update(self.root.glob("*.pt"))
        result = []
        for path in paths:
            if path.suffix != ".pt" or path.name.endswith(".tmp"):
                continue
            stat = path.stat()
            result.append({
                "path": str(path.relative_to(self.root)), "name": path.name,
                "bytes": stat.st_size, "modified": stat.st_mtime,
            })
        return sorted(result, key=lambda item: -item["modified"])

    # -- job provider (unified job list) ---------------------------------
    def _as_job(self, state: dict, logs: bool = False, since: int = 0) -> dict:
        total = float(state.get("total_steps") or 0)
        current = float(state.get("current_step") or 0)
        end = state.get("ended") or time.time()
        percent = max(0.0, min(100.0, 100.0 * current / total)) if total else 0.0
        if state["status"] == "completed":
            percent = 100.0
        result = {}
        if state.get("checkpoint_ready"):
            result = {"checkpoint": state["checkpoint"], "model_id": state["checkpoint"]}
        job = {
            "id": state["id"], "kind": "finetune" if state.get("finetune") else "train",
            "title": state.get("title") or "Training",
            "status": state["status"], "stage": state["stage"], "active": state["active"],
            "progress": {"current": current, "total": total, "unit": "steps",
                         "percent": round(percent, 2)},
            "created": state["created"], "started": state.get("started"),
            "ended": state.get("ended"),
            "elapsed": max(0.0, end - (state.get("started") or state["created"])),
            "error": state.get("error"), "result": result,
            "meta": {
                "technique": state.get("technique"), "mode": state.get("mode"),
                "base": state.get("base"), "checkpoint": state.get("checkpoint"),
                "backup": state.get("backup"), "view": "finetune" if state.get("finetune") else "train",
            },
            "can_pause": True,
            "cancel_requested": state["cancel_requested"],
            "pause_requested": state["pause_requested"],
            "log_count": state.get("log_count", len(state["logs"])),
        }
        if logs:
            with self.condition:
                kept = list(self.job["logs"]) if self.job and self.job["id"] == state["id"] else []
                count = self.job["log_count"] if kept else 0
            first = count - len(kept)
            job["logs"] = kept[max(0, since - first):]
            job["log_start"] = max(first, since)
            job["metrics"] = state.get("metrics")
        return job

    def jobs(self) -> list[dict]:
        state = self.state()
        return [self._as_job(state)] if state else []

    def job_snapshot(self, job_id: str, since: int = 0) -> dict | None:
        state = self.state()
        if not state or state["id"] != job_id:
            return None
        return self._as_job(state, logs=True, since=since)


class TrainingJobs:
    """Adapter that exposes the training manager to :class:`JobRegistry`."""

    def __init__(self, manager: TrainingManager):
        self.manager = manager

    def jobs(self) -> list[dict]:
        return self.manager.jobs()

    def job(self, job_id: str, since: int = 0) -> dict | None:
        return self.manager.job_snapshot(job_id, since)

    def cancel(self, job_id: str) -> dict:
        self.manager.cancel(job_id)
        return self.manager.job_snapshot(job_id)

    def pause(self, job_id: str) -> dict:
        self.manager.pause(job_id)
        return self.manager.job_snapshot(job_id)


def _last_error(logs: list[str], returncode: int) -> str:
    """Pick the most useful line of a failed run for the job summary."""
    for line in reversed(logs[-40:]):
        text = line.strip()
        if text.startswith("ERROR:"):
            return text[6:].strip()
    for line in reversed(logs[-40:]):
        text = line.strip()
        if re.match(r"^[A-Za-z_.]*(Error|Exception|SystemExit)\b", text) or "out of memory" in text.lower():
            return text[:400]
    for line in reversed(logs[-10:]):
        if line.strip() and not line.startswith("$ "):
            return line.strip()[:400]
    return f"The training process exited with code {returncode}"
