"""Training: command builders, the single-slot manager, and real runs over HTTP."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from mariana.cli import parse_args
from mariana.studio.jobs import JobRegistry
from mariana.studio.training import (
    MetricsHistory,
    TrainingJobs,
    TrainingManager,
    _checkpoint_step,
    build_training_command,
    build_training_commands,
    launcher,
)
from tests.helpers import ROOT, make_corpus, request, request_error, studio, wait_for_job

TINY_DRV = {
    "corpus_mode": "smart", "technique": "fast", "batch": 2, "grad_accum": 1, "ctx": 16,
    "dim": 32, "heads": 4, "device": "cpu", "eval_every": 1, "warmup": 1, "n_pre": 1,
    "n_core": 1, "n_post": 1, "loops": 1, "block_size": 8, "backup": False,
}


def test_studio_cli_and_ncm_command():
    args = parse_args(["--gui", "--port", "0", "--no-browser"])
    assert args.cmd == "gui" and args.port == 0 and args.no_browser
    assert parse_args(["train", "--gui"]).cmd == "gui"
    studio_args = parse_args(["studio", "--port", "0", "--no-browser"])
    assert studio_args.cmd == "studio" and studio_args.port == 0 and studio_args.no_browser
    command = build_training_command(
        {"technique": "ncm", "steps": 4, "tokenizer_steps": 3, "ctx": 32,
         "tokenizer_batch": 24, "tokenizer_grad_accum": 2},
        "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--ncm" in command
    assert command[command.index("--ncm-tokenizer-steps") + 1] == "3"
    assert command[command.index("--ncm-tokenizer-batch") + 1] == "24"
    assert command[command.index("--ncm-tokenizer-grad-accum") + 1] == "2"
    assert command[command.index("--steps") + 1] == "4"
    compiled = build_training_command(
        {"technique": "fast", "compile": True}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--compile" in compiled


def test_axiom_training_command_options():
    command = build_training_command(
        {"technique": "fast", "axiom": True, "axiom_profile": "reversible"},
        "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--axiom" in command
    assert command[command.index("--axiom-profile") + 1] == "reversible"
    assert "--axiom" not in build_training_command(
        {"technique": "fast"}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT
    )
    with pytest.raises(ValueError, match="DRV-LM"):
        build_training_command(
            {"technique": "ncm", "axiom": True},
            "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )
    with pytest.raises(ValueError, match="profile"):
        build_training_command(
            {"technique": "fast", "axiom": True, "axiom_profile": "extreme"},
            "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )


def test_moe_and_latent_command_options():
    config = {
        "technique": "fast", "moe": True, "latent_mode": "saint",
        "thoughts": 6, "latent_symbols": 64, "latent_window": 12,
        "latent_min_thoughts": 2, "latent_block_size": 3,
        "latent_continuous_slots": 2, "latent_refine_steps": 1,
        "latent_grammar_prob": 0.4, "w_latent_rl": 0.2,
    }
    command = build_training_command(config, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT)
    for flag in ("--moe", "--latent"):
        assert flag in command
    assert "--latent-legacy" not in command
    assert "--no-grouped-moe" not in command
    fallback = build_training_command(
        {**config, "grouped_moe": False}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--no-grouped-moe" in fallback
    expected = {
        "--thoughts": "6", "--latent-symbols": "64", "--latent-window": "12",
        "--latent-min-thoughts": "2", "--latent-block-size": "3",
        "--latent-continuous-slots": "2", "--latent-refine-steps": "1",
        "--latent-grammar-prob": "0.4", "--w-latent-rl": "0.2",
    }
    for flag, value in expected.items():
        assert command[command.index(flag) + 1] == value

    legacy = build_training_command(
        {**config, "latent_mode": "legacy"}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--latent-legacy" in legacy
    assert "--latent-min-thoughts" not in legacy

    ncm = build_training_command(
        {**config, "technique": "ncm"}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--moe" not in ncm and "--latent" not in ncm


def test_training_mode_command_matrix():
    base = {
        "technique": "ncm", "steps": 5, "tokenizer_steps": 3, "ctx": 32,
        "prefix": 16, "levels": "1,2,4,8,16,32", "device": "cpu",
        "pause_file": "/tmp/model.pt.pause",
    }
    continued = build_training_commands(
        {**base, "mode": "continue", "_generator_start_step": 11},
        "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert len(continued) == 1
    assert "--resume" in continued[0]
    assert continued[0][continued[0].index("--steps") + 1] == "16"

    updated = build_training_commands(
        {**base, "mode": "update", "_tokenizer_start_step": 7},
        "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert len(updated) == 2
    assert updated[0][3] == "tokenizer" and updated[1][3] == "generator"
    assert "--resume" in updated[0] and "--resume" not in updated[1]
    assert updated[0][updated[0].index("--steps") + 1] == "10"
    assert updated[1][updated[1].index("--steps") + 1] == "5"
    assert all("--pause-file" in command for command in updated)

    fresh_generator = build_training_commands(
        {**base, "mode": "fresh_generator"}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert len(fresh_generator) == 1 and fresh_generator[0][3] == "generator"
    assert "--resume" not in fresh_generator[0]

    for mode, resume in (("update_tokenizer", True), ("fresh_tokenizer", False)):
        commands = build_training_commands(
            {**base, "mode": mode, "_tokenizer_start_step": 7},
            "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )
        assert len(commands) == 1 and commands[0][3] == "tokenizer"
        assert ("--resume" in commands[0]) is resume

    drv = build_training_commands(
        {"technique": "fast", "mode": "continue", "steps": 5, "_generator_start_step": 9},
        "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--resume" in drv[0] and drv[0][drv[0].index("--steps") + 1] == "14"


def test_ncm_performance_options_cover_every_training_mode():
    base = {
        "technique": "ncm", "compile": True, "flex": True, "cache_codes": True,
        "steps": 2, "tokenizer_steps": 2,
    }
    for mode in ("fresh", "continue"):
        command = build_training_commands(
            {**base, "mode": mode}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )[0]
        assert "--compile" in command and "--flex" in command
        assert "--ncm-cache-codes" in command

    updated = build_training_commands(
        {**base, "mode": "update"}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert len(updated) == 2
    assert all(flag not in updated[0] for flag in ("--compile", "--flex", "--cache-codes"))
    assert all(flag in updated[1] for flag in ("--compile", "--flex", "--cache-codes"))

    generator = build_training_commands(
        {**base, "mode": "fresh_generator"}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )[0]
    assert all(flag in generator for flag in ("--compile", "--flex", "--cache-codes"))
    for mode in ("update_tokenizer", "fresh_tokenizer"):
        tokenizer = build_training_commands(
            {**base, "mode": mode}, "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )[0]
        assert all(flag not in tokenizer for flag in ("--compile", "--flex", "--cache-codes"))

    drv = build_training_command(
        {"technique": "fast", "compile": True, "flex": True, "cache_codes": True},
        "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
    )
    assert "--compile" in drv and "--flex" not in drv
    assert "--cache-codes" not in drv and "--ncm-cache-codes" not in drv

def test_launcher_prefers_the_workspace_script(tmp_path):
    assert launcher(tmp_path) == str(ROOT / "mariana.py")
    (tmp_path / "mariana.py").write_text("")
    assert launcher(tmp_path) == str(tmp_path / "mariana.py")


def test_metrics_history_tracks_stages_and_thins_long_runs():
    history = MetricsHistory(limit=8)
    for line in (
        "Tokenizer parameters: 1.00M",
        "step 1/3 | total 4.500 | lr 0.01 | pos/s 1200",
        "VAL total 3.900 | recon 1.000",
        "Generator trainable parameters: 2.00M",
        "step     1/2 | ar 1.000 | total 2.750 | lr 1.00e-03 | tok/s 900",
        "step     1/2 | ar 1.000 | total 2.750 | lr 1.00e-03 | tok/s 900",  # a repeat is ignored
        "VAL step 1: ar 1.000 | total 2.500",
        "not a metric line | total nan",
    ):
        history.feed(line)
    snapshot = history.snapshot()
    assert snapshot["points"] == [["tokenizer", 1, 4.5], ["generator", 1, 2.75]]
    assert snapshot["validation"] == [["tokenizer", 1, 3.9], ["generator", 1, 2.5]]
    assert snapshot["best"] == 2.75
    assert snapshot["latest"]["metrics"] == {"ar": 1.0, "total": 2.75, "lr": 0.001}
    assert snapshot["latest"]["throughput"] == 900 and snapshot["latest"]["throughput_unit"] == "tok/s"
    for step in range(2, 400):
        history.feed(f"step {step}/400 | total {1 / step:.6f} | tok/s 10")
    assert len(history.points) <= 8 and history.points[-1][1] > 300  # bounded, still recent
    assert history.best < 0.003


def test_a_cancelled_worker_cannot_touch_the_run_that_replaced_it(tmp_path):
    folder = make_corpus(tmp_path / "input")
    manager = TrainingManager(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def slow_corpus(folder, mode):
        entered.set()
        release.wait(10)
        raise ValueError("stale corpus failure")

    config = {**TINY_DRV, "folder": str(folder), "steps": 1}
    with patch.object(manager, "_corpus", side_effect=slow_corpus):
        first = manager.start({**config, "name": "first.pt"})
        assert entered.wait(5)
        assert manager.cancel(first["id"])["status"] == "cancelled"
        with patch.object(TrainingManager, "_run", return_value=None):
            second = manager.start({**config, "name": "second.pt"})
        release.set()
        for thread in threading.enumerate():
            if thread.name == "training-run":
                thread.join(timeout=5)
    state = manager.state()
    assert state["id"] == second["id"] and state["status"] == "preparing"
    assert not any("stale" in line for line in state["logs"])


def test_training_runs_appear_in_the_job_registry(tmp_path):
    folder = make_corpus(tmp_path / "input")
    registry = JobRegistry()
    manager = TrainingManager(tmp_path, registry=registry)
    registry.add_provider(TrainingJobs(manager))
    with patch.object(TrainingManager, "_run", return_value=None):
        started = manager.start({**TINY_DRV, "folder": str(folder), "name": "listed.pt", "steps": 7})
    (job,) = registry.list()
    assert job["id"] == started["id"] and job["kind"] == "train" and job["active"]
    assert job["title"] == "Sprint-DRV · fresh" and job["can_pause"] is True
    assert job["progress"] == {"current": 0.0, "total": 7.0, "unit": "steps", "percent": 0.0}
    assert registry.get(job["id"])["logs"] == []
    assert registry.cancel(job["id"])["status"] == "cancelled"
    assert registry.active() == []
    with pytest.raises(ValueError, match="not active"):
        manager.pause(job["id"])


def test_external_finetunes_stay_inside_the_models_folder(tmp_path):
    manager = TrainingManager(tmp_path)
    with pytest.raises(ValueError, match="under models/"):
        manager.start_external({"output": str(tmp_path / "elsewhere"), "folder": str(tmp_path)})
    with pytest.raises(ValueError, match="valid fine-tuning data"):
        manager.start_external({"output": str(tmp_path / "models" / "tuned"), "folder": ""})
    existing = tmp_path / "models" / "tuned"
    existing.mkdir(parents=True)
    (existing / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="enable backups"):
        manager.start_external({"output": str(existing), "folder": str(tmp_path), "backup": False})


def test_drv_continue_rejects_ncm_checkpoint(tmp_path):
    (tmp_path / "trained-model").mkdir()
    (tmp_path / "trained-model" / "ncm.pt").touch()
    with (
        patch("mariana.studio.training.torch.load", return_value={
            "stage": "generator", "step": 3, "cfg": {}, "model": {},
        }),
        patch.object(TrainingManager, "_run", return_value=None),
    ):
        with pytest.raises(ValueError, match="DRV-LM checkpoint"):
            TrainingManager(tmp_path).start({
                "technique": "fast", "mode": "continue", "name": "ncm.pt",
                "folder": str(tmp_path), "steps": 1,
            })


def test_invalid_model_geometry_is_rejected_at_the_boundary():
    with pytest.raises(ValueError, match="divisible by attention heads"):
        build_training_command(
            {"technique": "fast", "dim": 30, "heads": 4},
            "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )
    with pytest.raises(ValueError, match="even head size"):
        build_training_command(
            {"technique": "fast", "dim": 12, "heads": 4},
            "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )
    with pytest.raises(ValueError, match="latent width"):
        build_training_command(
            {"technique": "ncm", "dim": 512, "heads": 128, "steps": 1,
             "tokenizer_steps": 1},
            "/tmp/corpus.txt", Path("/tmp/model.pt"), ROOT,
        )


def test_corrupt_checkpoint_has_an_actionable_error(tmp_path):
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="could not be read as a checkpoint"):
        _checkpoint_step(broken)


def test_pause_request_does_not_mask_a_failed_run(tmp_path):
    manager = TrainingManager(tmp_path)
    job = manager._new_job(technique="fast", mode="fresh", status="running")
    manager.job = job
    job["pause_requested"] = True
    pause = tmp_path / "model.pt.pause"
    pause.write_text("pause\n", encoding="utf-8")
    manager._settle(job, 1, pause, False)
    assert job["status"] == "failed" and job["error"]
    assert not pause.exists()


def test_state_polling_stays_responsive_while_a_start_copies_a_backup(tmp_path):
    folder = make_corpus(tmp_path / "input")
    manager = TrainingManager(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def slow_backup(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return None

    started = {}

    def start():
        started["state"] = manager.start({
            **TINY_DRV, "folder": str(folder), "name": "slow.pt",
            "steps": 1, "backup": True,
        })

    with patch.object(TrainingManager, "_backup", side_effect=slow_backup), \
            patch.object(TrainingManager, "_run", return_value=None):
        worker = threading.Thread(target=start, daemon=True)
        worker.start()
        assert entered.wait(5)
        observed = {}
        probe = threading.Thread(
            target=lambda: observed.setdefault("state", manager.state()), daemon=True
        )
        probe.start()
        probe.join(timeout=2)
        responsive = "state" in observed
        release.set()
        worker.join(timeout=5)
    assert responsive, "state polling blocked behind backup disk I/O"
    assert started["state"]["status"] == "preparing"


def test_parquet_inspect_then_smart_does_not_crash(tmp_path):
    pytest.importorskip("pyarrow")
    folder = tmp_path / "input"
    folder.mkdir()
    subprocess.run([
        sys.executable, "-c",
        "import pyarrow as pa, pyarrow.parquet as pq, sys; "
        "pq.write_table(pa.table({'text': ['hello parquet']}), sys.argv[1])",
        str(folder / "table.parquet"),
    ], check=True)
    script = r"""
import sys
from pathlib import Path
from tests.helpers import request, studio
with studio(Path(sys.argv[2])) as (server, base):
    request(base + '/api/corpus/inspect', {'path': sys.argv[1]})
    result = request(base + '/api/corpus/smart', {
        'folder': sys.argv[1], 'corpus_mode': 'smart', 'technique': 'fast', 'device': 'cpu',
    })
    assert result['corpus']['output_bytes'] > 0
    print('inspect-then-smart OK')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(folder), str(tmp_path / "workspace")],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "inspect-then-smart OK" in completed.stdout


@pytest.mark.slow
def test_http_training_and_chat_workflow(tmp_path):
    folder = make_corpus(tmp_path / "input")
    with studio(tmp_path / "workspace") as (server, base):
        state = request(base + "/api/state")
        assert set(state["techniques"]) == {"fast", "legacy", "ncm"}
        assert state["name"] == "Mariana Studio" and state["job"] is None
        smart = request(base + "/api/corpus/smart", {
            "folder": str(folder), "corpus_mode": "smart", "technique": "fast", "device": "cpu",
        })
        assert smart["analysis"]["byte_tokens"] > 0
        assert smart["config"]["steps"] >= 10 and smart["config"]["batch"] >= 1
        signature = smart["analysis"]["signature"]
        with (folder / "notes.md").open("a", encoding="utf-8") as handle:
            handle.write("\nMore local content changes the recommendation signature.")
        updated = request(base + "/api/corpus/smart", {
            "folder": str(folder), "corpus_mode": "smart", "technique": "fast", "device": "cpu",
        })
        assert updated["analysis"]["signature"] != signature
        assert updated["analysis"]["byte_tokens"] > smart["analysis"]["byte_tokens"]

        job = request(base + "/api/train", {
            **TINY_DRV, "folder": str(folder), "name": "workflow.pt", "steps": 1,
        })["job"]
        assert job["title"] == "Sprint-DRV · fresh"
        job = wait_for_job(base, timeout=60)
        assert job["status"] == "completed", "\n".join(job["logs"])
        assert job["current_step"] == 1 and job["checkpoint_ready"] is True
        assert job["metrics"]["points"] and job["metrics"]["latest"]["step"] == 1
        assert (tmp_path / "workspace" / job["checkpoint"]).exists()
        assert any(item["path"].endswith("workflow.pt")
                   for item in request(base + "/api/checkpoints")["checkpoints"])
        (listed,) = request(base + "/api/jobs")["jobs"]
        assert listed["id"] == job["id"] and listed["status"] == "completed"
        assert listed["result"]["model_id"] == job["checkpoint"]

        status, error = request_error(base + "/api/train", {
            **TINY_DRV, "folder": str(folder), "name": "workflow.pt", "steps": 1,
        })
        assert status == 400 and "already exists" in error["error"]

        result = request(base + "/api/chat", {
            "checkpoint": job["checkpoint"], "message": "Hello", "history": [],
            "max_tokens": 2, "temperature": 0, "device": "cpu",
        })
        assert result["kind"] == "DRV-LM" and result["reply"]
        assert result["model"].endswith("workflow.pt")


@pytest.mark.slow
def test_moe_saint_training_workflow(tmp_path):
    folder = make_corpus(tmp_path / "input")
    with studio(tmp_path / "workspace") as (server, base):
        request(base + "/api/train", {
            **TINY_DRV, "folder": str(folder), "name": "moe-saint.pt", "steps": 1, "ctx": 64,
            "moe": True, "latent_mode": "saint", "thoughts": 4, "latent_symbols": 16,
            "latent_window": 8, "latent_min_thoughts": 1, "latent_block_size": 2,
            "latent_continuous_slots": 1, "latent_refine_steps": 0, "latent_grammar_prob": 0.0,
            "w_latent_rl": 0.1,
        })
        job = wait_for_job(base)
        assert job["status"] == "completed", "\n".join(job["logs"])
        state = torch.load(tmp_path / "workspace" / job["checkpoint"], weights_only=False)
        assert state["cfg"]["use_moe"] is True and state["cfg"]["use_latent"] is True
        assert state["cfg"]["latent_saint"] is True and state["cfg"]["n_latent_symbols"] == 16
        assert state["cfg"]["latent_continuous_slots"] == 1
        assert any("SAINT:" in line for line in job["logs"])


@pytest.mark.slow
def test_ncm_pause_continue_and_backup(tmp_path):
    folder = make_corpus(tmp_path / "input")
    workspace = tmp_path / "workspace"
    checkpoint = workspace / "trained-model" / "ncm-pause.pt"
    tokenizer = checkpoint.with_name("ncm-pause-tokenizer.pt")
    config = {
        "folder": str(folder), "corpus_mode": "smart", "technique": "ncm", "name": "ncm-pause.pt",
        "steps": 100000, "tokenizer_steps": 100000, "batch": 1, "grad_accum": 1,
        "tokenizer_batch": 1, "tokenizer_grad_accum": 1, "ctx": 8, "prefix": 4,
        "levels": "1,2,4,8", "dim": 32, "heads": 4, "device": "cpu", "lr": 0.001,
        "generator_lr": 0.001, "eval_every": 100000, "warmup": 1, "backup": False,
    }
    with studio(workspace) as (server, base):
        job = request(base + "/api/train", {**config, "mode": "fresh"})["job"]
        job = wait_for_job(base, progress=True)
        request(base + "/api/pause", {"id": job["id"]})
        job = wait_for_job(base)
        assert job["status"] == "paused", "\n".join(job["logs"])
        assert tokenizer.exists() and not checkpoint.exists()
        assert 1 <= torch.load(tokenizer, weights_only=False)["step"] < 100000

        request(base + "/api/train", {**config, "mode": "fresh_generator", "steps": 1})
        job = wait_for_job(base)
        assert job["status"] == "completed", "\n".join(job["logs"])
        assert torch.load(checkpoint, weights_only=False)["step"] == 1

        request(base + "/api/train", {**config, "mode": "continue", "steps": 1, "backup": True})
        job = wait_for_job(base)
        assert job["status"] == "completed", "\n".join(job["logs"])
        assert job["current_step"] == 1
        assert torch.load(checkpoint, weights_only=False)["step"] == 2
        assert job["backup"] and (workspace / job["backup"] / "ncm-pause.pt").exists()


@pytest.mark.slow
def test_drv_pause_and_continue(tmp_path):
    folder = make_corpus(tmp_path / "input")
    workspace = tmp_path / "workspace"
    checkpoint = workspace / "trained-model" / "drv-pause.pt"
    config = {**TINY_DRV, "folder": str(folder), "name": "drv-pause.pt", "steps": 100000,
              "eval_every": 100000}
    with studio(workspace) as (server, base):
        job = request(base + "/api/train", {**config, "mode": "fresh"})["job"]
        job = wait_for_job(base, progress=True)
        via_jobs = request(base + f"/api/jobs/{job['id']}/pause", {})["job"]
        assert via_jobs["pause_requested"] is True
        job = wait_for_job(base)
        assert job["status"] == "paused", "\n".join(job["logs"])
        paused_step = torch.load(checkpoint, weights_only=False)["step"]
        assert 1 <= paused_step < 100000
        assert not checkpoint.with_name("drv-pause.pt.pause").exists()

        request(base + "/api/train", {**config, "mode": "continue", "steps": 1})
        job = wait_for_job(base)
        assert job["status"] == "completed", "\n".join(job["logs"])
        assert job["current_step"] == 1
        assert torch.load(checkpoint, weights_only=False)["step"] == paused_step + 1
