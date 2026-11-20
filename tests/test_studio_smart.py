"""Smart Options: sizing a run from corpus bytes, epochs and device memory."""

from __future__ import annotations

from mariana.studio.smart import recommend_training_options


def test_smart_options_use_bytes_epochs_and_hardware():
    small = recommend_training_options(
        {"output_bytes": 100_000, "files": 10}, "fast", "cpu"
    )
    assert small["analysis"]["byte_tokens"] == 100_000
    assert small["config"]["ctx"] == 64
    assert small["config"]["dim"] == 192
    assert small["analysis"]["target_epochs"] == 3.0
    assert small["analysis"]["effective_batch"] == (
        small["config"]["batch"] * small["config"]["grad_accum"]
    )
    assert small["config"]["steps"] >= 10

    large = recommend_training_options(
        {"output_bytes": 1_973_737_337, "files": 1, "documents": 10_000},
        "ncm", "auto",
    )
    config, analysis = large["config"], large["analysis"]
    assert config["ctx"] == 512 and config["dim"] == 512
    assert config["heads"] == 8 and config["levels"].endswith("512")
    assert config["levels"].startswith("1,4,9,16")
    assert len(config["levels"].split(",")) == 14
    assert config["steps"] <= 100_000
    tokenizer_positions = config["tokenizer_batch"] * config["tokenizer_grad_accum"] * config["ctx"]
    assert config["tokenizer_steps"] == max(
        10, round(config["steps"] * analysis["positions_per_step"] * 0.5 / tokenizer_positions)
    )
    assert analysis["byte_tokens"] == 1_973_737_337
    assert analysis["estimated_parameters"] > 0
    assert analysis["estimated_memory"] > 0
    assert analysis["effective_batch"] == config["batch"] * config["grad_accum"]
    assert config["warmup"] <= config["steps"]
    assert config["eval_every"] >= 1


def test_smart_options_account_for_moe_and_saint_memory():
    from unittest.mock import patch

    import torch

    corpus = {"output_bytes": 1_787_900_000, "files": 1}
    options = {
        "moe": True, "latent_mode": "saint", "latent_symbols": 256,
        "latent_continuous_slots": 2, "latent_refine_steps": 2,
    }
    with (
        patch("mariana.studio.smart.pick_device", return_value=torch.device("cuda")),
        patch("mariana.studio.smart._device_memory", return_value=8 * 1024**3),
    ):
        plain = recommend_training_options(corpus, "fast", "cuda")
        enhanced = recommend_training_options(corpus, "fast", "cuda", options)
        legacy = recommend_training_options(corpus, "legacy", "cuda", options)
    assert plain["config"]["batch"] == 12
    assert enhanced["config"]["batch"] == 4
    assert enhanced["config"]["grad_accum"] == 8
    assert enhanced["config"]["moe"] is True
    assert enhanced["config"]["latent_mode"] == "saint"
    assert enhanced["config"]["compile"] is False
    assert 110_000_000 < enhanced["analysis"]["estimated_parameters"] < 111_000_000
    assert enhanced["analysis"]["estimated_memory"] < 6 * 1024**3
    assert legacy["config"]["batch"] == 2
    assert legacy["config"]["grad_accum"] == 16
    assert legacy["config"]["compile"] is False
    assert legacy["analysis"]["estimated_memory"] < 6 * 1024**3


def test_ncm_performance_options_follow_the_device():
    from unittest.mock import patch

    import torch
    from mariana.ncm.layers import HAS_FLEX

    with (
        patch("mariana.studio.smart.pick_device", return_value=torch.device("cuda")),
        patch("mariana.studio.smart._device_memory", return_value=8 * 1024**3),
    ):
        result = recommend_training_options(
            {"output_bytes": 100_000}, "ncm", "cuda", {"cache_codes": True},
        )
    assert result["config"]["compile"] is True
    assert result["config"]["flex"] is HAS_FLEX
    assert result["config"]["cache_codes"] is True

    with patch("mariana.studio.smart.pick_device", return_value=torch.device("cpu")):
        cpu = recommend_training_options({"output_bytes": 100_000}, "ncm", "cpu")
    assert cpu["config"]["compile"] is False and cpu["config"]["flex"] is False
    assert cpu["config"]["cache_codes"] is False
