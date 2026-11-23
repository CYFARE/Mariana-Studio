"""Low-VRAM fine-tuning estimates and automatic tuning decisions."""

from __future__ import annotations

from mariana.studio.finetune import memory

GIB = 1024 ** 3


def test_a_7b_model_fits_a_6gb_card_only_in_4bit():
    common = dict(parameters=7_000_000_000, method="lora", precision="bf16",
                  batch=4, block_size=512, trainable=40_000_000, layers=32, vocab=152_000)
    uncompressed = memory.estimate(quantization="none", **common)
    quantized = memory.estimate(quantization="4bit", **common)
    assert quantized["weights"] < uncompressed["weights"] / 3
    assert quantized["total"] < uncompressed["total"]

    tuned = memory.auto_tune(vram=6 * GIB, quantization="auto", grad_accum=2,
                             quantize_available=True, **common)
    assert tuned["quantization"] == "4bit"
    assert tuned["fits"] is True
    assert 1 <= tuned["batch"] <= 2
    assert tuned["gradient_checkpointing"] is True
    assert tuned["estimate"]["total"] <= 6 * GIB * memory.HEADROOM
    assert tuned["notes"]


def test_auto_tuning_leaves_a_small_run_and_an_explicit_choice_alone():
    small = memory.auto_tune(parameters=124_000_000, vram=8 * GIB, method="lora",
                             precision="bf16", quantization="auto", batch=4,
                             grad_accum=2, block_size=256, trainable=1_000_000,
                             layers=12, quantize_available=True)
    assert small["quantization"] == "none" and small["batch"] == 4
    assert small["gradient_checkpointing"] is False

    explicit = memory.auto_tune(parameters=7_000_000_000, vram=6 * GIB, method="lora",
                                precision="bf16", quantization="none", batch=4,
                                grad_accum=2, block_size=512, trainable=40_000_000,
                                layers=32, quantize_available=True)
    assert explicit["quantization"] == "none"
    assert explicit["fits"] is False
    assert any("estimate" in note for note in explicit["notes"])


def test_full_training_reports_that_it_does_not_fit_a_small_card():
    tuned = memory.auto_tune(parameters=7_000_000_000, vram=8 * GIB, method="full",
                             precision="bf16", quantization="auto", batch=2,
                             grad_accum=1, block_size=512, layers=32,
                             quantize_available=True)
    assert tuned["method"] == "full" and tuned["quantization"] == "none"
    assert tuned["fits"] is False
    assert tuned["batch"] == 1


def test_shrinking_the_batch_keeps_the_effective_batch_sane():
    tuned = memory.auto_tune(parameters=13_000_000_000, vram=8 * GIB, method="lora",
                             precision="bf16", quantization="auto", batch=8,
                             grad_accum=4, block_size=1024, trainable=80_000_000,
                             layers=40, vocab=128_000, quantize_available=True)
    assert 1 <= tuned["batch"] <= 8
    assert tuned["grad_accum"] >= 1
    assert tuned["effective_batch"] <= 8 * 4


def test_a_budget_without_a_gpu_never_rejects_a_plan():
    assert memory.fits({"total": 10 * GIB}, None)
    assert memory.fits({"total": 10 * GIB}, 0)
    report = memory.recommendation(7_000_000_000, method="lora", precision="bf16",
                                   quantization="4bit", batch=1, block_size=512,
                                   trainable=40_000_000, layers=32, vocab=152_000,
                                   gradient_checkpointing=True)
    assert report["fits_6gb"] is True and report["fits_8gb"] is True
    assert report["low_vram_total"] > 0
    assert "GB" in memory.describe_bytes(report["estimate"]["total"])
