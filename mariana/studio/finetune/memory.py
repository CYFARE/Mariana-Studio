"""Low-VRAM planning for fine-tuning: estimate bytes and pick a safe training shape.

The estimates are deliberately conservative and pure standard library.  They are
not a substitute for the allocator, but they let a 6-8 GB card choose 4-bit LoRA,
gradient checkpointing and a smaller micro-batch *before* loading a model that
would otherwise run out of memory.
"""

from __future__ import annotations

GIB = 1024 ** 3
HEADROOM = 0.9
QUANTIZATION_MODES = ("auto", "none", "4bit", "8bit")
QUANTIZED_BYTES = {"4bit": 0.55, "8bit": 1.05}
WEIGHT_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0}
LOSS_CHUNK_TOKENS = 256  # cross-entropy is computed in slices of this many positions
RESERVED_BYTES = 384 * 1024 * 1024  # CUDA context, allocator slack, temporary buffers


def describe_bytes(value: int | float) -> str:
    """A short human-readable size, e.g. ``3.4 GB``."""
    size = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def hidden_size(parameters: int, layers: int) -> int:
    """A rough transformer width from the parameter count (only used for estimates)."""
    if parameters <= 0 or layers <= 0:
        return 0
    return max(64, int((parameters / (12 * layers)) ** 0.5))


def estimate(parameters: int, *, method: str = "lora", precision: str = "bf16",
             quantization: str = "none", batch: int = 1, block_size: int = 512,
             trainable: int = 0, layers: int = 0, vocab: int = 0,
             gradient_checkpointing: bool = False) -> dict:
    """Estimated weights, gradients, optimizer and activation bytes for one run.

    ``parameters`` is the whole model; ``trainable`` is the number of trainable
    weights when it is known.  LoRA updates and AdamW moments stay in float32,
    so the method decides most of the footprint.
    """
    parameters = max(0, int(parameters or 0))
    method = method if method in {"lora", "full", "last_layers"} else "lora"
    quantization = quantization if quantization in QUANTIZATION_MODES else "none"
    if method != "lora" or quantization == "auto":
        quantization = "none"
    if trainable <= 0:
        trainable = parameters if method == "full" else max(1, int(parameters * 0.01))

    if method == "lora":
        per_weight = (QUANTIZED_BYTES[quantization]
                      if quantization in QUANTIZED_BYTES else WEIGHT_BYTES.get(precision, 2.0))
        weights = parameters * per_weight
    else:
        # Full and last-layer training keep fp32 master weights.
        weights = parameters * 4.0
    gradients = trainable * 4.0
    optimizer = trainable * 8.0  # AdamW first and second moments

    width = hidden_size(parameters, layers)
    if width:
        token_bytes = max(1, int(batch)) * max(8, int(block_size)) * width
        layers = max(1, int(layers))
        activations = token_bytes * ((layers * 2 + 16) if gradient_checkpointing
                                     else (layers * 20))
    else:
        activations = parameters * 0.01 * max(1, int(batch)) * max(8, int(block_size)) / 512
    # Chunked cross-entropy only materialises one slice of the vocabulary axis.
    logits = (max(0, int(vocab)) * max(1, int(batch))
              * min(max(8, int(block_size)), LOSS_CHUNK_TOKENS) * 4.0)
    total = weights + gradients + optimizer + activations + logits + RESERVED_BYTES
    return {
        "parameters": parameters, "trainable": int(trainable), "method": method,
        "precision": precision, "quantization": quantization,
        "weights": int(weights), "gradients": int(gradients),
        "optimizer": int(optimizer), "activations": int(activations),
        "logits": int(logits), "reserved": RESERVED_BYTES, "total": int(total),
    }


def fits(estimate_bytes: dict, vram_bytes: int | None, headroom: float = HEADROOM) -> bool:
    """Whether an estimate fits a VRAM budget with room for allocator fragmentation."""
    if not vram_bytes or vram_bytes <= 0:
        return True
    return estimate_bytes["total"] <= vram_bytes * max(0.5, min(1.0, headroom))


def auto_tune(parameters: int, *, vram: int | None, method: str, precision: str,
              quantization: str, batch: int, grad_accum: int, block_size: int,
              trainable: int = 0, layers: int = 0, vocab: int = 0,
              gradient_checkpointing: bool = False,
              quantize_available: bool = False) -> dict:
    """Choose quantization, batch and accumulation for the available VRAM.

    The smallest change that fits is preferred: 4-bit LoRA first, then gradient
    checkpointing, then a smaller micro-batch with more accumulation to keep the
    effective batch close to the request.  ``fits`` is advisory; the trainer
    still handles an unexpected allocator failure.
    """
    notes: list[str] = []
    method = method if method in {"lora", "full", "last_layers"} else "lora"
    precision = precision if precision in WEIGHT_BYTES else "bf16"
    if method != "lora" and quantization not in {"none", "auto"}:
        notes.append("Quantization only applies to LoRA; it was turned off.")
        quantization = "none"
    if method != "lora":
        quantization = "none"
    if quantization not in QUANTIZATION_MODES:
        quantization = "none"
    effective = max(1, int(batch) * max(1, int(grad_accum)))
    checkpointing = bool(gradient_checkpointing)
    options = dict(parameters=parameters, method=method, precision=precision,
                   batch=batch, block_size=block_size, trainable=trainable,
                   layers=layers, vocab=vocab, gradient_checkpointing=checkpointing)

    if quantization == "auto":
        quantization = "none"
        requested = estimate(quantization="none", **options)
        if method == "lora" and quantize_available and not fits(requested, vram):
            quantization = "4bit"
            notes.append("The base did not fit uncompressed; using 4-bit weights (QLoRA).")

    plan = estimate(quantization=quantization, **options)

    if not fits(plan, vram) and not checkpointing:
        checkpointing = True
        options["gradient_checkpointing"] = True
        plan = estimate(quantization=quantization, **options)
        notes.append("Gradient checkpointing is on to trade compute for activation memory.")

    if not fits(plan, vram) and batch > 1:
        while batch > 1 and not fits(plan, vram):
            batch = max(1, batch // 2)
            options["batch"] = batch
            plan = estimate(quantization=quantization, **options)
        grad_accum = max(1, min(4096, effective // max(1, batch)))
        notes.append(
            f"Micro-batch reduced to {batch} with accumulation {grad_accum} "
            f"(effective {batch * grad_accum}) to fit the GPU.")

    if not fits(plan, vram):
        if method == "lora" and not quantize_available and quantization == "none":
            notes.append(
                "Install bitsandbytes to enable 4-bit QLoRA; this model is too large "
                "for the GPU uncompressed.")
        notes.append(
            "The memory estimate is still above the GPU budget. The trainer will "
            "start at this size and stop with guidance if allocation fails.")
    return {
        "method": method, "precision": precision, "quantization": quantization,
        "batch": batch, "grad_accum": grad_accum,
        "gradient_checkpointing": checkpointing,
        "effective_batch": max(1, batch) * max(1, grad_accum),
        "fits": fits(plan, vram), "estimate": plan, "notes": notes,
    }


def recommendation(parameters: int, *, method: str, precision: str, quantization: str,
                   batch: int, block_size: int, trainable: int = 0, layers: int = 0,
                   vocab: int = 0, gradient_checkpointing: bool = False) -> dict:
    """Plan warnings for the UI: what a 6 GB or 8 GB card would do with this setup."""
    current = estimate(parameters, method=method, precision=precision,
                       quantization=quantization, batch=batch, block_size=block_size,
                       trainable=trainable, layers=layers, vocab=vocab,
                       gradient_checkpointing=gradient_checkpointing)
    low = estimate(parameters, method="lora", precision=precision, quantization="4bit",
                   batch=max(1, min(batch, 2)), block_size=block_size,
                   trainable=trainable, layers=layers, vocab=vocab,
                   gradient_checkpointing=True)
    return {
        "estimate": current,
        "fits_6gb": fits(current, 6 * GIB),
        "fits_8gb": fits(current, 8 * GIB),
        "low_vram_total": low["total"],
    }
