# Export and test with Ollama, llama.cpp or ONNX

> **Code:** `mariana/export/` · **Test:** `tests/test_export.py` ·
> **Index:** [documentation](README.md)

Mariana exports the ordinary AR decoding path losslessly to **GGUF**
(llama.cpp / Ollama), **ONNX**, and LLaMA-format **safetensors**.

## Export

```bash
pip install -r requirements-export.txt
python -m mariana export --checkpoint trained-model/mariana-cc.pt --format all --out exports
```

produces

```
exports/mariana-f16.gguf     # LLaMA-architecture GGUF (F16; F32/BF16 via --dtype)
exports/Modelfile            # ready for Ollama
exports/mariana.onnx         # dynamic batch/seq axes, onnxruntime-verified
exports/mariana-llama.safetensors
```

The export is **lossless by construction**: committed/AR tokens always use
diffusion time bucket 0, where (a) the time/region embeddings are defined
relative to their baseline, (b) adaLN modulation is the identity, and (c) the
RoPE time axis contributes zero rotation — so the AR path *is* a vanilla
LLaMA transformer (RMSNorm + RoPE θ=10000 + SwiGLU + tied embeddings). The
converter only unrolls the recurrent core and permutes Q/K rows from the
interleaved to the half-split RoPE layout. A numeric parity test
(`tests/test_export.py::test_llama_compat_parity`) verifies bit-exactness.

## Run it

```bash
# llama.cpp
llama-cli -m exports/mariana-f16.gguf -p "KING: " -n 64
llama-quantize exports/mariana-f16.gguf exports/mariana-q4.gguf Q4_K_M   # optional

# Ollama
cd exports && ollama create mariana -f Modelfile && ollama run mariana "KING: "

# ONNX Runtime (Python)
python - <<'PY'
import onnxruntime as ort, numpy as np
s = ort.InferenceSession("exports/mariana.onnx")
ids = np.array([[256] + list(b"KING: ")])          # BOS + prompt bytes
print(s.run(["logits"], {"tokens": ids})[0][0, -1].argmax())
PY
```

Tokenizer: byte-level (256 byte tokens + `<|bos|>` + `<|mask|>`) embedded in
the GGUF via the GPT-2 bytes-to-unicode table, so prompts encode/decode as
raw bytes end-to-end.

## What is not exported

MoE checkpoints and NCM checkpoints do not use this export path; train a dense
DRV-LM checkpoint to export. SAINT latent modules are omitted from every export,
which still contains the lossless ordinary AR path. Deploying compressed thought
requires the custom PyTorch graph.
