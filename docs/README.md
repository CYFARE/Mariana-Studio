# Mariana documentation

This directory holds the design notes, runbooks, and whitepapers for Mariana.
Start with the [main README](../README.md) for install, quick start, and the
Studio workspace overview.

## Whitepapers

See [`whitepapers/`](whitepapers/) for the index and citation details.

| Paper | Topic |
| --- | --- |
| [Sprint-DRV](whitepapers/sprint-drv.md) | Single-pass training for the hybrid diffusion-AR model, including the legacy two-pass baseline and measurements. |
| [SAINT](whitepapers/saint.md) | Adaptive hierarchical discrete/continuous thought memory, curricula, policy learning, and ablations. |
| [NCM](whitepapers/ncm.md) | Multiscale residual VQ and parallel next-concept generation. |
| [AML and AMP](whitepapers/aml-amp.md) | The portable agent document language and the local HTTP/JSON runtime protocol. |

## Runbooks and design notes

| Document | Contents |
| --- | --- |
| [fast-training.md](fast-training.md) | Sprint-DRV flags, benchmark command, tradeoffs, and MoE kernels. |
| [latent-language.md](latent-language.md) | SAINT commands, metrics, ablations, and limitations. |
| [ncm.md](ncm.md) | NCM training/inference recipes, controls, and evaluation requirements. |
| [distillation.md](distillation.md) | Sequence-level distillation from a local or cloud teacher. |
| [export.md](export.md) | GGUF, ONNX and safetensors export, plus llama.cpp, Ollama and ONNX Runtime runners. |
| [low-vram.md](low-vram.md) | External-disk Common Corpus run with MoE + MTP + SAINT on an 8 GB GPU. |
| [agentic.md](agentic.md) | Studio Agents workspace, AML/AMP quick loop, and security. |
| [ide.md](ide.md) | The coding workspace and its `/api/ide` protocol. |
| [../architecture.txt](../architecture.txt) | The full implemented DRV-LM architecture tree. |
| [../README_training.md](../README_training.md) | Training quickstart and Studio training workflow. |

## Engineering standard

[`AGENTS.md`](../AGENTS.md) is the project's backend and frontend engineering
standard. It covers the package map, naming, layering, error contracts,
security, tests, and the definition of done. [`tests/test_standards.py`](../tests/test_standards.py)
enforces the checkable parts.

## Contributing

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) for setup, the test commands, and
the pull-request checklist. Security reports are covered by
[`SECURITY.md`](../SECURITY.md).
