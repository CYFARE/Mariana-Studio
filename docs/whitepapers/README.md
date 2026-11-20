# Mariana whitepapers

These whitepapers document the techniques implemented in Mariana v2.3. They
replace the former single research-paper PDF with focused, independently
readable documents that can evolve with the code.

| Whitepaper | Technique | Implementation | Runbook |
| --- | --- | --- | --- |
| [Sprint-DRV](sprint-drv.md) | Single-pass training for the hybrid diffusion-AR model, with the legacy two-pass objective as its baseline | `mariana/train.py` | [fast-training.md](../fast-training.md) |
| [SAINT](saint.md) | Adaptive hierarchical discrete/continuous thought memory for a byte-level model | `mariana/saint.py` | [latent-language.md](../latent-language.md) |
| [NCM](ncm.md) | Multiscale residual VQ and parallel next-concept generation | `mariana/ncm/` | [ncm.md](../ncm.md) |
| [AML and AMP](aml-amp.md) | Portable agent document and local HTTP/JSON runtime protocol | `mariana/agents/` | [agentic.md](../agentic.md) |

## Scope and status

- **Implemented, not peer-reviewed.** Each paper describes code in this
  repository. It is a technical report, not a claim of publication or external
  replication.
- **Measurements are conditional.** Every number states the hardware, software,
  model, and protocol used. Local throughput or loss numbers are not predictions
  for another GPU, dataset, or model size.
- **No capability inflation.** A mechanism being implemented does not imply that
  it has solved language modeling, reasoning, or agent reliability. Where a
  technique has not been benchmarked at scale, the paper says so.
- **The runbooks are the how-to.** These papers explain design, objectives, and
  tradeoffs. Copy-paste commands, flags, and checkpoint layouts live in the
  linked runbooks and the main [README](../../README.md).

## Conventions

- `L_*` denotes a training objective; `w_*` denotes its configured weight.
- `x` is a byte sequence, `θ` the trainable parameters, and `p_θ` the model
  distribution.
- A **checkpoint** is a Mariana `.pt` file. A **model** is any catalog entry,
  including a remote endpoint.
- **AR** means the ordinary causal autoregressive path. **Draft** means the
  block-wise diffusion path. **Memory** is the SAINT latent summary.
- File and flag names match the implementation; if they diverge, the code wins.

## Citation

The repository ships a [`CITATION.cff`](../../CITATION.cff). For a specific
technique, cite the paper by name and version:

```bibtex
@techreport{mariana_sprint_drv_2026,
  title       = {Sprint-DRV: Single-Pass Training for a Hybrid
                 Diffusion-Autoregressive Language Model},
  author      = {{Mariana contributors}},
  institution = {Mariana},
  year        = {2026},
  version     = {2.3.0},
  url         = {https://github.com/CYFARE/Mariana/blob/main/docs/whitepapers/sprint-drv.md}
}
```

For the project as a whole:

```bibtex
@software{mariana2026,
  title   = {Mariana: a hybrid diffusion-AR language model and local studio},
  author  = {{Mariana contributors}},
  year    = {2026},
  version = {2.3.0},
  url     = {https://github.com/CYFARE/Mariana}
}
```

## Corrections

If a statement disagrees with the implementation, open an issue or a pull
request. Correctness fixes to a paper should reference the code path or test
that establishes the correct behavior.
