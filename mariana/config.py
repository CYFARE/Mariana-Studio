"""Configuration for the Mariana DRV-LM model.

The dataclass is the single source of truth for model hyper-parameters.
It serializes to/from plain dicts (JSON in checkpoints, GGUF metadata on
export) so checkpoints stay portable across versions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass
class Config:
    # --- vocabulary / sequence -------------------------------------------------
    vocab_size: int = 258          # 256 byte tokens + BOS + MASK
    ctx_len: int = 256
    bos_token: int = 256
    mask_token: int = 257

    # --- backbone --------------------------------------------------------------
    dim: int = 256
    n_heads: int = 4
    n_pre: int = 2                 # encoder blocks before the recurrent core
    n_core: int = 2                # shared core blocks (looped max_loops times)
    n_post: int = 2                # decoder blocks after the core
    mlp_mult: int = 4
    max_loops: int = 2             # recurrent-depth refinement iterations
    dropout: float = 0.0

    # --- diffusion -------------------------------------------------------------
    block_size: int = 32           # diffusion block width (block-causal attention)
    n_time_buckets: int = 64       # discrete noise-level buckets

    # --- conditioning ----------------------------------------------------------
    rope_base_pos: float = 10000.0
    rope_base_time: float = 100.0
    use_adaln: bool = True         # adaLN-zero FiLM timestep conditioning

    # --- optional capacity -----------------------------------------------------
    use_moe: bool = False          # sparse-MoE FFN in every block
    moe_experts: int = 4
    moe_topk: int = 2
    grouped_moe: bool = True         # grouped CUDA AMP expert GEMMs; reference fallback

    # --- auxiliary heads -------------------------------------------------------
    use_mtp: bool = True           # multi-token-prediction auxiliary head

    # --- latent code / internal language ---------------------------------------
    use_latent: bool = False       # learned discrete symbols for compressed thought
    n_latent_symbols: int = 256    # size of the model's private alphabet
    n_thoughts: int = 8            # symbols emitted before each compressed decode
    latent_local_window: int = 32  # raw bytes kept beside the latent summary
    latent_temperature: float = 1.0  # Gumbel temperature during training
    latent_window_dropout: float = 0.25  # per-example local-window mask probability
    latent_recall_prob: float = 0.25  # fraction of latent batches using a recall task
    latent_recall_pairs: int = 2   # key/value pairs to memorize in the recall task
    w_latent: float = 0.5          # compressed-context LM loss weight
    w_latent_entropy: float = 0.01  # encourage use of the whole alphabet

    # SAINT: adaptive, hierarchical, continuous, reinforcement-trained thought.
    # False keeps the original fixed discrete code and old checkpoints compatible.
    latent_saint: bool = False
    latent_min_thoughts: int = 2   # minimum content symbols before EOS is allowed
    latent_block_size: int = 2     # child thoughts summarized by each parent symbol
    latent_continuous_slots: int = 2  # variational continuous vectors per thought
    latent_refine_steps: int = 2   # bidirectional reasoning passes over continuous slots
    latent_grammar_prob: float = 0.25  # compositional key/operator/value curriculum
    latent_discrete_dropout: float = 0.05  # force use of the continuous channel
    latent_continuous_dropout: float = 0.05  # force use of discrete/parent codes
    latent_diffusion: bool = True  # draft-only cross-attention to the thought memory
    w_latent_grammar: float = 0.1  # next-symbol, pair-composition, and order losses
    w_latent_hierarchy: float = 0.05  # parent-to-child reconstruction loss
    w_latent_rl: float = 0.1       # baseline-centred policy gradient on code actions
    w_latent_value: float = 0.05   # compressed-return critic loss
    w_latent_length: float = 0.02  # return penalty per used thought (not per maximum)
    w_latent_task_reward: float = 0.1  # exact byte accuracy contribution to return
    w_latent_kl: float = 0.001     # variational continuous-slot information penalty
    w_latent_policy_entropy: float = 0.001  # exploration bonus for code actions

    # --- training architecture -------------------------------------------------
    # Sprint-DRV: share one hybrid forward across all objectives. AR supervision
    # uses the exact causal clean prefix; full-sequence AR alignment stays legacy.
    fast_train: bool = False

    # --- corpus encoding -------------------------------------------------------
    # AXIOM selects a text, dictionary, DNA/protein or identity codec from the
    # corpus and records it here so generation rewrites its prompt and decodes
    # its output with exactly the same rules. axiom_codec is the JSON codec
    # specification; an empty value means a pre-codec (v1) checkpoint.
    use_axiom: bool = False
    axiom_profile: str = "auto"
    axiom_codec: str = ""

    # --- loop regularisation ---------------------------------------------------
    loop_dropout: bool = True      # sample loop count in {1..max_loops} per step

    # --- loss weights ------------------------------------------------------------
    w_ar: float = 1.0
    w_diff: float = 1.0
    w_align: float = 0.1           # diffusion<->AR consistency (acceptance lever)
    w_ver: float = 0.1
    w_mtp: float = 0.1
    w_moe_lb: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        # tolerate checkpoints written by older/newer versions
        keys = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in keys})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls.from_dict(json.loads(s))

    @property
    def head_dim(self) -> int:
        if self.n_heads < 1 or self.dim % self.n_heads:
            raise ValueError(
                f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
            )
        return self.dim // self.n_heads

    @property
    def n_layers(self) -> int:
        """Flattened decoder depth (core blocks are weight-shared across loops)."""
        return self.n_pre + self.n_core * self.max_loops + self.n_post
