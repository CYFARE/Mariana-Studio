"""Native LoRA layers: zero start, freezing, merging, storage and PEFT interoperability."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from mariana.studio.finetune.lora import (
    LoRALinear,
    count_parameters,
    estimate_lora_parameters,
    find_target_modules,
    inject_lora,
    load_adapter,
    merge_adapter,
    save_adapter,
    select_target_names,
)


class Conv1D(nn.Module):
    """A stand-in for the GPT-2 layer of the same name: weight is ``(in, out)``."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.nf = out_features

    def forward(self, x):
        return x @ self.weight + self.bias


class Block(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return self.mlp(torch.tanh(self.q_proj(x) + self.v_proj(x)))


class TinyModel(nn.Module):
    def __init__(self, dim: int = 8, vocab: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([Block(dim), Block(dim)])
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, tokens):
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.lm_head(hidden)


def _train_a_little(model, tokens):
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for _ in range(3):
        optimizer.zero_grad()
        model(tokens).pow(2).mean().backward()
        optimizer.step()


def test_update_starts_at_zero_and_only_lora_weights_train():
    torch.manual_seed(0)
    model = TinyModel()
    tokens = torch.randint(0, 16, (2, 5))
    before = model(tokens).detach().clone()
    frozen = {name: value.detach().clone() for name, value in model.named_parameters()}
    names = inject_lora(model, r=4, alpha=8, dropout=0.0)
    assert names == ["layers.0.q_proj", "layers.0.v_proj", "layers.1.q_proj", "layers.1.v_proj"]
    assert torch.equal(model(tokens), before)
    total, trainable = count_parameters(model)
    assert trainable == 4 * 4 * (8 + 8) and 0 < trainable < total
    _train_a_little(model, tokens)
    assert not torch.allclose(model(tokens), before)
    for name, value in frozen.items():
        current = dict(model.named_parameters()).get(name.replace(".weight", ".base.weight")
                                                   .replace(".bias", ".base.bias"))
        if current is None:
            current = dict(model.named_parameters())[name]
        assert torch.equal(current, value), name


@pytest.mark.parametrize("layer", [nn.Linear(6, 4), Conv1D(6, 4)], ids=["linear", "conv1d"])
def test_merged_layer_reproduces_the_wrapped_forward(layer):
    torch.manual_seed(1)
    wrapped = LoRALinear(layer, r=2, alpha=4, dropout=0.0)
    nn.init.normal_(wrapped.lora_B.weight, std=0.3)
    x = torch.randn(3, 6)
    expected = wrapped(x).detach()
    merged = wrapped.merged()
    assert merged is layer
    assert torch.allclose(merged(x), expected, atol=1e-5)
    assert torch.allclose(wrapped(x), expected, atol=1e-5)  # a merged wrapper adds nothing twice


def test_half_precision_base_keeps_a_float32_update():
    base = nn.Linear(6, 4).to(torch.bfloat16)
    wrapped = LoRALinear(base, r=2, alpha=4, dropout=0.0)
    nn.init.normal_(wrapped.lora_B.weight, std=0.3)
    output = wrapped(torch.randn(2, 6, dtype=torch.bfloat16))
    assert output.dtype == torch.bfloat16
    assert wrapped.lora_A.weight.dtype == torch.float32
    output.float().sum().backward()
    assert wrapped.lora_B.weight.grad is not None
    assert wrapped.lora_B.weight.grad.dtype == torch.float32


def test_target_selection_modes_never_pick_the_output_head():
    model = TinyModel()
    every = find_target_modules(model, "all-linear")
    assert "lm_head" not in every and "layers.0.mlp" in every and len(every) == 6
    assert find_target_modules(model, ["mlp"]) == ["layers.0.mlp", "layers.1.mlp"]
    assert find_target_modules(model, "attention") == find_target_modules(model, "auto")
    assert select_target_names(["h.0.attn.c_proj", "h.0.mlp.c_proj", "h.0.attn.c_attn"], "attention") == [
        "h.0.attn.c_proj", "h.0.attn.c_attn",
    ]
    assert select_target_names(["a.fc", "b.fc"], "auto") == ["a.fc", "b.fc"]  # no attention names
    with pytest.raises(ValueError, match="No layer matches"):
        inject_lora(TinyModel(), targets=["missing"])
    with pytest.raises(ValueError, match="at least 1"):
        inject_lora(TinyModel(), r=0)


def test_adapter_round_trip_and_strict_loading(tmp_path):
    torch.manual_seed(2)
    model = TinyModel()
    reference = TinyModel()
    reference.load_state_dict(model.state_dict())
    tokens = torch.randint(0, 16, (2, 5))
    names = inject_lora(model, r=4, alpha=8, dropout=0.0)
    _train_a_little(model, tokens)
    tuned = model(tokens).detach()
    save_adapter(model, tmp_path / "adapter", base_model="tiny", r=4, alpha=8, dropout=0.0,
                 target_modules=names)
    config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert config["peft_type"] == "LORA" and config["r"] == 4 and config["target_modules"] == names
    assert load_adapter(reference, tmp_path / "adapter") == names
    assert torch.allclose(reference(tokens), tuned, atol=1e-6)
    merge_adapter(reference)
    assert not any(isinstance(module, LoRALinear) for module in reference.modules())
    assert torch.allclose(reference(tokens), tuned, atol=1e-5)
    with pytest.raises(ValueError, match="does not fit|shape"):
        load_adapter(TinyModel(dim=4), tmp_path / "adapter")
    with pytest.raises(ValueError, match="Cannot read"):
        load_adapter(TinyModel(), tmp_path / "missing")


def test_parameter_estimate_matches_injection():
    model = TinyModel()
    names = inject_lora(model, r=3)
    shapes = {name: (8, 8) for name in names}
    assert estimate_lora_parameters(shapes, 3) == count_parameters(model)[1]


def test_peft_loads_an_adapter_written_here(tmp_path):
    pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM

    from tests.helpers import make_tiny_transformer

    base = make_tiny_transformer(tmp_path / "base")
    model = AutoModelForCausalLM.from_pretrained(base)
    tokens = torch.randint(0, 200, (1, 8))
    names = inject_lora(model, r=4, alpha=8, dropout=0.0)
    assert names and all("attn" in name for name in names)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.normal_(module.lora_B.weight, std=0.2)
    model.eval()
    expected = model(tokens).logits.detach()
    save_adapter(model, tmp_path / "adapter", base_model=str(base), r=4, alpha=8, dropout=0.0,
                 target_modules=names)
    fresh = AutoModelForCausalLM.from_pretrained(base)
    assert not torch.allclose(fresh(tokens).logits, expected, atol=1e-4)
    loaded = peft.PeftModel.from_pretrained(fresh, str(tmp_path / "adapter")).eval()
    assert torch.allclose(loaded(tokens).logits, expected, atol=1e-4)
