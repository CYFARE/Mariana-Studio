"""Context-window metadata: provider fields, catalog entries and nested configs."""

from __future__ import annotations

from mariana.studio.models.context import entry_context, model_context, positive_int


def test_positive_int_ignores_bools_and_non_numbers():
    assert positive_int(128000) == 128000
    assert positive_int("131072") == 131072
    assert positive_int(32768.0) == 32768
    assert positive_int(True) == 0
    assert positive_int(-1) == 0
    assert positive_int("unknown") == 0
    assert positive_int(None) == 0


def test_model_context_reads_the_widest_field_a_provider_reports():
    assert model_context({"id": "a", "context_length": 8192}) == 8192
    assert model_context({"id": "b", "max_model_len": 32768}) == 32768
    assert model_context({"id": "c", "context_window": 65536, "meta": {"n_ctx_train": 131072}}) == 131072
    assert model_context({"id": "d", "model_info": {"llama.context_length": 200000}}) == 200000
    assert model_context({"id": "e", "parameters": "num_ctx 16384\ntemperature 0.7"}) == 16384
    assert model_context({"id": "f"}) is None
    assert model_context(None) is None


def test_entry_context_combines_catalog_detail_and_capabilities():
    entry = {
        "id": "endpoint:local",
        "kind": "openai",
        "detail": {"ctx_len": 32768, "metadata": {"max_position_embeddings": 131072}},
        "capabilities": {"context_window": 65536},
    }
    assert entry_context(entry) == 131072
    assert entry_context({"detail": {"ctx_len": 2048, "ctx_unit": "bytes"}}) == 2048
    assert entry_context({"detail": {"n_ctx": 4096}}) == 4096
    assert entry_context({}) == 0
    assert entry_context(None) == 0
