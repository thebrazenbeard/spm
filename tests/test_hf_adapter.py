from __future__ import annotations

from pathlib import Path

import pytest

import spm_bench.hf_adapter as hf_adapter
from spm_bench.hf_adapter import LocalHFAdapter, local_inventory_digest


def test_missing_local_model_path_fails_closed(tmp_path: Path):
    missing = tmp_path / "not-present"

    with pytest.raises(FileNotFoundError, match="local model path"):
        LocalHFAdapter(missing, "model-x", "rev-1", {})


def test_local_inventory_digest_is_content_bound_and_stable(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"a":1}', encoding="utf-8")
    (model / "weights.bin").write_bytes(b"abc")

    first = local_inventory_digest(model)
    second = local_inventory_digest(model)
    (model / "weights.bin").write_bytes(b"abd")
    changed = local_inventory_digest(model)

    assert first == second
    assert len(first) == 64
    assert changed != first


def test_adapter_loads_strictly_local_and_keeps_explicit_revision(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return object()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(("model", path, kwargs))
            return object()

    monkeypatch.setattr(
        hf_adapter,
        "_load_hf_classes",
        lambda: (FakeTokenizerClass, FakeModelClass),
    )
    adapter = LocalHFAdapter(model_path, "model-x", "immutable-rev", {})
    adapter._load()

    assert adapter.revision == "immutable-rev"
    assert len(adapter.model_digest) == 64
    assert [call[0] for call in calls] == ["tokenizer", "model"]
    assert all(call[2]["local_files_only"] is True for call in calls)
    assert all(call[2]["trust_remote_code"] is False for call in calls)


def test_generate_uses_deterministic_defaults_and_decodes_only_new_tokens(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}

    class FakeTensor:
        shape = (1, 3)
        def to(self, device):
            observed["device"] = device
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            observed["messages"] = messages
            return "PROMPT"
        def __call__(self, prompt, return_tensors):
            observed["prompt"] = prompt
            return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}
        def decode(self, tokens, skip_special_tokens):
            observed["decoded"] = tokens
            return "b"

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return FakeTokenizer()

    class FakeOutput:
        def __getitem__(self, item):
            return [10, 11, 12, 99][item]

    class FakeModel:
        device = "cpu"
        def generate(self, **kwargs):
            observed["generate"] = kwargs
            return [FakeOutput()]

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return FakeModel()

    monkeypatch.setattr(hf_adapter, "_load_hf_classes", lambda: (FakeTokenizerClass, FakeModelClass))
    adapter = LocalHFAdapter(model_path, "model-x", "rev-1", {"max_new_tokens": 7})
    output = adapter.generate(({"role": "user", "content": "x"},), {"max_new_tokens": 4})

    assert output == "b"
    generation = observed["generate"]
    assert generation["do_sample"] is False
    assert generation["max_new_tokens"] == 4
    assert "temperature" not in generation
    assert generation["pad_token_id"] == 0
