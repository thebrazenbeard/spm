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


def test_choose_constrains_decision_to_declared_single_token_ids(tmp_path, monkeypatch):
    import torch

    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "PROMPT"
        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        def encode(self, value, add_special_tokens=False):
            return {"a": [10], "b": [20], "c": [30]}[value]

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return FakeTokenizer()

    class FakeOutputs:
        def __init__(self):
            self.logits = torch.zeros((1, 2, 64))
            self.logits[0, -1, 10] = 1.0
            self.logits[0, -1, 20] = 5.0
            self.logits[0, -1, 30] = 2.0

    class FakeModel:
        device = "cpu"
        def eval(self):
            return self
        def __call__(self, **kwargs):
            return FakeOutputs()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return FakeModel()

    monkeypatch.setattr(hf_adapter, "_load_hf_classes", lambda: (FakeTokenizerClass, FakeModelClass))
    adapter = LocalHFAdapter(model_path, "model-x", "rev-1", {})
    chosen = adapter.choose(({"role": "user", "content": "x"},), ("a", "b", "c"))
    assert chosen == "b"


def test_choose_rejects_multitoken_choice_ids(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "PROMPT"
        def __call__(self, prompt, return_tensors):
            import torch
            return {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        def encode(self, value, add_special_tokens=False):
            return [1, 2]

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return FakeTokenizer()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            class FakeModel:
                device = "cpu"
                def eval(self): return self
            return FakeModel()

    monkeypatch.setattr(hf_adapter, "_load_hf_classes", lambda: (FakeTokenizerClass, FakeModelClass))
    adapter = LocalHFAdapter(model_path, "model-x", "rev-1", {})
    with pytest.raises(ValueError, match="single tokenizer token"):
        adapter.choose(({"role": "user", "content": "x"},), ("aa",))


def test_choose_scores_labels_after_forced_shared_bracket_prefix(tmp_path, monkeypatch):
    import torch
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    observed = {}

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt): return "PROMPT"
        def __call__(self, prompt, return_tensors):
            observed["prompt"] = prompt
            return {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        def encode(self, value, add_special_tokens=False): return {"a": [10], "b": [20]}[value]

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs): return FakeTokenizer()

    class FakeOutputs:
        def __init__(self):
            self.logits = torch.zeros((1, 2, 32)); self.logits[0, -1, 20] = 3

    class FakeModel:
        device = "cpu"
        def eval(self): return self
        def __call__(self, **kwargs): return FakeOutputs()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs): return FakeModel()

    monkeypatch.setattr(hf_adapter, "_load_hf_classes", lambda: (FakeTokenizerClass, FakeModelClass))
    adapter = LocalHFAdapter(model_path, "model-x", "rev-1", {})
    assert adapter.choose(({"role": "user", "content": "x"},), ("a", "b")) == "b"
    assert observed["prompt"] == "PROMPT["


def test_choose_many_batches_prompts_without_changing_choice_semantics(tmp_path, monkeypatch):
    import torch
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    observed = {"calls": 0}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return messages[0]["content"]
        def __call__(self, prompt, return_tensors):
            ids = [1, 2] if prompt.startswith("short") else [1, 2, 3]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        def encode(self, value, add_special_tokens=False):
            return {"a": [10], "b": [20]}[value]

    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs): return FakeTokenizer()

    class FakeOutputs:
        def __init__(self, batch):
            self.logits = torch.zeros((2, 3, 32))
            self.logits[0, 1, 20] = 4
            self.logits[1, 2, 10] = 5

    class FakeModel:
        device = "cpu"
        def eval(self): return self
        def __call__(self, **kwargs):
            observed["calls"] += 1
            observed["input_ids"] = kwargs["input_ids"].tolist()
            observed["attention_mask"] = kwargs["attention_mask"].tolist()
            return FakeOutputs(kwargs["input_ids"])

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs): return FakeModel()

    monkeypatch.setattr(hf_adapter, "_load_hf_classes", lambda: (FakeTokenizerClass, FakeModelClass))
    adapter = LocalHFAdapter(model_path, "model-x", "rev-1", {})
    batches = (({"role": "user", "content": "short"},), ({"role": "user", "content": "longer"},))
    assert adapter.choose_many(batches, ("a", "b")) == ("b", "a")
    assert observed["calls"] == 1
    assert observed["input_ids"] == [[1, 2, 0], [1, 2, 3]]
    assert observed["attention_mask"] == [[1, 1, 0], [1, 1, 1]]


def test_score_many_returns_declared_choice_probabilities_per_prompt(tmp_path, monkeypatch):
    import torch
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2
        def apply_chat_template(self, messages, tokenize, add_generation_prompt): return messages[0]["content"]
        def __call__(self, prompt, return_tensors):
            ids = [1, 2] if prompt.startswith("short") else [1, 2, 3]
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        def encode(self, value, add_special_tokens=False): return {"a": [10], "b": [20]}[value]
    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs): return FakeTokenizer()

    class FakeOutputs:
        def __init__(self):
            self.logits = torch.zeros((2, 3, 32))
            self.logits[0, 1, 10] = 1.0; self.logits[0, 1, 20] = 3.0
            self.logits[1, 2, 10] = 4.0; self.logits[1, 2, 20] = 2.0

    class FakeModel:
        device = "cpu"
        def eval(self): return self
        def __call__(self, **kwargs): return FakeOutputs()
    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs): return FakeModel()

    monkeypatch.setattr(hf_adapter, "_load_hf_classes", lambda: (FakeTokenizerClass, FakeModelClass))
    adapter = LocalHFAdapter(model_path, "model-x", "rev-1", {})
    batches = (({"role": "user", "content": "short"},), ({"role": "user", "content": "longer"},))
    scores = adapter.score_many(batches, ("a", "b"))
    assert set(scores[0]) == {"a", "b"}
    assert scores[0]["b"] > scores[0]["a"]
    assert scores[1]["a"] > scores[1]["b"]
    assert all(abs(sum(row.values()) - 1.0) < 1e-6 for row in scores)
