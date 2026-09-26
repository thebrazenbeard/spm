from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from spm_bench.case import BenchmarkCase
from spm_bench.vector_state import VectorStateModule
from spm_bench.vector_training import stateful_base_forward, vector_training_presentations


def sample_case() -> BenchmarkCase:
    return BenchmarkCase.from_dict({
        "case_id": "train-case",
        "version": 1,
        "family": "referent_preservation",
        "turns": [
            {"role": "user", "content": "A prior fact lives here."},
            {"role": "user", "content": "Which option follows?"},
        ],
        "choices": [
            {"id": "a", "text": "first"},
            {"id": "b", "text": "second"},
            {"id": "c", "text": "third"},
        ],
        "expected_choice": "b",
        "risk_class": "low",
        "tags": [],
    })


def test_three_training_presentations_balance_target_label_and_position():
    presentations = vector_training_presentations(sample_case())
    assert len(presentations) == 3
    assert {item.target_label for item in presentations} == {"a", "b", "c"}
    assert {item.target_position for item in presentations} == {0, 1, 2}
    assert all(item.prefix_turns == ("A prior fact lives here.",) for item in presentations)
    assert all("Which option follows?" in item.final_text for item in presentations)


class AddLayer(nn.Module):
    def __init__(self, amount: float) -> None:
        super().__init__()
        self.amount = amount

    def forward(self, hidden_states, **kwargs):
        return hidden_states + self.amount


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0), AddLayer(3.0)])

    def forward(self, input_ids, attention_mask, use_cache=False):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeModel(nn.Module):
    base_model_prefix = "model"

    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()


def test_stateful_forward_updates_at_declared_layer_and_reinjects_only_when_requested():
    torch.manual_seed(1729)
    model = FakeModel()
    state_module = VectorStateModule(hidden_size=4, state_size=2)
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    prior = torch.zeros(1, 2)

    plain, observed = stateful_base_forward(
        model,
        state_module,
        inputs,
        prior,
        persistent=True,
        injection_layer_index=1,
        reinject=False,
    )
    conditioned, observed_again = stateful_base_forward(
        model,
        state_module,
        inputs,
        prior,
        persistent=True,
        injection_layer_index=1,
        reinject=True,
    )

    assert torch.allclose(observed, observed_again)
    assert not torch.allclose(plain.last_hidden_state[:, -1], conditioned.last_hidden_state[:, -1])
    assert torch.allclose(plain.last_hidden_state[:, 0], conditioned.last_hidden_state[:, 0])


class FakeTokenizer:
    pad_token_id = 3
    eos_token_id = 3

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        return {"a": [0], "b": [1], "c": [2]}[text]

    def __call__(self, prompts, return_tensors="pt", padding=False):
        if isinstance(prompts, str):
            prompts = [prompts]
        rows = []
        for prompt in prompts:
            base = [0, 1] if "prior" in prompt.lower() else [1, 2, 0]
            rows.append(base)
        width = max(len(row) for row in rows)
        ids = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
        mask = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


class FakeTrainingModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(4))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def get_output_embeddings(self):
        return self.head


def test_case_choice_loss_backpropagates_into_state_module_only():
    from spm_bench.vector_training import case_choice_loss

    torch.manual_seed(1729)
    model = FakeTrainingModel()
    tokenizer = FakeTokenizer()
    state_module = VectorStateModule(hidden_size=4, state_size=2)

    loss = case_choice_loss(
        model,
        tokenizer,
        state_module,
        sample_case(),
        persistent=True,
        injection_layer_index=1,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in state_module.parameters()
    )
    assert all(parameter.grad is None for parameter in model.parameters())


def test_seeded_vector_modules_start_byte_identical():
    from spm_bench.vector_training import initialized_vector_state

    first = initialized_vector_state(hidden_size=4, state_size=2, seed=1729)
    second = initialized_vector_state(hidden_size=4, state_size=2, seed=1729)
    assert all(
        torch.equal(first.state_dict()[key], second.state_dict()[key])
        for key in first.state_dict()
    )


def test_training_loop_is_deterministic_and_refuses_trainable_backbone():
    from spm_bench.vector_training import initialized_vector_state, train_vector_state_arm

    tokenizer = FakeTokenizer()
    cases = (sample_case(),)

    def run_once():
        model = FakeTrainingModel()
        module = initialized_vector_state(hidden_size=4, state_size=2, seed=1729)
        report = train_vector_state_arm(
            model, tokenizer, module, cases,
            persistent=True,
            injection_layer_index=1,
            epochs=1,
            accumulation_cases=1,
            learning_rate=3e-4,
            weight_decay=0.01,
            gradient_norm_ceiling=1.0,
            data_order_seed=20260917,
        )
        return report, {key: value.detach().clone() for key, value in module.state_dict().items()}

    first_report, first_state = run_once()
    second_report, second_state = run_once()
    assert first_report == second_report
    assert first_report["optimizer_steps"] == 1
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)

    unsafe_model = FakeTrainingModel()
    unsafe_model.head.weight.requires_grad_(True)
    module = initialized_vector_state(hidden_size=4, state_size=2, seed=1729)
    try:
        train_vector_state_arm(unsafe_model, tokenizer, module, cases, persistent=True, injection_layer_index=1, epochs=1, accumulation_cases=1, learning_rate=3e-4, weight_decay=0.01, gradient_norm_ceiling=1.0, data_order_seed=20260917)
    except ValueError as error:
        assert "backbone" in str(error)
    else:
        raise AssertionError("trainable backbone was accepted")


def test_vector_evaluation_uses_nine_balanced_presentations_and_returns_semantic_scores():
    from spm_bench.vector_training import evaluate_vector_state_case

    model = FakeTrainingModel()
    tokenizer = FakeTokenizer()
    module = VectorStateModule(hidden_size=4, state_size=2)
    result = evaluate_vector_state_case(
        model,
        tokenizer,
        module,
        sample_case(),
        persistent=True,
        injection_layer_index=1,
    )

    assert set(result["semantic_scores"]) == {"a", "b", "c"}
    assert abs(sum(result["semantic_scores"].values()) - 1.0) < 1e-6
    assert result["expected_choice"] == "b"
    assert isinstance(result["correct"], bool)


def test_stateful_forward_is_compatible_with_real_smollm3_layer_api():
    from transformers import SmolLM3Config, SmolLM3ForCausalLM

    config = SmolLM3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SmolLM3ForCausalLM(config)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    state_module = VectorStateModule(hidden_size=32, state_size=8)
    inputs = {
        "input_ids": torch.tensor([[1, 4, 5, 6]], dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    outputs, next_state = stateful_base_forward(
        model, state_module, inputs, torch.zeros(1, 8),
        persistent=True, injection_layer_index=1, reinject=True,
    )
    assert outputs.last_hidden_state.shape == (1, 4, 32)
    assert next_state.shape == (1, 8)
