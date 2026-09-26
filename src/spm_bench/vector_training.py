"""Training-time presentation and causal injection helpers for vector-state V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .case import BenchmarkCase
from .vector_state import VectorStateModule


@dataclass(frozen=True, slots=True)
class TrainingPresentation:
    prefix_turns: tuple[str, ...]
    final_text: str
    choice_labels: tuple[str, ...]
    target_label: str
    target_position: int


def vector_training_presentations(case: BenchmarkCase) -> tuple[TrainingPresentation, ...]:
    if len(case.choices) != 3:
        raise ValueError("vector-state V1 requires exactly three choices")
    labels = tuple(choice.choice_id for choice in case.choices)
    expected_index = labels.index(case.expected_choice)
    layouts = ((0, 0), (1, 2), (2, 1))
    prefix_turns = tuple(turn.content for turn in case.turns[:-1])
    final_question = case.turns[-1].content
    presentations: list[TrainingPresentation] = []
    for order_offset, label_offset in layouts:
        order = tuple(range(3))[order_offset:] + tuple(range(3))[:order_offset]
        assigned = labels[label_offset:] + labels[:label_offset]
        choice_lines = "\n".join(
            f"[{label}] {case.choices[index].text}"
            for label, index in zip(assigned, order, strict=True)
        )
        position = order.index(expected_index)
        target_label = assigned[position]
        presentations.append(TrainingPresentation(
            prefix_turns=prefix_turns,
            final_text=(
                f"{final_question}\n\nChoices:\n{choice_lines}\n"
                f"Reply with exactly one bracketed choice ID, such as [{assigned[0]}], and no other text."
            ),
            choice_labels=labels,
            target_label=target_label,
            target_position=position,
        ))
    return tuple(presentations)


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("unsupported transformer-layer output type")


def _replace_layer_hidden(output: Any, hidden_states: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden_states
    if isinstance(output, tuple):
        return (hidden_states, *output[1:])
    if isinstance(output, list):
        return [hidden_states, *output[1:]]
    raise TypeError("unsupported transformer-layer output type")


def stateful_base_forward(
    model: Any,
    state_module: VectorStateModule,
    inputs: Mapping[str, torch.Tensor],
    prior_state: torch.Tensor,
    *,
    persistent: bool,
    injection_layer_index: int,
    reinject: bool,
):
    prefix = getattr(model, "base_model_prefix", None)
    backbone = getattr(model, prefix, None) if prefix else None
    layers = getattr(backbone, "layers", None)
    if backbone is None or layers is None:
        raise RuntimeError("model does not expose transformer layers")
    if injection_layer_index < 0 or injection_layer_index >= len(layers):
        raise ValueError("injection_layer_index is out of range")
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None or attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seq]")
    last_positions = attention_mask.long().sum(dim=-1) - 1
    if torch.any(last_positions < 0):
        raise ValueError("every sequence must contain at least one real token")

    captured: dict[str, torch.Tensor] = {}
    state_dtype = state_module.observation_projection.weight.dtype
    state_device = state_module.observation_projection.weight.device

    def hook(_layer, _args, output):
        hidden = _hidden_from_layer_output(output)
        row_ids = torch.arange(hidden.shape[0], device=hidden.device)
        positions = last_positions.to(hidden.device)
        observation = hidden[row_ids, positions].to(device=state_device, dtype=state_dtype)
        effective_prior = prior_state.to(device=state_device, dtype=state_dtype)
        next_state = state_module.update(
            observation,
            effective_prior,
            persistent=persistent,
        )
        captured["next_state"] = next_state
        if not reinject:
            return output
        conditioned = state_module.reinject(hidden, next_state, positions)
        return _replace_layer_hidden(output, conditioned)

    handle = layers[injection_layer_index].register_forward_hook(hook)
    try:
        outputs = backbone(**dict(inputs), use_cache=False)
    finally:
        handle.remove()
    if "next_state" not in captured:
        raise RuntimeError("state injection layer was not executed")
    return outputs, captured["next_state"]


def _prompt(tokenizer: Any, text: str, *, bracket_prefix: bool) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered + ("[" if bracket_prefix else "")


def _encode_prompt_batch(tokenizer: Any, prompts: tuple[str, ...], *, device) -> dict[str, torch.Tensor]:
    encoded_rows = [tokenizer(prompt, return_tensors="pt") for prompt in prompts]
    lengths = [int(row["input_ids"].shape[-1]) for row in encoded_rows]
    width = max(lengths)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)
    input_rows = []
    mask_rows = []
    for encoded, length in zip(encoded_rows, lengths, strict=True):
        padding = width - length
        input_rows.append(torch.cat((
            encoded["input_ids"][0],
            torch.full((padding,), pad_token_id, dtype=encoded["input_ids"].dtype),
        )))
        mask_rows.append(torch.cat((
            encoded["attention_mask"][0],
            torch.zeros((padding,), dtype=encoded["attention_mask"].dtype),
        )))
    return {
        "input_ids": torch.stack(input_rows).to(device),
        "attention_mask": torch.stack(mask_rows).to(device),
    }


def case_choice_loss(
    model: Any,
    tokenizer: Any,
    state_module: VectorStateModule,
    case: BenchmarkCase,
    *,
    persistent: bool,
    injection_layer_index: int,
) -> torch.Tensor:
    presentations = vector_training_presentations(case)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    state_device = state_module.observation_projection.weight.device
    state_dtype = state_module.observation_projection.weight.dtype
    prior = state_module.zero_state(1, device=state_device, dtype=state_dtype)

    for text in presentations[0].prefix_turns:
        inputs = _encode_prompt_batch(
            tokenizer,
            (_prompt(tokenizer, text, bracket_prefix=False),),
            device=model_device,
        )
        _outputs, prior = stateful_base_forward(
            model,
            state_module,
            inputs,
            prior,
            persistent=persistent,
            injection_layer_index=injection_layer_index,
            reinject=False,
        )

    final_prompts = tuple(
        _prompt(tokenizer, presentation.final_text, bracket_prefix=True)
        for presentation in presentations
    )
    final_inputs = _encode_prompt_batch(tokenizer, final_prompts, device=model_device)
    prior_batch = prior.expand(len(presentations), -1)
    outputs, _next_state = stateful_base_forward(
        model,
        state_module,
        final_inputs,
        prior_batch,
        persistent=persistent,
        injection_layer_index=injection_layer_index,
        reinject=True,
    )
    hidden = outputs.last_hidden_state
    positions = final_inputs["attention_mask"].long().sum(dim=-1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    final_hidden = hidden[rows, positions.to(hidden.device)]

    labels = presentations[0].choice_labels
    token_ids: list[int] = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError("every choice label must encode to exactly one token")
        token_ids.append(encoded[0])
    head = model.get_output_embeddings()
    if head is None or not hasattr(head, "weight"):
        raise RuntimeError("model does not expose output embedding weights")
    token_index = torch.tensor(token_ids, device=head.weight.device)
    selected_weight = head.weight.index_select(0, token_index).to(final_hidden.device)
    selected_bias = getattr(head, "bias", None)
    if selected_bias is not None:
        selected_bias = selected_bias.index_select(0, token_index).to(final_hidden.device)
    logits = torch.nn.functional.linear(
        final_hidden.float(),
        selected_weight.float(),
        selected_bias.float() if selected_bias is not None else None,
    )
    targets = torch.tensor(
        [labels.index(presentation.target_label) for presentation in presentations],
        device=logits.device,
    )
    return torch.nn.functional.cross_entropy(logits, targets)


def initialized_vector_state(*, hidden_size: int, state_size: int, seed: int) -> VectorStateModule:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return VectorStateModule(hidden_size=hidden_size, state_size=state_size)


def train_vector_state_arm(
    model: Any,
    tokenizer: Any,
    state_module: VectorStateModule,
    cases,
    *,
    persistent: bool,
    injection_layer_index: int,
    epochs: int,
    accumulation_cases: int,
    learning_rate: float,
    weight_decay: float,
    gradient_norm_ceiling: float,
    data_order_seed: int,
) -> dict[str, Any]:
    import random

    ordered_cases = tuple(cases)
    if not ordered_cases:
        raise ValueError("training cases must not be empty")
    if epochs <= 0 or accumulation_cases <= 0:
        raise ValueError("epochs and accumulation_cases must be positive")
    if len(ordered_cases) % accumulation_cases:
        raise ValueError("case count must be divisible by accumulation_cases")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("backbone/output-head parameters must be frozen before training")
    model.eval()
    state_module.train()
    optimizer = torch.optim.AdamW(
        state_module.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer_steps = 0
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        indices = list(range(len(ordered_cases)))
        random.Random(data_order_seed + epoch).shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        pending = 0
        for case_index in indices:
            loss = case_choice_loss(
                model,
                tokenizer,
                state_module,
                ordered_cases[case_index],
                persistent=persistent,
                injection_layer_index=injection_layer_index,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            total_loss += float(loss.detach().cpu())
            (loss / accumulation_cases).backward()
            pending += 1
            if pending == accumulation_cases:
                torch.nn.utils.clip_grad_norm_(
                    state_module.parameters(), gradient_norm_ceiling
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                pending = 0
        if pending:
            raise RuntimeError("unexpected incomplete accumulation bucket")
        epoch_losses.append(total_loss / len(ordered_cases))

    return {
        "persistent": persistent,
        "case_count": len(ordered_cases),
        "epochs": epochs,
        "optimizer_steps": optimizer_steps,
        "mean_case_loss_by_epoch": epoch_losses,
    }


def evaluate_vector_state_case(
    model: Any,
    tokenizer: Any,
    state_module: VectorStateModule,
    case: BenchmarkCase,
    *,
    persistent: bool,
    injection_layer_index: int,
) -> dict[str, Any]:
    from .runner import _balanced_layouts, _balanced_score_result

    labels, layouts, message_batches = _balanced_layouts(case)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    state_device = state_module.observation_projection.weight.device
    state_dtype = state_module.observation_projection.weight.dtype
    prior = state_module.zero_state(1, device=state_device, dtype=state_dtype)
    state_module.eval()
    model.eval()

    with torch.no_grad():
        for turn in case.turns[:-1]:
            inputs = _encode_prompt_batch(
                tokenizer,
                (_prompt(tokenizer, turn.content, bracket_prefix=False),),
                device=model_device,
            )
            _outputs, prior = stateful_base_forward(
                model, state_module, inputs, prior,
                persistent=persistent,
                injection_layer_index=injection_layer_index,
                reinject=False,
            )
        final_texts = tuple(
            messages[-2]["content"] + "\n\n" + messages[-1]["content"]
            for messages in message_batches
        )
        prompts = tuple(
            _prompt(tokenizer, text, bracket_prefix=True) for text in final_texts
        )
        final_inputs = _encode_prompt_batch(tokenizer, prompts, device=model_device)
        prior_batch = prior.expand(len(prompts), -1)
        outputs, _ = stateful_base_forward(
            model, state_module, final_inputs, prior_batch,
            persistent=persistent,
            injection_layer_index=injection_layer_index,
            reinject=True,
        )
        hidden = outputs.last_hidden_state
        positions = final_inputs["attention_mask"].long().sum(dim=-1) - 1
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        final_hidden = hidden[rows, positions.to(hidden.device)]

        token_ids = []
        for label in labels:
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("every choice label must encode to exactly one token")
            token_ids.append(encoded[0])
        head = model.get_output_embeddings()
        token_index = torch.tensor(token_ids, device=head.weight.device)
        weight = head.weight.index_select(0, token_index).to(final_hidden.device)
        bias = getattr(head, "bias", None)
        if bias is not None:
            bias = bias.index_select(0, token_index).to(final_hidden.device)
        logits = torch.nn.functional.linear(
            final_hidden.float(), weight.float(), bias.float() if bias is not None else None
        )
        probabilities = torch.softmax(logits, dim=-1)
        score_rows = tuple(
            {label: float(probabilities[row, col].item()) for col, label in enumerate(labels)}
            for row in range(probabilities.shape[0])
        )
    return _balanced_score_result(case, labels, layouts, score_rows)
