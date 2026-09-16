"""Deterministic benchmark execution and result manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from .adapter import ChatAdapter
from .case import BenchmarkCase


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a run manifest without environment-dependent formatting."""
    return _canonical_json(dict(manifest))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def benchmark_digest(cases: Sequence[BenchmarkCase]) -> str:
    """Digest the ordered semantic benchmark content, independent of file layout."""
    payload = [case.to_dict() for case in cases]
    return _sha256_text(_canonical_json(payload))


def _case_messages(case: BenchmarkCase) -> tuple[dict[str, str], ...]:
    messages = [turn.to_dict() for turn in case.turns]
    choices = "\n".join(
        f"[{choice.choice_id}] {choice.text}" for choice in case.choices
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Choices:\n"
                f"{choices}\n"
                "Reply with exactly one choice ID and no other text."
            ),
        }
    )
    return tuple(messages)


def _choice_messages_with_layout(
    case: BenchmarkCase,
    order_indices: Sequence[int],
    labels: Sequence[str],
) -> tuple[dict[str, str], ...]:
    assigned = tuple(labels)
    order = tuple(order_indices)
    expected_indices = tuple(range(len(case.choices)))
    if sorted(order) != list(expected_indices):
        raise ValueError("order_indices must be a permutation of every choice")
    if len(assigned) != len(case.choices) or len(set(assigned)) != len(assigned):
        raise ValueError("labels must uniquely cover every choice")
    messages = [turn.to_dict() for turn in case.turns]
    choices = "\n".join(
        f"[{label}] {case.choices[index].text}"
        for label, index in zip(assigned, order, strict=True)
    )
    example = assigned[0]
    messages.append({
        "role": "user",
        "content": (
            "Choices:\n"
            f"{choices}\n"
            f"Reply with exactly one bracketed choice ID, such as [{example}], and no other text."
        ),
    })
    return tuple(messages)


def _choice_messages_with_labels(
    case: BenchmarkCase, labels: Sequence[str]
) -> tuple[dict[str, str], ...]:
    return _choice_messages_with_layout(
        case, tuple(range(len(case.choices))), labels
    )


def _choice_case_messages(case: BenchmarkCase) -> tuple[dict[str, str], ...]:
    return _choice_messages_with_labels(
        case, tuple(choice.choice_id for choice in case.choices)
    )


def _parse_choice(output: str, case: BenchmarkCase) -> str | None:
    candidate = output.strip()
    valid = {choice.choice_id for choice in case.choices}
    return candidate if candidate in valid else None


def run_suite(
    cases: Sequence[BenchmarkCase],
    adapter: ChatAdapter,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run cases in order and return an exact-subject deterministic manifest."""
    ordered_cases = tuple(cases)
    config = dict(generation_config)
    config_digest = _sha256_text(_canonical_json(config))

    results: list[dict[str, Any]] = []
    for case in ordered_cases:
        output = adapter.generate(_case_messages(case), config)
        if not isinstance(output, str):
            raise TypeError("adapter.generate must return str")
        parsed_choice = _parse_choice(output, case)
        results.append(
            {
                "case_id": case.case_id,
                "case_digest": case.digest(),
                "family": case.family,
                "output": output,
                "parsed_choice": parsed_choice,
                "expected_choice": case.expected_choice,
                "correct": parsed_choice == case.expected_choice,
            }
        )

    summary = {
        "case_count": len(results),
        "correct_count": sum(1 for result in results if result["correct"]),
        "malformed_count": sum(
            1 for result in results if result["parsed_choice"] is None
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_digest": benchmark_digest(ordered_cases),
        "model": {
            "id": adapter.model_id,
            "digest": adapter.model_digest,
        },
        "generation_config": config,
        "generation_config_digest": config_digest,
        "results": results,
        "summary": summary,
    }
    manifest["run_digest"] = _sha256_text(_canonical_json(manifest))
    return manifest


def run_choice_suite(
    cases: Sequence[BenchmarkCase],
    adapter: Any,
) -> dict[str, Any]:
    """Evaluate declared choice IDs by constrained next-token selection."""
    ordered_cases = tuple(cases)
    protocol = {
        "method": "forced_bracket_prefix_label_argmax",
        "version": 1,
    }
    results: list[dict[str, Any]] = []
    for case in ordered_cases:
        choice_ids = tuple(choice.choice_id for choice in case.choices)
        chosen = adapter.choose(_choice_case_messages(case), choice_ids)
        if chosen not in choice_ids:
            raise ValueError(f"adapter returned undeclared choice: {chosen!r}")
        results.append({
            "case_id": case.case_id,
            "case_digest": case.digest(),
            "family": case.family,
            "parsed_choice": chosen,
            "expected_choice": case.expected_choice,
            "correct": chosen == case.expected_choice,
        })

    summary = {
        "case_count": len(results),
        "correct_count": sum(1 for result in results if result["correct"]),
        "malformed_count": 0,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_mode": "constrained_choice_v1",
        "choice_protocol": protocol,
        "benchmark_digest": benchmark_digest(ordered_cases),
        "model": {
            "id": adapter.model_id,
            "digest": adapter.model_digest,
        },
        "results": results,
        "summary": summary,
    }
    manifest["run_digest"] = _sha256_text(_canonical_json(manifest))
    return manifest


def run_permutation_choice_suite(
    cases: Sequence[BenchmarkCase],
    adapter: Any,
) -> dict[str, Any]:
    """Require semantic choice stability across cyclic label permutations."""
    ordered_cases = tuple(cases)
    protocol = {
        "method": "cyclic_label_permutation_consensus",
        "version": 1,
        "base_selector": "forced_bracket_prefix_label_argmax",
        "selector_execution": "batch_if_supported",
    }
    results: list[dict[str, Any]] = []
    for case in ordered_cases:
        labels = tuple(choice.choice_id for choice in case.choices)
        assignments = tuple(
            labels[offset:] + labels[:offset] for offset in range(len(labels))
        )
        message_batches = tuple(
            _choice_messages_with_labels(case, assigned) for assigned in assignments
        )
        choose_many = getattr(adapter, "choose_many", None)
        if callable(choose_many):
            chosen_labels = tuple(choose_many(message_batches, labels))
        else:
            chosen_labels = tuple(
                adapter.choose(messages, labels) for messages in message_batches
            )
        if len(chosen_labels) != len(assignments):
            raise ValueError("adapter returned wrong number of choices")

        rotation_choices: list[str] = []
        for assigned, chosen in zip(assignments, chosen_labels, strict=True):
            if chosen not in labels:
                raise ValueError(f"adapter returned undeclared choice: {chosen!r}")
            semantic_index = assigned.index(chosen)
            rotation_choices.append(case.choices[semantic_index].choice_id)
        stable = (
            rotation_choices[0]
            if rotation_choices and len(set(rotation_choices)) == 1
            else None
        )
        results.append({
            "case_id": case.case_id,
            "case_digest": case.digest(),
            "family": case.family,
            "rotation_choices": rotation_choices,
            "parsed_choice": stable,
            "label_invariant": stable is not None,
            "expected_choice": case.expected_choice,
            "correct": stable == case.expected_choice,
        })

    summary = {
        "case_count": len(results),
        "correct_count": sum(1 for result in results if result["correct"]),
        "unstable_count": sum(1 for result in results if not result["label_invariant"]),
        "malformed_count": sum(1 for result in results if result["parsed_choice"] is None),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_mode": "permutation_balanced_choice_v1",
        "choice_protocol": protocol,
        "benchmark_digest": benchmark_digest(ordered_cases),
        "model": {"id": adapter.model_id, "digest": adapter.model_digest},
        "results": results,
        "summary": summary,
    }
    manifest["run_digest"] = _sha256_text(_canonical_json(manifest))
    return manifest


def run_presentation_invariance_choice_suite(
    cases: Sequence[BenchmarkCase],
    adapter: Any,
) -> dict[str, Any]:
    """Require semantic choice stability across cyclic labels and option order."""
    ordered_cases = tuple(cases)
    protocol = {
        "method": "cyclic_label_and_order_consensus",
        "version": 1,
        "base_selector": "forced_bracket_prefix_label_argmax",
        "selector_execution": "batch_if_supported",
    }
    results: list[dict[str, Any]] = []
    for case in ordered_cases:
        labels = tuple(choice.choice_id for choice in case.choices)
        count = len(labels)
        layouts = []
        for order_offset in range(count):
            order = tuple(range(count))[order_offset:] + tuple(range(count))[:order_offset]
            for label_offset in range(count):
                assigned = labels[label_offset:] + labels[:label_offset]
                layouts.append((order, assigned))
        message_batches = tuple(
            _choice_messages_with_layout(case, order, assigned)
            for order, assigned in layouts
        )
        choose_many = getattr(adapter, "choose_many", None)
        if callable(choose_many):
            chosen_labels = tuple(choose_many(message_batches, labels))
        else:
            chosen_labels = tuple(
                adapter.choose(messages, labels) for messages in message_batches
            )
        if len(chosen_labels) != len(layouts):
            raise ValueError("adapter returned wrong number of choices")

        presentation_choices: list[str] = []
        for (order, assigned), chosen in zip(layouts, chosen_labels, strict=True):
            if chosen not in labels:
                raise ValueError(f"adapter returned undeclared choice: {chosen!r}")
            position = assigned.index(chosen)
            semantic_index = order[position]
            presentation_choices.append(case.choices[semantic_index].choice_id)
        stable = (
            presentation_choices[0]
            if presentation_choices and len(set(presentation_choices)) == 1
            else None
        )
        results.append({
            "case_id": case.case_id,
            "case_digest": case.digest(),
            "family": case.family,
            "presentation_choices": presentation_choices,
            "parsed_choice": stable,
            "presentation_invariant": stable is not None,
            "expected_choice": case.expected_choice,
            "correct": stable == case.expected_choice,
        })

    summary = {
        "case_count": len(results),
        "correct_count": sum(1 for result in results if result["correct"]),
        "unstable_count": sum(
            1 for result in results if not result["presentation_invariant"]
        ),
        "malformed_count": sum(
            1 for result in results if result["parsed_choice"] is None
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_mode": "presentation_invariant_choice_v1",
        "choice_protocol": protocol,
        "benchmark_digest": benchmark_digest(ordered_cases),
        "model": {"id": adapter.model_id, "digest": adapter.model_digest},
        "results": results,
        "summary": summary,
    }
    manifest["run_digest"] = _sha256_text(_canonical_json(manifest))
    return manifest
