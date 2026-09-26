"""Deterministic persistence-required training cases for SPM vector-state V2."""

from __future__ import annotations

import hashlib

from .case import BenchmarkCase, BenchmarkChoice, BenchmarkTurn

FAMILIES = (
    "initial_assignment",
    "supersession",
    "correction",
    "alias",
    "preference",
    "authority_change",
    "decision_finality",
    "project_state",
)

_LABELS = ("a", "b", "c")


def _opaque(seed: int, index: int, role: str, prefix: str) -> str:
    digest = hashlib.sha256(f"{seed}:{index}:{role}".encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}-{digest}"


def _place_choices(index: int, correct: str, other_one: str, other_two: str):
    expected_position = index % 3
    values = [None, None, None]
    values[expected_position] = correct
    remaining = [position for position in range(3) if position != expected_position]
    values[remaining[0]] = other_one
    values[remaining[1]] = other_two
    choices = tuple(
        BenchmarkChoice(choice_id=label, text=value)
        for label, value in zip(_LABELS, values, strict=True)
    )
    return choices, _LABELS[expected_position]


def _case(index: int, seed: int) -> BenchmarkCase:
    family = FAMILIES[index % len(FAMILIES)]
    subject = _opaque(seed, index, "subject", "SUBJ")
    correct = _opaque(seed, index, "correct", "VAL")
    stale = _opaque(seed, index, "stale", "VAL")
    distractor = _opaque(seed, index, "distractor", "VAL")
    noise_subject = _opaque(seed, index, "noise_subject", "NOISE")
    noise_value = _opaque(seed, index, "noise_value", "AUX")
    alias = _opaque(seed, index, "alias", "ALIAS")

    if family == "initial_assignment":
        prefix = (
            f"The registered value for {subject} is {correct}.",
            f"Unrelated record: {noise_subject} is paired with {noise_value}.",
        )
        query = f"What registered value belongs to {subject}?"
        other_one, other_two = stale, distractor
    elif family == "supersession":
        prefix = (
            f"The earlier registered value for {subject} is {stale}.",
            f"Update: replace {stale} with {correct} for {subject}. {correct} is current and {stale} is obsolete.",
        )
        query = f"What is the current registered value for {subject}?"
        other_one, other_two = stale, distractor
    elif family == "correction":
        prefix = (
            f"I previously recorded {subject} as {stale}.",
            f"Correction: that record was wrong. The correct value for {subject} is {correct}, not {stale}.",
        )
        query = f"What value should now be used for {subject}?"
        other_one, other_two = stale, distractor
    elif family == "alias":
        prefix = (
            f"The alias {alias} refers to device code {correct}.",
            f"Unrelated record: {noise_subject} is paired with {noise_value}.",
        )
        query = f"Which device code does alias {alias} refer to?"
        other_one, other_two = stale, distractor
    elif family == "preference":
        prefix = (
            f"For task {subject}, the selected format code is {correct}.",
            f"Unrelated record: {noise_subject} is paired with {noise_value}.",
        )
        query = f"What format code was selected for task {subject}?"
        other_one, other_two = stale, distractor
    elif family == "authority_change":
        prefix = (
            f"Authority code {stale} currently controls {subject}.",
            f"Authority changed: {correct} now controls {subject}. {stale} no longer has authority.",
        )
        query = f"Which authority code currently controls {subject}?"
        other_one, other_two = stale, distractor
    elif family == "decision_finality":
        prefix = (
            f"The tentative decision code for {subject} is {stale}.",
            f"Final decision: use {correct} for {subject}. The tentative code {stale} is abandoned.",
        )
        query = f"What is the final decision code for {subject}?"
        other_one, other_two = stale, distractor
    elif family == "project_state":
        prefix = (
            f"The earlier status code for {subject} is {stale}.",
            f"Current status update: {subject} is now {correct}; the earlier {stale} status is stale.",
        )
        query = f"What is the current status code for {subject}?"
        other_one, other_two = stale, distractor
    else:  # pragma: no cover
        raise AssertionError(family)

    choices, expected_choice = _place_choices(index, correct, other_one, other_two)
    turns = tuple(BenchmarkTurn(role="user", content=text) for text in (*prefix, query))
    return BenchmarkCase(
        case_id=f"spm0_persistence_v2_{index:04d}",
        version=2,
        family=family,
        turns=turns,
        choices=choices,
        expected_choice=expected_choice,
        risk_class="low",
        tags=("synthetic_vector_state_v2", "memory_necessary", "opaque_choices"),
    )


def generate_persistence_cases(*, start_index: int, count: int, seed: int) -> tuple[BenchmarkCase, ...]:
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return tuple(_case(index, seed) for index in range(start_index, start_index + count))
