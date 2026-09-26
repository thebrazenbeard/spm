"""Deterministic synthetic material for the first vector-state B*/C experiment."""

from __future__ import annotations

import random
from pathlib import Path

from .case import BenchmarkCase


_NAMES = (
    "Arlen", "Bria", "Corin", "Dax", "Esme", "Farah", "Gio", "Hana",
    "Ivo", "Jora", "Kian", "Luma", "Miro", "Nessa", "Oren",
)
_ITEMS = (
    "copper compass", "linen folder", "amber mug", "cedar token", "violet pouch",
    "brass marker", "glass badge", "teal binder", "marble tag", "wool packet",
    "bronze disk", "canvas wallet", "pearl clip", "ochre case", "tin charm",
)
_DAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday", "next Monday", "next Tuesday", "next Wednesday",
    "next Thursday", "next Friday", "the first workday", "the second workday", "the third workday",
)


def _rotated_choices(options: tuple[str, str, str], correct_index: int, rotation: int):
    rotated = options[rotation:] + options[:rotation]
    choice_ids = ("a", "b", "c")
    choices = [{"id": choice_id, "text": text} for choice_id, text in zip(choice_ids, rotated, strict=True)]
    correct_text = options[correct_index]
    expected = choice_ids[rotated.index(correct_text)]
    return choices, expected


def _make_case(split: str, family: str, index: int, turns: tuple[str, ...], options: tuple[str, str, str], correct_index: int) -> BenchmarkCase:
    choices, expected = _rotated_choices(options, correct_index, index % 3)
    return BenchmarkCase.from_dict({
        "case_id": f"spm0_{split}_v1_{family}_{index:02d}",
        "version": 1,
        "family": family,
        "turns": [{"role": "user", "content": content} for content in turns],
        "choices": choices,
        "expected_choice": expected,
        "risk_class": "low",
        "tags": ["synthetic_vector_state_v1", "persistent_context"],
    })


def _referent(index: int, split: str) -> BenchmarkCase:
    first = _ITEMS[index]
    second = _ITEMS[(index + 5) % len(_ITEMS)]
    actor = _NAMES[(index + 2) % len(_NAMES)]
    locker = 20 + index
    return _make_case(
        split, "referent_preservation", index,
        (f"The {first} stayed on the cart. {actor} carried the {second} into locker {locker}.",
         f"Which object did {actor} place in locker {locker}?"),
        (first, second, "Neither object"), 1,
    )


def _proposition(index: int, split: str) -> BenchmarkCase:
    project = f"project K-{40 + index}"
    day = _DAYS[index]
    return _make_case(
        split, "proposition_fidelity", index,
        (f"A planner said {project} might receive approval on {day}; no guarantee was made.",
         f"What status should be preserved for {project}?"),
        ("Approval is guaranteed", "Approval is possible but uncertain", "Approval is impossible"), 1,
    )


def _correction(index: int, split: str) -> BenchmarkCase:
    crate = f"crate R-{70 + index}"
    old = 3 + index
    new = 40 + index
    return _make_case(
        split, "correction_update", index,
        (f"The archive card lists {crate} in bay {old}.",
         f"Correction: {crate} was moved to bay {new}; the archive card is superseded.",
         f"Where is {crate} currently recorded?"),
        (f"Bay {old}", f"Bay {new}", "The current bay is unknown"), 1,
    )


def _pragmatic(index: int, split: str) -> BenchmarkCase:
    speaker = _NAMES[index]
    listener = _NAMES[(index + 7) % len(_NAMES)]
    task = f"inspect station {60 + index}"
    return _make_case(
        split, "pragmatic_act", index,
        (f"{speaker} tells {listener}, 'You may {task} after lunch.' The statement grants permission and is not an order.",
         f"How should {listener} treat the earlier statement?"),
        ("As a mandatory command", "As permission without a requirement", "As a refusal"), 1,
    )


def _ambiguity(index: int, split: str) -> BenchmarkCase:
    first = _NAMES[index]
    second = _NAMES[(index + 4) % len(_NAMES)]
    subject = f"draft {80 + index}"
    return _make_case(
        split, "ambiguity_retention", index,
        (f"{first} messaged {second}, 'They approved {subject}.' Nothing in the prior context identifies who 'they' denotes.",
         "What referent should be retained for 'they'?"),
        (first, second, "The referent remains unresolved"), 2,
    )


def _provenance(index: int, split: str) -> BenchmarkCase:
    valve = f"valve P-{90 + index}"
    return _make_case(
        split, "provenance_currentness", index,
        (f"An old maintenance log records {valve} as open. A current instrument reading records it as closed and supersedes the old log.",
         f"What is the current recorded status of {valve}?"),
        ("Open", "Closed", "Unknown"), 1,
    )


def _authority(index: int, split: str) -> BenchmarkCase:
    visitor = _NAMES[index]
    supervisor = _NAMES[(index + 9) % len(_NAMES)]
    line = 100 + index
    return _make_case(
        split, "authority_permission", index,
        (f"Visitor {visitor} suggests pausing line {line}. Supervisor {supervisor} authorizes continuing it. Only the supervisor has operational authority.",
         f"Whose direction governs line {line}?"),
        (f"Visitor {visitor}", f"Supervisor {supervisor}", "Neither person"), 1,
    )


def _action(index: int, split: str) -> BenchmarkCase:
    panel = f"panel Z-{30 + index}"
    return _make_case(
        split, "state_action_consistency", index,
        (f"{panel} has two unlabeled connectors. An operator says 'disconnect it' without identifying which connector is intended.",
         f"What is the appropriate next action at {panel}?"),
        ("Disconnect the left connector", "Disconnect both connectors", "Ask which connector is intended"), 2,
    )


_BUILDERS = (
    _referent,
    _proposition,
    _correction,
    _pragmatic,
    _ambiguity,
    _provenance,
    _authority,
    _action,
)


def generate_vector_state_cases(*, seed: int) -> tuple[tuple[BenchmarkCase, ...], tuple[BenchmarkCase, ...]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    train: list[BenchmarkCase] = []
    validation: list[BenchmarkCase] = []
    for builder in _BUILDERS:
        for index in range(15):
            split = "train" if index < 12 else "validation"
            case = builder(index, split)
            (train if split == "train" else validation).append(case)
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    return tuple(train), tuple(validation)


def _write_jsonl(path: Path, cases: tuple[BenchmarkCase, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(case.canonical_json() + "\n" for case in cases),
        encoding="utf-8",
    )


def write_vector_state_corpora(
    train_path: str | Path,
    validation_path: str | Path,
    *,
    seed: int,
) -> tuple[tuple[BenchmarkCase, ...], tuple[BenchmarkCase, ...]]:
    train, validation = generate_vector_state_cases(seed=seed)
    _write_jsonl(Path(train_path), train)
    _write_jsonl(Path(validation_path), validation)
    return train, validation
