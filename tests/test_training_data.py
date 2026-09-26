from __future__ import annotations

from collections import Counter
from pathlib import Path

from spm_bench.case import load_jsonl_cases
from spm_bench.training_data import generate_vector_state_cases, write_vector_state_corpora


EXPECTED_FAMILIES = {
    "referent_preservation",
    "proposition_fidelity",
    "correction_update",
    "pragmatic_act",
    "ambiguity_retention",
    "provenance_currentness",
    "authority_permission",
    "state_action_consistency",
}


def test_vector_state_training_split_is_fixed_and_balanced():
    train, validation = generate_vector_state_cases(seed=20260917)
    assert len(train) == 96
    assert len(validation) == 24
    assert Counter(case.family for case in train) == {family: 12 for family in EXPECTED_FAMILIES}
    assert Counter(case.family for case in validation) == {family: 3 for family in EXPECTED_FAMILIES}
    assert all(len(case.turns) >= 2 for case in train + validation)


def test_training_corpus_has_no_exact_public_dev_case_or_turn_leakage():
    train, validation = generate_vector_state_cases(seed=20260917)
    public = load_jsonl_cases(Path("benchmarks") / "spm_v0_public_dev.jsonl")
    generated = train + validation

    assert {case.case_id for case in generated}.isdisjoint(case.case_id for case in public)
    assert {case.digest() for case in generated}.isdisjoint(case.digest() for case in public)
    generated_turns = {turn.content for case in generated for turn in case.turns}
    public_turns = {turn.content for case in public for turn in case.turns}
    assert generated_turns.isdisjoint(public_turns)


def test_training_generation_is_deterministic_and_materializes_round_trip(tmp_path):
    first = generate_vector_state_cases(seed=20260917)
    second = generate_vector_state_cases(seed=20260917)
    assert [case.digest() for split in first for case in split] == [
        case.digest() for split in second for case in split
    ]

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    write_vector_state_corpora(train_path, validation_path, seed=20260917)
    assert load_jsonl_cases(train_path) == first[0]
    assert load_jsonl_cases(validation_path) == first[1]


def test_choice_positions_are_balanced_within_each_family():
    train, validation = generate_vector_state_cases(seed=20260917)
    for cases in (train, validation):
        by_family: dict[str, list[str]] = {}
        for case in cases:
            by_family.setdefault(case.family, []).append(case.expected_choice)
        for expected in by_family.values():
            counts = Counter(expected)
            assert max(counts.values()) - min(counts.values()) <= 1
