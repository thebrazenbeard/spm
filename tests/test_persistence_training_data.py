from collections import Counter

from spm_bench.persistence_training_data import (
    FAMILIES,
    generate_persistence_cases,
)


def _choice_texts(cases):
    return {choice.text for case in cases for choice in case.choices}


def test_v2_split_sizes_family_balance_and_choice_balance():
    train = generate_persistence_cases(start_index=0, count=96, seed=20260918)
    validation = generate_persistence_cases(start_index=96, count=24, seed=20260918)
    holdout = generate_persistence_cases(start_index=120, count=24, seed=20260918)

    assert len(train) == 96
    assert len(validation) == 24
    assert len(holdout) == 24
    assert Counter(case.family for case in train) == {family: 12 for family in FAMILIES}
    assert Counter(case.family for case in validation) == {family: 3 for family in FAMILIES}
    assert Counter(case.family for case in holdout) == {family: 3 for family in FAMILIES}
    assert Counter(case.expected_choice for case in train) == {"a": 32, "b": 32, "c": 32}
    assert Counter(case.expected_choice for case in validation) == {"a": 8, "b": 8, "c": 8}
    assert Counter(case.expected_choice for case in holdout) == {"a": 8, "b": 8, "c": 8}


def test_v2_final_turn_is_information_insufficient_by_construction():
    cases = generate_persistence_cases(start_index=0, count=144, seed=20260918)
    for case in cases:
        final = case.turns[-1].content.casefold()
        assert len(case.turns) >= 3
        assert "memory_necessary" in case.tags
        assert "opaque_choices" in case.tags
        for choice in case.choices:
            assert choice.text.casefold() not in final
        expected = next(c.text for c in case.choices if c.choice_id == case.expected_choice)
        assert any(expected in turn.content for turn in case.turns[:-1])


def test_v2_splits_have_no_choice_value_overlap():
    train = generate_persistence_cases(start_index=0, count=96, seed=20260918)
    validation = generate_persistence_cases(start_index=96, count=24, seed=20260918)
    holdout = generate_persistence_cases(start_index=120, count=24, seed=20260918)
    assert _choice_texts(train).isdisjoint(_choice_texts(validation))
    assert _choice_texts(train).isdisjoint(_choice_texts(holdout))
    assert _choice_texts(validation).isdisjoint(_choice_texts(holdout))


def test_v2_generation_is_deterministic_and_index_sensitive():
    a = generate_persistence_cases(start_index=0, count=12, seed=20260918)
    b = generate_persistence_cases(start_index=0, count=12, seed=20260918)
    c = generate_persistence_cases(start_index=1, count=12, seed=20260918)
    assert [case.digest() for case in a] == [case.digest() for case in b]
    assert [case.digest() for case in a] != [case.digest() for case in c]
