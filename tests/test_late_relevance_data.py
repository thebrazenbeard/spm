from __future__ import annotations

from spm_bench.late_relevance_data import generate_late_relevance_examples


def test_generator_is_deterministic_for_same_seed():
    first = generate_late_relevance_examples(seed=1729, count=20, max_chunks=5)
    second = generate_late_relevance_examples(seed=1729, count=20, max_chunks=5)

    assert first == second


def test_generator_changes_subject_for_different_seed():
    first = generate_late_relevance_examples(seed=1, count=10, max_chunks=5)
    second = generate_late_relevance_examples(seed=2, count=10, max_chunks=5)

    assert first != second


def test_generated_exact_cases_bind_answer_to_decisive_chunk():
    examples = generate_late_relevance_examples(
        seed=42,
        count=50,
        max_chunks=6,
    )

    assert len(examples) == 50
    for example in examples:
        assert example.exact_required is True
        assert 0 <= example.decisive_chunk < len(example.source_chunks)
        assert example.exact_answer in example.source_chunks[example.decisive_chunk]
        assert example.exact_answer not in example.compact_state
        assert len(example.source_chunks) <= 6


def test_generated_case_ids_are_unique():
    examples = generate_late_relevance_examples(
        seed=9,
        count=100,
        max_chunks=5,
    )

    assert len({item.case_id for item in examples}) == len(examples)


def test_generator_covers_multiple_late_relevance_families():
    examples = generate_late_relevance_examples(
        seed=20261002,
        count=100,
        max_chunks=5,
    )

    assert {
        "color",
        "number",
        "identifier",
        "supersession",
        "code_token",
    }.issubset({item.family for item in examples})
