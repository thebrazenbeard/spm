from __future__ import annotations

import pytest

from spm_bench.arm_m_memory import ArmMMemory, MemoryRecord


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def record(sequence: int, text: str, source: str = "user") -> MemoryRecord:
    return MemoryRecord.create(sequence=sequence, source_class=source, content=text)


def test_memory_record_identity_is_deterministic_and_rejects_model_output():
    one = record(1, "Blue notebook goes to Niko")
    two = record(1, "Blue notebook goes to Niko")
    assert one == two
    assert len(one.digest) == 64
    with pytest.raises(ValueError, match="exogenous"):
        record(2, "assistant generated answer", source="assistant")


def test_append_is_idempotent_and_conflicting_sequence_fails_closed():
    memory = ArmMMemory.empty().append(record(1, "alpha"), byte_ceiling=4096)
    assert memory.append(record(1, "alpha"), byte_ceiling=4096) == memory
    with pytest.raises(ValueError, match="sequence"):
        memory.append(record(1, "different"), byte_ceiling=4096)


def test_active_memory_is_byte_bounded_by_oldest_first_eviction():
    first = record(1, "one " * 20)
    second = record(2, "two " * 20)
    third = record(3, "three " * 20)
    one_record_ceiling = len(third.canonical_bytes()) + 8
    memory = ArmMMemory.empty()
    memory = memory.append(first, byte_ceiling=4096)
    memory = memory.append(second, byte_ceiling=4096)
    memory = memory.append(third, byte_ceiling=one_record_ceiling)
    assert [item.sequence for item in memory.records] == [3]
    assert memory.byte_size <= one_record_ceiling


def test_single_record_larger_than_ceiling_fails_before_training_evidence():
    with pytest.raises(ValueError, match="byte ceiling"):
        ArmMMemory.empty().append(record(1, "oversized observation"), byte_ceiling=8)


def test_retrieval_combines_recency_and_bm25_and_never_exceeds_token_budget():
    tokenizer = WordTokenizer()
    memory = ArmMMemory.empty()
    for sequence, text in (
        (1, "contract review archive"),
        (2, "blue notebook assigned to Niko"),
        (3, "weather unrelated note"),
        (4, "latest ordinary update"),
    ):
        memory = memory.append(record(sequence, text), byte_ceiling=4096)
    receipt = memory.retrieve(
        current_text="Where is the blue notebook now",
        tokenizer=tokenizer,
        max_retrieved_tokens=20,
    )
    selected_sequences = {item.sequence for item in receipt.selected_records}
    assert 4 in selected_sequences
    assert 2 in selected_sequences
    assert receipt.token_count <= 20
    assert receipt.rendered_text.startswith("MEMORY_CONTEXT_V1")


def test_retrieval_is_deterministic_and_can_exclude_current_context_records():
    tokenizer = WordTokenizer()
    memory = ArmMMemory.empty()
    for sequence, text in ((1, "alpha beta"), (2, "beta gamma"), (3, "gamma delta")):
        memory = memory.append(record(sequence, text), byte_ceiling=4096)
    kwargs = dict(
        current_text="beta question",
        tokenizer=tokenizer,
        max_retrieved_tokens=16,
        excluded_record_digests={memory.records[2].digest},
    )
    one = memory.retrieve(**kwargs)
    two = memory.retrieve(**kwargs)
    assert one == two
    assert memory.records[2].digest not in {item.digest for item in one.selected_records}
    assert one.receipt_digest == two.receipt_digest


def test_tight_budget_delivers_content_from_both_retrieval_lanes():
    tokenizer = WordTokenizer()
    memory = ArmMMemory.empty()
    for sequence, text in (
        (1, "contract review archive"),
        (2, "blue notebook assigned to Niko"),
        (3, "weather unrelated note"),
        (4, "latest ordinary update"),
    ):
        memory = memory.append(record(sequence, text), byte_ceiling=4096)
    receipt = memory.retrieve(
        current_text="Where is the blue notebook now",
        tokenizer=tokenizer,
        max_retrieved_tokens=12,
    )
    assert "RECENCY" in receipt.rendered_text
    assert "LEXICAL" in receipt.rendered_text
    assert "latest" in receipt.rendered_text
    assert "blue" in receipt.rendered_text
    assert receipt.token_count <= 12
