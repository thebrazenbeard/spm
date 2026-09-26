from spm_bench.continuity_data import generate_continuity_cases
from spm_bench.continuity_runner import build_condition_case


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))
    def decode(self, ids, skip_special_tokens=True):
        return " ".join("tok" for _ in ids)


def test_build_condition_case_reset_excludes_history():
    source=generate_continuity_cases()[0]
    built, receipt=build_condition_case(source, condition="RESET", tokenizer=None)
    assert receipt is None
    assert len(built.turns)==1
    assert all(event.content not in built.turns[0].content for event in source.events)
    assert built.expected_choice==source.expected_choice


def test_build_condition_case_full_history_contains_every_event():
    source=generate_continuity_cases()[1]
    built, receipt=build_condition_case(source, condition="FULL_HISTORY", tokenizer=None)
    assert receipt is None
    assert all(event.content in built.turns[0].content for event in source.events)


def test_build_condition_case_persistent_uses_memory_receipt():
    source=generate_continuity_cases()[2]
    built, receipt=build_condition_case(
        source, condition="PERSISTENT", tokenizer=TinyTokenizer(),
        byte_ceiling=8192, max_retrieved_tokens=512,
    )
    assert receipt is not None
    assert receipt.selected_records
    assert "MEMORY_CONTEXT_V1" in built.turns[0].content
    assert built.expected_choice==source.expected_choice
