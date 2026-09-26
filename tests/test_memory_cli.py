from spm_bench.memory_cli import build_records, resolution_to_dict
from spm_bench.memory_specialist import MemoryResolution


def test_memory_cli_build_records_assigns_monotonic_sequences():
    records = build_records(("old value", "new value"), source_class="user")
    assert [record.sequence for record in records] == [1, 2]
    assert [record.content for record in records] == ["old value", "new value"]


def test_memory_cli_serializes_resolution():
    resolution = MemoryResolution(
        chosen_index=1,
        chosen_text="CURRENT",
        semantic_scores=(0.1, 0.8, 0.1),
        adapter_active=True,
        retrieval_receipt_digest="a" * 64,
        selected_record_digests=("b" * 64,),
        token_count=20,
        truncated=False,
    )
    payload = resolution_to_dict(resolution)
    assert payload["chosen_index"] == 1
    assert payload["chosen_text"] == "CURRENT"
    assert payload["adapter_active"] is True
    assert payload["semantic_scores"] == [0.1, 0.8, 0.1]
