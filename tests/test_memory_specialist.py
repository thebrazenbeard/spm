import pytest

from spm_bench.memory_specialist import (\n    memory_adapter_active,\n    verify_adapter_artifacts,\n    verify_base_inventory,\n)


def test_base_inventory_verification_binds_loaded_model_directory(tmp_path):
    from spm_bench.hf_adapter import local_inventory_digest

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model":"qualified"}', encoding="utf-8")
    (base / "weights.bin").write_bytes(b"qualified-base")

    expected = local_inventory_digest(base)
    verify_base_inventory(base, expected)

    (base / "weights.bin").write_bytes(b"substituted-base")
    with pytest.raises(RuntimeError, match="base model inventory digest mismatch"):
        verify_base_inventory(base, expected)


def test_adapter_artifact_verification_binds_weights_and_config(tmp_path):
    weights = tmp_path / "adapter_model.safetensors"
    config = tmp_path / "adapter_config.json"
    weights.write_bytes(b"qualified-weights")
    config.write_bytes(b'{"qualified":true}')

    import hashlib

    verify_adapter_artifacts(
        tmp_path,
        adapter_digest=hashlib.sha256(weights.read_bytes()).hexdigest(),
        adapter_config_digest=hashlib.sha256(config.read_bytes()).hexdigest(),
    )


def test_adapter_artifact_verification_rejects_weight_substitution(tmp_path):
    weights = tmp_path / "adapter_model.safetensors"
    config = tmp_path / "adapter_config.json"
    weights.write_bytes(b"substituted-weights")
    config.write_bytes(b'{"qualified":true}')

    import hashlib

    with pytest.raises(RuntimeError, match="adapter weights digest mismatch"):
        verify_adapter_artifacts(
            tmp_path,
            adapter_digest="0" * 64,
            adapter_config_digest=hashlib.sha256(config.read_bytes()).hexdigest(),
        )


def test_adapter_artifact_verification_rejects_config_substitution(tmp_path):
    weights = tmp_path / "adapter_model.safetensors"
    config = tmp_path / "adapter_config.json"
    weights.write_bytes(b"qualified-weights")
    config.write_bytes(b'{"substituted":true}')

    import hashlib

    with pytest.raises(RuntimeError, match="adapter config digest mismatch"):
        verify_adapter_artifacts(
            tmp_path,
            adapter_digest=hashlib.sha256(weights.read_bytes()).hexdigest(),
            adapter_config_digest="0" * 64,
        )


def test_memory_adapter_activation_depends_only_on_retrieval_state():
    assert memory_adapter_active(0) is False
    assert memory_adapter_active(1) is True
    assert memory_adapter_active(7) is True


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
def test_memory_adapter_activation_fails_closed_on_invalid_count(value):
    with pytest.raises((TypeError, ValueError)):
        memory_adapter_active(value)


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


class _FakeView:
    def __init__(self, active):
        self.memory_active = active

    def score_many_selected(self, message_batches, choice_ids):
        rows = []
        for messages in message_batches:
            choices_block = messages[-1]["content"].split("Choices:\n", 1)[1]
            target = None
            for line in choices_block.splitlines():
                if line.startswith("[") and "CURRENT" in line:
                    target = line[1:line.index("]")]
                    break
            if target is None:
                target = choice_ids[0]
            rows.append({
                choice: 0.9 if choice == target else 0.05
                for choice in choice_ids
            })
        return tuple(rows)


class _FakeRuntime:
    def __init__(self):
        self.tokenizer = _WordTokenizer()

    def view_for_retrieval(self, retrieved_record_count):
        return _FakeView(retrieved_record_count > 0)


def test_resolve_memory_choice_returns_semantic_winner_and_provenance():
    from spm_bench.arm_m_memory import MemoryRecord
    from spm_bench.memory_specialist import resolve_memory_choice

    runtime = _FakeRuntime()
    records = (
        MemoryRecord.create(
            sequence=1, source_class="user", content="The old status was STALE."
        ),
        MemoryRecord.create(
            sequence=2, source_class="user", content="Update: the status is now CURRENT."
        ),
    )
    result = resolve_memory_choice(
        runtime,
        memory_records=records,
        query="What is the current status?",
        choices=("STALE", "CURRENT", "UNKNOWN"),
    )

    assert result.chosen_index == 1
    assert result.chosen_text == "CURRENT"
    assert result.adapter_active is True
    assert len(result.selected_record_digests) == 2
    assert len(result.retrieval_receipt_digest) == 64
    assert result.semantic_scores[1] > result.semantic_scores[0]


def test_resolve_memory_choice_without_records_fails_closed_to_base_view():
    from spm_bench.memory_specialist import resolve_memory_choice

    result = resolve_memory_choice(
        _FakeRuntime(),
        memory_records=(),
        query="Choose one.",
        choices=("CURRENT", "OTHER", "UNKNOWN"),
    )

    assert result.adapter_active is False
    assert result.selected_record_digests == ()
