from __future__ import annotations

import pytest

from spm_bench.interventions import (
    freeze_state,
    reset_state,
    scramble_state,
    substitute_state,
)
from spm_bench.state_runtime import CheckpointEnvelope, canonical_prefix_digest


MODEL = "1" * 64
SCHEMA = "2" * 64


def checkpoint(state: bytes, *, prefix=(), model=MODEL, schema=SCHEMA, sequence=0, parent=None):
    return CheckpointEnvelope.create(
        model_digest=model,
        schema_digest=schema,
        prefix_digest=canonical_prefix_digest(prefix),
        parent_digest=parent,
        sequence=sequence,
        state_payload=state,
    )


def target_prefix(text: str):
    return ({"role": "user", "content": text},)


def test_freeze_preserves_payload_but_binds_new_prefix_and_source():
    source = checkpoint(b"semantic-state")
    prefix = target_prefix("new turn")

    result = freeze_state(source, prefix)

    assert result.kind == "freeze"
    assert result.source_checkpoint_digest == source.digest
    assert result.target_prefix_digest == canonical_prefix_digest(prefix)
    assert result.state_payload == source.state_payload
    assert result.state_payload_digest == source.state_payload_digest
    assert result.digest != source.digest
    assert source.state_payload == b"semantic-state"


def test_substitute_uses_donor_payload_without_mutating_sources():
    source = checkpoint(b"source")
    donor = checkpoint(b"donor")
    prefix = target_prefix("counterfactual")

    result = substitute_state(source, donor, prefix)

    assert result.kind == "substitute"
    assert result.state_payload == b"donor"
    assert result.descriptor["donor_checkpoint_digest"] == donor.digest
    assert source.state_payload == b"source"
    assert donor.state_payload == b"donor"


def test_substitute_rejects_cross_subject_donor():
    source = checkpoint(b"source")
    donor = checkpoint(b"donor", model="9" * 64)
    with pytest.raises(ValueError, match="subject"):
        substitute_state(source, donor, target_prefix("x"))


def test_reset_uses_root_payload_and_requires_root_subject_match():
    source = checkpoint(b"current")
    root = checkpoint(b"root")
    result = reset_state(source, root, target_prefix("reset"))
    assert result.kind == "reset"
    assert result.state_payload == b"root"
    assert result.descriptor["root_checkpoint_digest"] == root.digest

    wrong_root = checkpoint(b"root", schema="8" * 64)
    with pytest.raises(ValueError, match="subject"):
        reset_state(source, wrong_root, target_prefix("reset"))


def test_scramble_is_deterministic_changes_order_and_preserves_byte_multiset():
    source = checkpoint(bytes(range(32)))
    prefix = target_prefix("scramble")
    first = scramble_state(source, prefix, seed="trial-7")
    second = scramble_state(source, prefix, seed="trial-7")
    other = scramble_state(source, prefix, seed="trial-8")

    assert first == second
    assert first.state_payload != source.state_payload
    assert sorted(first.state_payload) == sorted(source.state_payload)
    assert first.state_payload != other.state_payload


def test_intervention_digest_binds_source_prefix_kind_and_descriptor():
    source = checkpoint(b"state")
    a = scramble_state(source, target_prefix("one"), seed="s")
    b = scramble_state(source, target_prefix("two"), seed="s")
    c = scramble_state(source, target_prefix("one"), seed="t")

    assert a.digest != b.digest
    assert a.digest != c.digest
    assert len(a.digest) == 64
    assert len(a.state_payload_digest) == 64
