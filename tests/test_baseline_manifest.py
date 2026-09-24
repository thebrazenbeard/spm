from __future__ import annotations

import pytest

from spm_bench.baseline_manifest import ArtifactIdentity, BaselineSubject


MODEL_REV = "a" * 40
TOKENIZER_REV = "b" * 40
MODEL_INVENTORY = "c" * 64
TOKENIZER_INVENTORY = "d" * 64


def identity(name: str, *, revision: str | None = None, inventory: str | None = None):
    return ArtifactIdentity(
        artifact_id=name,
        immutable_revision=revision,
        inventory_digest=inventory,
    )


def test_identity_requires_immutable_revision_or_inventory_digest():
    with pytest.raises(ValueError, match="immutable identity"):
        identity("model-x")

    with pytest.raises(ValueError, match="immutable revision"):
        identity("model-x", revision="main")

    with pytest.raises(ValueError, match="inventory_digest"):
        identity("model-x", inventory="not-a-digest")


def test_subject_binds_model_tokenizer_and_generation_config():
    subject = BaselineSubject(
        model=identity("model-x", revision=MODEL_REV),
        tokenizer=identity("tokenizer-x", revision=TOKENIZER_REV),
        parameter_count=3_000_000_000,
        generation_config={"do_sample": False, "max_new_tokens": 4},
    )

    payload = subject.to_dict()
    assert payload["model"]["artifact_id"] == "model-x"
    assert payload["tokenizer"]["artifact_id"] == "tokenizer-x"
    assert payload["parameter_count"] == 3_000_000_000
    assert payload["generation_config"] == {"do_sample": False, "max_new_tokens": 4}
    assert len(subject.digest()) == 64


def test_subject_digest_is_deterministic_across_mapping_order():
    first = BaselineSubject(
        model=identity("model-x", inventory=MODEL_INVENTORY),
        tokenizer=identity("tokenizer-x", inventory=TOKENIZER_INVENTORY),
        parameter_count=None,
        generation_config={"max_new_tokens": 4, "do_sample": False},
    )
    second = BaselineSubject(
        model=identity("model-x", inventory=MODEL_INVENTORY),
        tokenizer=identity("tokenizer-x", inventory=TOKENIZER_INVENTORY),
        parameter_count=None,
        generation_config={"do_sample": False, "max_new_tokens": 4},
    )

    assert first.digest() == second.digest()
    assert first.canonical_json() == second.canonical_json()


def test_subject_rejects_invalid_parameter_count_and_generation_config():
    model = identity("model-x", revision=MODEL_REV)
    tokenizer = identity("tokenizer-x", revision=TOKENIZER_REV)

    with pytest.raises(ValueError, match="parameter_count"):
        BaselineSubject(
            model=model,
            tokenizer=tokenizer,
            parameter_count=0,
            generation_config={"do_sample": False},
        )

    with pytest.raises(ValueError, match="generation_config"):
        BaselineSubject(
            model=model,
            tokenizer=tokenizer,
            parameter_count=1,
            generation_config="not-a-mapping",
        )


def test_identity_can_bind_both_revision_and_inventory_digest():
    item = identity(
        "model-x",
        revision=MODEL_REV,
        inventory=MODEL_INVENTORY,
    )

    assert item.to_dict() == {
        "artifact_id": "model-x",
        "immutable_revision": MODEL_REV,
        "inventory_digest": MODEL_INVENTORY,
    }
