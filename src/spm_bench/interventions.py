"""Non-mutating causal intervention artifacts for SPM state experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

from .state_runtime import CheckpointEnvelope, canonical_prefix_digest, payload_digest


@dataclass(frozen=True, slots=True)
class InterventionResult:
    kind: str
    source_checkpoint_digest: str
    model_digest: str
    schema_digest: str
    target_prefix_digest: str
    descriptor: Mapping[str, str]
    state_payload_digest: str
    state_payload: bytes
    digest: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _same_subject(left: CheckpointEnvelope, right: CheckpointEnvelope) -> None:
    if left.model_digest != right.model_digest or left.schema_digest != right.schema_digest:
        raise ValueError("intervention checkpoints must have the same subject")


def _result(
    kind: str,
    source: CheckpointEnvelope,
    target_prefix: Sequence[Mapping[str, Any]],
    state_payload: bytes,
    descriptor: Mapping[str, str],
) -> InterventionResult:
    prefix_digest = canonical_prefix_digest(target_prefix)
    state_digest = payload_digest(state_payload)
    frozen_descriptor = dict(sorted(descriptor.items()))
    identity = {
        "kind": kind,
        "source_checkpoint_digest": source.digest,
        "model_digest": source.model_digest,
        "schema_digest": source.schema_digest,
        "target_prefix_digest": prefix_digest,
        "descriptor": frozen_descriptor,
        "state_payload_digest": state_digest,
        "state_payload_hex": state_payload.hex(),
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return InterventionResult(
        kind=kind,
        source_checkpoint_digest=source.digest,
        model_digest=source.model_digest,
        schema_digest=source.schema_digest,
        target_prefix_digest=prefix_digest,
        descriptor=MappingProxyType(frozen_descriptor),
        state_payload_digest=state_digest,
        state_payload=state_payload,
        digest=digest,
    )


def freeze_state(
    source: CheckpointEnvelope,
    target_prefix: Sequence[Mapping[str, Any]],
) -> InterventionResult:
    return _result("freeze", source, target_prefix, source.state_payload, {})


def substitute_state(
    source: CheckpointEnvelope,
    donor: CheckpointEnvelope,
    target_prefix: Sequence[Mapping[str, Any]],
) -> InterventionResult:
    _same_subject(source, donor)
    return _result(
        "substitute",
        source,
        target_prefix,
        donor.state_payload,
        {"donor_checkpoint_digest": donor.digest},
    )


def reset_state(
    source: CheckpointEnvelope,
    root: CheckpointEnvelope,
    target_prefix: Sequence[Mapping[str, Any]],
) -> InterventionResult:
    _same_subject(source, root)
    if root.parent_digest is not None or root.sequence != 0:
        raise ValueError("reset intervention requires a root checkpoint")
    return _result(
        "reset",
        source,
        target_prefix,
        root.state_payload,
        {"root_checkpoint_digest": root.digest},
    )


def scramble_state(
    source: CheckpointEnvelope,
    target_prefix: Sequence[Mapping[str, Any]],
    *,
    seed: str,
) -> InterventionResult:
    if not isinstance(seed, str) or not seed:
        raise ValueError("scramble seed must be a non-empty string")
    indices = list(range(len(source.state_payload)))
    seed_bytes = seed.encode("utf-8")
    indices.sort(
        key=lambda index: hashlib.sha256(
            seed_bytes + index.to_bytes(8, "big", signed=False)
        ).digest()
    )
    scrambled = bytes(source.state_payload[index] for index in indices)
    return _result(
        "scramble",
        source,
        target_prefix,
        scrambled,
        {"algorithm": "sha256-index-permutation-v1", "seed": seed},
    )
