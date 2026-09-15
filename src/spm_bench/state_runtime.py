"""Content-addressed opaque-state runtime primitives for SPM-0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
import threading
from typing import Any


_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def payload_digest(payload: bytes) -> str:
    if not isinstance(payload, bytes):
        raise TypeError("state payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_prefix_digest(messages: Sequence[Mapping[str, str]]) -> str:
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("each message must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("each message requires string role and content")
        normalized.append({"role": role, "content": content})
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointEnvelope:
    model_digest: str
    schema_digest: str
    prefix_digest: str
    parent_digest: str | None
    sequence: int
    state_payload_digest: str
    state_payload: bytes
    digest: str

    @classmethod
    def create(
        cls,
        *,
        model_digest: str,
        schema_digest: str,
        prefix_digest: str,
        parent_digest: str | None,
        sequence: int,
        state_payload: bytes,
    ) -> "CheckpointEnvelope":
        model = _require_digest(model_digest, "model_digest")
        schema = _require_digest(schema_digest, "schema_digest")
        prefix = _require_digest(prefix_digest, "prefix_digest")
        parent = None if parent_digest is None else _require_digest(parent_digest, "parent_digest")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        state_hash = payload_digest(state_payload)
        identity = {
            "model_digest": model,
            "schema_digest": schema,
            "prefix_digest": prefix,
            "parent_digest": parent,
            "sequence": sequence,
            "state_payload_digest": state_hash,
            "state_payload_hex": state_payload.hex(),
        }
        digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
        return cls(
            model_digest=model,
            schema_digest=schema,
            prefix_digest=prefix,
            parent_digest=parent,
            sequence=sequence,
            state_payload_digest=state_hash,
            state_payload=state_payload,
            digest=digest,
        )


class InMemoryCheckpointStore:
    """Content-addressed checkpoints with explicit lineage CAS semantics."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, CheckpointEnvelope] = {}
        self._heads: dict[str, str] = {}
        self._idempotency: dict[tuple[str, str], tuple[str, str | None]] = {}
        self._lock = threading.RLock()

    def get(self, digest: str) -> CheckpointEnvelope | None:
        with self._lock:
            return self._checkpoints.get(digest)

    def head(self, lineage_id: str) -> str | None:
        with self._lock:
            return self._heads.get(lineage_id)

    def fork_lineage(self, lineage_id: str, from_digest: str) -> None:
        if not isinstance(lineage_id, str) or not lineage_id:
            raise ValueError("lineage_id must be non-empty")
        with self._lock:
            if lineage_id in self._heads:
                raise ValueError("lineage already exists")
            if from_digest not in self._checkpoints:
                raise ValueError("fork source checkpoint is unknown")
            self._heads[lineage_id] = from_digest

    def commit(
        self,
        checkpoint: CheckpointEnvelope,
        *,
        lineage_id: str,
        expected_head: str | None,
        idempotency_key: str,
    ) -> CheckpointEnvelope:
        if not isinstance(lineage_id, str) or not lineage_id:
            raise ValueError("lineage_id must be non-empty")
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise ValueError("idempotency_key must be non-empty")
        rebuilt = CheckpointEnvelope.create(
            model_digest=checkpoint.model_digest,
            schema_digest=checkpoint.schema_digest,
            prefix_digest=checkpoint.prefix_digest,
            parent_digest=checkpoint.parent_digest,
            sequence=checkpoint.sequence,
            state_payload=checkpoint.state_payload,
        )
        if rebuilt != checkpoint:
            raise ValueError("checkpoint envelope integrity mismatch")

        with self._lock:
            scoped_key = (lineage_id, idempotency_key)
            prior = self._idempotency.get(scoped_key)
            identity = (checkpoint.digest, expected_head)
            if prior is not None:
                if prior != identity:
                    raise ValueError("idempotency key reused for a different transition")
                stored = self._checkpoints.get(checkpoint.digest)
                if stored is None:
                    raise ValueError("idempotent transition checkpoint was evicted")
                return stored

            current_head = self._heads.get(lineage_id)
            if current_head != expected_head:
                raise ValueError("stale head: expected lineage head does not match")
            if checkpoint.parent_digest != expected_head:
                raise ValueError("checkpoint parent does not match expected head")
            if expected_head is None:
                if checkpoint.sequence != 0:
                    raise ValueError("root sequence must be zero")
            else:
                parent = self._checkpoints.get(expected_head)
                if parent is None:
                    raise ValueError("expected parent checkpoint is unknown")
                if checkpoint.model_digest != parent.model_digest or checkpoint.schema_digest != parent.schema_digest:
                    raise ValueError("checkpoint subject does not match parent subject")
                if checkpoint.sequence != parent.sequence + 1:
                    raise ValueError("checkpoint sequence must advance parent by exactly one")

            existing = self._checkpoints.get(checkpoint.digest)
            if existing is not None and existing != checkpoint:
                raise ValueError("checkpoint digest collision")
            self._checkpoints.setdefault(checkpoint.digest, checkpoint)
            self._heads[lineage_id] = checkpoint.digest
            self._idempotency[scoped_key] = identity
            return self._checkpoints[checkpoint.digest]

    def resolve(self, model_digest: str, schema_digest: str, prefix_digest: str) -> CheckpointEnvelope | None:
        model = _require_digest(model_digest, "model_digest")
        schema = _require_digest(schema_digest, "schema_digest")
        prefix = _require_digest(prefix_digest, "prefix_digest")
        with self._lock:
            matches = [
                checkpoint for checkpoint in self._checkpoints.values()
                if checkpoint.model_digest == model
                and checkpoint.schema_digest == schema
                and checkpoint.prefix_digest == prefix
            ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("ambiguous checkpoint resolution for model/schema/prefix")
        return matches[0]

    def evict(self, digest: str) -> None:
        with self._lock:
            self._checkpoints.pop(digest, None)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            checkpoints = []
            for checkpoint in sorted(self._checkpoints.values(), key=lambda item: item.digest):
                checkpoints.append({
                    "model_digest": checkpoint.model_digest,
                    "schema_digest": checkpoint.schema_digest,
                    "prefix_digest": checkpoint.prefix_digest,
                    "parent_digest": checkpoint.parent_digest,
                    "sequence": checkpoint.sequence,
                    "state_payload_digest": checkpoint.state_payload_digest,
                    "state_payload_hex": checkpoint.state_payload.hex(),
                    "digest": checkpoint.digest,
                })
            return {
                "schema_version": 1,
                "checkpoints": checkpoints,
                "heads": dict(sorted(self._heads.items())),
                "idempotency": [
                    {
                        "lineage_id": lineage_id,
                        "idempotency_key": idempotency_key,
                        "checkpoint_digest": digest,
                        "expected_head": expected_head,
                    }
                    for (lineage_id, idempotency_key), (digest, expected_head)
                    in sorted(self._idempotency.items())
                ],
            }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "InMemoryCheckpointStore":
        if not isinstance(snapshot, Mapping) or snapshot.get("schema_version") != 1:
            raise ValueError("unsupported checkpoint-store snapshot")
        store = cls()
        checkpoints = snapshot.get("checkpoints")
        heads = snapshot.get("heads")
        idempotency = snapshot.get("idempotency")
        if not isinstance(checkpoints, list) or not isinstance(heads, Mapping) or not isinstance(idempotency, list):
            raise ValueError("malformed checkpoint-store snapshot")

        for item in checkpoints:
            if not isinstance(item, Mapping):
                raise ValueError("malformed checkpoint snapshot entry")
            try:
                payload = bytes.fromhex(item["state_payload_hex"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid checkpoint payload encoding") from error
            rebuilt = CheckpointEnvelope.create(
                model_digest=item.get("model_digest"),
                schema_digest=item.get("schema_digest"),
                prefix_digest=item.get("prefix_digest"),
                parent_digest=item.get("parent_digest"),
                sequence=item.get("sequence"),
                state_payload=payload,
            )
            if rebuilt.digest != item.get("digest") or rebuilt.state_payload_digest != item.get("state_payload_digest"):
                raise ValueError("checkpoint snapshot integrity mismatch")
            if rebuilt.digest in store._checkpoints:
                raise ValueError("duplicate checkpoint digest in snapshot")
            store._checkpoints[rebuilt.digest] = rebuilt

        for checkpoint in store._checkpoints.values():
            if checkpoint.parent_digest is None:
                if checkpoint.sequence != 0:
                    raise ValueError("snapshot root sequence must be zero")
                continue
            parent = store._checkpoints.get(checkpoint.parent_digest)
            if parent is None:
                raise ValueError("snapshot checkpoint parent is missing")
            if checkpoint.model_digest != parent.model_digest or checkpoint.schema_digest != parent.schema_digest:
                raise ValueError("snapshot checkpoint subject does not match parent subject")
            if checkpoint.sequence != parent.sequence + 1:
                raise ValueError("snapshot checkpoint sequence is inconsistent with parent")

        for lineage_id, digest in heads.items():
            if not isinstance(lineage_id, str) or not lineage_id or digest not in store._checkpoints:
                raise ValueError("snapshot head references unknown checkpoint")
            store._heads[lineage_id] = digest

        for item in idempotency:
            if not isinstance(item, Mapping):
                raise ValueError("malformed snapshot idempotency record")
            lineage_id = item.get("lineage_id")
            key = item.get("idempotency_key")
            digest = item.get("checkpoint_digest")
            expected_head = item.get("expected_head")
            if not isinstance(lineage_id, str) or not lineage_id or not isinstance(key, str) or not key:
                raise ValueError("malformed snapshot idempotency record")
            if digest not in store._checkpoints:
                raise ValueError("snapshot idempotency references unknown checkpoint")
            if expected_head is not None and expected_head not in store._checkpoints:
                raise ValueError("snapshot idempotency expected head is unknown")
            store._idempotency[(lineage_id, key)] = (digest, expected_head)
        return store


class LineageRuntime:
    """Model/schema-bound lifecycle over an opaque deterministic state updater."""

    def __init__(self, store, *, model_digest: str, schema_digest: str, updater) -> None:
        self.store = store
        self.model_digest = _require_digest(model_digest, "model_digest")
        self.schema_digest = _require_digest(schema_digest, "schema_digest")
        if not callable(updater):
            raise ValueError("updater must be callable")
        self.updater = updater

    def _validate_subject(self, checkpoint: CheckpointEnvelope) -> None:
        if checkpoint.model_digest != self.model_digest:
            raise ValueError("checkpoint model digest does not match runtime")
        if checkpoint.schema_digest != self.schema_digest:
            raise ValueError("checkpoint schema digest does not match runtime")

    def root(self, lineage_id: str, *, initial_state: bytes, idempotency_key: str) -> CheckpointEnvelope:
        checkpoint = CheckpointEnvelope.create(
            model_digest=self.model_digest,
            schema_digest=self.schema_digest,
            prefix_digest=canonical_prefix_digest(()),
            parent_digest=None,
            sequence=0,
            state_payload=initial_state,
        )
        return self.store.commit(
            checkpoint, lineage_id=lineage_id, expected_head=None,
            idempotency_key=idempotency_key,
        )

    def advance(
        self,
        lineage_id: str,
        *,
        expected_parent: str,
        prefix: Sequence[Mapping[str, str]],
        idempotency_key: str,
    ) -> CheckpointEnvelope:
        parent = self.store.get(expected_parent)
        if parent is None:
            raise ValueError("expected parent checkpoint is unknown")
        self._validate_subject(parent)
        next_state = self.updater(parent.state_payload, prefix)
        if not isinstance(next_state, bytes):
            raise TypeError("updater must return bytes")
        checkpoint = CheckpointEnvelope.create(
            model_digest=self.model_digest,
            schema_digest=self.schema_digest,
            prefix_digest=canonical_prefix_digest(prefix),
            parent_digest=parent.digest,
            sequence=parent.sequence + 1,
            state_payload=next_state,
        )
        return self.store.commit(
            checkpoint, lineage_id=lineage_id,
            expected_head=expected_parent, idempotency_key=idempotency_key,
        )

    def resume(self, checkpoint_digest: str) -> CheckpointEnvelope:
        checkpoint = self.store.get(checkpoint_digest)
        if checkpoint is None:
            raise ValueError("checkpoint is unknown")
        self._validate_subject(checkpoint)
        return checkpoint

    def reset(self, root_digest: str) -> CheckpointEnvelope:
        checkpoint = self.resume(root_digest)
        if checkpoint.sequence != 0 or checkpoint.parent_digest is not None:
            raise ValueError("reset target is not a root checkpoint")
        return checkpoint

    def replay(
        self,
        lineage_id: str,
        root_digest: str,
        prefixes: Sequence[Sequence[Mapping[str, str]]],
        *,
        idempotency_prefix: str,
    ) -> CheckpointEnvelope:
        root = self.reset(root_digest)
        self.store.fork_lineage(lineage_id, root.digest)
        current = root
        for index, prefix in enumerate(prefixes, start=1):
            current = self.advance(
                lineage_id,
                expected_parent=current.digest,
                prefix=prefix,
                idempotency_key=f"{idempotency_prefix}:{index}",
            )
        return current
