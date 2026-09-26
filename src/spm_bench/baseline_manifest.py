"""Immutable subject identities for reproducible baseline evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any


_HEX_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    artifact_id: str
    immutable_revision: str | None = None
    inventory_digest: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.artifact_id, "artifact_id")
        if self.immutable_revision is not None and not _HEX_REVISION.fullmatch(self.immutable_revision):
            raise ValueError("immutable revision must be a 40-64 character hexadecimal revision")
        if self.inventory_digest is not None and not _HEX_SHA256.fullmatch(self.inventory_digest):
            raise ValueError("inventory_digest must be a 64-character hexadecimal SHA-256")
        if self.immutable_revision is None and self.inventory_digest is None:
            raise ValueError("artifact requires immutable identity via revision or inventory_digest")

    def to_dict(self) -> dict[str, str]:
        payload = {"artifact_id": self.artifact_id}
        if self.immutable_revision is not None:
            payload["immutable_revision"] = self.immutable_revision.lower()
        if self.inventory_digest is not None:
            payload["inventory_digest"] = self.inventory_digest.lower()
        return payload


@dataclass(frozen=True, slots=True)
class BaselineSubject:
    model: ArtifactIdentity
    tokenizer: ArtifactIdentity
    parameter_count: int | None
    generation_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.parameter_count is not None:
            if isinstance(self.parameter_count, bool) or not isinstance(self.parameter_count, int) or self.parameter_count <= 0:
                raise ValueError("parameter_count must be a positive integer or None")
        if not isinstance(self.generation_config, Mapping):
            raise ValueError("generation_config must be a mapping")
        try:
            json.dumps(dict(self.generation_config), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("generation_config must be JSON-serializable") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "parameter_count": self.parameter_count,
            "generation_config": dict(self.generation_config),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
