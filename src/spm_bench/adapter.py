"""Model adapter protocol for deterministic SPM benchmark execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ChatAdapter(Protocol):
    """Minimal interface required by the benchmark runner."""

    model_id: str
    model_digest: str

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, Any],
    ) -> str:
        """Generate one response for a canonical benchmark prompt."""
        ...
