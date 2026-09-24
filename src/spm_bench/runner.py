"""Deterministic benchmark execution and result manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from .adapter import ChatAdapter
from .case import BenchmarkCase


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a run manifest without environment-dependent formatting."""
    return _canonical_json(dict(manifest))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def benchmark_digest(cases: Sequence[BenchmarkCase]) -> str:
    """Digest the ordered semantic benchmark content, independent of file layout."""
    payload = [case.to_dict() for case in cases]
    return _sha256_text(_canonical_json(payload))


def _case_messages(case: BenchmarkCase) -> tuple[dict[str, str], ...]:
    messages = [turn.to_dict() for turn in case.turns]
    choices = "\n".join(
        f"[{choice.choice_id}] {choice.text}" for choice in case.choices
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Choices:\n"
                f"{choices}\n"
                "Reply with exactly one choice ID and no other text."
            ),
        }
    )
    return tuple(messages)


def _parse_choice(output: str, case: BenchmarkCase) -> str | None:
    candidate = output.strip()
    valid = {choice.choice_id for choice in case.choices}
    return candidate if candidate in valid else None


def run_suite(
    cases: Sequence[BenchmarkCase],
    adapter: ChatAdapter,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run cases in order and return an exact-subject deterministic manifest."""
    ordered_cases = tuple(cases)
    config = dict(generation_config)
    config_digest = _sha256_text(_canonical_json(config))

    results: list[dict[str, Any]] = []
    for case in ordered_cases:
        output = adapter.generate(_case_messages(case), config)
        if not isinstance(output, str):
            raise TypeError("adapter.generate must return str")
        parsed_choice = _parse_choice(output, case)
        results.append(
            {
                "case_id": case.case_id,
                "case_digest": case.digest(),
                "family": case.family,
                "output": output,
                "parsed_choice": parsed_choice,
                "expected_choice": case.expected_choice,
                "correct": parsed_choice == case.expected_choice,
            }
        )

    summary = {
        "case_count": len(results),
        "correct_count": sum(1 for result in results if result["correct"]),
        "malformed_count": sum(
            1 for result in results if result["parsed_choice"] is None
        ),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_digest": benchmark_digest(ordered_cases),
        "model": {
            "id": adapter.model_id,
            "digest": adapter.model_digest,
        },
        "generation_config": config,
        "generation_config_digest": config_digest,
        "results": results,
        "summary": summary,
    }
    manifest["run_digest"] = _sha256_text(_canonical_json(manifest))
    return manifest
