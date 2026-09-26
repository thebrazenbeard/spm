"""Cross-session continuity and drift benchmark contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from .case import BenchmarkChoice


@dataclass(frozen=True, slots=True)
class ContinuityEvent:
    session: int
    source_class: str
    content: str

    @classmethod
    def from_dict(cls, data: Any) -> "ContinuityEvent":
        if not isinstance(data, Mapping):
            raise ValueError("continuity event must be an object")
        session = data.get("session")
        if isinstance(session, bool) or not isinstance(session, int) or session <= 0:
            raise ValueError("session must be a positive integer")
        source = data.get("source_class")
        content = data.get("content")
        if not isinstance(source, str) or not source:
            raise ValueError("source_class must be a non-empty string")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        return cls(session=session, source_class=source, content=content)

    def to_dict(self) -> dict[str, Any]:
        return {"session": self.session, "source_class": self.source_class, "content": self.content}


@dataclass(frozen=True, slots=True)
class ContinuityCase:
    case_id: str
    version: int | str
    family: str
    events: tuple[ContinuityEvent, ...]
    query: str
    choices: tuple[BenchmarkChoice, ...]
    expected_choice: str
    stale_choice: str | None
    risk_class: str
    tags: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: Any) -> "ContinuityCase":
        if not isinstance(data, Mapping):
            raise ValueError("continuity case must be an object")
        required = ("case_id","version","family","events","query","choices",
                    "expected_choice","risk_class","tags")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")
        events = tuple(ContinuityEvent.from_dict(item) for item in data["events"])
        if not events:
            raise ValueError("events must not be empty")
        choices = tuple(BenchmarkChoice.from_dict(item) for item in data["choices"])
        ids = tuple(choice.choice_id for choice in choices)
        expected = data["expected_choice"]
        if expected not in ids:
            raise ValueError("expected_choice must identify a choice")
        stale = data.get("stale_choice")
        if stale is not None and stale not in ids:
            raise ValueError("stale_choice must identify a choice or be null")
        return cls(
            case_id=str(data["case_id"]), version=data["version"], family=str(data["family"]),
            events=events, query=str(data["query"]), choices=choices,
            expected_choice=expected, stale_choice=stale,
            risk_class=str(data["risk_class"]), tags=tuple(str(tag) for tag in data["tags"]),
        )

    @property
    def expected_text(self) -> str:
        return next(choice.text for choice in self.choices if choice.choice_id == self.expected_choice)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id, "version": self.version, "family": self.family,
            "events": [event.to_dict() for event in self.events], "query": self.query,
            "choices": [choice.to_dict() for choice in self.choices],
            "expected_choice": self.expected_choice, "stale_choice": self.stale_choice,
            "risk_class": self.risk_class, "tags": list(self.tags),
        }

    def digest(self) -> str:
        payload=json.dumps(self.to_dict(),ensure_ascii=False,sort_keys=True,separators=(",",":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_condition_context(case: ContinuityCase, *, condition: str, memory_text: str | None) -> str:
    condition = condition.upper()
    if condition == "RESET":
        history = ""
    elif condition == "PERSISTENT":
        if not memory_text:
            raise ValueError("persistent condition requires memory_text")
        history = f"Persistent memory from earlier interactions:\n{memory_text}\n\n"
    elif condition == "FULL_HISTORY":
        history = "Earlier interactions:\n" + "\n".join(
            f"[session {event.session}|{event.source_class}] {event.content}" for event in case.events
        ) + "\n\n"
    else:
        raise ValueError(f"unknown continuity condition: {condition}")
    return history + case.query


def _condition_metrics(cases: Sequence[ContinuityCase], results: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if len(cases) != len(results):
        raise ValueError("case/result count mismatch")
    correct = 0
    stale = 0
    for case, result in zip(cases, results, strict=True):
        if bool(result.get("correct")):
            correct += 1
        if case.stale_choice is not None and result.get("parsed_choice") == case.stale_choice:
            stale += 1
    return {
        "case_count": len(cases),
        "correct_count": correct,
        "user_correction_burden": len(cases) - correct,
        "stale_error_count": stale,
    }


def continuity_metrics(
    cases: Sequence[ContinuityCase],
    *,
    reset_results: Sequence[Mapping[str, Any]],
    persistent_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reset = _condition_metrics(cases, reset_results)
    persistent = _condition_metrics(cases, persistent_results)
    persistent_only = reset_only = both_correct = both_wrong = 0
    for reset_row, persistent_row in zip(reset_results, persistent_results, strict=True):
        r = bool(reset_row.get("correct"))
        p = bool(persistent_row.get("correct"))
        if p and not r:
            persistent_only += 1
        elif r and not p:
            reset_only += 1
        elif r and p:
            both_correct += 1
        else:
            both_wrong += 1
    return {
        "reset": reset,
        "persistent": persistent,
        "paired": {
            "persistent_only_correct": persistent_only,
            "reset_only_correct": reset_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "net_persistence_gain": persistent_only - reset_only,
            "correction_burden_reduction":
                reset["user_correction_burden"] - persistent["user_correction_burden"],
        },
    }
