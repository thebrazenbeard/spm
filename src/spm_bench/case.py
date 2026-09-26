"""Immutable benchmark case contracts and JSONL loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from os import PathLike
from pathlib import Path
from typing import Any


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_fields(data: Mapping[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"missing required field(s): {', '.join(missing)}")


@dataclass(frozen=True, slots=True)
class BenchmarkTurn:
    role: str
    content: str

    @classmethod
    def from_dict(cls, data: Any) -> BenchmarkTurn:
        if not isinstance(data, Mapping):
            raise ValueError("each turn must be an object")
        _require_fields(data, ("role", "content"))
        return cls(
            role=_required_string(data["role"], "turn role"),
            content=_required_string(data["content"], "turn content"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class BenchmarkChoice:
    choice_id: str
    text: str

    @classmethod
    def from_dict(cls, data: Any) -> BenchmarkChoice:
        if not isinstance(data, Mapping):
            raise ValueError("each choice must be an object")
        _require_fields(data, ("id", "text"))
        return cls(
            choice_id=_required_string(data["id"], "choice id"),
            text=_required_string(data["text"], "choice text"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"id": self.choice_id, "text": self.text}


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    version: int | str
    family: str
    turns: tuple[BenchmarkTurn, ...]
    choices: tuple[BenchmarkChoice, ...]
    expected_choice: str
    risk_class: str
    tags: tuple[str, ...]

    _REQUIRED_FIELDS = (
        "case_id",
        "version",
        "family",
        "turns",
        "choices",
        "expected_choice",
        "risk_class",
        "tags",
    )

    @classmethod
    def from_dict(cls, data: Any) -> BenchmarkCase:
        if not isinstance(data, Mapping):
            raise ValueError("benchmark case must be an object")
        _require_fields(data, cls._REQUIRED_FIELDS)

        version = data["version"]
        if isinstance(version, bool) or not isinstance(version, (int, str)):
            raise ValueError("version must be an integer or non-empty string")
        if isinstance(version, str) and not version.strip():
            raise ValueError("version must be an integer or non-empty string")

        turns_data = data["turns"]
        if not isinstance(turns_data, list) or not turns_data:
            raise ValueError("turns must be a non-empty list")
        turns = tuple(BenchmarkTurn.from_dict(turn) for turn in turns_data)

        choices_data = data["choices"]
        if not isinstance(choices_data, list) or not choices_data:
            raise ValueError("choices must be a non-empty list")
        choices = tuple(BenchmarkChoice.from_dict(choice) for choice in choices_data)
        choice_ids = [choice.choice_id for choice in choices]
        if len(choice_ids) != len(set(choice_ids)):
            raise ValueError("duplicate choice id")

        expected_choice = _required_string(
            data["expected_choice"], "expected_choice"
        )
        if expected_choice not in choice_ids:
            raise ValueError("expected_choice must identify one of the choices")

        tags_data = data["tags"]
        if not isinstance(tags_data, list):
            raise ValueError("tags must be a list")
        tags = tuple(_required_string(tag, "tag") for tag in tags_data)

        return cls(
            case_id=_required_string(data["case_id"], "case_id"),
            version=version,
            family=_required_string(data["family"], "family"),
            turns=turns,
            choices=choices,
            expected_choice=expected_choice,
            risk_class=_required_string(data["risk_class"], "risk_class"),
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "version": self.version,
            "family": self.family,
            "turns": [turn.to_dict() for turn in self.turns],
            "choices": [choice.to_dict() for choice in self.choices],
            "expected_choice": self.expected_choice,
            "risk_class": self.risk_class,
            "tags": list(self.tags),
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


def load_jsonl_cases(path: str | PathLike[str]) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    source = Path(path)
    with source.open(encoding="utf-8") as lines:
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                cases.append(BenchmarkCase.from_dict(data))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    return tuple(cases)
