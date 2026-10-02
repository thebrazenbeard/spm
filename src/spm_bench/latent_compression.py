from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Iterable


class ExactnessEvidence(StrEnum):
    COMPACT_ONLY = "COMPACT_ONLY"
    SOURCE_REHYDRATED = "SOURCE_REHYDRATED"
    INSUFFICIENT_FIDELITY = "INSUFFICIENT_FIDELITY"


@dataclass(frozen=True, slots=True)
class CompressionCase:
    case_id: str
    source: bytes
    query: str
    expected_answer: str
    exact_required: bool
    source_digest: str

    def validate(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("case_id must be a non-empty exact string")
        if type(self.source) is not bytes or not self.source:
            raise ValueError("source must be non-empty exact bytes")
        if type(self.query) is not str or not self.query:
            raise ValueError("query must be a non-empty exact string")
        if type(self.expected_answer) is not str or not self.expected_answer:
            raise ValueError("expected_answer must be a non-empty exact string")
        if type(self.exact_required) is not bool:
            raise ValueError("exact_required must be an exact bool")
        expected_digest = sha256(self.source).hexdigest()
        if self.source_digest != expected_digest:
            raise ValueError("source digest does not bind source bytes")


@dataclass(frozen=True, slots=True)
class CompressionTrial:
    latent: bytes
    answer: str
    latent_budget_bytes: int
    exactness_evidence: ExactnessEvidence
    recovered_source: bytes | None = None

    def validate(self) -> None:
        if type(self.latent) is not bytes or not self.latent:
            raise ValueError("latent must be non-empty exact bytes")
        if type(self.answer) is not str or not self.answer:
            raise ValueError("answer must be a non-empty exact string")
        if (
            type(self.latent_budget_bytes) is not int
            or isinstance(self.latent_budget_bytes, bool)
            or self.latent_budget_bytes < 1
        ):
            raise ValueError("latent_budget_bytes must be a positive exact int")
        if len(self.latent) > self.latent_budget_bytes:
            raise ValueError("latent budget exceeded")
        if type(self.exactness_evidence) is not ExactnessEvidence:
            raise ValueError("exactness_evidence must be exact ExactnessEvidence")
        if self.recovered_source is not None and type(self.recovered_source) is not bytes:
            raise ValueError("recovered_source must be exact bytes when present")
        if (
            self.exactness_evidence is ExactnessEvidence.SOURCE_REHYDRATED
            and self.recovered_source is None
        ):
            raise ValueError("SOURCE_REHYDRATED requires recovered_source")
        if (
            self.exactness_evidence is not ExactnessEvidence.SOURCE_REHYDRATED
            and self.recovered_source is not None
        ):
            raise ValueError("recovered_source is only valid for SOURCE_REHYDRATED")


@dataclass(frozen=True, slots=True)
class CompressionTrialResult:
    case_id: str
    answer_correct: bool
    exact_recovered: bool
    false_reconstruction: bool
    safe: bool
    source_bytes: int
    latent_bytes: int
    rehydration_bytes: int
    compression_ratio: float


@dataclass(frozen=True, slots=True)
class CompressionSuiteReport:
    case_count: int
    answer_accuracy: float
    exact_recovery_count: int
    false_reconstruction_count: int
    safe_count: int
    source_bytes_total: int
    latent_bytes_total: int
    rehydration_bytes_total: int
    compression_ratio: float
    claim_ceiling: str = "EXPERIMENT_PROTOCOL_ONLY_NOT_MODEL_LEVEL_COMPRESSION_PROOF"


def evaluate_trial(
    case: CompressionCase,
    trial: CompressionTrial,
) -> CompressionTrialResult:
    if type(case) is not CompressionCase:
        raise TypeError("case must be exact CompressionCase")
    if type(trial) is not CompressionTrial:
        raise TypeError("trial must be exact CompressionTrial")
    case.validate()
    trial.validate()

    answer_correct = trial.answer == case.expected_answer
    rehydration_bytes = 0
    exact_recovered = False
    false_reconstruction = False

    if trial.exactness_evidence is ExactnessEvidence.SOURCE_REHYDRATED:
        assert trial.recovered_source is not None
        recovered_digest = sha256(trial.recovered_source).hexdigest()
        if recovered_digest != case.source_digest:
            raise ValueError("source digest mismatch during exact rehydration")
        rehydration_bytes = len(trial.recovered_source)
        exact_recovered = answer_correct if case.exact_required else False
    elif trial.exactness_evidence is ExactnessEvidence.INSUFFICIENT_FIDELITY:
        if trial.answer != "INSUFFICIENT_FIDELITY":
            raise ValueError(
                "INSUFFICIENT_FIDELITY evidence requires matching answer"
            )
    elif case.exact_required:
        false_reconstruction = trial.answer != "INSUFFICIENT_FIDELITY"

    if case.exact_required:
        safe = (
            exact_recovered
            or trial.exactness_evidence is ExactnessEvidence.INSUFFICIENT_FIDELITY
        )
    else:
        safe = answer_correct

    return CompressionTrialResult(
        case_id=case.case_id,
        answer_correct=answer_correct,
        exact_recovered=exact_recovered,
        false_reconstruction=false_reconstruction,
        safe=safe,
        source_bytes=len(case.source),
        latent_bytes=len(trial.latent),
        rehydration_bytes=rehydration_bytes,
        compression_ratio=len(case.source) / len(trial.latent),
    )


def evaluate_suite(
    trials: Iterable[tuple[CompressionCase, CompressionTrial]],
) -> CompressionSuiteReport:
    pairs = tuple(trials)
    if not pairs:
        raise ValueError("compression suite requires at least one trial")
    results = tuple(evaluate_trial(case, trial) for case, trial in pairs)
    source_total = sum(item.source_bytes for item in results)
    latent_total = sum(item.latent_bytes for item in results)
    return CompressionSuiteReport(
        case_count=len(results),
        answer_accuracy=(
            sum(item.answer_correct for item in results) / len(results)
        ),
        exact_recovery_count=sum(item.exact_recovered for item in results),
        false_reconstruction_count=sum(
            item.false_reconstruction for item in results
        ),
        safe_count=sum(item.safe for item in results),
        source_bytes_total=source_total,
        latent_bytes_total=latent_total,
        rehydration_bytes_total=sum(
            item.rehydration_bytes for item in results
        ),
        compression_ratio=source_total / latent_total,
    )
