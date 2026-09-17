"""Comparison and fail-closed baseline readiness gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _accuracy(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_runs(evidences: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare exact-subject-bound runs from one frozen benchmark."""
    if not evidences:
        raise ValueError("at least one evidence record is required")

    benchmark_digest: str | None = None
    subjects: dict[str, Any] = {}
    for evidence in evidences:
        subject_digest = evidence.get("subject_digest")
        run = evidence.get("run")
        if not isinstance(subject_digest, str) or not subject_digest:
            raise ValueError("evidence requires subject_digest")
        if subject_digest in subjects:
            raise ValueError(f"duplicate subject digest: {subject_digest}")
        if not isinstance(run, Mapping):
            raise ValueError("evidence requires run manifest")
        observed_benchmark = run.get("benchmark_digest")
        if benchmark_digest is None:
            benchmark_digest = observed_benchmark
        elif observed_benchmark != benchmark_digest:
            raise ValueError("benchmark digest mismatch across runs")

        results = run.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError("run results must be a non-empty list")
        family_values: dict[str, list[bool]] = defaultdict(list)
        all_values: list[bool] = []
        malformed = 0
        for result in results:
            if not isinstance(result, Mapping):
                raise ValueError("each run result must be a mapping")
            family = result.get("family")
            correct = result.get("correct")
            if not isinstance(family, str) or not isinstance(correct, bool):
                raise ValueError("run result requires family and boolean correct")
            all_values.append(correct)
            family_values[family].append(correct)
            if result.get("parsed_choice") is None:
                malformed += 1

        family_accuracy = {
            family: _accuracy(values)
            for family, values in sorted(family_values.items())
        }
        subjects[subject_digest] = {
            "run_digest": run.get("run_digest"),
            "overall_accuracy": _accuracy(all_values),
            "ordinary_competence_accuracy": family_accuracy.get(
                "ordinary_competence", 0.0
            ),
            "malformed_rate": malformed / len(results),
            "family_accuracy": family_accuracy,
        }

    return {
        "benchmark_digest": benchmark_digest,
        "subjects": subjects,
    }


def baseline_readiness(
    *,
    suite_frozen: bool,
    benchmark_digest: str,
    subject_digests: Sequence[str],
    inherited_subject_digest: str,
    evidences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a fail-closed baseline readiness verdict."""
    blockers: list[str] = []
    subjects = list(subject_digests)
    if not suite_frozen:
        blockers.append("public suite is not frozen")
    if len(subjects) < 3 or len(set(subjects)) < 3:
        blockers.append("at least three distinct bound subjects are required")
    if inherited_subject_digest not in subjects:
        blockers.append("inherited backbone subject is not in the subject set")

    evidence_by_subject = {
        item.get("subject_digest"): item for item in evidences
        if isinstance(item, Mapping)
    }
    for subject in subjects:
        evidence = evidence_by_subject.get(subject)
        if evidence is None:
            blockers.append(f"missing result for subject {subject}")
            continue
        run = evidence.get("run")
        if not isinstance(run, Mapping):
            blockers.append(f"missing result for subject {subject}")
            continue
        if run.get("benchmark_digest") != benchmark_digest:
            blockers.append(f"benchmark digest mismatch for subject {subject}")

    return {
        "status": "READY" if not blockers else "NOT_READY",
        "subject_count": len(set(subjects)),
        "blockers": blockers,
    }


def baseline_readiness_exact(
    *,
    suite_frozen: bool,
    benchmark_digest: str,
    subject_model_digests: Mapping[str, str],
    inherited_subject_digest: str,
    evaluator_commit: str,
    score_method: str,
    evidences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed unless subject, model, evaluator, and scoring method all bind exactly."""
    subjects = tuple(subject_model_digests)
    base = baseline_readiness(
        suite_frozen=suite_frozen,
        benchmark_digest=benchmark_digest,
        subject_digests=subjects,
        inherited_subject_digest=inherited_subject_digest,
        evidences=evidences,
    )
    blockers = list(base["blockers"])
    if not isinstance(evaluator_commit, str) or not evaluator_commit:
        blockers.append("evaluator commit is not bound")
    if not isinstance(score_method, str) or not score_method:
        blockers.append("score method is not bound")

    evidence_by_subject = {
        item.get("subject_digest"): item for item in evidences
        if isinstance(item, Mapping)
    }
    for subject, expected_model_digest in subject_model_digests.items():
        evidence = evidence_by_subject.get(subject)
        if evidence is None:
            continue
        if evidence.get("evaluator_commit") != evaluator_commit:
            blockers.append(f"evaluator commit mismatch for subject {subject}")
        run = evidence.get("run")
        if not isinstance(run, Mapping):
            continue
        model = run.get("model")
        if not isinstance(model, Mapping) or model.get("digest") != expected_model_digest:
            blockers.append(f"model digest mismatch for subject {subject}")
        execution = run.get("execution")
        if not isinstance(execution, Mapping) or execution.get("score_method") != score_method:
            blockers.append(f"score method mismatch for subject {subject}")
        run_digest = run.get("run_digest")
        if not isinstance(run_digest, str) or len(run_digest) != 64:
            blockers.append(f"run digest is not bound for subject {subject}")

    return {
        "status": "READY" if not blockers else "NOT_READY",
        "subject_count": len(set(subjects)),
        "blockers": blockers,
        "evaluator_commit": evaluator_commit,
        "score_method": score_method,
    }
