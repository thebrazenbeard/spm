from __future__ import annotations

import pytest

from spm_bench.compare import baseline_readiness, compare_runs


BENCHMARK = "a" * 64


def evidence(subject: str, *, correct=(True, False), malformed=(False, False), benchmark=BENCHMARK):
    results = []
    families = ("ordinary_competence", "referent_preservation")
    for index, is_correct in enumerate(correct):
        results.append({
            "case_id": f"case-{index}",
            "family": families[index % len(families)],
            "correct": is_correct,
            "parsed_choice": None if malformed[index] else "a",
        })
    return {
        "subject_digest": subject,
        "run": {
            "benchmark_digest": benchmark,
            "run_digest": subject[::-1],
            "results": results,
            "summary": {
                "case_count": len(results),
                "correct_count": sum(correct),
                "malformed_count": sum(malformed),
            },
        },
    }


def test_compare_runs_reports_family_overall_and_malformed_rates():
    report = compare_runs([
        evidence("1" * 64, correct=(True, False), malformed=(False, True)),
        evidence("2" * 64, correct=(True, True), malformed=(False, False)),
    ])

    first = report["subjects"]["1" * 64]
    assert first["overall_accuracy"] == 0.5
    assert first["ordinary_competence_accuracy"] == 1.0
    assert first["malformed_rate"] == 0.5
    assert first["family_accuracy"] == {
        "ordinary_competence": 1.0,
        "referent_preservation": 0.0,
    }


def test_compare_runs_rejects_benchmark_digest_mismatch():
    with pytest.raises(ValueError, match="benchmark digest mismatch"):
        compare_runs([
            evidence("1" * 64, benchmark="a" * 64),
            evidence("2" * 64, benchmark="b" * 64),
        ])


def test_compare_runs_rejects_duplicate_subject_digest():
    item = evidence("1" * 64)
    with pytest.raises(ValueError, match="duplicate subject"):
        compare_runs([item, item])


def test_baseline_readiness_requires_three_distinct_bound_subjects_including_backbone():
    subjects = ["1" * 64, "2" * 64, "3" * 64]
    evidences = [evidence(subject) for subject in subjects]

    status = baseline_readiness(
        suite_frozen=True,
        benchmark_digest=BENCHMARK,
        subject_digests=subjects,
        inherited_subject_digest="3" * 64,
        evidences=evidences,
    )

    assert status["status"] == "READY"
    assert status["subject_count"] == 3
    assert status["blockers"] == []


def test_baseline_readiness_fails_closed_on_missing_run_or_digest_mismatch():
    subjects = ["1" * 64, "2" * 64, "3" * 64]
    evidences = [evidence("1" * 64), evidence("2" * 64, benchmark="b" * 64)]

    status = baseline_readiness(
        suite_frozen=True,
        benchmark_digest=BENCHMARK,
        subject_digests=subjects,
        inherited_subject_digest="3" * 64,
        evidences=evidences,
    )

    assert status["status"] == "NOT_READY"
    assert any("missing result" in blocker for blocker in status["blockers"])
    assert any("benchmark digest mismatch" in blocker for blocker in status["blockers"])


def test_baseline_readiness_requires_frozen_suite_and_backbone_membership():
    status = baseline_readiness(
        suite_frozen=False,
        benchmark_digest=BENCHMARK,
        subject_digests=["1" * 64, "2" * 64, "3" * 64],
        inherited_subject_digest="4" * 64,
        evidences=[evidence("1" * 64), evidence("2" * 64), evidence("3" * 64)],
    )

    assert status["status"] == "NOT_READY"
    assert "public suite is not frozen" in status["blockers"]
    assert "inherited backbone subject is not in the subject set" in status["blockers"]


def test_exact_baseline_readiness_binds_model_evaluator_and_score_method():
    from spm_bench.compare import baseline_readiness_exact

    subjects = ["1" * 64, "2" * 64, "3" * 64]
    models = {subject: str(index) * 64 for index, subject in enumerate(subjects, start=4)}
    evaluator = "e" * 40
    evidences = []
    for subject in subjects:
        item = evidence(subject)
        item["evaluator_commit"] = evaluator
        item["run"]["model"] = {"id": subject, "digest": models[subject]}
        item["run"]["execution"] = {"score_method": "selected_output_projection_v1"}
        evidences.append(item)
    status = baseline_readiness_exact(
        suite_frozen=True,
        benchmark_digest=BENCHMARK,
        subject_model_digests=models,
        inherited_subject_digest="3" * 64,
        evaluator_commit=evaluator,
        score_method="selected_output_projection_v1",
        evidences=evidences,
    )
    assert status["status"] == "READY"
    assert status["blockers"] == []


def test_exact_baseline_readiness_fails_on_evaluator_or_model_drift():
    from spm_bench.compare import baseline_readiness_exact

    subjects = ["1" * 64, "2" * 64, "3" * 64]
    models = {subject: str(index) * 64 for index, subject in enumerate(subjects, start=4)}
    evaluator = "e" * 40
    evidences = []
    for subject in subjects:
        item = evidence(subject)
        item["evaluator_commit"] = evaluator
        item["run"]["model"] = {"id": subject, "digest": models[subject]}
        item["run"]["execution"] = {"score_method": "selected_output_projection_v1"}
        evidences.append(item)
    evidences[1]["evaluator_commit"] = "f" * 40
    evidences[2]["run"]["model"]["digest"] = "9" * 64
    status = baseline_readiness_exact(
        suite_frozen=True,
        benchmark_digest=BENCHMARK,
        subject_model_digests=models,
        inherited_subject_digest="3" * 64,
        evaluator_commit=evaluator,
        score_method="selected_output_projection_v1",
        evidences=evidences,
    )
    assert status["status"] == "NOT_READY"
    assert any("evaluator commit mismatch" in blocker for blocker in status["blockers"])
    assert any("model digest mismatch" in blocker for blocker in status["blockers"])


def test_exact_baseline_readiness_can_require_runtime_precision():
    from spm_bench.compare import baseline_readiness_exact

    subjects = {"1" * 64: "a" * 64, "2" * 64: "b" * 64, "3" * 64: "c" * 64}
    evidences = []
    for subject, model_digest in subjects.items():
        item = evidence(subject)
        item["evaluator_commit"] = "commit-1"
        item["run"]["model"] = {"digest": model_digest}
        item["run"]["execution"] = {"score_method": "selected_output_projection_v1"}
        item["runtime"] = {"model_parameter_dtype": "torch.bfloat16"}
        evidences.append(item)

    ready = baseline_readiness_exact(
        suite_frozen=True,
        benchmark_digest=BENCHMARK,
        subject_model_digests=subjects,
        inherited_subject_digest="3" * 64,
        evaluator_commit="commit-1",
        score_method="selected_output_projection_v1",
        evidences=evidences,
        runtime_requirements={"model_parameter_dtype": "torch.bfloat16"},
    )
    assert ready["status"] == "READY"

    evidences[0]["runtime"]["model_parameter_dtype"] = "torch.float16"
    not_ready = baseline_readiness_exact(
        suite_frozen=True, benchmark_digest=BENCHMARK,
        subject_model_digests=subjects, inherited_subject_digest="3" * 64,
        evaluator_commit="commit-1", score_method="selected_output_projection_v1",
        evidences=evidences,
        runtime_requirements={"model_parameter_dtype": "torch.bfloat16"},
    )
    assert not_ready["status"] == "NOT_READY"
    assert any("runtime model_parameter_dtype mismatch" in item for item in not_ready["blockers"])
