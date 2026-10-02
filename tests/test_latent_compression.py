from __future__ import annotations

from hashlib import sha256

import pytest

from spm_bench.latent_compression import (
    CompressionCase,
    CompressionTrial,
    ExactnessEvidence,
    evaluate_suite,
    evaluate_trial,
)


def _case() -> CompressionCase:
    source = b"Patrick bought a red screwdriver on Tuesday at 15:17."
    return CompressionCase(
        case_id="late-color",
        source=source,
        query="What color was the screwdriver?",
        expected_answer="red",
        exact_required=True,
        source_digest=sha256(source).hexdigest(),
    )


def test_compact_only_exact_answer_is_not_allowed_to_impersonate_source_recovery():
    case = _case()
    trial = CompressionTrial(
        latent=b"purchase(tool)",
        answer="red",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.COMPACT_ONLY,
    )

    result = evaluate_trial(case, trial)

    assert result.answer_correct is True
    assert result.exact_recovered is False
    assert result.false_reconstruction is True
    assert result.safe is False


def test_verified_source_rehydration_can_satisfy_exact_query():
    case = _case()
    trial = CompressionTrial(
        latent=b"purchase(tool)",
        answer="red",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.SOURCE_REHYDRATED,
        recovered_source=case.source,
    )

    result = evaluate_trial(case, trial)

    assert result.answer_correct is True
    assert result.exact_recovered is True
    assert result.false_reconstruction is False
    assert result.safe is True
    assert result.rehydration_bytes == len(case.source)


def test_insufficient_fidelity_is_a_safe_exactness_failure():
    case = _case()
    trial = CompressionTrial(
        latent=b"purchase(tool)",
        answer="INSUFFICIENT_FIDELITY",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.INSUFFICIENT_FIDELITY,
    )

    result = evaluate_trial(case, trial)

    assert result.answer_correct is False
    assert result.exact_recovered is False
    assert result.false_reconstruction is False
    assert result.safe is True


def test_rehydration_digest_mismatch_fails_closed():
    case = _case()
    trial = CompressionTrial(
        latent=b"purchase(tool)",
        answer="red",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.SOURCE_REHYDRATED,
        recovered_source=b"tampered",
    )

    with pytest.raises(ValueError, match="source digest"):
        evaluate_trial(case, trial)


def test_latent_budget_is_hard_not_advisory():
    case = _case()
    trial = CompressionTrial(
        latent=b"x" * 33,
        answer="INSUFFICIENT_FIDELITY",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.INSUFFICIENT_FIDELITY,
    )

    with pytest.raises(ValueError, match="latent budget"):
        evaluate_trial(case, trial)


def test_suite_reports_false_reconstruction_separately_from_accuracy():
    case = _case()
    unsafe = CompressionTrial(
        latent=b"purchase(tool)",
        answer="red",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.COMPACT_ONLY,
    )
    safe = CompressionTrial(
        latent=b"purchase(tool)",
        answer="red",
        latent_budget_bytes=32,
        exactness_evidence=ExactnessEvidence.SOURCE_REHYDRATED,
        recovered_source=case.source,
    )

    report = evaluate_suite(((case, unsafe), (case, safe)))

    assert report.case_count == 2
    assert report.answer_accuracy == 1.0
    assert report.false_reconstruction_count == 1
    assert report.exact_recovery_count == 1
    assert report.safe_count == 1
    assert report.source_bytes_total == 2 * len(case.source)
    assert report.latent_bytes_total == 2 * len(b"purchase(tool)")
    assert report.claim_ceiling == "EXPERIMENT_PROTOCOL_ONLY_NOT_MODEL_LEVEL_COMPRESSION_PROOF"
