from __future__ import annotations

from spm_bench.causal_evidence import NuisanceMatch, semantic_specificity_receipt


def matched() -> NuisanceMatch:
    return NuisanceMatch(
        same_sequence=True,
        same_update_count=True,
        same_state_shape=True,
        same_dtype=True,
        same_text_context=True,
        perturbation_l2_relative_difference=0.01,
        state_norm_relative_difference=0.02,
    )


def test_semantic_specificity_pass_requires_direction_beyond_nuisance_and_null():
    receipt = semantic_specificity_receipt(
        relevant_effect=0.12,
        nuisance_effect=0.02,
        null_effect=0.01,
        delta_min=0.03,
        nuisance_match=matched(),
        predicted_direction="increase",
    )
    assert receipt.verdict == "PASS"
    assert receipt.evidence_class == "SEMANTIC_RELATION_SPECIFICITY"
    assert len(receipt.digest) == 64


def test_semantic_specificity_is_unknown_when_required_nuisance_match_fails():
    mismatch = NuisanceMatch(
        same_sequence=True,
        same_update_count=True,
        same_state_shape=True,
        same_dtype=True,
        same_text_context=True,
        perturbation_l2_relative_difference=0.06,
        state_norm_relative_difference=0.01,
    )
    receipt = semantic_specificity_receipt(
        relevant_effect=0.20,
        nuisance_effect=0.00,
        null_effect=0.00,
        delta_min=0.03,
        nuisance_match=mismatch,
        predicted_direction="increase",
    )
    assert receipt.verdict == "UNKNOWN"
    assert "perturbation_l2_relative_difference" in receipt.mismatches


def test_semantic_specificity_fails_when_directional_effect_is_not_material():
    receipt = semantic_specificity_receipt(
        relevant_effect=0.02,
        nuisance_effect=0.00,
        null_effect=0.00,
        delta_min=0.03,
        nuisance_match=matched(),
        predicted_direction="increase",
    )
    assert receipt.verdict == "FAIL"
