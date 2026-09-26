import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments" / "spm0_control_plan_v1.json"


def load_plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_control_plan_requires_strong_conventional_persistent_memory_arm():
    plan = load_plan()
    control = plan["persistent_memory_control"]
    assert control["arm"] == "M"
    assert control["required"] is True
    assert set(control["comparisons"]) == {"C-B*", "C-M", "M-B"}
    assert {"inference_tokens", "memory_or_state_bytes", "update_retrieval_compute"} <= set(
        control["budget_match"]
    )

def test_control_plan_separates_causal_evidence_classes():
    plan = load_plan()
    classes = {entry["id"]: entry for entry in plan["evidence_classes"]}
    assert set(classes) == {
        "STATE_DEPENDENCE",
        "PERSISTENCE_DEPENDENCE",
        "REPRESENTATION_STRUCTURE_DEPENDENCE",
        "SEMANTIC_RELATION_SPECIFICITY",
    }
    assert "C_vs_B_star" in classes["PERSISTENCE_DEPENDENCE"]["requires"]
    assert "capacity_compute_matched_structure_control" in classes[
        "REPRESENTATION_STRUCTURE_DEPENDENCE"
    ]["requires"]
    assert "nuisance_matched_donor_control" in classes[
        "SEMANTIC_RELATION_SPECIFICITY"
    ]["requires"]
    assert "predeclared_directional_prediction" in classes[
        "SEMANTIC_RELATION_SPECIFICITY"
    ]["requires"]


def test_control_plan_blocks_training_without_frozen_controls():
    gate = set(load_plan()["training_gate"])
    assert {"baseline_readiness_pass", "arm_M_frozen", "nuisance_intervention_plan_frozen"} <= gate


def test_specificity_control_matches_reviewed_nuisance_variables():
    classes = {entry["id"]: entry for entry in load_plan()["evidence_classes"]}
    requirements = set(classes["SEMANTIC_RELATION_SPECIFICITY"]["requires"])
    assert {
        "matched_state_age_update_count",
        "comparable_magnitude_norm",
        "same_representation_capacity",
        "same_allowed_textual_context",
        "within_case_permutation_or_null_substitution",
    } <= requirements

def test_arm_m_protocol_is_concrete_and_budget_derived_before_results():
    control = load_plan()["persistent_memory_control"]
    assert control["protocol_version"] == "M_HYBRID_EXTRACTIVE_V1"
    assert control["memory_source"] == "accepted_exogenous_observations_only"
    assert control["retrieval_policy"] == {
        "recency_fraction": 0.5,
        "lexical_fraction": 0.5,
        "lexical_method": "bm25_k1_1.2_b_0.75",
    }
    assert control["budget_derivation"]["memory_bytes"] == "C_persistent_state_payload_bytes_per_lineage"
    assert control["budget_derivation"]["retrieved_tokens"] == "max_integer_within_C_incremental_state_compute_ceiling"
    assert control["budget_derivation"]["freeze_timing"] == "before_training_or_qualification_outcomes"


def test_specificity_protocol_freezes_matching_and_directional_threshold():
    specificity = next(
        item for item in load_plan()["evidence_classes"]
        if item["id"] == "SEMANTIC_RELATION_SPECIFICITY"
    )
    protocol = specificity["protocol"]
    assert protocol["perturbation_l2_relative_tolerance"] == 0.05
    assert protocol["state_norm_relative_tolerance"] == 0.05
    assert protocol["delta_min"] == "max(0.02,3*sigma_null)"
    assert protocol["on_unmatched_required_nuisance"] == "UNKNOWN"
