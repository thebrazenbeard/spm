from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "spm0_vector_state_bstar_c_v1.json"


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_vector_experiment_binds_exact_subject_architecture_and_parameter_budget():
    manifest = load_manifest()
    assert manifest["subject"]["model_id"] == "HuggingFaceTB/SmolLM3-3B"
    assert manifest["subject"]["revision"] == "a07cc9a04f16550a088caea529712d1d335b0ac1"
    assert manifest["subject"]["inventory_digest"] == "fd0ee8f56c88636d77521cf9082b14cc499cf11f34f8467c189b11791dc1decb"
    architecture = manifest["architecture"]
    assert architecture["state_size"] == 128
    assert architecture["injection_layer_index"] == 31
    assert architecture["remaining_backbone_blocks"] == [32, 33, 34, 35]
    assert architecture["trainable_parameter_count"] == 623488
    assert manifest["arms"]["B*"]["persistent_prior"] is False
    assert manifest["arms"]["C"]["persistent_prior"] is True


def test_vector_experiment_binds_materialized_corpus_and_optimizer_budget():
    manifest = load_manifest()
    data = manifest["data"]
    train = ROOT / data["train_path"]
    validation = ROOT / data["validation_path"]
    assert hashlib.sha256(train.read_bytes()).hexdigest() == data["train_sha256"]
    assert hashlib.sha256(validation.read_bytes()).hexdigest() == data["validation_sha256"]
    assert data["train_case_count"] == 96
    assert data["validation_case_count"] == 24
    budget = manifest["optimization"]
    assert budget == {
        "accumulation_cases": 4,
        "data_order_seed": 20260917,
        "epochs": 2,
        "gradient_norm_ceiling": 1.0,
        "initialization_seed": 1729,
        "learning_rate": 0.0003,
        "optimizer": "AdamW",
        "optimizer_steps_per_arm": 48,
        "weight_decay": 0.01,
    }


def test_vector_experiment_freezes_pre_result_decision_rules_and_training_gate():
    manifest = load_manifest()
    assert manifest["decision_rules"] == {
        "validation_net_paired_wins_min": 3,
        "public_target_net_paired_wins_min": 4,
        "ordinary_competence_max_c_loss_cases": 0,
        "c_vs_m_public_target_net_paired_wins_min": 2,
    }
    assert set(manifest["training_gate"]) == {
        "exact_baseline_readiness_READY",
        "arm_M_and_nuisance_controls_frozen",
        "training_data_digest_verified",
        "B_star_C_budget_frozen",
    }
    assert manifest["retune_after_comparison"] is False


def test_vector_experiment_freezes_training_precision():
    assert load_manifest()["precision"] == {
        "backbone": "bfloat16",
        "state_module": "float32",
        "selected_choice_logits": "float32",
    }
