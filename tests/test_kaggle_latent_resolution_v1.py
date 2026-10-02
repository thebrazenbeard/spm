from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "kaggle" / "latent_resolution_v1" / "kernel_main.py"
BUILDER = ROOT / "scripts" / "build_kaggle_latent_resolution_v1.py"


def _load_kernel():
    spec = importlib.util.spec_from_file_location("latent_resolution_kernel", KERNEL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_binds_exact_spm_and_qwen_subjects():
    module = _load_kernel()

    assert module.SPM_COMMIT == "dbd4cb10be61711b349ccad5c900037d7ac68ea1"
    assert module.MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert module.MODEL_REVISION == "7ae557604adf67be50417f59c2c2f167def9a775"
    assert module.MODEL_INVENTORY_DIGEST == "6080fc05cb5e0ccfa35e64523b11a902cc1f3e35672f85135a19eb16b722f8b8"
    assert module.CLAIM_CEILING == "KAGGLE_EXPERIMENT_ONLY_NOT_VERA_RUNTIME_OR_PRODUCTION_VRAM_PROOF"


def test_dry_run_manifest_is_deterministic_and_requires_no_gpu_or_network():
    module = _load_kernel()

    first = module.build_dry_run_manifest()
    second = module.build_dry_run_manifest()

    assert first == second
    assert first["schema"] == "SPM_LATENT_RESOLUTION_KAGGLE_V1"
    assert first["spm_commit"] == module.SPM_COMMIT
    assert first["model_revision"] == module.MODEL_REVISION
    assert first["model_inventory_digest"] == module.MODEL_INVENTORY_DIGEST
    assert first["slot_budgets"] == [2, 4, 8, 16]
    assert first["train_case_count"] > first["dev_case_count"] > 0
    assert len(first["train_subject_digest"]) == 64
    assert len(first["dev_subject_digest"]) == 64


def test_kernel_source_contains_no_credential_literals():
    text = KERNEL.read_text(encoding="utf-8").lower()

    forbidden = (
        "kaggle_key",
        "kaggle_username",
        "api_token",
        "access_token=",
        "password=",
    )
    assert not any(item in text for item in forbidden)


def test_builder_creates_private_gpu_kernel_metadata_without_secrets(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--user",
            "example-user",
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "example-user/vera-latent-resolution-v1"
    assert metadata["code_file"] == "kernel_main.py"
    assert metadata["kernel_type"] == "script"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["is_private"] is True
    assert (tmp_path / "kernel_main.py").is_file()
    assert "NvidiaTeslaT4" in completed.stdout
    assert "kaggle kernels push" in completed.stdout


def test_kernel_dry_run_cli_outputs_json():
    completed = subprocess.run(
        [sys.executable, str(KERNEL), "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["schema"] == "SPM_LATENT_RESOLUTION_KAGGLE_V1"
    assert payload["claim_ceiling"] == "KAGGLE_EXPERIMENT_ONLY_NOT_VERA_RUNTIME_OR_PRODUCTION_VRAM_PROOF"
