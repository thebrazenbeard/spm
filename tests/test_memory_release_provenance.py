from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
RELEASE = ROOT / "releases" / "SPM_MEMORY_SPECIALIST_QWEN1P5_RC1.json"


def _source_bytes(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _windows_execution_bytes(path: Path) -> bytes:
    canonical = _source_bytes(path)
    return canonical.replace(b"\n", b"\r\n")


def test_rc1_release_binds_canonical_git_source_bytes_and_legacy_execution_bytes():
    release = json.loads(RELEASE.read_text(encoding="utf-8"))

    pairs = (
        (
            ROOT / "artifacts/memory_adapter_qwen1p5_v1/adapter_config.json",
            release["adapter"]["config_source_sha256"],
            release["adapter"]["config_sha256"],
        ),
        (
            ROOT / "state/adapters/SPM_MEMORY_ADAPTER_QWEN1P5_V1_TRAIN.json",
            release["training"]["receipt_source_sha256"],
            release["training"]["receipt_sha256"],
        ),
        (
            ROOT / "state/adapters/SPM_MEMORY_ADAPTER_QWEN1P5_V1_EVAL.json",
            release["training"]["evaluation_receipt_source_sha256"],
            release["training"]["evaluation_receipt_sha256"],
        ),
        (
            ROOT / "state/adapters/SPM_MEMORY_SPECIALIST_GATED_RUNTIME_V1_QUALIFICATION.json",
            release["qualification"]["receipt_source_sha256"],
            release["qualification"]["receipt_sha256"],
        ),
        (
            ROOT / "state/adapters/SPM_MEMORY_SPECIALIST_RC1_SMOKE.json",
            release["smoke"]["receipt_source_sha256"],
            release["smoke"]["receipt_sha256"],
        ),
    )

    for path, expected_source, expected_execution in pairs:
        assert _sha256(_source_bytes(path)) == expected_source
        assert _sha256(_windows_execution_bytes(path)) == expected_execution


def test_rc1_release_declares_digest_semantics():
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    semantics = release["digest_semantics"]
    assert "exact binary bytes" in semantics["binary_sha256"]
    assert "Windows CRLF" in semantics["legacy_text_sha256"]
    assert "canonical Git source bytes" in semantics["source_sha256"]
