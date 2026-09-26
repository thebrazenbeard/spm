from __future__ import annotations

import json
from pathlib import Path

from spm_bench.baseline_manifest import ArtifactIdentity, BaselineSubject

SUBJECTS = {
    "qwen2.5-0.5b-instruct.json": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        "6080fc05cb5e0ccfa35e64523b11a902cc1f3e35672f85135a19eb16b722f8b8",
        "460a513145ad2f88d8b5165e8100a9f620eb30bad9257427d71cb7db83479fba",
    ),
    "smollm2-360m-instruct.json": (
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "a10cc1512eabd3dde888204e902eca88bddb4951",
        "3ac36bcfe007decda631d7d47b6503b3469a8f6d1bef090c662f0acfc449efb3",
        "bff315e568107dfb37bb6dcaf57c668ee50f05e1229a628aa0e31a6de8845513",
    ),
}
SUBJECTS.update({
    "qwen2.5-1.5b-instruct.json": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        "866e9c64bd94ae56fbaf5f2a46c0d6578b9a44baeab1ebc269e43018f3b1766d",
        "dd3f034efe14e340ea1de88eda7aac4b41a23e3168e286066bf914920ec9c80f",
    ),
    "smollm3-3b.json": (
        "HuggingFaceTB/SmolLM3-3B",
        "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "fd0ee8f56c88636d77521cf9082b14cc499cf11f34f8467c189b11791dc1decb",
        "d70a0bcbe7cd83cd174ae7af3d7f0c7ee92ac6ced302e4d8afa8bf273d1b96bd",
    ),
})


def test_source_controlled_baseline_subjects_match_bound_identities():
    root = Path("baselines/subjects")
    assert {path.name for path in root.glob("*.json")} == set(SUBJECTS)
    for filename, (artifact_id, revision, inventory, expected_digest) in SUBJECTS.items():
        payload = json.loads((root / filename).read_text(encoding="utf-8"))
        subject = BaselineSubject(
            model=ArtifactIdentity(**payload["model"]),
            tokenizer=ArtifactIdentity(**payload["tokenizer"]),
            parameter_count=payload["parameter_count"],
            generation_config=payload["generation_config"],
        )
        assert subject.model.artifact_id == artifact_id
        assert subject.model.immutable_revision == revision
        assert subject.model.inventory_digest == inventory
        assert subject.tokenizer.artifact_id == artifact_id
        assert subject.tokenizer.immutable_revision == revision
        assert subject.digest() == expected_digest
        assert json.loads(subject.canonical_json()) == payload
