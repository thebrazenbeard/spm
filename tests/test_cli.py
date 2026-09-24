from __future__ import annotations

import json
from pathlib import Path

import spm_bench.cli as cli


def _write_suite(path: Path) -> None:
    case = {
        "case_id": "cli-case",
        "version": 1,
        "family": "ordinary_competence",
        "turns": [{"role": "user", "content": "Choose."}],
        "choices": [{"id": "a", "text": "wrong"}, {"id": "b", "text": "right"}],
        "expected_choice": "b",
        "risk_class": "low",
        "tags": ["control"],
    }
    path.write_text(json.dumps(case, separators=(",", ":")) + "\n", encoding="utf-8")


def test_validate_prints_deterministic_suite_identity(tmp_path, capsys):
    suite = tmp_path / "suite.jsonl"
    _write_suite(suite)

    assert cli.main(["validate", str(suite)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["case_count"] == 1
    assert len(output["benchmark_digest"]) == 64
    assert "timestamp" not in output


def test_run_writes_canonical_result_file_with_explicit_subject(tmp_path, monkeypatch):
    suite = tmp_path / "suite.jsonl"
    model_path = tmp_path / "model"
    output_path = tmp_path / "result.json"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    _write_suite(suite)

    class FakeAdapter:
        def __init__(self, model_path_arg, model_id, revision, generation_config):
            assert Path(model_path_arg) == model_path
            self.model_id = model_id
            self.model_digest = "2" * 64
            self.revision = revision
        def generate(self, messages, generation_config):
            return "b"

    monkeypatch.setattr(cli, "LocalHFAdapter", FakeAdapter)

    rc = cli.main([
        "run", "--suite", str(suite), "--model-path", str(model_path),
        "--model-id", "fake/model", "--revision", "rev-123",
        "--output", str(output_path), "--max-new-tokens", "4",
    ])

    assert rc == 0
    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["model"]["id"] == "fake/model"
    assert manifest["summary"]["correct_count"] == 1
    assert manifest["generation_config"]["do_sample"] is False
    assert manifest["generation_config"]["max_new_tokens"] == 4


def test_summarize_prints_only_bound_summary(tmp_path, capsys):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({
        "run_digest": "3" * 64,
        "benchmark_digest": "4" * 64,
        "model": {"id": "m", "digest": "5" * 64},
        "summary": {"case_count": 2, "correct_count": 1, "malformed_count": 0},
    }), encoding="utf-8")

    assert cli.main(["summarize", str(result_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "benchmark_digest": "4" * 64,
        "model": {"id": "m", "digest": "5" * 64},
        "run_digest": "3" * 64,
        "summary": {"case_count": 2, "correct_count": 1, "malformed_count": 0},
    }
