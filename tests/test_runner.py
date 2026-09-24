from __future__ import annotations

import json

from spm_bench.case import BenchmarkCase
from spm_bench.runner import benchmark_digest, canonical_manifest_json, run_suite


class ScriptedAdapter:
    model_id = "scripted-test-model"
    model_digest = "sha256:" + "1" * 64

    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.calls: list[tuple[tuple[dict[str, str], ...], dict[str, object]]] = []

    def generate(self, messages, generation_config):
        frozen_messages = tuple(dict(message) for message in messages)
        self.calls.append((frozen_messages, dict(generation_config)))
        return next(self.outputs)


def make_case(case_id: str, expected: str = "b") -> BenchmarkCase:
    return BenchmarkCase.from_dict({
        "case_id": case_id,
        "version": 1,
        "family": "test_family",
        "turns": [{"role": "user", "content": "Which option is correct?"}],
        "choices": [
            {"id": "a", "text": "wrong"},
            {"id": "b", "text": "right"},
        ],
        "expected_choice": expected,
        "risk_class": "low",
        "tags": ["test"],
    })


def test_benchmark_digest_is_stable_and_order_sensitive():
    first = make_case("case-1")
    second = make_case("case-2")

    assert benchmark_digest((first, second)) == benchmark_digest((first, second))
    assert benchmark_digest((first, second)) != benchmark_digest((second, first))


def test_run_suite_preserves_case_order_and_scores_exact_choices():
    cases = (make_case("case-1"), make_case("case-2", expected="a"))
    adapter = ScriptedAdapter(["b", "a"])

    manifest = run_suite(cases, adapter, {"temperature": 0, "max_new_tokens": 4})

    assert [result["case_id"] for result in manifest["results"]] == ["case-1", "case-2"]
    assert [result["parsed_choice"] for result in manifest["results"]] == ["b", "a"]
    assert [result["correct"] for result in manifest["results"]] == [True, True]
    assert manifest["summary"] == {"case_count": 2, "correct_count": 2, "malformed_count": 0}


def test_runner_appends_deterministic_choice_instruction():
    adapter = ScriptedAdapter(["b"])
    run_suite((make_case("case-1"),), adapter, {"temperature": 0})

    messages, _ = adapter.calls[0]
    assert messages[-1]["role"] == "user"
    assert "[a] wrong" in messages[-1]["content"]
    assert "[b] right" in messages[-1]["content"]
    assert "Reply with exactly one choice ID" in messages[-1]["content"]


def test_verbose_or_unknown_output_is_malformed_not_helpfully_parsed():
    cases = (make_case("case-1"), make_case("case-2"))
    adapter = ScriptedAdapter(["The answer is b", "z"])

    manifest = run_suite(cases, adapter, {"temperature": 0})

    assert [result["parsed_choice"] for result in manifest["results"]] == [None, None]
    assert [result["correct"] for result in manifest["results"]] == [False, False]
    assert manifest["summary"]["malformed_count"] == 2


def test_run_manifest_binds_model_benchmark_and_generation_config():
    cases = (make_case("case-1"),)
    adapter = ScriptedAdapter(["b"])
    generation_config = {"max_new_tokens": 4, "temperature": 0}

    manifest = run_suite(cases, adapter, generation_config)

    assert manifest["schema_version"] == 1
    assert manifest["benchmark_digest"] == benchmark_digest(cases)
    assert manifest["model"] == {
        "id": adapter.model_id,
        "digest": adapter.model_digest,
    }
    assert len(manifest["generation_config_digest"]) == 64
    assert len(manifest["run_digest"]) == 64


def test_run_digest_is_stable_and_manifest_contains_no_wall_clock_fields():
    case = make_case("case-1")
    config = {"temperature": 0, "max_new_tokens": 4}

    first = run_suite((case,), ScriptedAdapter(["b"]), config)
    second = run_suite((case,), ScriptedAdapter(["b"]), dict(reversed(list(config.items()))))

    assert first["run_digest"] == second["run_digest"]
    encoded = canonical_manifest_json(first)
    assert json.loads(encoded) == first
    assert "timestamp" not in encoded.lower()
    assert "created_at" not in encoded.lower()
    assert "duration" not in encoded.lower()
