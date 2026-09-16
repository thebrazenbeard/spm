from __future__ import annotations

from spm_bench.case import BenchmarkCase
from spm_bench.runner import run_choice_suite


class ScriptedChoiceAdapter:
    model_id = "choice-test-model"
    model_digest = "7" * 64

    def __init__(self, choices: list[str]):
        self.choices = iter(choices)
        self.calls = []

    def choose(self, messages, choice_ids):
        self.calls.append((tuple(messages), tuple(choice_ids)))
        return next(self.choices)


def make_case(case_id: str, expected: str = "b") -> BenchmarkCase:
    return BenchmarkCase.from_dict({
        "case_id": case_id,
        "version": 1,
        "family": "test_family",
        "turns": [{"role": "user", "content": "Choose."}],
        "choices": [{"id": "a", "text": "wrong"}, {"id": "b", "text": "right"}],
        "expected_choice": expected,
        "risk_class": "low",
        "tags": ["test"],
    })


def test_choice_suite_scores_without_answer_key_in_adapter_call():
    cases = (make_case("case-1"), make_case("case-2", expected="a"))
    adapter = ScriptedChoiceAdapter(["b", "a"])

    manifest = run_choice_suite(cases, adapter)

    assert manifest["evaluation_mode"] == "constrained_choice_v1"
    assert manifest["summary"] == {"case_count": 2, "correct_count": 2, "malformed_count": 0}
    assert [item["parsed_choice"] for item in manifest["results"]] == ["b", "a"]
    assert adapter.calls[0][1] == ("a", "b")
    assert all("expected_choice" not in str(call) for call in adapter.calls)


def test_choice_suite_rejects_adapter_choice_outside_declared_ids():
    adapter = ScriptedChoiceAdapter(["z"])
    try:
        run_choice_suite((make_case("case-1"),), adapter)
    except ValueError as error:
        assert "undeclared choice" in str(error)
    else:
        raise AssertionError("undeclared choice must fail closed")


def test_choice_suite_digest_is_stable():
    case = make_case("case-1")
    first = run_choice_suite((case,), ScriptedChoiceAdapter(["b"]))
    second = run_choice_suite((case,), ScriptedChoiceAdapter(["b"]))
    assert first["run_digest"] == second["run_digest"]
    assert first["choice_protocol"] == {
        "method": "forced_bracket_prefix_label_argmax",
        "version": 1,
    }


def test_choice_suite_uses_bracketed_choice_prompt():
    adapter = ScriptedChoiceAdapter(["b"])
    run_choice_suite((make_case("case-1"),), adapter)
    messages, _choice_ids = adapter.calls[0]
    final = messages[-1]["content"]
    assert "bracketed choice ID" in final
    assert "[a]" in final
