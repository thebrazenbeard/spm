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


def test_permutation_choice_suite_tracks_semantic_option_across_label_rotation():
    from spm_bench.runner import run_permutation_choice_suite
    adapter = ScriptedChoiceAdapter(["b", "a"])
    manifest = run_permutation_choice_suite((make_case("case-1"),), adapter)
    result = manifest["results"][0]
    assert result["rotation_choices"] == ["b", "b"]
    assert result["parsed_choice"] == "b"
    assert result["label_invariant"] is True
    assert result["correct"] is True


def test_permutation_choice_suite_marks_fixed_label_prior_unstable():
    from spm_bench.runner import run_permutation_choice_suite
    adapter = ScriptedChoiceAdapter(["a", "a"])
    manifest = run_permutation_choice_suite((make_case("case-1"),), adapter)
    result = manifest["results"][0]
    assert result["rotation_choices"] == ["a", "b"]
    assert result["parsed_choice"] is None
    assert result["label_invariant"] is False
    assert result["correct"] is False
    assert manifest["summary"]["unstable_count"] == 1


def test_permutation_choice_suite_uses_batch_selector_when_available():
    from spm_bench.runner import run_permutation_choice_suite

    class BatchAdapter:
        model_id = "batch-test-model"
        model_digest = "8" * 64
        def __init__(self):
            self.batch_calls = []
        def choose_many(self, message_batches, choice_ids):
            self.batch_calls.append((tuple(tuple(m) for m in message_batches), tuple(choice_ids)))
            return ("b", "a")
        def choose(self, messages, choice_ids):
            raise AssertionError("scalar choose should not be used when choose_many exists")

    adapter = BatchAdapter()
    manifest = run_permutation_choice_suite((make_case("case-1"),), adapter)
    assert len(adapter.batch_calls) == 1
    assert len(adapter.batch_calls[0][0]) == 2
    assert manifest["results"][0]["rotation_choices"] == ["b", "b"]
    assert manifest["results"][0]["parsed_choice"] == "b"


def test_presentation_invariance_suite_survives_label_and_order_rotation():
    import re
    from spm_bench.runner import run_presentation_invariance_choice_suite

    class SemanticAdapter:
        model_id = "semantic-test-model"
        model_digest = "9" * 64
        def choose_many(self, message_batches, choice_ids):
            out = []
            for messages in message_batches:
                text = messages[-1]["content"]
                match = re.search(r"\[([^]]+)\] right", text)
                out.append(match.group(1))
            return tuple(out)

    manifest = run_presentation_invariance_choice_suite((make_case("case-1"),), SemanticAdapter())
    result = manifest["results"][0]
    assert result["parsed_choice"] == "b"
    assert result["presentation_invariant"] is True
    assert result["correct"] is True
    assert len(result["presentation_choices"]) == 4


def test_presentation_invariance_suite_rejects_fixed_first_option_bias():
    import re
    from spm_bench.runner import run_presentation_invariance_choice_suite

    class FirstOptionAdapter:
        model_id = "first-option-test-model"
        model_digest = "a" * 64
        def choose_many(self, message_batches, choice_ids):
            out = []
            for messages in message_batches:
                text = messages[-1]["content"]
                out.append(re.search(r"Choices:\n\[([^]]+)\]", text).group(1))
            return tuple(out)

    manifest = run_presentation_invariance_choice_suite((make_case("case-1"),), FirstOptionAdapter())
    result = manifest["results"][0]
    assert result["parsed_choice"] is None
    assert result["presentation_invariant"] is False
    assert result["correct"] is False
    assert manifest["summary"]["unstable_count"] == 1


def test_balanced_score_suite_cancels_fixed_first_position_bias():
    import re
    from spm_bench.runner import run_balanced_score_choice_suite

    class FirstPositionScoreAdapter:
        model_id = "position-bias"
        model_digest = "b" * 64
        def score_many(self, message_batches, choice_ids):
            rows = []
            for messages in message_batches:
                first = re.search(r"Choices:\n\[([^]]+)\]", messages[-1]["content"]).group(1)
                rows.append({choice: (0.9 if choice == first else 0.1) for choice in choice_ids})
            return tuple(rows)

    result = run_balanced_score_choice_suite((make_case("case-1"),), FirstPositionScoreAdapter())["results"][0]
    assert result["parsed_choice"] is None
    assert result["tie"] is True
    assert result["correct"] is False


def test_balanced_score_suite_recovers_semantic_signal_across_presentations():
    import re
    from spm_bench.runner import run_balanced_score_choice_suite

    class SemanticScoreAdapter:
        model_id = "semantic-score"
        model_digest = "c" * 64
        def score_many(self, message_batches, choice_ids):
            rows = []
            for messages in message_batches:
                text = messages[-1]["content"]
                right = re.search(r"\[([^]]+)\] right", text).group(1)
                rows.append({choice: (0.75 if choice == right else 0.25) for choice in choice_ids})
            return tuple(rows)

    manifest = run_balanced_score_choice_suite((make_case("case-1"),), SemanticScoreAdapter())
    result = manifest["results"][0]
    assert result["parsed_choice"] == "b"
    assert result["tie"] is False
    assert result["correct"] is True
    assert result["semantic_scores"]["b"] > result["semantic_scores"]["a"]
    assert manifest["evaluation_mode"] == "balanced_semantic_score_v1"


def test_balanced_score_suite_case_batching_preserves_results_and_reduces_calls():
    import re
    from spm_bench.runner import run_balanced_score_choice_suite

    class CountingSemanticScoreAdapter:
        model_id = "batch-score"
        model_digest = "d" * 64
        def __init__(self): self.calls = 0
        def score_many(self, message_batches, choice_ids):
            self.calls += 1
            rows = []
            for messages in message_batches:
                text = messages[-1]["content"]
                right = re.search(r"\[([^]]+)\] right", text).group(1)
                rows.append({choice: (0.75 if choice == right else 0.25) for choice in choice_ids})
            return tuple(rows)

    cases = tuple(make_case(f"case-{i}") for i in range(3))
    scalar = CountingSemanticScoreAdapter()
    batched = CountingSemanticScoreAdapter()
    one = run_balanced_score_choice_suite(cases, scalar, case_batch_size=1)
    two = run_balanced_score_choice_suite(cases, batched, case_batch_size=2)
    assert [r["semantic_scores"] for r in one["results"]] == [r["semantic_scores"] for r in two["results"]]
    assert [r["parsed_choice"] for r in one["results"]] == [r["parsed_choice"] for r in two["results"]]
    assert scalar.calls == 3
    assert batched.calls == 2
    assert two["execution"] == {"case_batch_size": 2}

def test_balanced_score_suite_prefers_selected_projection_when_available():
    from spm_bench.runner import run_balanced_score_choice_suite

    class SelectedAdapter:
        model_id = "selected-score"
        model_digest = "e" * 64
        def __init__(self):
            self.full_calls = 0
            self.selected_calls = 0
        def score_many(self, message_batches, choice_ids):
            self.full_calls += 1
            raise AssertionError("full score_many path must not be used")
        def score_many_selected(self, message_batches, choice_ids):
            self.selected_calls += 1
            return tuple({choice: (0.8 if choice == "b" else 0.2) for choice in choice_ids}
                         for _ in message_batches)

    adapter = SelectedAdapter()
    manifest = run_balanced_score_choice_suite((make_case("case-1"),), adapter)
    assert adapter.full_calls == 0
    assert adapter.selected_calls == 1
    assert manifest["execution"]["score_method"] == "selected_output_projection_v1"
