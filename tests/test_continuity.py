import json

from spm_bench.continuity import (
    ContinuityCase,
    continuity_metrics,
    render_condition_context,
)
from spm_bench.continuity_data import generate_continuity_cases


def test_continuity_data_is_balanced_and_memory_necessary():
    cases = generate_continuity_cases()
    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == 24
    assert {case.family for case in cases} == {
        "initial_assignment", "supersession", "correction", "alias",
        "preference", "authority_change", "decision_finality", "project_state",
    }
    positions = {"a": 0, "b": 0, "c": 0}
    for case in cases:
        positions[case.expected_choice] += 1
        assert case.expected_text.casefold() not in case.query.casefold()
        assert case.events
    assert positions == {"a": 8, "b": 8, "c": 8}


def test_reset_context_omits_history_and_persistent_includes_memory():
    case = generate_continuity_cases()[0]
    reset = render_condition_context(case, condition="RESET", memory_text=None)
    assert all(event.content not in reset for event in case.events)
    persistent = render_condition_context(
        case, condition="PERSISTENT", memory_text="remembered earlier fact"
    )
    assert "remembered earlier fact" in persistent
    full = render_condition_context(case, condition="FULL_HISTORY", memory_text=None)
    assert all(event.content in full for event in case.events)


def test_metrics_measure_correction_burden_drift_and_paired_gain():
    cases = (
        ContinuityCase.from_dict({
            "case_id":"x1","version":1,"family":"supersession",
            "events":[
                {"session":1,"source_class":"user","content":"Old is R."},
                {"session":2,"source_class":"user","content":"Current is T; R is obsolete."},
            ],
            "query":"What is current?","choices":[
                {"id":"a","text":"R"},{"id":"b","text":"T"},{"id":"c","text":"Q"}],
            "expected_choice":"b","stale_choice":"a","risk_class":"low","tags":["currentness"],
        }),
        ContinuityCase.from_dict({
            "case_id":"x2","version":1,"family":"initial_assignment",
            "events":[{"session":1,"source_class":"user","content":"Marker is V."}],
            "query":"What is the marker?","choices":[
                {"id":"a","text":"U"},{"id":"b","text":"V"},{"id":"c","text":"W"}],
            "expected_choice":"b","stale_choice":None,"risk_class":"low","tags":["persistence"],
        }),
    )
    reset=[{"parsed_choice":"a","correct":False},{"parsed_choice":"c","correct":False}]
    persistent=[{"parsed_choice":"b","correct":True},{"parsed_choice":"b","correct":True}]
    metrics=continuity_metrics(cases, reset_results=reset, persistent_results=persistent)
    assert metrics["reset"]["user_correction_burden"] == 2
    assert metrics["persistent"]["user_correction_burden"] == 0
    assert metrics["reset"]["stale_error_count"] == 1
    assert metrics["persistent"]["stale_error_count"] == 0
    assert metrics["paired"]["persistent_only_correct"] == 2
    assert metrics["paired"]["reset_only_correct"] == 0
    assert metrics["paired"]["net_persistence_gain"] == 2
