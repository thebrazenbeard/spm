import hashlib
import json

import pytest

from spm_bench.case import BenchmarkCase, load_jsonl_cases


VALID_CASE = {
    "case_id": "referent-001",
    "version": "1.0",
    "family": "referent_preservation",
    "turns": [
        {"role": "user", "content": "Morgan handed Riley the blue folder."},
        {"role": "user", "content": "Who has the folder?"},
    ],
    "choices": [
        {"id": "a", "text": "Morgan"},
        {"id": "b", "text": "Riley"},
    ],
    "expected_choice": "b",
    "risk_class": "low",
    "tags": ["referent", "transfer"],
}


@pytest.mark.parametrize("field", VALID_CASE)
def test_from_dict_requires_every_case_field(field):
    data = dict(VALID_CASE)
    del data[field]

    with pytest.raises(ValueError, match=field):
        BenchmarkCase.from_dict(data)


def test_from_dict_rejects_duplicate_choice_ids():
    data = dict(VALID_CASE)
    data["choices"] = [
        {"id": "a", "text": "Morgan"},
        {"id": "a", "text": "Riley"},
    ]

    with pytest.raises(ValueError, match="duplicate choice id"):
        BenchmarkCase.from_dict(data)


def test_from_dict_rejects_expected_choice_not_in_choices():
    data = dict(VALID_CASE)
    data["expected_choice"] = "missing"

    with pytest.raises(ValueError, match="expected_choice"):
        BenchmarkCase.from_dict(data)


def test_canonical_json_and_digest_are_stable_across_mapping_order():
    reordered = {key: VALID_CASE[key] for key in reversed(VALID_CASE)}
    case = BenchmarkCase.from_dict(VALID_CASE)
    reordered_case = BenchmarkCase.from_dict(reordered)

    expected_json = (
        '{"case_id":"referent-001","choices":[{"id":"a","text":"Morgan"},'
        '{"id":"b","text":"Riley"}],"expected_choice":"b",'
        '"family":"referent_preservation","risk_class":"low",'
        '"tags":["referent","transfer"],"turns":['
        '{"content":"Morgan handed Riley the blue folder.","role":"user"},'
        '{"content":"Who has the folder?","role":"user"}],"version":"1.0"}'
    )

    assert case.canonical_json() == expected_json
    assert reordered_case.canonical_json() == expected_json
    assert case.digest() == hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    assert reordered_case.digest() == case.digest()


def test_load_jsonl_cases_loads_cases_in_file_order(tmp_path):
    second = dict(VALID_CASE)
    second["case_id"] = "referent-002"
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join((json.dumps(VALID_CASE), "", json.dumps(second))) + "\n",
        encoding="utf-8",
    )

    cases = load_jsonl_cases(path)

    assert isinstance(cases, tuple)
    assert [case.case_id for case in cases] == ["referent-001", "referent-002"]
    assert all(isinstance(case, BenchmarkCase) for case in cases)
