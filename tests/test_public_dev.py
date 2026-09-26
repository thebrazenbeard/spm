from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re

import pytest

from spm_bench.case import load_jsonl_cases


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = REPOSITORY_ROOT / "benchmarks" / "spm_v0_public_dev.jsonl"
MANIFEST_PATH = (
    REPOSITORY_ROOT / "benchmarks" / "SPM_V0_PUBLIC_DEV_MANIFEST.json"
)

REQUIRED_FAMILIES = {
    "referent_preservation",
    "proposition_fidelity",
    "correction_update",
    "pragmatic_act",
    "ambiguity_retention",
    "provenance_currentness",
    "authority_permission",
    "state_action_consistency",
    "ordinary_competence",
}
MINIMUM_CASES = 32
MINIMUM_CASES_PER_FAMILY = 3
PROJECT_SPECIFIC_NAME = re.compile(r"\b(?:patrick|vera)\b", re.IGNORECASE)


def _missing_artifacts() -> list[Path]:
    return [path for path in (SUITE_PATH, MANIFEST_PATH) if not path.is_file()]


@pytest.fixture(scope="module")
def public_dev_cases():
    if not SUITE_PATH.is_file():
        pytest.skip(f"public development fixture is not present: {SUITE_PATH}")
    return load_jsonl_cases(SUITE_PATH)


@pytest.fixture(scope="module")
def public_dev_manifest():
    if not MANIFEST_PATH.is_file():
        pytest.skip(f"public development manifest is not present: {MANIFEST_PATH}")
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        pytest.fail(f"public development manifest is not valid JSON: {error}")
    assert isinstance(manifest, Mapping), "public development manifest must be an object"
    return manifest


def test_frozen_public_dev_artifacts_exist():
    missing = _missing_artifacts()

    assert not missing, "missing frozen public development artifact(s): " + ", ".join(
        str(path.relative_to(REPOSITORY_ROOT)) for path in missing
    )


def test_public_dev_has_at_least_32_cases(public_dev_cases):
    assert len(public_dev_cases) >= MINIMUM_CASES


def test_public_dev_case_ids_are_unique(public_dev_cases):
    case_ids = [case.case_id for case in public_dev_cases]
    duplicates = sorted(
        case_id for case_id, count in Counter(case_ids).items() if count > 1
    )

    assert not duplicates, f"duplicate public development case_id values: {duplicates}"


def test_public_dev_covers_each_required_family(public_dev_cases):
    family_counts = Counter(case.family for case in public_dev_cases)
    underfilled = {
        family: family_counts[family]
        for family in sorted(REQUIRED_FAMILIES)
        if family_counts[family] < MINIMUM_CASES_PER_FAMILY
    }

    assert not underfilled, (
        "required families must each contain at least "
        f"{MINIMUM_CASES_PER_FAMILY} cases; observed: {underfilled}"
    )


def test_public_dev_expected_choices_identify_a_choice(public_dev_cases):
    invalid = {
        case.case_id: case.expected_choice
        for case in public_dev_cases
        if case.expected_choice not in {choice.choice_id for choice in case.choices}
    }

    assert not invalid, f"expected_choice does not identify a choice: {invalid}"


def test_public_dev_has_no_project_specific_lexical_dependency(public_dev_cases):
    occurrences: list[str] = []
    for case in public_dev_cases:
        lexical_fields = [
            (f"turns[{index}].content", turn.content)
            for index, turn in enumerate(case.turns)
        ]
        lexical_fields.extend(
            (f"choices[{index}].text", choice.text)
            for index, choice in enumerate(case.choices)
        )
        lexical_fields.extend(
            (f"tags[{index}]", tag) for index, tag in enumerate(case.tags)
        )
        occurrences.extend(
            f"{case.case_id}.{field}"
            for field, value in lexical_fields
            if PROJECT_SPECIFIC_NAME.search(value)
        )

    assert not occurrences, (
        "public cases must not depend on Patrick/Vera-specific vocabulary; found: "
        + ", ".join(occurrences)
    )


def test_public_dev_manifest_matches_frozen_suite(
    public_dev_cases, public_dev_manifest
):
    expected_sha256 = hashlib.sha256(SUITE_PATH.read_bytes()).hexdigest()
    expected_family_counts = dict(
        sorted(Counter(case.family for case in public_dev_cases).items())
    )

    assert public_dev_manifest.get("sha256") == expected_sha256
    assert public_dev_manifest.get("case_count") == len(public_dev_cases)
    assert public_dev_manifest.get("family_counts") == expected_family_counts
