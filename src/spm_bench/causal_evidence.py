"""Fail-closed causal-evidence receipts for SPM interventions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


SPECIFICITY_EVIDENCE_CLASS = "SEMANTIC_RELATION_SPECIFICITY"
_RELATIVE_TOLERANCE = 0.05


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NuisanceMatch:
    same_sequence: bool
    same_update_count: bool
    same_state_shape: bool
    same_dtype: bool
    same_text_context: bool
    perturbation_l2_relative_difference: float
    state_norm_relative_difference: float

    def mismatch_fields(self) -> tuple[str, ...]:
        mismatches: list[str] = []
        for field in (
            "same_sequence",
            "same_update_count",
            "same_state_shape",
            "same_dtype",
            "same_text_context",
        ):
            if getattr(self, field) is not True:
                mismatches.append(field)
        if self.perturbation_l2_relative_difference > _RELATIVE_TOLERANCE:
            mismatches.append("perturbation_l2_relative_difference")
        if self.state_norm_relative_difference > _RELATIVE_TOLERANCE:
            mismatches.append("state_norm_relative_difference")
        return tuple(mismatches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "same_sequence": self.same_sequence,
            "same_update_count": self.same_update_count,
            "same_state_shape": self.same_state_shape,
            "same_dtype": self.same_dtype,
            "same_text_context": self.same_text_context,
            "perturbation_l2_relative_difference": self.perturbation_l2_relative_difference,
            "state_norm_relative_difference": self.state_norm_relative_difference,
        }


@dataclass(frozen=True, slots=True)
class CausalEvidenceReceipt:
    evidence_class: str
    verdict: str
    predicted_direction: str
    relevant_effect: float
    nuisance_effect: float
    null_effect: float
    delta_min: float
    mismatches: tuple[str, ...]
    digest: str

def semantic_specificity_receipt(
    *,
    relevant_effect: float,
    nuisance_effect: float,
    null_effect: float,
    delta_min: float,
    nuisance_match: NuisanceMatch,
    predicted_direction: str,
) -> CausalEvidenceReceipt:
    if predicted_direction not in {"increase", "decrease"}:
        raise ValueError("predicted_direction must be increase or decrease")
    if delta_min <= 0.0:
        raise ValueError("delta_min must be positive")
    mismatches = nuisance_match.mismatch_fields()
    sign = 1.0 if predicted_direction == "increase" else -1.0
    directional_relevant = sign * relevant_effect
    directional_nuisance = sign * nuisance_effect
    directional_null = sign * null_effect

    if mismatches:
        verdict = "UNKNOWN"
    elif directional_relevant < delta_min:
        verdict = "FAIL"
    elif directional_relevant - directional_nuisance < delta_min:
        verdict = "FAIL"
    elif directional_null >= delta_min:
        verdict = "FAIL"
    else:
        verdict = "PASS"

    identity = {
        "evidence_class": SPECIFICITY_EVIDENCE_CLASS,
        "verdict": verdict,
        "predicted_direction": predicted_direction,
        "relevant_effect": relevant_effect,
        "nuisance_effect": nuisance_effect,
        "null_effect": null_effect,
        "delta_min": delta_min,
        "nuisance_match": nuisance_match.to_dict(),
        "mismatches": list(mismatches),
    }
    return CausalEvidenceReceipt(
        evidence_class=SPECIFICITY_EVIDENCE_CLASS,
        verdict=verdict,
        predicted_direction=predicted_direction,
        relevant_effect=relevant_effect,
        nuisance_effect=nuisance_effect,
        null_effect=null_effect,
        delta_min=delta_min,
        mismatches=mismatches,
        digest=_sha256(identity),
    )
