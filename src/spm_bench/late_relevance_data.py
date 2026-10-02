from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import random


@dataclass(frozen=True, slots=True)
class LateRelevanceExample:
    case_id: str
    family: str
    source_chunks: tuple[str, ...]
    compact_state: str
    query: str
    exact_answer: str
    exact_required: bool
    decisive_chunk: int


_COLORS = (
    "red",
    "blue",
    "green",
    "amber",
    "violet",
    "orange",
    "silver",
    "teal",
)

_DISTRACTORS = (
    "The following note discusses routine preventive maintenance scheduling.",
    "A separate paragraph records a vendor callback with no bearing on the query.",
    "The operator then reviewed an unrelated safety checklist.",
    "A later status update concerns inventory counts for a different work area.",
    "The discussion briefly shifts to weather and travel timing.",
    "An unrelated diagnostic note records normal operating temperature.",
    "The next entry concerns a different artifact and should not answer the query.",
)


def _identifier(rng: random.Random, prefix: str) -> str:
    value = rng.getrandbits(48)
    return f"{prefix}-{value:012x}"


def _family_payload(
    rng: random.Random,
    family: str,
) -> tuple[str, str, str, str]:
    if family == "color":
        answer = rng.choice(_COLORS)
        decisive = (
            f"The exact tool-handle color recorded in the source is {answer}."
        )
        compact = "A tool-handle color was recorded."
        query = "What exact tool-handle color was recorded?"
        return decisive, compact, query, answer

    if family == "number":
        numerator = rng.randint(1, 9999)
        answer = f"{numerator / 10000:.4f} volts"
        decisive = (
            f"The exact approved calibration threshold is {answer}."
        )
        compact = "An exact calibration threshold was approved."
        query = "What exact calibration threshold was approved?"
        return decisive, compact, query, answer

    if family == "identifier":
        answer = _identifier(rng, "artifact-sha256")
        decisive = f"The exact recovery artifact identifier is {answer}."
        compact = "A recovery artifact identifier exists."
        query = "What is the exact recovery artifact identifier?"
        return decisive, compact, query, answer

    if family == "supersession":
        old = rng.choice(_COLORS)
        replacement_choices = tuple(color for color in _COLORS if color != old)
        answer = rng.choice(replacement_choices)
        decisive = (
            f"Correction: supersede the earlier value {old}; "
            f"the current exact policy value is {answer}."
        )
        compact = "A prior policy value was superseded by a correction."
        query = "What is the current exact policy value?"
        return decisive, compact, query, answer

    if family == "code_token":
        answer = f"__VeraState_{rng.getrandbits(24):06x}__"
        decisive = f"The exact sentinel token in code is {answer}."
        compact = "The code contains an exact sentinel token."
        query = "What exact sentinel token appears in the code?"
        return decisive, compact, query, answer

    raise ValueError(f"unknown late-relevance family: {family}")


def generate_late_relevance_examples(
    *,
    seed: int,
    count: int,
    max_chunks: int,
) -> tuple[LateRelevanceExample, ...]:
    if type(seed) is not int or isinstance(seed, bool):
        raise ValueError("seed must be an exact int")
    if type(count) is not int or isinstance(count, bool) or count < 1:
        raise ValueError("count must be a positive exact int")
    if (
        type(max_chunks) is not int
        or isinstance(max_chunks, bool)
        or max_chunks < 3
    ):
        raise ValueError("max_chunks must be an exact int >= 3")

    rng = random.Random(seed)
    families = (
        "color",
        "number",
        "identifier",
        "supersession",
        "code_token",
    )
    out: list[LateRelevanceExample] = []

    for index in range(count):
        family = families[index % len(families)]
        decisive_text, compact, query, answer = _family_payload(rng, family)
        chunk_count = rng.randint(3, max_chunks)
        decisive_chunk = rng.randrange(chunk_count)
        chunks = [
            rng.choice(_DISTRACTORS)
            + f" [case={index} distractor={position}]"
            for position in range(chunk_count)
        ]
        chunks[decisive_chunk] = decisive_text
        source_chunks = tuple(chunks)
        identity = (
            f"{seed}|{index}|{family}|{decisive_chunk}|"
            + "\n".join(source_chunks)
            + f"|{query}|{answer}"
        ).encode("utf-8")
        case_id = (
            f"{family}-{index:06d}-"
            f"{sha256(identity).hexdigest()[:10]}"
        )
        out.append(
            LateRelevanceExample(
                case_id=case_id,
                family=family,
                source_chunks=source_chunks,
                compact_state=compact,
                query=query,
                exact_answer=answer,
                exact_required=True,
                decisive_chunk=decisive_chunk,
            )
        )

    return tuple(out)
