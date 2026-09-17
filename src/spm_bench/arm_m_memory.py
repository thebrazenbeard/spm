"""Deterministic conventional persistent-memory control for SPM experiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable


POLICY_VERSION = "M_HYBRID_EXTRACTIVE_V1"
_ALLOWED_SOURCES = frozenset({"user", "tool", "environment", "system_observation"})
_TERM_RE = re.compile(r"\w+", flags=re.UNICODE)
_BM25_K1 = 1.2
_BM25_B = 0.75


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _terms(text: str) -> list[str]:
    return _TERM_RE.findall(text.casefold())

@dataclass(frozen=True, slots=True)
class MemoryRecord:
    sequence: int
    source_class: str
    content: str
    digest: str

    @classmethod
    def create(cls, *, sequence: int, source_class: str, content: str) -> "MemoryRecord":
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ValueError("sequence must be a positive integer")
        if source_class not in _ALLOWED_SOURCES:
            raise ValueError("memory source must be an accepted exogenous observation")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        identity = {"sequence": sequence, "source_class": source_class, "content": content}
        digest = _sha256_text(_canonical_json(identity))
        return cls(sequence=sequence, source_class=source_class, content=content, digest=digest)

    def canonical_bytes(self) -> bytes:
        return _canonical_json({
            "sequence": self.sequence,
            "source_class": self.source_class,
            "content": self.content,
            "digest": self.digest,
        }).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RetrievalReceipt:
    selected_records: tuple[MemoryRecord, ...]
    recency_records: tuple[MemoryRecord, ...]
    lexical_records: tuple[MemoryRecord, ...]
    rendered_text: str
    token_count: int
    truncated: bool
    receipt_digest: str

def _token_ids(tokenizer: Any, text: str) -> list[Any]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _render_record(record: MemoryRecord) -> str:
    return f"[{record.sequence}|{record.source_class}]\n{record.content}"


def _bm25_scores(records: tuple[MemoryRecord, ...], query: str) -> dict[str, float]:
    if not records:
        return {}
    document_terms = {record.digest: _terms(record.content) for record in records}
    lengths = [len(terms) for terms in document_terms.values()]
    average_length = sum(lengths) / len(lengths) if lengths else 1.0
    document_frequency: Counter[str] = Counter()
    for terms in document_terms.values():
        document_frequency.update(set(terms))
    query_counts = Counter(_terms(query))
    total_documents = len(records)
    scores: dict[str, float] = {}
    for record in records:
        terms = document_terms[record.digest]
        frequencies = Counter(terms)
        score = 0.0
        for term, query_frequency in query_counts.items():
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_count = document_frequency[term]
            inverse = math.log(1.0 + (total_documents - document_count + 0.5) / (document_count + 0.5))
            denominator = frequency + _BM25_K1 * (
                1.0 - _BM25_B + _BM25_B * (len(terms) / average_length if average_length else 0.0)
            )
            score += query_frequency * inverse * (frequency * (_BM25_K1 + 1.0) / denominator)
        scores[record.digest] = score
    return scores

@dataclass(frozen=True, slots=True)
class ArmMMemory:
    records: tuple[MemoryRecord, ...]

    @classmethod
    def empty(cls) -> "ArmMMemory":
        return cls(records=())

    @property
    def byte_size(self) -> int:
        return sum(len(record.canonical_bytes()) + 1 for record in self.records)

    def append(self, record: MemoryRecord, *, byte_ceiling: int) -> "ArmMMemory":
        if isinstance(byte_ceiling, bool) or not isinstance(byte_ceiling, int) or byte_ceiling <= 0:
            raise ValueError("byte ceiling must be a positive integer")
        if self.records:
            last = self.records[-1]
            if record.sequence == last.sequence:
                if record.digest == last.digest:
                    return self
                raise ValueError("conflicting record for existing sequence")
            if record.sequence != last.sequence + 1:
                raise ValueError("sequence must advance by exactly one")
        elif record.sequence != 1:
            raise ValueError("first sequence must be one")

        retained = list(self.records) + [record]
        while len(retained) > 1 and sum(len(item.canonical_bytes()) + 1 for item in retained) > byte_ceiling:
            retained.pop(0)
        result = ArmMMemory(records=tuple(retained))
        if result.byte_size > byte_ceiling:
            raise ValueError("single observation exceeds memory byte ceiling")
        return result

    def retrieve(
        self,
        *,
        current_text: str,
        tokenizer: Any,
        max_retrieved_tokens: int,
        excluded_record_digests: Iterable[str] = (),
    ) -> RetrievalReceipt:
        if not isinstance(current_text, str):
            raise ValueError("current_text must be a string")
        if isinstance(max_retrieved_tokens, bool) or not isinstance(max_retrieved_tokens, int) or max_retrieved_tokens <= 0:
            raise ValueError("max_retrieved_tokens must be a positive integer")
        excluded = set(excluded_record_digests)
        candidates = tuple(record for record in self.records if record.digest not in excluded)
        header = "MEMORY_CONTEXT_V1\nRECENCY\n\nLEXICAL\n"
        header_tokens = len(_token_ids(tokenizer, header))
        content_budget = max_retrieved_tokens - header_tokens
        if candidates and content_budget < 2:
            raise ValueError("token budget is too small to preserve both retrieval lanes")

        def pack(records, budget):
            selected = []
            fragments = []
            used = 0
            partial = False
            for record in records:
                if used >= budget:
                    break
                ids = _token_ids(tokenizer, record.content)
                if not ids:
                    continue
                take = ids[: budget - used]
                if len(take) < len(ids):
                    partial = True
                selected.append(record)
                fragments.append(tokenizer.decode(take, skip_special_tokens=True))
                used += len(take)
            return tuple(selected), "\n".join(fragments), used, partial

        recency_budget = (content_budget + 1) // 2
        lexical_budget = content_budget - recency_budget
        recency_order = tuple(reversed(candidates))
        recency_records, recency_text, recency_used, recency_partial = pack(recency_order, recency_budget)
        recency_ids = {record.digest for record in recency_records}
        lexical_candidates = tuple(record for record in candidates if record.digest not in recency_ids)
        scores = _bm25_scores(lexical_candidates, current_text)
        lexical_order = tuple(sorted(
            lexical_candidates,
            key=lambda record: (-scores[record.digest], -record.sequence, record.digest),
        ))
        lexical_records, lexical_text, lexical_used, lexical_partial = pack(lexical_order, lexical_budget)

        # Reallocate an unused lane allowance before rendering, without changing rank order.
        spare = content_budget - recency_used - lexical_used
        if spare > 0 and lexical_used < lexical_budget:
            recency_budget += spare
            recency_records, recency_text, recency_used, recency_partial = pack(recency_order, recency_budget)
            recency_ids = {record.digest for record in recency_records}
        elif spare > 0 and recency_used < recency_budget:
            lexical_budget += spare
            lexical_candidates = tuple(record for record in candidates if record.digest not in recency_ids)
            scores = _bm25_scores(lexical_candidates, current_text)
            lexical_order = tuple(sorted(
                lexical_candidates,
                key=lambda record: (-scores[record.digest], -record.sequence, record.digest),
            ))
            lexical_records, lexical_text, lexical_used, lexical_partial = pack(lexical_order, lexical_budget)

        def render():
            return f"MEMORY_CONTEXT_V1\nRECENCY\n{recency_text}\nLEXICAL\n{lexical_text}"

        rendered = render()
        encoded = _token_ids(tokenizer, rendered)
        if len(encoded) > max_retrieved_tokens:
            raise RuntimeError("retrieval lane packing exceeded the hard token ceiling")
        ordered = tuple(sorted(
            {record.digest: record for record in recency_records + lexical_records}.values(),
            key=lambda record: record.sequence,
        ))
        truncated = recency_partial or lexical_partial or len(ordered) < len(candidates)
        identity = {
            "policy_version": POLICY_VERSION,
            "current_text_digest": _sha256_text(current_text),
            "excluded_record_digests": sorted(excluded),
            "recency_record_digests": [record.digest for record in recency_records],
            "lexical_record_digests": [record.digest for record in lexical_records],
            "selected_record_digests": [record.digest for record in ordered],
            "max_retrieved_tokens": max_retrieved_tokens,
            "token_count": len(encoded),
            "truncated": truncated,
            "rendered_digest": _sha256_text(rendered),
        }
        return RetrievalReceipt(
            selected_records=ordered,
            recency_records=recency_records,
            lexical_records=lexical_records,
            rendered_text=rendered,
            token_count=len(encoded),
            truncated=truncated,
            receipt_digest=_sha256_text(_canonical_json(identity)),
        )
