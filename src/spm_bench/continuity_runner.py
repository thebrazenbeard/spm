"""Execution helpers for cross-session continuity conditions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .arm_m_memory import ArmMMemory, MemoryRecord
from .case import BenchmarkCase, BenchmarkTurn
from .continuity import ContinuityCase, continuity_metrics, render_condition_context
from .runner import run_balanced_score_choice_suite


def build_condition_case(
    source: ContinuityCase,
    *,
    condition: str,
    tokenizer: Any,
    byte_ceiling: int = 8192,
    max_retrieved_tokens: int = 512,
):
    condition = condition.upper()
    receipt = None
    memory_text = None
    if condition == "PERSISTENT":
        if tokenizer is None:
            raise ValueError("persistent condition requires a tokenizer")
        memory = ArmMMemory.empty()
        for sequence, event in enumerate(source.events, start=1):
            memory = memory.append(
                MemoryRecord.create(
                    sequence=sequence,
                    source_class=event.source_class,
                    content=event.content,
                ),
                byte_ceiling=byte_ceiling,
            )
        receipt = memory.retrieve(
            current_text=source.query,
            tokenizer=tokenizer,
            max_retrieved_tokens=max_retrieved_tokens,
        )
        memory_text = receipt.rendered_text
    context = render_condition_context(source, condition=condition, memory_text=memory_text)
    return BenchmarkCase(
        case_id=f"{source.case_id}__{condition.lower()}",
        version=source.version,
        family=source.family,
        turns=(BenchmarkTurn(role="user", content=context),),
        choices=source.choices,
        expected_choice=source.expected_choice,
        risk_class=source.risk_class,
        tags=source.tags + (f"condition:{condition.lower()}",),
    ), receipt


def _summary(cases: Sequence[ContinuityCase], results):
    correct=sum(1 for row in results if row["correct"])
    stale=sum(
        1 for case,row in zip(cases,results,strict=True)
        if case.stale_choice is not None and row["parsed_choice"] == case.stale_choice
    )
    return {
        "case_count": len(cases),
        "correct_count": correct,
        "user_correction_burden": len(cases)-correct,
        "stale_error_count": stale,
    }


def run_continuity_benchmark(
    cases: Sequence[ContinuityCase],
    adapter: Any,
    *,
    tokenizer: Any,
    case_batch_size: int = 1,
    byte_ceiling: int = 8192,
    max_retrieved_tokens: int = 512,
):
    ordered=tuple(cases)
    manifests={}
    retrieval_receipts={}
    for condition in ("RESET","PERSISTENT","FULL_HISTORY"):
        built=[]
        receipts=[]
        for case in ordered:
            converted, receipt=build_condition_case(
                case, condition=condition, tokenizer=tokenizer,
                byte_ceiling=byte_ceiling,
                max_retrieved_tokens=max_retrieved_tokens,
            )
            built.append(converted)
            receipts.append(receipt)
        manifest=run_balanced_score_choice_suite(
            tuple(built), adapter, case_batch_size=case_batch_size
        )
        manifests[condition]=manifest
        if condition=="PERSISTENT":
            retrieval_receipts[condition]=[
                {
                    "case_id": case.case_id,
                    "receipt_digest": receipt.receipt_digest,
                    "token_count": receipt.token_count,
                    "truncated": receipt.truncated,
                    "selected_record_digests": [r.digest for r in receipt.selected_records],
                }
                for case,receipt in zip(ordered,receipts,strict=True)
            ]
    reset=manifests["RESET"]["results"]
    persistent=manifests["PERSISTENT"]["results"]
    full=manifests["FULL_HISTORY"]["results"]
    metrics=continuity_metrics(
        ordered, reset_results=reset, persistent_results=persistent
    )
    metrics["full_history"]=_summary(ordered,full)
    metrics["persistent_vs_full_history"]={
        "correct_gap": metrics["full_history"]["correct_count"] - metrics["persistent"]["correct_count"],
        "correction_burden_gap":
            metrics["persistent"]["user_correction_burden"] - metrics["full_history"]["user_correction_burden"],
    }
    return {
        "schema_version": 1,
        "conditions": manifests,
        "metrics": metrics,
        "retrieval": retrieval_receipts,
        "memory_policy": {
            "byte_ceiling": byte_ceiling,
            "max_retrieved_tokens": max_retrieved_tokens,
        },
    }
