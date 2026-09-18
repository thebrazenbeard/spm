"""Command-line interface for the qualified gated memory specialist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .arm_m_memory import MemoryRecord
from .memory_specialist import (
    MemoryResolution,
    MemorySpecialistRuntime,
    resolve_memory_choice,
)


DEFAULT_BASE_INVENTORY = (
    "866e9c64bd94ae56fbaf5f2a46c0d6578b9a44baeab1ebc269e43018f3b1766d"
)
DEFAULT_ADAPTER_DIGEST = (
    "c8a835ccd2ab2547fcf2bc3fc5757a8b0b76584f670a3ad925470f5e3e7d1098"
)
DEFAULT_ADAPTER_CONFIG_DIGEST = (
    "10afd4d6a8153d948cf4e8e6ecddf9057a595c4c465e38f93afbdd5ce5781adf"
)
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def build_records(
    contents: Sequence[str],
    *,
    source_class: str = "user",
) -> tuple[MemoryRecord, ...]:
    return tuple(
        MemoryRecord.create(
            sequence=index,
            source_class=source_class,
            content=content,
        )
        for index, content in enumerate(contents, start=1)
    )


def resolution_to_dict(resolution: MemoryResolution) -> dict:
    return {
        "chosen_index": resolution.chosen_index,
        "chosen_text": resolution.chosen_text,
        "semantic_scores": list(resolution.semantic_scores),
        "adapter_active": resolution.adapter_active,
        "retrieval_receipt_digest": resolution.retrieval_receipt_digest,
        "selected_record_digests": list(resolution.selected_record_digests),
        "token_count": resolution.token_count,
        "truncated": resolution.truncated,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spm-memory-resolve",
        description=(
            "Resolve one of three candidate states using bounded persistent memory "
            "and the qualified Qwen1.5B gated memory specialist."
        ),
    )
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--memory",
        action="append",
        default=[],
        help="Prior observation. Repeat in chronological order.",
    )
    parser.add_argument(
        "--choice",
        action="append",
        required=True,
        help="Candidate answer. Supply exactly three times.",
    )
    parser.add_argument(
        "--source-class",
        default="user",
        choices=("user", "tool", "environment", "system_observation"),
    )
    parser.add_argument("--byte-ceiling", type=int, default=8192)
    parser.add_argument("--max-retrieved-tokens", type=int, default=512)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.choice) != 3:
        raise SystemExit("--choice must be supplied exactly three times")

    runtime = MemorySpecialistRuntime.load_quantized(
        base_path=args.base_path,
        adapter_path=args.adapter_dir,
        model_id=DEFAULT_MODEL_ID,
        base_inventory_digest=DEFAULT_BASE_INVENTORY,
        adapter_digest=DEFAULT_ADAPTER_DIGEST,
        adapter_config_digest=DEFAULT_ADAPTER_CONFIG_DIGEST,
        microbatch=3,
    )
    records = build_records(args.memory, source_class=args.source_class)
    resolution = resolve_memory_choice(
        runtime,
        memory_records=records,
        query=args.query,
        choices=tuple(args.choice),
        byte_ceiling=args.byte_ceiling,
        max_retrieved_tokens=args.max_retrieved_tokens,
    )
    print(json.dumps(resolution_to_dict(resolution), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
