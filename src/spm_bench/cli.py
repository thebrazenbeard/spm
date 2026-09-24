"""Command-line interface for local SPM benchmark work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .case import load_jsonl_cases
from .hf_adapter import LocalHFAdapter
from .runner import benchmark_digest, canonical_manifest_json, run_suite


def _emit(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spm-bench")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("suite")

    run = commands.add_parser("run")
    run.add_argument("--suite", required=True)
    run.add_argument("--model-path", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--max-new-tokens", type=int, default=4)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("result")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "validate":
        cases = load_jsonl_cases(args.suite)
        _emit({"benchmark_digest": benchmark_digest(cases), "case_count": len(cases)})
        return 0

    if args.command == "run":
        cases = load_jsonl_cases(args.suite)
        generation_config = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
        }
        adapter = LocalHFAdapter(
            args.model_path,
            args.model_id,
            args.revision,
            generation_config,
        )
        manifest = run_suite(cases, adapter, generation_config)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            canonical_manifest_json(manifest) + "\n", encoding="utf-8"
        )
        _emit({"output": str(output_path), "run_digest": manifest["run_digest"]})
        return 0

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    _emit({
        "benchmark_digest": result["benchmark_digest"],
        "model": result["model"],
        "run_digest": result["run_digest"],
        "summary": result["summary"],
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
