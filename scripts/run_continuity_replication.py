from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spm_bench.continuity_data import generate_continuity_cases
from spm_bench.continuity_runner import run_continuity_benchmark
from spm_bench.hf_adapter import LocalHFAdapter

BENCHMARK_PATH = ROOT / "benchmarks" / "continuity_v1.jsonl"
BENCHMARK_SHA256 = "a3f5d26fff7e742ce14ca0cb1e22b4bbeb28eef4f80b45d6ab664d68abed41c2"
BENCHMARK_COMMIT = "d6b9c5690af6e0e35c7a4c62f77c64650bcf7e63"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--inventory-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--case-batch-size", type=int, default=1)
    parser.add_argument("--byte-ceiling", type=int, default=8192)
    parser.add_argument("--max-retrieved-tokens", type=int, default=512)
    args = parser.parse_args()

    if sha256_file(BENCHMARK_PATH) != BENCHMARK_SHA256:
        raise RuntimeError("continuity benchmark digest drift")

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    adapter = LocalHFAdapter(
        args.model_path,
        args.model_id,
        args.revision,
        {},
    )
    if adapter.model_digest != args.inventory_digest:
        raise RuntimeError(
            f"model inventory mismatch: {adapter.model_digest} != {args.inventory_digest}"
        )

    tokenizer, model = adapter._load()
    dtype = torch.bfloat16
    model.to(device=args.device, dtype=dtype)
    model.eval()

    result = run_continuity_benchmark(
        generate_continuity_cases(),
        adapter,
        tokenizer=tokenizer,
        case_batch_size=args.case_batch_size,
        byte_ceiling=args.byte_ceiling,
        max_retrieved_tokens=args.max_retrieved_tokens,
    )
    result["subject"] = {
        "model_id": adapter.model_id,
        "revision": adapter.revision,
        "inventory_digest": adapter.model_digest,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
    }
    result["benchmark_commit"] = BENCHMARK_COMMIT
    result["benchmark_file_sha256"] = BENCHMARK_SHA256
    result["execution"] = {
        "runner_head": git_head(),
        "torch_threads": args.threads,
        "case_batch_size": args.case_batch_size,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps({
        "metrics": result["metrics"],
        "subject": result["subject"],
        "receipt": str(args.output),
        "sha256": sha256_file(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
