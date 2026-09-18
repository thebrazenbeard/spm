from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from spm_bench.case import load_jsonl_cases
from spm_bench.continuity import continuity_metrics
from spm_bench.continuity_data import generate_continuity_cases
from spm_bench.continuity_runner import build_condition_case
from spm_bench.memory_specialist import MemorySpecialistRuntime
from spm_bench.runner import run_balanced_score_choice_suite

BASE_PATH=Path(r"C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
BASE_INVENTORY="866e9c64bd94ae56fbaf5f2a46c0d6578b9a44baeab1ebc269e43018f3b1766d"
ADAPTER_CONFIG_DIGEST="3ad675932f62fd5aa6a10d8893ca11fb19011e8519bc5379d7a6129026a362d3"
QUALIFICATION_PATH=ROOT/"experiments"/"spm_memory_specialist_gated_runtime_v1.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()


def condition_summary(cases, results):
    correct=sum(1 for row in results if row["correct"])
    stale=sum(
        1 for case,row in zip(cases,results,strict=True)
        if case.stale_choice is not None and row["parsed_choice"]==case.stale_choice
    )
    return {
        "case_count":len(cases),
        "correct_count":correct,
        "user_correction_burden":len(cases)-correct,
        "stale_error_count":stale,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--adapter-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()

    spec=json.loads(QUALIFICATION_PATH.read_text(encoding="utf-8"))
    adapter_weights=args.adapter_dir/"adapter_model.safetensors"
    if sha256_file(adapter_weights)!=spec["subject"]["adapter_sha256"]:
        raise RuntimeError("adapter digest mismatch")

    runtime=MemorySpecialistRuntime.load_quantized(
        base_path=BASE_PATH,
        adapter_path=args.adapter_dir,
        model_id=MODEL_ID,
        base_inventory_digest=BASE_INVENTORY,
        adapter_digest=spec["subject"]["adapter_sha256"],
        adapter_config_digest=ADAPTER_CONFIG_DIGEST,
        microbatch=3,
    )

    source_cases=generate_continuity_cases()
    manifests={}
    for condition in ("RESET","PERSISTENT","FULL_HISTORY"):
        built=[]
        receipts=[]
        for case in source_cases:
            converted,receipt=build_condition_case(
                case,
                condition=condition,
                tokenizer=runtime.tokenizer,
                byte_ceiling=8192,
                max_retrieved_tokens=512,
            )
            built.append(converted)
            receipts.append(receipt)
        retrieved_count=1 if condition=="PERSISTENT" else 0
        view=runtime.view_for_retrieval(retrieved_count)
        manifest=run_balanced_score_choice_suite(
            tuple(built),view,case_batch_size=1
        )
        manifests[condition]=manifest

    reset=manifests["RESET"]["results"]
    persistent=manifests["PERSISTENT"]["results"]
    full=manifests["FULL_HISTORY"]["results"]
    metrics=continuity_metrics(
        source_cases,reset_results=reset,persistent_results=persistent
    )
    metrics["full_history"]=condition_summary(source_cases,full)
    metrics["persistent_vs_full_history"]={
        "correct_gap":metrics["full_history"]["correct_count"]-metrics["persistent"]["correct_count"],
        "correction_burden_gap":
            metrics["persistent"]["user_correction_burden"]-metrics["full_history"]["user_correction_burden"],
    }

    public_cases=load_jsonl_cases(ROOT/"benchmarks"/"spm_v0_public_dev.jsonl")
    public_manifest=run_balanced_score_choice_suite(
        public_cases,runtime.base_view(),case_batch_size=1
    )
    public_correct=public_manifest["summary"]["correct_count"]

    gates_spec=spec["success_gates"]
    gates={
        "persistent_correct":
            metrics["persistent"]["correct_count"]>=gates_spec["persistent_correct_min"],
        "persistent_stale_errors":
            metrics["persistent"]["stale_error_count"]<=gates_spec["persistent_stale_errors_max"],
        "persistent_correction_burden":
            metrics["persistent"]["user_correction_burden"]<=gates_spec["persistent_user_correction_burden_max"],
        "full_history_correct":
            metrics["full_history"]["correct_count"]>=gates_spec["full_history_correct_min"],
        "general_public36_correct":
            public_correct>=gates_spec["general_public36_correct_min"],
        "reset_uses_base":
            runtime.view_for_retrieval(0).memory_active is False,
        "general_uses_base":
            runtime.base_view().memory_active is False,
        "persistent_uses_adapter":
            runtime.view_for_retrieval(1).memory_active is True,
    }
    receipt={
        "schema_version":1,
        "qualification_manifest_sha256":sha256_file(QUALIFICATION_PATH),
        "qualification_manifest_commit":"ab94f19a517bacdf496900bac316062c0cb0e903",
        "runtime_commit":"d23456a285eadad9dd2b60ab3879be9d4d9aa3cf",
        "qualification_harness_commit":git_head(),
        "qualification_harness_sha256":sha256_file(Path(__file__).resolve()),
        "adapter_sha256":spec["subject"]["adapter_sha256"],
        "activation_policy":spec["activation_policy"],
        "continuity":{"conditions":manifests,"metrics":metrics},
        "general_public36":public_manifest,
        "gates":gates,
        "status":"PASS" if all(gates.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(receipt,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status":receipt["status"],
        "gates":gates,
        "continuity_metrics":metrics,
        "public36_correct":public_correct,
        "receipt":str(args.output),
        "receipt_sha256":sha256_file(args.output),
    },indent=2))


if __name__=="__main__":
    main()
