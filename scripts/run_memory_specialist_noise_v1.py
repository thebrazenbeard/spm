from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HARNESS_PATH=Path(__file__).resolve()
HARNESS_ROOT=HARNESS_PATH.parents[1]
sys.path.insert(0,str(HARNESS_ROOT/"src"))

from spm_bench.arm_m_memory import ArmMMemory, MemoryRecord
from spm_bench.case import BenchmarkCase, BenchmarkTurn, load_jsonl_cases
from spm_bench.memory_specialist import MemorySpecialistRuntime
from spm_bench.runner import run_balanced_score_choice_suite

EXPERIMENT_COMMIT="406ffae04c888b795039ebdf2fc93793cc5dde5c"
MANIFEST_SHA="cd24384a496c68f20d3fdcea266d2fa477577c12fa8bdb6b1c54de26052ff1c8"
BASE_PATH=Path(r"C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
BASE_ID="Qwen/Qwen2.5-1.5B-Instruct"
BASE_DIGEST="866e9c64bd94ae56fbaf5f2a46c0d6578b9a44baeab1ebc269e43018f3b1766d"
ADAPTER_DIGEST="c8a835ccd2ab2547fcf2bc3fc5757a8b0b76584f670a3ad925470f5e3e7d1098"

def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def head(path:Path)->str:
    return subprocess.check_output(["git","rev-parse","HEAD"],cwd=path,text=True).strip()

def reset_case(case:BenchmarkCase)->BenchmarkCase:
    return BenchmarkCase(
        case_id=case.case_id+"__reset_base",version=case.version,family=case.family,
        turns=(case.turns[-1],),choices=case.choices,expected_choice=case.expected_choice,
        risk_class=case.risk_class,tags=case.tags+("noise:reset_base",),
    )

def correct_text(case:BenchmarkCase)->str:
    return next(choice.text for choice in case.choices if choice.choice_id==case.expected_choice)

def stale_text(case:BenchmarkCase)->str:
    prior="\n".join(turn.content for turn in case.turns[:-1])
    candidates=[
        choice.text for choice in case.choices
        if choice.choice_id!=case.expected_choice and choice.text in prior
    ]
    if len(candidates)!=1:
        raise RuntimeError(f"could not identify unique stale choice for {case.case_id}")
    return candidates[0]

def persistent_case(case:BenchmarkCase,tokenizer,byte_ceiling:int,max_tokens:int):
    memory=ArmMMemory.empty()
    for sequence,turn in enumerate(case.turns[:-1],start=1):
        memory=memory.append(
            MemoryRecord.create(sequence=sequence,source_class="user",content=turn.content),
            byte_ceiling=byte_ceiling,
        )
    query=case.turns[-1].content
    receipt=memory.retrieve(
        current_text=query,tokenizer=tokenizer,max_retrieved_tokens=max_tokens,
    )
    if not receipt.selected_records:
        raise RuntimeError(f"no records retrieved for {case.case_id}")
    context=(
        "Persistent memory from earlier interactions:\n"
        f"{receipt.rendered_text}\n\n{query}"
    )
    converted=BenchmarkCase(
        case_id=case.case_id+"__persistent",version=case.version,family=case.family,
        turns=(BenchmarkTurn(role="user",content=context),),choices=case.choices,
        expected_choice=case.expected_choice,risk_class=case.risk_class,
        tags=case.tags+("noise:persistent",),
    )
    return converted,receipt,len(memory.records)

def summary(results):
    correct=sum(1 for row in results if row["correct"])
    return {"case_count":len(results),"correct_count":correct,
            "correction_burden":len(results)-correct}

def paired(base,adapted):
    adapter_only=base_only=both=neither=0
    for b,a in zip(base,adapted,strict=True):
        if a["correct"] and not b["correct"]: adapter_only+=1
        elif b["correct"] and not a["correct"]: base_only+=1
        elif b["correct"] and a["correct"]: both+=1
        else: neither+=1
    return {"adapter_only_correct":adapter_only,"base_only_correct":base_only,
            "both_correct":both,"both_wrong":neither,
            "net_adapter_gain":adapter_only-base_only}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--adapter-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    if head(root)!=EXPERIMENT_COMMIT:
        raise RuntimeError("experiment worktree HEAD mismatch")
    manifest_path=root/"experiments"/"spm_memory_specialist_noise_v1.json"
    if sha(manifest_path)!=MANIFEST_SHA:
        raise RuntimeError("manifest digest mismatch")
    spec=json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path=root/spec["data"]["path"]
    if sha(data_path)!=spec["data"]["sha256"]:
        raise RuntimeError("noise data digest mismatch")
    adapter_path=args.adapter_dir.resolve()
    if sha(adapter_path/"adapter_model.safetensors")!=ADAPTER_DIGEST:
        raise RuntimeError("adapter digest mismatch")

    runtime=MemorySpecialistRuntime.load_quantized(
        base_path=BASE_PATH,adapter_path=adapter_path,model_id=BASE_ID,
        base_inventory_digest=BASE_DIGEST,adapter_digest=ADAPTER_DIGEST,microbatch=3,
    )
    source=load_jsonl_cases(data_path)
    resets=tuple(reset_case(case) for case in source)
    policy=spec["memory_policy"]
    pairs=tuple(
        persistent_case(
            case,runtime.tokenizer,
            byte_ceiling=policy["byte_ceiling"],
            max_tokens=policy["max_retrieved_tokens"],
        )
        for case in source
    )
    persistent=tuple(item[0] for item in pairs)
    receipts=tuple(item[1] for item in pairs)
    retained_counts=tuple(item[2] for item in pairs)

    current_hits=0
    stale_hits=0
    for case,receipt in zip(source,receipts,strict=True):
        rendered="\n".join(record.content for record in receipt.selected_records)
        current_hits+=int(correct_text(case) in rendered)
        stale_hits+=int(stale_text(case) in rendered)

    base=runtime.base_view()
    memory=runtime.memory_view()
    reset_manifest=run_balanced_score_choice_suite(resets,base,case_batch_size=1)
    persistent_base_manifest=run_balanced_score_choice_suite(persistent,base,case_batch_size=1)
    persistent_adapter_manifest=run_balanced_score_choice_suite(persistent,memory,case_batch_size=1)
    full_manifest=run_balanced_score_choice_suite(source,base,case_batch_size=1)

    rb=reset_manifest["results"]
    pb=persistent_base_manifest["results"]
    pa=persistent_adapter_manifest["results"]
    fh=full_manifest["results"]
    summaries={
        "RESET_BASE":summary(rb),
        "PERSISTENT_BASE":summary(pb),
        "PERSISTENT_ADAPTER":summary(pa),
        "FULL_HISTORY_BASE":summary(fh),
    }
    pair=paired(pb,pa)
    retrieval={
        "current_record_hit_count":current_hits,
        "stale_record_hit_count":stale_hits,
        "truncated_count":sum(1 for r in receipts if r.truncated),
        "max_token_count":max(r.token_count for r in receipts),
        "min_retained_record_count":min(retained_counts),
        "max_retained_record_count":max(retained_counts),
    }
    g=spec["success_gates"]
    gates={
        "retrieval_current_hit":current_hits>=g["retrieval_current_hit_min"],
        "persistent_adapter_correct":
            summaries["PERSISTENT_ADAPTER"]["correct_count"]>=g["persistent_adapter_correct_min"],
        "persistent_adapter_net_gain":
            pair["net_adapter_gain"]>=g["persistent_adapter_net_gain_over_base_min"],
        "full_history_base_correct":
            summaries["FULL_HISTORY_BASE"]["correct_count"]>=g["full_history_base_correct_min"],
        "truncation_is_real":
            retrieval["truncated_count"]>=g["truncated_retrieval_count_min"],
    }
    receipt={
        "schema_version":1,
        "experiment_commit":EXPERIMENT_COMMIT,
        "execution_harness_commit":head(HARNESS_ROOT),
        "execution_harness_sha256":sha(HARNESS_PATH),
        "manifest_sha256":sha(manifest_path),
        "data_sha256":sha(data_path),
        "adapter_sha256":ADAPTER_DIGEST,
        "summaries":summaries,
        "persistent_adapter_vs_base":pair,
        "retrieval":retrieval,
        "gates":gates,
        "status":"PASS" if all(gates.values()) else "FAIL",
        "conditions":{
            "RESET_BASE":reset_manifest,
            "PERSISTENT_BASE":persistent_base_manifest,
            "PERSISTENT_ADAPTER":persistent_adapter_manifest,
            "FULL_HISTORY_BASE":full_manifest,
        },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(receipt,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    print(json.dumps({
        "status":receipt["status"],"gates":gates,"summaries":summaries,
        "paired":pair,"retrieval":retrieval,
        "receipt":str(args.output),"sha256":sha(args.output),
    },indent=2))

if __name__=="__main__":
    main()
