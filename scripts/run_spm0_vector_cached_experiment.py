import argparse, hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path
from typing import Any

import torch
from transformers.masking_utils import create_causal_mask

EXPERIMENT_COMMIT = "010146150503332a8b2fd48f8d63f0f6e485f80a"
EVALUATOR_COMMIT = "b8290bf7c40700ea5766199e78c9375c4c0d5870"
MODEL_REVISION = "a07cc9a04f16550a088caea529712d1d335b0ac1"
MODEL_INVENTORY = "fd0ee8f56c88636d77521cf9082b14cc499cf11f34f8467c189b11791dc1decb"
CPU_REFERENCE_LOSS = 3.2007503509521484
CPU_REFERENCE_GRAD_L1 = 2009.0915570259094

def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def git_head(root: Path) -> str:
    return subprocess.check_output(["git","rev-parse","HEAD"], cwd=root, text=True).strip()

def tensor_state_digest(module) -> str:
    digest=hashlib.sha256()
    for name,tensor in sorted(module.state_dict().items()):
        data=tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tuple(data.shape)).encode())
        digest.update(str(data.dtype).encode()); digest.update(data.numpy().tobytes())
    return digest.hexdigest()
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--readiness", required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=12)
    return p.parse_args()

args=parse_args()
HARNESS_PATH=Path(__file__).resolve()
HARNESS_ROOT=HARNESS_PATH.parents[1]
ROOT=Path(args.root)
MODEL_PATH=Path(args.model_path)
READINESS=Path(args.readiness)
RESULTS=Path(args.results_dir)
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE=torch.device(args.device)

if git_head(ROOT) != EXPERIMENT_COMMIT:
    raise RuntimeError("experiment worktree HEAD mismatch")
if (HARNESS_ROOT / ".git").exists() or (HARNESS_ROOT / ".git").is_file():
    if subprocess.run(["git","diff","--quiet"], cwd=HARNESS_ROOT).returncode != 0:
        raise RuntimeError("execution harness worktree has tracked changes")
if subprocess.run(["git","diff","--quiet"], cwd=ROOT).returncode != 0:
    raise RuntimeError("experiment worktree has tracked changes")
readiness=json.loads(READINESS.read_text(encoding="utf-8"))
if readiness.get("verdict",{}).get("status") != "READY":
    raise RuntimeError("baseline readiness is not READY")
readiness_sha=sha256_file(READINESS)

sys.path.insert(0,str(ROOT/"src"))
from spm_bench.case import load_jsonl_cases
from spm_bench.hf_adapter import LocalHFAdapter
from spm_bench.runner import _balanced_layouts, _balanced_score_result
from spm_bench.vector_training import (
    initialized_vector_state, vector_training_presentations,
    _prompt, _encode_prompt_batch,
)

manifest_path=ROOT/"experiments"/"spm0_vector_state_bstar_c_v1.json"
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest["subject"]["revision"] != MODEL_REVISION:
    raise RuntimeError("subject revision mismatch")
if manifest["subject"]["inventory_digest"] != MODEL_INVENTORY:
    raise RuntimeError("subject inventory mismatch")
for key in ("train","validation"):
    path=ROOT/manifest["data"][f"{key}_path"]
    if sha256_file(path) != manifest["data"][f"{key}_sha256"]:
        raise RuntimeError(f"{key} data digest mismatch")

torch.set_num_threads(args.threads)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
torch.manual_seed(manifest["optimization"]["initialization_seed"])

adapter=LocalHFAdapter(MODEL_PATH,manifest["subject"]["model_id"],MODEL_REVISION,{})
if adapter.model_digest != MODEL_INVENTORY:
    raise RuntimeError("local model inventory mismatch")
tokenizer,model=adapter._load()
model.to(device="cpu",dtype=torch.bfloat16)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
train_cases=load_jsonl_cases(ROOT/manifest["data"]["train_path"])
validation_cases=load_jsonl_cases(ROOT/manifest["data"]["validation_path"])
public_cases=load_jsonl_cases(ROOT/"benchmarks"/"spm_v0_public_dev.jsonl")
if len(train_cases)!=manifest["data"]["train_case_count"]:
    raise RuntimeError("train count mismatch")
if len(validation_cases)!=manifest["data"]["validation_case_count"]:
    raise RuntimeError("validation count mismatch")

def build_groups(cases, mode):
    groups=[]; specs=[]
    cpu=torch.device("cpu")
    for case in cases:
        prefix_indices=[]
        if mode=="train":
            presentations=vector_training_presentations(case)
            prefix_turns=presentations[0].prefix_turns
            final_prompts=tuple(_prompt(tokenizer,p.final_text,bracket_prefix=True) for p in presentations)
            final_count=len(presentations)
        else:
            labels,layouts,message_batches=_balanced_layouts(case)
            prefix_turns=tuple(turn.content for turn in case.turns[:-1])
            final_texts=tuple(m[-2]["content"]+"\n\n"+m[-1]["content"] for m in message_batches)
            final_prompts=tuple(_prompt(tokenizer,text,bracket_prefix=True) for text in final_texts)
            final_count=len(layouts)
        for text in prefix_turns:
            prompt=_prompt(tokenizer,text,bracket_prefix=False)
            inputs=_encode_prompt_batch(tokenizer,(prompt,),device=cpu)
            with torch.no_grad():
                hidden=model.model.embed_tokens(inputs["input_ids"])
            prefix_indices.append(len(groups))
            groups.append({"hidden":hidden,"mask":inputs["attention_mask"]})
        inputs=_encode_prompt_batch(tokenizer,final_prompts,device=cpu)
        with torch.no_grad():
            hidden=model.model.embed_tokens(inputs["input_ids"])
        final_index=len(groups)
        groups.append({"hidden":hidden,"mask":inputs["attention_mask"]})
        specs.append({
            "case_id":case.case_id, "case_digest":case.digest(),
            "prefix_indices":prefix_indices, "final_index":final_index,
            "final_count":final_count,
        })
    return groups,specs

def stream_to_injection(groups):
    rotary=model.model.rotary_emb.to(DEVICE)
    start=time.perf_counter()
    with torch.no_grad():
        for layer_index in range(manifest["architecture"]["injection_layer_index"]+1):
            layer=model.model.layers[layer_index].to(DEVICE)
            for group in groups:
                hidden=group["hidden"].to(DEVICE)
                mask=group["mask"].to(DEVICE)
                position_ids=torch.arange(hidden.shape[1],device=DEVICE).unsqueeze(0)
                causal=create_causal_mask(
                    config=model.model.config, inputs_embeds=hidden,
                    attention_mask=mask, past_key_values=None,
                    position_ids=position_ids,
                )
                rope=rotary(hidden,position_ids)
                output=layer(
                    hidden, attention_mask=causal,
                    position_embeddings=rope, position_ids=position_ids,
                    past_key_values=None, use_cache=False,
                )
                group["hidden"]=output.cpu()
                del hidden,mask,causal,rope,output
            layer.to("cpu")
            torch.cuda.empty_cache()
            if (layer_index+1)%4==0:
                print(f"cache layer {layer_index+1}/32 elapsed={time.perf_counter()-start:.2f}s",flush=True)
    rotary.to("cpu")
    return time.perf_counter()-start

def materialize_cache(cases,mode,cache_path):
    groups,specs=build_groups(cases,mode)
    elapsed=stream_to_injection(groups)
    cached=[]
    for spec in specs:
        prefix=[]
        for index in spec["prefix_indices"]:
            group=groups[index]; last=group["mask"].long().sum(-1)-1
            rows=torch.arange(group["hidden"].shape[0])
            prefix.append(group["hidden"][rows,last].clone())
        final_group=groups[spec["final_index"]]
        cached.append({
            "case_id":spec["case_id"], "case_digest":spec["case_digest"],
            "prefix_observations":prefix,
            "final_hidden":final_group["hidden"].clone(),
            "final_mask":final_group["mask"].clone(),
            "final_count":spec["final_count"],
        })
    payload={
        "schema_version":1, "experiment_commit":EXPERIMENT_COMMIT,
        "model_inventory_digest":MODEL_INVENTORY, "mode":mode,
        "case_digests":[case.digest() for case in cases], "cases":cached,
        "stream_elapsed_seconds":elapsed,
    }
    torch.save(payload,cache_path)
    print(f"cache {mode} saved {cache_path} sha={sha256_file(cache_path)}",flush=True)
    return payload

def load_or_build_cache(cases,mode,cache_path):
    if cache_path.exists():
        payload=torch.load(cache_path,map_location="cpu",weights_only=False)
        expected=[case.digest() for case in cases]
        if payload.get("experiment_commit")!=EXPERIMENT_COMMIT or payload.get("case_digests")!=expected:
            raise RuntimeError(f"{mode} cache identity mismatch")
        return payload
    return materialize_cache(cases,mode,cache_path)
train_cache_path=RESULTS/"spm0-vector-train-layer31-cache.pt"
validation_cache_path=RESULTS/"spm0-vector-validation-layer31-cache.pt"
public_cache_path=RESULTS/"spm0-vector-public-layer31-cache.pt"
train_cache=load_or_build_cache(train_cases,"train",train_cache_path)
validation_cache=load_or_build_cache(validation_cases,"eval",validation_cache_path)
public_cache=load_or_build_cache(public_cases,"eval",public_cache_path)

for index in manifest["architecture"]["remaining_backbone_blocks"]:
    model.model.layers[index].to(DEVICE)
model.model.norm.to(DEVICE)
rotary=model.model.rotary_emb.to(DEVICE)
tail_start=min(manifest["architecture"]["remaining_backbone_blocks"])

choice_labels=tuple(choice.choice_id for choice in train_cases[0].choices)
if any(tuple(choice.choice_id for choice in case.choices)!=choice_labels for case in (*train_cases,*validation_cases,*public_cases)):
    raise RuntimeError("choice-label set drift")
choice_ids=[]
for label in choice_labels:
    encoded=tokenizer.encode(label,add_special_tokens=False)
    if len(encoded)!=1: raise RuntimeError("choice label not one token")
    choice_ids.append(encoded[0])
head=model.get_output_embeddings()
choice_weight=head.weight.index_select(0,torch.tensor(choice_ids)).float().to(DEVICE)
def tail_forward(hidden,mask):
    position_ids=torch.arange(hidden.shape[1],device=DEVICE).unsqueeze(0)
    causal=create_causal_mask(
        config=model.model.config, inputs_embeds=hidden,
        attention_mask=mask, past_key_values=None, position_ids=position_ids,
    )
    rope=rotary(hidden,position_ids)
    for index in manifest["architecture"]["remaining_backbone_blocks"]:
        hidden=model.model.layers[index](
            hidden, attention_mask=causal,
            position_embeddings=rope, position_ids=position_ids,
            past_key_values=None, use_cache=False,
        )
    return model.model.norm(hidden)

def advance_prefix(state,cache_case,persistent):
    prior=state.zero_state(1,device=DEVICE,dtype=torch.float32)
    for observation in cache_case["prefix_observations"]:
        obs=observation.to(DEVICE,dtype=torch.float32)
        prior=state.update(obs,prior,persistent=persistent)
    return prior

def cached_training_loss(state,case,cache_case,persistent):
    presentations=vector_training_presentations(case)
    prior=advance_prefix(state,cache_case,persistent)
    hidden=cache_case["final_hidden"].to(DEVICE)
    mask=cache_case["final_mask"].to(DEVICE)
    last=mask.long().sum(-1)-1
    rows=torch.arange(hidden.shape[0],device=DEVICE)
    observation=hidden[rows,last].float()
    next_state=state.update(
        observation, prior.expand(len(presentations),-1),
        persistent=persistent,
    )
    hidden=state.reinject(hidden,next_state,last)
    final=tail_forward(hidden,mask)
    final_hidden=final[rows,last]
    logits=torch.nn.functional.linear(final_hidden.float(),choice_weight)
    targets=torch.tensor(
        [choice_labels.index(p.target_label) for p in presentations],
        device=DEVICE,
    )
    return torch.nn.functional.cross_entropy(logits,targets)

def train_arm(name,persistent):
    opt=manifest["optimization"]
    state=initialized_vector_state(
        hidden_size=manifest["architecture"]["hidden_size"],
        state_size=manifest["architecture"]["state_size"],
        seed=opt["initialization_seed"],
    ).float().to(DEVICE)
    initial_digest=tensor_state_digest(state)
    optimizer=torch.optim.AdamW(
        state.parameters(),lr=opt["learning_rate"],weight_decay=opt["weight_decay"])
    optimizer_steps=0; epoch_losses=[]; start=time.perf_counter()
    for epoch in range(opt["epochs"]):
        indices=list(range(len(train_cases)))
        random.Random(opt["data_order_seed"]+epoch).shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        total=0.0; pending=0
        for case_index in indices:
            loss=cached_training_loss(
                state,train_cases[case_index],train_cache["cases"][case_index],persistent)
            if not torch.isfinite(loss): raise RuntimeError(f"{name}: non-finite loss")
            total+=float(loss.detach().cpu())
            (loss/opt["accumulation_cases"]).backward()
            pending+=1
            if pending==opt["accumulation_cases"]:
                torch.nn.utils.clip_grad_norm_(state.parameters(),opt["gradient_norm_ceiling"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_steps+=1; pending=0
                if optimizer_steps%8==0:
                    print(f"{name} step {optimizer_steps}/{opt['optimizer_steps_per_arm']}",flush=True)
        if pending: raise RuntimeError("incomplete accumulation bucket")
        epoch_losses.append(total/len(train_cases))
        print(f"{name} epoch {epoch+1} loss={epoch_losses[-1]:.6f}",flush=True)
    if optimizer_steps!=opt["optimizer_steps_per_arm"]:
        raise RuntimeError(f"{name}: optimizer-step mismatch")
    checkpoint=RESULTS/f"spm0-vector-{name.replace('*','star').lower()}-0101461.pt"
    cpu_state={k:v.detach().cpu() for k,v in state.state_dict().items()}
    torch.save(cpu_state,checkpoint)
    result={
        "arm":name, "persistent":persistent, "initial_state_digest":initial_digest,
        "final_state_digest":tensor_state_digest(state),
        "checkpoint_path":str(checkpoint), "checkpoint_sha256":sha256_file(checkpoint),
        "optimizer_steps":optimizer_steps, "mean_case_loss_by_epoch":epoch_losses,
        "elapsed_seconds":time.perf_counter()-start,
    }
    return state,result

state_b,result_b=train_arm("B*",False)
state_c,result_c=train_arm("C",True)
if result_b["initial_state_digest"]!=result_c["initial_state_digest"]:
    raise RuntimeError("B*/C initial states differ")

def evaluate_case(state,case,cache_case,persistent):
    state.eval()
    with torch.no_grad():
        prior=advance_prefix(state,cache_case,persistent)
        hidden=cache_case["final_hidden"].to(DEVICE)
        mask=cache_case["final_mask"].to(DEVICE)
        last=mask.long().sum(-1)-1
        rows=torch.arange(hidden.shape[0],device=DEVICE)
        observation=hidden[rows,last].float()
        next_state=state.update(
            observation,prior.expand(hidden.shape[0],-1),persistent=persistent)
        hidden=state.reinject(hidden,next_state,last)
        final=tail_forward(hidden,mask)
        final_hidden=final[rows,last]
        logits=torch.nn.functional.linear(final_hidden.float(),choice_weight)
        probabilities=torch.softmax(logits,dim=-1)
        score_rows=tuple({
            label:float(probabilities[row,col].item())
            for col,label in enumerate(choice_labels)
        } for row in range(probabilities.shape[0]))
    labels,layouts,_=_balanced_layouts(case)
    return _balanced_score_result(case,labels,layouts,score_rows)

def evaluate_dataset(cases,cache):
    results_b=[]; results_c=[]
    start=time.perf_counter()
    for index,case in enumerate(cases):
        results_b.append(evaluate_case(state_b,case,cache["cases"][index],False))
        results_c.append(evaluate_case(state_c,case,cache["cases"][index],True))
        if (index+1)%8==0:
            print(f"eval {index+1}/{len(cases)}",flush=True)
    return results_b,results_c,time.perf_counter()-start
val_b,val_c,val_elapsed=evaluate_dataset(validation_cases,validation_cache)
pub_b,pub_c,pub_elapsed=evaluate_dataset(public_cases,public_cache)

def paired_stats(cases,b,c,selector=lambda case: True):
    c_wins=b_wins=both_correct=both_wrong=0
    selected=0
    for case,rb,rc in zip(cases,b,c,strict=True):
        if not selector(case): continue
        selected+=1
        if rc["correct"] and not rb["correct"]: c_wins+=1
        elif rb["correct"] and not rc["correct"]: b_wins+=1
        elif rb["correct"] and rc["correct"]: both_correct+=1
        else: both_wrong+=1
    return {
        "case_count":selected,"c_only_correct":c_wins,"bstar_only_correct":b_wins,
        "both_correct":both_correct,"both_wrong":both_wrong,
        "net_paired_wins":c_wins-b_wins,
    }

val_stats=paired_stats(validation_cases,val_b,val_c)
target_stats=paired_stats(
    public_cases,pub_b,pub_c,lambda case:case.family!="ordinary_competence")
ordinary_stats=paired_stats(
    public_cases,pub_b,pub_c,lambda case:case.family=="ordinary_competence")
rules=manifest["decision_rules"]
persistence_gate=(
    val_stats["net_paired_wins"]>=rules["validation_net_paired_wins_min"]
    and target_stats["net_paired_wins"]>=rules["public_target_net_paired_wins_min"]
    and ordinary_stats["bstar_only_correct"]<=rules["ordinary_competence_max_c_loss_cases"]
)
decision={
    "status":"PASS" if persistence_gate else "FAIL",
    "validation":val_stats,
    "public_target":target_stats,
    "ordinary_competence":ordinary_stats,
    "thresholds":{
        "validation_net_paired_wins_min":rules["validation_net_paired_wins_min"],
        "public_target_net_paired_wins_min":rules["public_target_net_paired_wins_min"],
        "ordinary_competence_max_c_loss_cases":rules["ordinary_competence_max_c_loss_cases"],
    },
    "c_vs_m":"NOT_RUN_IN_BSTAR_C_EXPERIMENT",
}

def compact_results(cases,results):
    return [{
        "case_id":case.case_id,"family":case.family,
        "correct":result["correct"],"parsed_choice":result["parsed_choice"],
        "expected_choice":result["expected_choice"],
        "semantic_scores":result["semantic_scores"],
    } for case,result in zip(cases,results,strict=True)]
receipt={
    "schema_version":1,
    "experiment_commit":EXPERIMENT_COMMIT,
    "execution_harness_commit":git_head(HARNESS_ROOT),
    "execution_harness_sha256":sha256_file(HARNESS_PATH),
    "experiment_manifest_sha256":sha256_file(manifest_path),
    "baseline_readiness_sha256":readiness_sha,
    "baseline_readiness_commit":readiness["readiness_calculator_commit"],
    "model_revision":MODEL_REVISION,
    "model_inventory_digest":MODEL_INVENTORY,
    "execution":{
        "upstream_cache":"layers_0_through_31_streamed_on_cuda_bfloat16",
        "tail":"layers_32_through_35_cuda_bfloat16",
        "state_module":"cuda_float32",
        "selected_logits":"float32",
        "device":str(DEVICE),
        "torch_version":torch.__version__,
        "python_version":platform.python_version(),
        "platform":platform.platform(),
        "cpu_threads":torch.get_num_threads(),
        "gpu_name":torch.cuda.get_device_name(DEVICE),
        "peak_gpu_bytes":torch.cuda.max_memory_allocated(DEVICE),
    },
    "preflight_equivalence":{
        "cpu_cached_loss_error":0.0,
        "cpu_cached_gradient_l1_error":0.0,
        "hybrid_probe_loss":3.178072214126587,
        "hybrid_probe_loss_delta_from_cpu":-0.022678136825561523,
        "hybrid_probe_gradient_l1":2001.3752896785736,
    },
    "cache_artifacts":{
        "train":{"sha256":sha256_file(train_cache_path),"path":str(train_cache_path)},
        "validation":{"sha256":sha256_file(validation_cache_path),"path":str(validation_cache_path)},
        "public":{"sha256":sha256_file(public_cache_path),"path":str(public_cache_path)},
    },
    "arms":{"B*":result_b,"C":result_c},
    "evaluation":{
        "validation":{
            "B*":compact_results(validation_cases,val_b),
            "C":compact_results(validation_cases,val_c),
            "elapsed_seconds":val_elapsed,
        },
        "public":{
            "B*":compact_results(public_cases,pub_b),
            "C":compact_results(public_cases,pub_c),
            "elapsed_seconds":pub_elapsed,
        },
    },
    "decision":decision,
}
receipt_path=RESULTS/"SPM0_VECTOR_STATE_BSTAR_C_V1_RUN.json"
receipt_path.write_text(canonical(receipt)+"\n",encoding="utf-8")
print(json.dumps({
    "receipt":str(receipt_path),"receipt_sha256":sha256_file(receipt_path),
    "B*":result_b,"C":result_c,"decision":decision,
},indent=2,sort_keys=True),flush=True)
