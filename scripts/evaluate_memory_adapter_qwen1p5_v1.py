from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spm_bench.case import load_jsonl_cases
from spm_bench.continuity_data import generate_continuity_cases
from spm_bench.continuity_runner import run_continuity_benchmark
from spm_bench.runner import run_balanced_score_choice_suite

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
BASE_INVENTORY = "866e9c64bd94ae56fbaf5f2a46c0d6578b9a44baeab1ebc269e43018f3b1766d"
BASE_PATH = Path(r"C:\Users\patri\.cache\huggingface\hub\models--Qwen--Qwen2.5-1.5B-Instruct\snapshots\989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
CONTINUITY_SHA = "a3f5d26fff7e742ce14ca0cb1e22b4bbeb28eef4f80b45d6ab664d68abed41c2"
PUBLIC36_SHA = "6947c936892bbfff300326ecc0bb6e208274a4243222805a9c5905711cbac30b"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


class LoadedChoiceAdapter:
    def __init__(self, tokenizer, model, *, model_id: str, model_digest: str, microbatch: int = 3):
        self.tokenizer = tokenizer
        self.model = model
        self.model_id = model_id
        self.model_digest = model_digest
        self.microbatch = microbatch

    def score_many_selected(self, message_batches, choice_ids):
        choices = tuple(choice_ids)
        token_ids = []
        for choice in choices:
            encoded = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("choice IDs must be single tokens")
            token_ids.append(encoded[0])
        results = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        device = next(p for p in self.model.parameters() if p.is_cuda).device
        selected_index = torch.tensor(token_ids, dtype=torch.long, device=device)

        for start in range(0, len(message_batches), self.microbatch):
            chunk = message_batches[start:start + self.microbatch]
            rows = []
            masks = []
            for messages in chunk:
                prompt = self.tokenizer.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True
                ) + "["
                encoded = self.tokenizer(prompt, add_special_tokens=False)
                rows.append(encoded["input_ids"])
            max_len = max(len(row) for row in rows)
            padded = []
            for row in rows:
                pad = max_len - len(row)
                padded.append([pad_id] * pad + row)
                masks.append([0] * pad + [1] * len(row))
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                    use_cache=False,
                ).logits[:, -1, :]
                selected = logits.index_select(-1, selected_index)
                probs = torch.softmax(selected.float(), dim=-1)
            for row in range(probs.shape[0]):
                results.append({
                    choice: float(probs[row, i].item())
                    for i, choice in enumerate(choices)
                })
        return tuple(results)


def load_quantized_base():
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quant,
        device_map={"": 0},
    )
    model.eval()
    return tokenizer, model


def evaluate_subject(tokenizer, model, *, digest, case_batch_size=1):
    adapter = LoadedChoiceAdapter(
        tokenizer, model, model_id=BASE_MODEL_ID, model_digest=digest, microbatch=3
    )
    continuity = run_continuity_benchmark(
        generate_continuity_cases(),
        adapter,
        tokenizer=tokenizer,
        case_batch_size=case_batch_size,
        byte_ceiling=8192,
        max_retrieved_tokens=512,
    )
    public_cases = load_jsonl_cases(ROOT / "benchmarks" / "spm_v0_public_dev.jsonl")
    public = run_balanced_score_choice_suite(
        public_cases, adapter, case_batch_size=case_batch_size
    )
    return {"continuity": continuity, "public36": public}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(ROOT / "benchmarks" / "continuity_v1.jsonl") != CONTINUITY_SHA:
        raise RuntimeError("continuity benchmark drift")
    # PUBLIC36_SHA is the semantic benchmark digest, not raw-file digest.
    train_receipt = json.loads(args.training_receipt.read_text(encoding="utf-8"))
    expected_adapter = train_receipt["adapter_files"]["adapter_model.safetensors"]["sha256"]
    adapter_weights = args.adapter_dir / "adapter_model.safetensors"
    if sha256_file(adapter_weights) != expected_adapter:
        raise RuntimeError("adapter weights digest mismatch")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tokenizer, base = load_quantized_base()
    base_result = evaluate_subject(
        tokenizer, base, digest=f"{BASE_INVENTORY}:bnb_nf4"
    )

    adapted = PeftModel.from_pretrained(
        base, args.adapter_dir, is_trainable=False
    )
    adapted.eval()
    combined_digest = hashlib.sha256(
        f"{BASE_INVENTORY}:{expected_adapter}".encode("utf-8")
    ).hexdigest()
    adapted_result = evaluate_subject(
        tokenizer, adapted, digest=combined_digest
    )

    base_cont = base_result["continuity"]["metrics"]
    adapted_cont = adapted_result["continuity"]["metrics"]
    base_public = base_result["public36"]["summary"]["correct_count"]
    adapted_public = adapted_result["public36"]["summary"]["correct_count"]

    gates = {
        "persistent_correct": adapted_cont["persistent"]["correct_count"] >= 23,
        "persistent_stale_errors": adapted_cont["persistent"]["stale_error_count"] <= 1,
        "persistent_correction_burden":
            adapted_cont["persistent"]["user_correction_burden"] <= 1,
        "full_history_correct": adapted_cont["full_history"]["correct_count"] >= 23,
        "general_public36_regression": adapted_public >= base_public - 1,
    }
    receipt = {
        "schema_version": 1,
        "evaluation_harness_commit": git_head(),
        "evaluation_harness_sha256": sha256_file(Path(__file__).resolve()),
        "base_model_id": BASE_MODEL_ID,
        "base_revision": BASE_REVISION,
        "base_inventory_digest": BASE_INVENTORY,
        "adapter_model_sha256": expected_adapter,
        "training_receipt_sha256": sha256_file(args.training_receipt),
        "base_quantized": base_result,
        "adapted_quantized": adapted_result,
        "comparison": {
            "base_persistent_correct": base_cont["persistent"]["correct_count"],
            "adapted_persistent_correct": adapted_cont["persistent"]["correct_count"],
            "base_persistent_stale_errors": base_cont["persistent"]["stale_error_count"],
            "adapted_persistent_stale_errors": adapted_cont["persistent"]["stale_error_count"],
            "base_correction_burden": base_cont["persistent"]["user_correction_burden"],
            "adapted_correction_burden": adapted_cont["persistent"]["user_correction_burden"],
            "base_public36_correct": base_public,
            "adapted_public36_correct": adapted_public,
        },
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": receipt["status"],
        "gates": gates,
        "comparison": receipt["comparison"],
        "adapted_continuity_metrics": adapted_cont,
        "base_continuity_metrics": base_cont,
        "receipt": str(args.output),
        "receipt_sha256": sha256_file(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
