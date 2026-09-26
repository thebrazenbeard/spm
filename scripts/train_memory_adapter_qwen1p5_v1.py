from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

HARNESS_PATH = Path(__file__).resolve()
HARNESS_ROOT = HARNESS_PATH.parents[1]
sys.path.insert(0, str(HARNESS_ROOT / "src"))

from spm_bench.arm_m_memory import ArmMMemory, MemoryRecord
from spm_bench.case import BenchmarkCase, BenchmarkTurn, load_jsonl_cases
from spm_bench.runner import _choice_messages_with_layout

EXPERIMENT_COMMIT = "7605d337d9b131c9a26d13eb798f10921a8a0499"
EXPERIMENT_PATH = "experiments/spm_memory_adapter_qwen1p5_v1.json"
EXPECTED_MANIFEST_SHA256 = "ea24744d562d10c337df427305e5deb18d9f110c8291af3839f54028a44029b3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    return parser.parse_args()


def rotate(values, offset):
    values = tuple(values)
    return values[offset:] + values[:offset]


def persistent_case(source: BenchmarkCase, tokenizer, *, byte_ceiling: int, max_tokens: int):
    if len(source.turns) < 2:
        raise ValueError("training case requires prior memory and a final query")
    memory = ArmMMemory.empty()
    for sequence, turn in enumerate(source.turns[:-1], start=1):
        memory = memory.append(
            MemoryRecord.create(
                sequence=sequence,
                source_class=turn.role,
                content=turn.content,
            ),
            byte_ceiling=byte_ceiling,
        )
    query = source.turns[-1].content
    receipt = memory.retrieve(
        current_text=query,
        tokenizer=tokenizer,
        max_retrieved_tokens=max_tokens,
    )
    context = (
        "Persistent memory from earlier interactions:\n"
        f"{receipt.rendered_text}\n\n"
        f"{query}"
    )
    converted = BenchmarkCase(
        case_id=source.case_id,
        version=source.version,
        family=source.family,
        turns=(BenchmarkTurn(role="user", content=context),),
        choices=source.choices,
        expected_choice=source.expected_choice,
        risk_class=source.risk_class,
        tags=source.tags + ("memory_adapter_training",),
    )
    return converted, receipt


def build_example(source, index, tokenizer, *, byte_ceiling, max_tokens):
    case, receipt = persistent_case(
        source, tokenizer, byte_ceiling=byte_ceiling, max_tokens=max_tokens
    )
    labels = tuple(choice.choice_id for choice in case.choices)
    count = len(labels)
    if count != 3:
        raise ValueError("memory adapter v1 requires exactly three choices")
    order = rotate(tuple(range(count)), index % count)
    assigned = rotate(labels, (index // count) % count)
    messages = _choice_messages_with_layout(case, order, assigned)
    semantic_index = next(
        i for i, choice in enumerate(case.choices)
        if choice.choice_id == case.expected_choice
    )
    target_position = order.index(semantic_index)
    target_label = assigned[target_position]
    prompt = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    ) + "["
    encoded = tokenizer(prompt, add_special_tokens=False)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "input_ids": encoded["input_ids"],
        "target_label": target_label,
        "memory_receipt_digest": receipt.receipt_digest,
    }


class Collator:
    def __init__(self, pad_token_id: int, label_to_index: dict[str, int]):
        self.pad_token_id = pad_token_id
        self.label_to_index = label_to_index

    def __call__(self, rows):
        max_len = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        targets = []
        for row in rows:
            ids = row["input_ids"]
            pad = max_len - len(ids)
            input_ids.append([self.pad_token_id] * pad + ids)
            attention_mask.append([0] * pad + [1] * len(ids))
            targets.append(self.label_to_index[row["target_label"]])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


def evaluate(model, loader, choice_token_ids, device):
    model.eval()
    correct = 0
    count = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
                use_cache=False,
            )
            logits = output.logits[:, -1, :].index_select(-1, choice_token_ids)
            loss = F.cross_entropy(logits.float(), targets)
            total_loss += float(loss.item()) * targets.numel()
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            count += targets.numel()
    return {
        "case_count": count,
        "correct_count": correct,
        "accuracy": correct / count if count else 0.0,
        "mean_loss": total_loss / count if count else math.nan,
    }


def main():
    args = parse_args()
    root = args.root.resolve()
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)

    if git_head(root) != EXPERIMENT_COMMIT:
        raise RuntimeError("experiment worktree HEAD mismatch")
    if subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode != 0:
        raise RuntimeError("experiment worktree has tracked changes")
    if subprocess.run(["git", "diff", "--quiet"], cwd=HARNESS_ROOT).returncode != 0:
        raise RuntimeError("execution harness worktree has tracked changes")

    manifest_path = root / EXPERIMENT_PATH
    if sha256_file(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("experiment manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for split in ("train", "dev"):
        info = manifest["data"][split]
        if sha256_file(root / info["path"]) != info["sha256"]:
            raise RuntimeError(f"{split} data digest mismatch")

    seed = int(manifest["training"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    subject = manifest["base_subject"]
    model_path = Path(subject["local_snapshot"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    labels = ("a", "b", "c")
    choice_ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"choice label {label!r} is not one token")
        choice_ids.append(encoded[0])

    memory_policy = manifest["data"]["memory_policy"]
    train_cases = load_jsonl_cases(root / manifest["data"]["train"]["path"])
    dev_cases = load_jsonl_cases(root / manifest["data"]["dev"]["path"])
    train_rows = [
        build_example(
            case, i, tokenizer,
            byte_ceiling=memory_policy["byte_ceiling"],
            max_tokens=memory_policy["max_retrieved_tokens"],
        )
        for i, case in enumerate(train_cases)
    ]
    dev_rows = [
        build_example(
            case, i, tokenizer,
            byte_ceiling=memory_policy["byte_ceiling"],
            max_tokens=memory_policy["max_retrieved_tokens"],
        )
        for i, case in enumerate(dev_cases)
    ]
    if max(len(row["input_ids"]) for row in train_rows + dev_rows) > 512:
        raise RuntimeError("unexpected prompt length > 512 tokens")

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        quantization_config=quant,
        device_map={"": 0},
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True
    )

    lora = manifest["training"]["lora"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias=lora["bias"],
    )
    model = get_peft_model(base, config)
    model.train()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    device = next(p for p in model.parameters() if p.is_cuda).device
    choice_token_ids = torch.tensor(choice_ids, dtype=torch.long, device=device)

    batch_size = int(manifest["training"]["micro_batch_size"])
    accumulation = int(manifest["training"]["gradient_accumulation_steps"])
    generator = torch.Generator()
    generator.manual_seed(seed)
    collator = Collator(tokenizer.pad_token_id, {label: i for i, label in enumerate(labels)})
    train_loader = DataLoader(
        train_rows,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(manifest["training"]["learning_rate"]),
        weight_decay=float(manifest["training"]["weight_decay"]),
    )
    total_optimizer_steps = math.ceil(len(train_loader) / accumulation)
    if total_optimizer_steps != 108:
        raise RuntimeError(f"optimizer step count drift: {total_optimizer_steps}")
    warmup = int(manifest["training"]["warmup_optimizer_steps"])

    def lr_scale(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        remaining = max(total_optimizer_steps - warmup, 1)
        return max(
            0.0,
            float(total_optimizer_steps - (step + 1)) / float(remaining),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    if args.probe:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
            use_cache=False,
        )
        logits = output.logits[:, -1, :].index_select(-1, choice_token_ids)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        print(json.dumps({
            "probe_loss": float(loss.item()),
            "trainable_parameters": trainable,
            "total_parameters_reported": total,
            "prompt_max_tokens": max(len(row["input_ids"]) for row in train_rows),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        }, indent=2))
        return

    start = time.perf_counter()
    optimizer_step = 0
    accumulated_examples = 0
    epoch_losses = []
    batch_losses = []
    for epoch in range(int(manifest["training"]["epochs"])):
        epoch_loss_sum = 0.0
        epoch_example_count = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
                use_cache=False,
            )
            selected = output.logits[:, -1, :].index_select(-1, choice_token_ids)
            loss = F.cross_entropy(selected.float(), targets)
            (loss / accumulation).backward()
            examples = targets.numel()
            epoch_loss_sum += float(loss.item()) * examples
            epoch_example_count += examples
            accumulated_examples += examples
            batch_losses.append(float(loss.item()))

            if batch_index % accumulation == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    float(manifest["training"]["max_grad_norm"]),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % 12 == 0:
                    print(
                        f"optimizer_step {optimizer_step}/{total_optimizer_steps} "
                        f"loss={loss.item():.6f}",
                        flush=True,
                    )
        epoch_losses.append(epoch_loss_sum / epoch_example_count)

    if optimizer_step != total_optimizer_steps:
        raise RuntimeError("final optimizer step count mismatch")

    dev_metrics = evaluate(model, dev_loader, choice_token_ids, device)
    elapsed = time.perf_counter() - start
    adapter_dir = results / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)

    adapter_files = {}
    for path in sorted(p for p in adapter_dir.rglob("*") if p.is_file()):
        adapter_files[path.relative_to(adapter_dir).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    receipt = {
        "schema_version": 1,
        "experiment_commit": EXPERIMENT_COMMIT,
        "execution_harness_commit": git_head(HARNESS_ROOT),
        "execution_harness_sha256": sha256_file(HARNESS_PATH),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "base_subject": subject,
        "quantization": manifest["training"]["quantization"],
        "lora": lora,
        "objective": manifest["training"]["objective"],
        "train_case_count": len(train_rows),
        "dev_case_count": len(dev_rows),
        "optimizer_steps": optimizer_step,
        "effective_batch_size": manifest["training"]["effective_batch_size"],
        "trainable_parameters": trainable,
        "reported_total_parameters": total,
        "epoch_mean_loss": epoch_losses,
        "last_12_batch_mean_loss": (
            sum(batch_losses[-12:]) / min(len(batch_losses), 12)
        ),
        "dev": dev_metrics,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "adapter_files": adapter_files,
        "adapter_dir": str(adapter_dir),
    }
    receipt_path = results / "SPM_MEMORY_ADAPTER_QWEN1P5_V1_TRAIN.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "optimizer_steps": optimizer_step,
        "epoch_mean_loss": epoch_losses,
        "dev": dev_metrics,
        "trainable_parameters": trainable,
        "peak_vram_bytes": receipt["peak_vram_bytes"],
        "elapsed_seconds": elapsed,
        "adapter_files": adapter_files,
    }, indent=2))


if __name__ == "__main__":
    main()
