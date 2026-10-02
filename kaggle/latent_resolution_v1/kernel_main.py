from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time


SPM_COMMIT = "dbd4cb10be61711b349ccad5c900037d7ac68ea1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_INVENTORY_DIGEST = "6080fc05cb5e0ccfa35e64523b11a902cc1f3e35672f85135a19eb16b722f8b8"
CLAIM_CEILING = "KAGGLE_EXPERIMENT_ONLY_NOT_VERA_RUNTIME_OR_PRODUCTION_VRAM_PROOF"

SCHEMA = "SPM_LATENT_RESOLUTION_KAGGLE_V1"
SLOT_BUDGETS = (2, 4, 8, 16)
TRAIN_SEED = 20261002
DEV_SEED = 20261003
TRAIN_CASE_COUNT = 240
DEV_CASE_COUNT = 80
MAX_CHUNKS = 5
MAX_CHUNK_TOKENS = 96
MAX_QUERY_TOKENS = 64
TRAIN_EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TRANSFORMERS_VERSION = "4.57.1"
OUTPUT_NAME = "latent_resolution_v1_result.json"


def _repo_root_if_present() -> Path | None:
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "src" / "spm_bench").is_dir():
        return candidate
    return None


def _ensure_spm_importable() -> None:
    try:
        importlib.import_module("spm_bench")
        return
    except ImportError:
        pass

    root = _repo_root_if_present()
    if root is not None:
        sys.path.insert(0, str(root / "src"))
        importlib.import_module("spm_bench")
        return

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "git+https://github.com/thebrazenbeard/spm.git@" + SPM_COMMIT,
        ],
        check=True,
    )
    importlib.import_module("spm_bench")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _example_payload(example) -> dict[str, object]:
    return {
        "case_id": example.case_id,
        "family": example.family,
        "source_chunks": list(example.source_chunks),
        "compact_state": example.compact_state,
        "query": example.query,
        "exact_answer": example.exact_answer,
        "exact_required": example.exact_required,
        "decisive_chunk": example.decisive_chunk,
    }


def _examples_digest(examples) -> str:
    return hashlib.sha256(
        _canonical_bytes([_example_payload(item) for item in examples])
    ).hexdigest()


def _experiment_examples():
    _ensure_spm_importable()
    from spm_bench.late_relevance_data import generate_late_relevance_examples

    train = generate_late_relevance_examples(
        seed=TRAIN_SEED,
        count=TRAIN_CASE_COUNT,
        max_chunks=MAX_CHUNKS,
    )
    dev = generate_late_relevance_examples(
        seed=DEV_SEED,
        count=DEV_CASE_COUNT,
        max_chunks=MAX_CHUNKS,
    )
    return train, dev


def build_dry_run_manifest() -> dict[str, object]:
    train, dev = _experiment_examples()
    return {
        "schema": SCHEMA,
        "spm_commit": SPM_COMMIT,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_inventory_digest": MODEL_INVENTORY_DIGEST,
        "slot_budgets": list(SLOT_BUDGETS),
        "train_seed": TRAIN_SEED,
        "dev_seed": DEV_SEED,
        "train_case_count": len(train),
        "dev_case_count": len(dev),
        "train_subject_digest": _examples_digest(train),
        "dev_subject_digest": _examples_digest(dev),
        "max_chunks": MAX_CHUNKS,
        "max_chunk_tokens": MAX_CHUNK_TOKENS,
        "max_query_tokens": MAX_QUERY_TOKENS,
        "train_epochs": TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "claim_ceiling": CLAIM_CEILING,
    }


def _ensure_transformers() -> None:
    installed = None
    if importlib.util.find_spec("transformers") is not None:
        try:
            installed = importlib.import_module("transformers").__version__
        except Exception:
            installed = None
    if installed == TRANSFORMERS_VERSION:
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"transformers=={TRANSFORMERS_VERSION}",
        ],
        check=True,
    )


def _download_verified_snapshot() -> tuple[Path, str]:
    from huggingface_hub import snapshot_download
    from spm_bench.hf_adapter import local_inventory_digest

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
        )
    )
    inventory_digest = local_inventory_digest(snapshot)
    if inventory_digest != MODEL_INVENTORY_DIGEST:
        raise RuntimeError(
            "downloaded model inventory digest does not match the bound SPM baseline subject"
        )
    return snapshot, inventory_digest


def _configure_seed(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _attention_heads_for(hidden_size: int) -> int:
    for candidate in (16, 14, 8, 7, 4, 2, 1):
        if hidden_size % candidate == 0:
            return candidate
    raise RuntimeError("no supported attention-head divisor for hidden size")


def _tokenize(tokenizer, text: str, *, max_tokens: int, device):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _backbone(model):
    candidate = getattr(model, "model", None)
    if candidate is None:
        raise RuntimeError("base model does not expose a decoder backbone")
    return candidate


def _encode_hidden(torch, model, tokenizer, text: str, *, max_tokens: int, device):
    inputs = _tokenize(
        tokenizer,
        text,
        max_tokens=max_tokens,
        device=device,
    )
    with torch.inference_mode():
        output = _backbone(model)(
            **inputs,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.last_hidden_state
    mask = inputs["attention_mask"]
    return (
        hidden.detach().to("cpu", dtype=torch.float16),
        mask.detach().to("cpu"),
        int(mask.sum().item()),
    )


def _query_summary(torch, hidden, mask):
    weights = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    total = (hidden * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (total / denom).squeeze(0)


def _precompute_records(torch, model, tokenizer, examples, *, device):
    records: list[dict[str, object]] = []
    source_tokens_total = 0
    for index, example in enumerate(examples, 1):
        query_hidden, query_mask, query_tokens = _encode_hidden(
            torch,
            model,
            tokenizer,
            example.query,
            max_tokens=MAX_QUERY_TOKENS,
            device=device,
        )
        query_summary = _query_summary(torch, query_hidden, query_mask)
        chunks = []
        masks = []
        source_tokens = 0
        for chunk in example.source_chunks:
            hidden, mask, token_count = _encode_hidden(
                torch,
                model,
                tokenizer,
                chunk,
                max_tokens=MAX_CHUNK_TOKENS,
                device=device,
            )
            chunks.append(hidden.squeeze(0))
            masks.append(mask.squeeze(0))
            source_tokens += token_count
        source_tokens_total += source_tokens
        records.append(
            {
                "case_id": example.case_id,
                "family": example.family,
                "query_summary": query_summary,
                "chunks": tuple(chunks),
                "masks": tuple(masks),
                "target": int(example.decisive_chunk),
                "source_tokens": source_tokens,
                "query_tokens": query_tokens,
            }
        )
        if index % 50 == 0:
            print(f"precompute {index}/{len(examples)}", flush=True)
    return records, source_tokens_total


def _record_to_device(torch, record, device):
    query = record["query_summary"].to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    chunks = tuple(
        item.to(device=device, dtype=torch.float32).unsqueeze(0)
        for item in record["chunks"]
    )
    masks = tuple(
        item.to(device=device).unsqueeze(0)
        for item in record["masks"]
    )
    target = torch.tensor(
        [record["target"]],
        dtype=torch.long,
        device=device,
    )
    return query, chunks, masks, target


def _evaluate_router(torch, router, records, *, device):
    router.eval()
    correct = 0
    wrong_chunk = 0
    direct = 0
    insufficient = 0
    with torch.inference_mode():
        for record in records:
            query, chunks, masks, target = _record_to_device(
                torch,
                record,
                device,
            )
            output = router(query, chunks, masks)
            predicted = int(output.logits.argmax(dim=-1).item())
            expected = int(target.item())
            if predicted == expected:
                correct += 1
            elif predicted < router.config.max_chunks:
                wrong_chunk += 1
            elif predicted == router.config.direct_target:
                direct += 1
            else:
                insufficient += 1
    count = len(records)
    return {
        "case_count": count,
        "correct_chunk_count": correct,
        "wrong_chunk_count": wrong_chunk,
        "unsafe_direct_count": direct,
        "safe_insufficient_count": insufficient,
        "route_accuracy": correct / count if count else 0.0,
        "strict_safe_rate": (
            (correct + insufficient) / count if count else 0.0
        ),
    }


def _train_router(
    torch,
    *,
    hidden_size: int,
    slot_budget: int,
    train_records,
    dev_records,
    device,
):
    from spm_bench.latent_router import (
        LatentResolutionRouter,
        ResolutionRouterConfig,
    )

    _configure_seed(torch, TRAIN_SEED + slot_budget * 101)
    config = ResolutionRouterConfig(
        hidden_size=hidden_size,
        latent_slots=slot_budget,
        attention_heads=_attention_heads_for(hidden_size),
        max_chunks=MAX_CHUNKS,
    )
    router = LatentResolutionRouter(config).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    started = time.time()
    epoch_rows = []
    indices = list(range(len(train_records)))
    for epoch in range(TRAIN_EPOCHS):
        router.train()
        random.Random(
            TRAIN_SEED + slot_budget * 1000 + epoch
        ).shuffle(indices)
        total_loss = 0.0
        correct = 0
        for row_index in indices:
            record = train_records[row_index]
            query, chunks, masks, target = _record_to_device(
                torch,
                record,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            output = router(query, chunks, masks)
            loss = torch.nn.functional.cross_entropy(
                output.logits.float(),
                target,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                router.parameters(),
                GRAD_CLIP,
            )
            optimizer.step()
            total_loss += float(loss.item())
            correct += int(
                output.logits.argmax(dim=-1).item()
                == int(target.item())
            )
        dev = _evaluate_router(
            torch,
            router,
            dev_records,
            device=device,
        )
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": total_loss / len(train_records),
                "train_route_accuracy": correct / len(train_records),
                "dev_route_accuracy": dev["route_accuracy"],
                "dev_strict_safe_rate": dev["strict_safe_rate"],
            }
        )
        print(
            (
                f"slots={slot_budget} epoch={epoch + 1}/{TRAIN_EPOCHS} "
                f"train_acc={epoch_rows[-1]['train_route_accuracy']:.3f} "
                f"dev_acc={dev['route_accuracy']:.3f}"
            ),
            flush=True,
        )

    evaluation = _evaluate_router(
        torch,
        router,
        dev_records,
        device=device,
    )
    parameter_count = sum(
        parameter.numel() for parameter in router.parameters()
    )
    return router, {
        "slot_budget": slot_budget,
        "parameter_count": parameter_count,
        "training_seconds": time.time() - started,
        "epochs": epoch_rows,
        "dev": evaluation,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inflate_chunks(example, repetitions: int = 10):
    return tuple(
        " ".join([chunk] * repetitions)
        for chunk in example.source_chunks
    )


def _memory_probe_one(torch, model, tokenizer, router, example, *, device):
    if device.type != "cuda":
        raise RuntimeError("memory probe requires CUDA")

    long_chunks = _inflate_chunks(example)
    full_text = (
        "\n\n".join(long_chunks)
        + "\n\nQUESTION:\n"
        + example.query
    )

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    resident_before = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    full_inputs = _tokenize(
        tokenizer,
        full_text,
        max_tokens=2048,
        device=device,
    )
    with torch.inference_mode():
        full_output = _backbone(model)(
            **full_inputs,
            use_cache=False,
            return_dict=True,
        )
        _ = full_output.last_hidden_state[:, -1, :].sum().item()
    torch.cuda.synchronize(device)
    full_peak = int(torch.cuda.max_memory_allocated(device))
    full_tokens = int(full_inputs["attention_mask"].sum().item())
    del full_output, full_inputs
    torch.cuda.empty_cache()

    torch.cuda.synchronize(device)
    resident_latent = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    query_hidden, query_mask, query_tokens = _encode_hidden(
        torch,
        model,
        tokenizer,
        example.query,
        max_tokens=MAX_QUERY_TOKENS,
        device=device,
    )
    query = _query_summary(
        torch,
        query_hidden,
        query_mask,
    ).to(device=device, dtype=torch.float32).unsqueeze(0)
    state = router.memory.initialize(query)
    chunk_tokens_total = 0
    with torch.inference_mode():
        for chunk in long_chunks:
            inputs = _tokenize(
                tokenizer,
                chunk,
                max_tokens=256,
                device=device,
            )
            output = _backbone(model)(
                **inputs,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state.to(dtype=torch.float32)
            mask = inputs["attention_mask"]
            state = router.memory.update(state, hidden, mask)
            chunk_tokens_total += int(mask.sum().item())
            del output, hidden, inputs, mask
        logits = router.route_head(state.mean(dim=1))
        _ = logits.argmax(dim=-1).item()
    torch.cuda.synchronize(device)
    latent_peak = int(torch.cuda.max_memory_allocated(device))
    del state, query, query_hidden, query_mask, logits
    torch.cuda.empty_cache()

    return {
        "case_id": example.case_id,
        "slot_budget": router.config.latent_slots,
        "full_context_tokens": full_tokens,
        "sequential_chunk_tokens_total": chunk_tokens_total,
        "query_tokens": query_tokens,
        "resident_before_full_bytes": resident_before,
        "full_context_peak_bytes": full_peak,
        "full_context_incremental_peak_bytes": max(
            0, full_peak - resident_before
        ),
        "resident_before_latent_bytes": resident_latent,
        "latent_path_peak_bytes": latent_peak,
        "latent_path_incremental_peak_bytes": max(
            0, latent_peak - resident_latent
        ),
    }


def _hardware_report(torch, device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_type": "cuda",
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "device_index": device.index,
        "device_name": torch.cuda.get_device_name(device),
        "total_memory_bytes": int(properties.total_memory),
        "cuda_version": torch.version.cuda,
    }


def run_experiment(output_path: Path) -> dict[str, object]:
    _ensure_spm_importable()
    _ensure_transformers()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Kaggle latent-resolution run requires a CUDA accelerator"
        )
    device = torch.device("cuda:0")
    _configure_seed(torch, TRAIN_SEED)

    train_examples, dev_examples = _experiment_examples()
    manifest = build_dry_run_manifest()
    snapshot_path, runtime_inventory_digest = _download_verified_snapshot()

    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print("precomputing train hidden states", flush=True)
    train_records, train_source_tokens = _precompute_records(
        torch,
        model,
        tokenizer,
        train_examples,
        device=device,
    )
    print("precomputing dev hidden states", flush=True)
    dev_records, dev_source_tokens = _precompute_records(
        torch,
        model,
        tokenizer,
        dev_examples,
        device=device,
    )

    hidden_size = int(model.config.hidden_size)
    trained = {}
    routers = {}
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for slot_budget in SLOT_BUDGETS:
        router, report = _train_router(
            torch,
            hidden_size=hidden_size,
            slot_budget=slot_budget,
            train_records=train_records,
            dev_records=dev_records,
            device=device,
        )
        artifact = output_dir / (
            f"latent_resolution_router_slots_{slot_budget}.pt"
        )
        torch.save(
            {
                "schema": SCHEMA,
                "spm_commit": SPM_COMMIT,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_inventory_digest": runtime_inventory_digest,
                "slot_budget": slot_budget,
                "max_chunks": MAX_CHUNKS,
                "hidden_size": hidden_size,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in router.state_dict().items()
                },
            },
            artifact,
        )
        report["artifact"] = artifact.name
        report["artifact_sha256"] = _sha256_file(artifact)
        trained[str(slot_budget)] = report
        routers[slot_budget] = router

    memory_rows = []
    for slot_budget in SLOT_BUDGETS:
        router = routers[slot_budget]
        router.eval()
        for example in dev_examples[:3]:
            memory_rows.append(
                _memory_probe_one(
                    torch,
                    model,
                    tokenizer,
                    router,
                    example,
                    device=device,
                )
            )

    result = {
        **manifest,
        "software": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "hardware": _hardware_report(torch, device),
        "runtime_model_inventory_digest": runtime_inventory_digest,
        "hidden_size": hidden_size,
        "train_source_tokens_total": train_source_tokens,
        "dev_source_tokens_total": dev_source_tokens,
        "budgets": trained,
        "memory_probe": memory_rows,
        "completed_at_unix": time.time(),
    }
    output_path.write_text(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(output_path),
                "sha256": _sha256_file(output_path),
                "budgets": {
                    key: value["dev"]["route_accuracy"]
                    for key, value in trained.items()
                },
                "claim_ceiling": CLAIM_CEILING,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/kaggle/working") / OUTPUT_NAME,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print(json.dumps(build_dry_run_manifest(), sort_keys=True))
        return 0
    run_experiment(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
