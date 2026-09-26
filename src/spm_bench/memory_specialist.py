"""Runtime support for a gated persistent-memory LoRA specialist."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any, Sequence, Mapping


from .hf_adapter import local_inventory_digest


_VERIFIED_ARTIFACT_CONSTRUCTION = object()


def memory_adapter_active(retrieved_record_count: int) -> bool:
    """Enable the specialist only when retrieval explicitly returned records."""
    if isinstance(retrieved_record_count, bool) or not isinstance(retrieved_record_count, int):
        raise TypeError("retrieved_record_count must be an integer")
    if retrieved_record_count < 0:
        raise ValueError("retrieved_record_count must be non-negative")
    return retrieved_record_count > 0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def verify_base_inventory(
    base_path: str | Path,
    base_inventory_digest: str,
) -> None:
    """Bind the loaded local base model to the qualified content inventory."""
    expected = _require_sha256(base_inventory_digest, "base_inventory_digest")
    observed = local_inventory_digest(base_path)
    if observed != expected:
        raise RuntimeError("base model inventory digest mismatch")


def verify_adapter_artifacts(
    adapter_path: str | Path,
    *,
    adapter_digest: str,
    adapter_config_digest: str,
) -> None:
    """Bind the loaded PEFT adapter to the qualified weights and config bytes."""
    adapter_path = Path(adapter_path)
    expected_weights = _require_sha256(adapter_digest, "adapter_digest")
    expected_config = _require_sha256(
        adapter_config_digest, "adapter_config_digest"
    )
    weights_path = adapter_path / "adapter_model.safetensors"
    config_path = adapter_path / "adapter_config.json"
    if not weights_path.is_file():
        raise RuntimeError("qualified adapter weights are missing")
    if not config_path.is_file():
        raise RuntimeError("qualified adapter config is missing")
    if _sha256_file(weights_path) != expected_weights:
        raise RuntimeError("adapter weights digest mismatch")
    config_bytes = config_path.read_bytes()
    canonical_config_bytes = config_bytes.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if hashlib.sha256(canonical_config_bytes).hexdigest() != expected_config:
        raise RuntimeError("adapter config digest mismatch")


class _MemorySpecialistView:
    def __init__(self, runtime: "MemorySpecialistRuntime", *, active: bool) -> None:
        self._runtime = runtime
        self._active = bool(active)
        self.model_id = runtime.model_id
        self.model_digest = (
            runtime.combined_digest if self._active else runtime.base_runtime_digest
        )

    @property
    def memory_active(self) -> bool:
        return self._active

    def score_many_selected(
        self,
        message_batches: Sequence[Sequence[Mapping[str, str]]],
        choice_ids: Sequence[str],
    ):
        return self._runtime._score_many_selected(
            message_batches, choice_ids, adapter_enabled=self._active
        )


class MemorySpecialistRuntime:
    """One quantized base model with an explicitly gated PEFT memory adapter."""

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        *,
        model_id: str,
        base_inventory_digest: str,
        adapter_digest: str,
        adapter_config_digest: str,
        microbatch: int = 3,
        _artifact_verification_token: object | None = None,
    ) -> None:
        if _artifact_verification_token is not _VERIFIED_ARTIFACT_CONSTRUCTION:
            raise RuntimeError(
                "MemorySpecialistRuntime must be constructed through load_quantized()"
            )
        if microbatch <= 0:
            raise ValueError("microbatch must be positive")
        self.tokenizer = tokenizer
        self.model = model
        self.model_id = model_id
        self.base_inventory_digest = base_inventory_digest
        self.adapter_digest = adapter_digest
        self.adapter_config_digest = adapter_config_digest
        self.microbatch = microbatch
        self.base_runtime_digest = f"{base_inventory_digest}:bnb_nf4"
        self.combined_digest = _sha256_text(
            f"{base_inventory_digest}:{adapter_digest}:{adapter_config_digest}:bnb_nf4"
        )

    @classmethod
    def load_quantized(
        cls,
        *,
        base_path: str | Path,
        adapter_path: str | Path,
        model_id: str,
        base_inventory_digest: str,
        adapter_digest: str,
        adapter_config_digest: str,
        microbatch: int = 3,
    ) -> "MemorySpecialistRuntime":
        verify_base_inventory(base_path, base_inventory_digest)
        verify_adapter_artifacts(
            adapter_path,
            adapter_digest=adapter_digest,
            adapter_config_digest=adapter_config_digest,
        )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        base_path = Path(base_path)
        adapter_path = Path(adapter_path)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_path, local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            local_files_only=True,
            trust_remote_code=False,
            quantization_config=quant,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(
            base, adapter_path, is_trainable=False
        )
        model.eval()
        return cls(
            tokenizer,
            model,
            model_id=model_id,
            base_inventory_digest=base_inventory_digest,
            adapter_digest=adapter_digest,
            adapter_config_digest=adapter_config_digest,
            microbatch=microbatch,
            _artifact_verification_token=_VERIFIED_ARTIFACT_CONSTRUCTION,
        )

    def view_for_retrieval(self, retrieved_record_count: int) -> _MemorySpecialistView:
        return _MemorySpecialistView(
            self, active=memory_adapter_active(retrieved_record_count)
        )

    def base_view(self) -> _MemorySpecialistView:
        return _MemorySpecialistView(self, active=False)

    def memory_view(self) -> _MemorySpecialistView:
        return _MemorySpecialistView(self, active=True)

    def _score_many_selected(
        self,
        message_batches: Sequence[Sequence[Mapping[str, str]]],
        choice_ids: Sequence[str],
        *,
        adapter_enabled: bool,
    ):
        import torch

        choices = tuple(choice_ids)
        if not choices:
            return ()
        token_ids = []
        for choice in choices:
            encoded = self.tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("choice IDs must each encode to one token")
            token_ids.append(encoded[0])

        device = next(parameter for parameter in self.model.parameters() if parameter.is_cuda).device
        selected_index = torch.tensor(token_ids, dtype=torch.long, device=device)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        rows_out = []
        adapter_context = nullcontext if adapter_enabled else self.model.disable_adapter
        for start in range(0, len(message_batches), self.microbatch):
            chunk = message_batches[start:start + self.microbatch]
            encoded_rows = []
            for messages in chunk:
                prompt = self.tokenizer.apply_chat_template(
                    list(messages),
                    tokenize=False,
                    add_generation_prompt=True,
                ) + "["
                encoded = self.tokenizer(prompt, add_special_tokens=False)
                encoded_rows.append(encoded["input_ids"])
            max_len = max(len(row) for row in encoded_rows)
            padded = []
            masks = []
            for row in encoded_rows:
                pad = max_len - len(row)
                padded.append([pad_id] * pad + row)
                masks.append([0] * pad + [1] * len(row))
            input_ids = torch.tensor(padded, dtype=torch.long, device=device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
            with adapter_context(), torch.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                    use_cache=False,
                ).logits[:, -1, :]
                selected = logits.index_select(-1, selected_index)
                probabilities = torch.softmax(selected.float(), dim=-1)
            for row_index in range(probabilities.shape[0]):
                rows_out.append({
                    choice: float(probabilities[row_index, column].item())
                    for column, choice in enumerate(choices)
                })
        return tuple(rows_out)


from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MemoryResolution:
    chosen_index: int | None
    chosen_text: str | None
    semantic_scores: tuple[float, float, float]
    adapter_active: bool
    retrieval_receipt_digest: str
    selected_record_digests: tuple[str, ...]
    token_count: int
    truncated: bool


def resolve_memory_choice(
    runtime: MemorySpecialistRuntime,
    *,
    memory_records,
    query: str,
    choices,
    byte_ceiling: int = 8192,
    max_retrieved_tokens: int = 512,
) -> MemoryResolution:
    """Resolve one of three semantic candidates using bounded retrieved memory."""
    from .arm_m_memory import ArmMMemory
    from .case import BenchmarkCase, BenchmarkChoice, BenchmarkTurn
    from .runner import _balanced_layouts

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    choice_texts = tuple(choices)
    if len(choice_texts) != 3:
        raise ValueError("exactly three choices are required")
    if any(not isinstance(choice, str) or not choice.strip() for choice in choice_texts):
        raise ValueError("choices must be non-empty strings")
    if len(set(choice_texts)) != 3:
        raise ValueError("choices must be distinct")

    memory = ArmMMemory.empty()
    records = tuple(memory_records)
    for record in records:
        memory = memory.append(record, byte_ceiling=byte_ceiling)

    receipt = memory.retrieve(
        current_text=query,
        tokenizer=runtime.tokenizer,
        max_retrieved_tokens=max_retrieved_tokens,
    )
    adapter_active = bool(receipt.selected_records)
    if adapter_active:
        context = (
            "Persistent memory from earlier interactions:\n"
            f"{receipt.rendered_text}\n\n"
            f"{query}"
        )
    else:
        context = query

    case = BenchmarkCase(
        case_id="memory_resolution_runtime",
        version=1,
        family="runtime_resolution",
        turns=(BenchmarkTurn(role="user", content=context),),
        choices=tuple(
            BenchmarkChoice(choice_id=choice_id, text=text)
            for choice_id, text in zip(("a", "b", "c"), choice_texts, strict=True)
        ),
        expected_choice="a",
        risk_class="runtime",
        tags=("memory_specialist_runtime",),
    )
    labels, layouts, message_batches = _balanced_layouts(case)
    view = runtime.view_for_retrieval(len(receipt.selected_records))
    score_rows = tuple(view.score_many_selected(message_batches, labels))
    if len(score_rows) != len(layouts):
        raise RuntimeError("memory specialist returned wrong number of score rows")

    totals = [0.0, 0.0, 0.0]
    for (order, assigned), row in zip(layouts, score_rows, strict=True):
        if set(row) != set(labels):
            raise RuntimeError("score row did not cover declared labels")
        for label in labels:
            position = assigned.index(label)
            semantic_index = order[position]
            totals[semantic_index] += float(row[label])
    semantic_scores = tuple(value / len(layouts) for value in totals)
    best = max(semantic_scores)
    winners = [
        index for index, value in enumerate(semantic_scores)
        if abs(value - best) <= 1e-12
    ]
    chosen_index = winners[0] if len(winners) == 1 else None
    chosen_text = choice_texts[chosen_index] if chosen_index is not None else None

    return MemoryResolution(
        chosen_index=chosen_index,
        chosen_text=chosen_text,
        semantic_scores=semantic_scores,
        adapter_active=adapter_active,
        retrieval_receipt_digest=receipt.receipt_digest,
        selected_record_digests=tuple(
            record.digest for record in receipt.selected_records
        ),
        token_count=receipt.token_count,
        truncated=receipt.truncated,
    )
