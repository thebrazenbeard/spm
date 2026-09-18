"""Runtime support for a gated persistent-memory LoRA specialist."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any, Sequence, Mapping


def memory_adapter_active(retrieved_record_count: int) -> bool:
    """Enable the specialist only when retrieval explicitly returned records."""
    if isinstance(retrieved_record_count, bool) or not isinstance(retrieved_record_count, int):
        raise TypeError("retrieved_record_count must be an integer")
    if retrieved_record_count < 0:
        raise ValueError("retrieved_record_count must be non-negative")
    return retrieved_record_count > 0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
        microbatch: int = 3,
    ) -> None:
        if microbatch <= 0:
            raise ValueError("microbatch must be positive")
        self.tokenizer = tokenizer
        self.model = model
        self.model_id = model_id
        self.base_inventory_digest = base_inventory_digest
        self.adapter_digest = adapter_digest
        self.microbatch = microbatch
        self.base_runtime_digest = f"{base_inventory_digest}:bnb_nf4"
        self.combined_digest = _sha256_text(
            f"{base_inventory_digest}:{adapter_digest}:bnb_nf4"
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
        microbatch: int = 3,
    ) -> "MemorySpecialistRuntime":
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
            microbatch=microbatch,
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
