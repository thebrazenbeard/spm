"""Strictly local Hugging Face chat-model adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any


def _load_hf_classes():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "LocalHFAdapter requires the optional 'transformers' package"
        ) from error
    return AutoTokenizer, AutoModelForCausalLM


def local_inventory_digest(path: str | Path) -> str:
    """Content-bind every regular file under a local model directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"local model path is not a directory: {root}")
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _deterministic_generation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    if normalized.get("do_sample") is True:
        raise ValueError("sampling is not permitted in deterministic baseline runs")
    normalized["do_sample"] = False
    for key in ("temperature", "top_p", "top_k"):
        normalized.pop(key, None)
    max_new_tokens = normalized.get("max_new_tokens", 8)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    normalized["max_new_tokens"] = max_new_tokens
    return normalized


class LocalHFAdapter:
    """Load an explicitly identified model only from an existing local path."""

    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        revision: str,
        generation_config: Mapping[str, Any],
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"local model path is not a directory: {self.model_path}"
            )
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be a non-empty string")
        self.model_id = model_id
        self.revision = revision
        self.model_digest = local_inventory_digest(self.model_path)
        self.generation_config = _deterministic_generation_config(generation_config)
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        tokenizer_class, model_class = _load_hf_classes()
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        self._tokenizer = tokenizer_class.from_pretrained(str(self.model_path), **kwargs)
        self._model = model_class.from_pretrained(str(self.model_path), **kwargs)
        if hasattr(self._model, "eval"):
            self._model.eval()
        return self._tokenizer, self._model

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        generation_config: Mapping[str, Any],
    ) -> str:
        tokenizer, model = self._load()
        merged = dict(self.generation_config)
        merged.update(generation_config)
        config = _deterministic_generation_config(merged)

        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        input_length = inputs["input_ids"].shape[-1]
        if "pad_token_id" not in config:
            pad_token_id = getattr(tokenizer, "pad_token_id", None)
            if pad_token_id is None:
                pad_token_id = getattr(tokenizer, "eos_token_id", None)
            config["pad_token_id"] = pad_token_id
        generated = model.generate(**inputs, **config)
        new_tokens = generated[0][input_length:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def choose(
        self,
        messages: Sequence[Mapping[str, str]],
        choice_ids: Sequence[str],
    ) -> str:
        tokenizer, model = self._load()
        choices = tuple(choice_ids)
        if not choices or any(not isinstance(choice, str) or not choice for choice in choices):
            raise ValueError("choice_ids must contain non-empty strings")

        token_ids: list[int] = []
        for choice in choices:
            encoded = tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("each choice ID must encode to a single tokenizer token")
            token_ids.append(encoded[0])

        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        ) + "["
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

        import torch
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        scores = [float(logits[token_id].item()) for token_id in token_ids]
        winner = max(range(len(choices)), key=scores.__getitem__)
        return choices[winner]

    def choose_many(
        self,
        message_batches: Sequence[Sequence[Mapping[str, str]]],
        choice_ids: Sequence[str],
    ) -> tuple[str, ...]:
        tokenizer, model = self._load()
        choices = tuple(choice_ids)
        if not choices or any(not isinstance(choice, str) or not choice for choice in choices):
            raise ValueError("choice_ids must contain non-empty strings")

        token_ids: list[int] = []
        for choice in choices:
            encoded = tokenizer.encode(choice, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError("each choice ID must encode to a single tokenizer token")
            token_ids.append(encoded[0])

        import torch
        encoded_prompts = []
        lengths: list[int] = []
        for messages in message_batches:
            prompt = tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            ) + "["
            encoded = tokenizer(prompt, return_tensors="pt")
            encoded_prompts.append(encoded)
            lengths.append(int(encoded["input_ids"].shape[-1]))
        if not encoded_prompts:
            return ()

        max_length = max(lengths)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", 0)
        input_rows = []
        mask_rows = []
        for encoded, length in zip(encoded_prompts, lengths, strict=True):
            padding = max_length - length
            input_rows.append(torch.cat((encoded["input_ids"][0], torch.full((padding,), pad_token_id, dtype=encoded["input_ids"].dtype))))
            mask_rows.append(torch.cat((encoded["attention_mask"][0], torch.zeros((padding,), dtype=encoded["attention_mask"].dtype))))
        inputs = {"input_ids": torch.stack(input_rows), "attention_mask": torch.stack(mask_rows)}
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        winners: list[str] = []
        for row, length in enumerate(lengths):
            scores = [float(logits[row, length - 1, token_id].item()) for token_id in token_ids]
            winners.append(choices[max(range(len(choices)), key=scores.__getitem__)])
        return tuple(winners)
