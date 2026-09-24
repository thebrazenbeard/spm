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
