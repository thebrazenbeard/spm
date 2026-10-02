from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LatentMemoryConfig:
    hidden_size: int
    latent_slots: int
    attention_heads: int

    def __post_init__(self) -> None:
        for value, label in (
            (self.hidden_size, "hidden_size"),
            (self.latent_slots, "latent_slots"),
            (self.attention_heads, "attention_heads"),
        ):
            if (
                type(value) is not int
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive exact int")
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by attention_heads"
            )


@dataclass(frozen=True, slots=True)
class LatentPrefixBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None


class QueryConditionedLatentMemory(nn.Module):
    """Fixed-slot recurrent latent memory updated one source chunk at a time.

    The module does not define truth or exactness. It is only a trainable
    representation mechanism. Exact-detail claims must be evaluated through
    a separate source-backed protocol.
    """

    def __init__(self, config: LatentMemoryConfig) -> None:
        super().__init__()
        if type(config) is not LatentMemoryConfig:
            raise TypeError("config must be exact LatentMemoryConfig")
        self.config = config
        self.base_slots = nn.Parameter(
            torch.empty(config.latent_slots, config.hidden_size)
        )
        nn.init.normal_(self.base_slots, mean=0.0, std=0.02)

        self.query_projection = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.attention_heads,
            batch_first=True,
        )
        self.update_gate = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
        )
        self.candidate_projection = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    @property
    def latent_slots(self) -> int:
        return self.config.latent_slots

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def initialize(self, query_summary: torch.Tensor) -> torch.Tensor:
        if (
            query_summary.ndim != 2
            or query_summary.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "query_summary must have shape [batch, hidden_size]"
            )
        batch_size = query_summary.shape[0]
        base = self.base_slots.unsqueeze(0).expand(batch_size, -1, -1)
        query = self.query_projection(query_summary).unsqueeze(1)
        return self.layer_norm(base + query)

    def update(
        self,
        state: torch.Tensor,
        chunk_hidden: torch.Tensor,
        chunk_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            state.ndim != 3
            or state.shape[1:] != (self.latent_slots, self.hidden_size)
        ):
            raise ValueError(
                "state must have shape [batch, latent_slots, hidden_size]"
            )
        if (
            chunk_hidden.ndim != 3
            or chunk_hidden.shape[0] != state.shape[0]
            or chunk_hidden.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "chunk_hidden must have shape [batch, chunk_tokens, hidden_size]"
            )
        if chunk_hidden.shape[1] < 1:
            raise ValueError("chunk_hidden must contain at least one token")

        key_padding_mask = None
        if chunk_attention_mask is not None:
            if (
                chunk_attention_mask.ndim != 2
                or chunk_attention_mask.shape
                != chunk_hidden.shape[:2]
            ):
                raise ValueError(
                    "chunk_attention_mask must have shape [batch, chunk_tokens]"
                )
            if not torch.all(
                (chunk_attention_mask == 0)
                | (chunk_attention_mask == 1)
            ):
                raise ValueError(
                    "chunk_attention_mask must contain only 0/1 values"
                )
            if torch.any(chunk_attention_mask.sum(dim=1) == 0):
                raise ValueError(
                    "each chunk must contain at least one visible token"
                )
            key_padding_mask = ~chunk_attention_mask.to(
                dtype=torch.bool,
                device=chunk_hidden.device,
            )

        attended, _ = self.cross_attention(
            query=state,
            key=chunk_hidden,
            value=chunk_hidden,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        combined = torch.cat((state, attended), dim=-1)
        gate = torch.sigmoid(self.update_gate(combined))
        candidate = torch.tanh(self.candidate_projection(combined))
        return self.layer_norm(
            gate * state + (1.0 - gate) * candidate
        )

    def compress(
        self,
        query_summary: torch.Tensor,
        chunks: tuple[torch.Tensor, ...],
        chunk_attention_masks: tuple[torch.Tensor | None, ...] | None = None,
    ) -> torch.Tensor:
        if type(chunks) is not tuple or not chunks:
            raise ValueError("chunks must be a non-empty exact tuple")
        if chunk_attention_masks is None:
            masks = (None,) * len(chunks)
        else:
            if (
                type(chunk_attention_masks) is not tuple
                or len(chunk_attention_masks) != len(chunks)
            ):
                raise ValueError(
                    "chunk_attention_masks must align exactly with chunks"
                )
            masks = chunk_attention_masks

        state = self.initialize(query_summary)
        for chunk_hidden, mask in zip(chunks, masks, strict=True):
            state = self.update(state, chunk_hidden, mask)
        return state


def compose_latent_prefix(
    latent_state: torch.Tensor,
    token_embeddings: torch.Tensor,
    token_attention_mask: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
) -> LatentPrefixBatch:
    if latent_state.ndim != 3:
        raise ValueError(
            "latent_state must have shape [batch, latent_slots, hidden]"
        )
    if token_embeddings.ndim != 3:
        raise ValueError(
            "token_embeddings must have shape [batch, tokens, hidden]"
        )
    if latent_state.shape[0] != token_embeddings.shape[0]:
        raise ValueError("latent and token batch sizes must match")
    if latent_state.shape[-1] != token_embeddings.shape[-1]:
        raise ValueError("latent and token hidden widths must match")
    if token_attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError(
            "token_attention_mask must have shape [batch, tokens]"
        )

    batch_size, latent_slots, _ = latent_state.shape
    latent_mask = torch.ones(
        batch_size,
        latent_slots,
        dtype=token_attention_mask.dtype,
        device=token_attention_mask.device,
    )
    combined_embeddings = torch.cat(
        (
            latent_state.to(
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            ),
            token_embeddings,
        ),
        dim=1,
    )
    combined_mask = torch.cat(
        (latent_mask, token_attention_mask),
        dim=1,
    )

    combined_labels = None
    if labels is not None:
        if labels.shape != token_attention_mask.shape:
            raise ValueError(
                "labels must have shape [batch, tokens]"
            )
        latent_labels = torch.full(
            (batch_size, latent_slots),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        combined_labels = torch.cat((latent_labels, labels), dim=1)

    return LatentPrefixBatch(
        inputs_embeds=combined_embeddings,
        attention_mask=combined_mask,
        labels=combined_labels,
    )
