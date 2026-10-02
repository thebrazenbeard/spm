from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .query_latent import (
    LatentMemoryConfig,
    QueryConditionedLatentMemory,
)


DIRECT_ROUTE = "DIRECT"
INSUFFICIENT_ROUTE = "INSUFFICIENT_FIDELITY"


@dataclass(frozen=True, slots=True)
class ResolutionRouterConfig:
    hidden_size: int
    latent_slots: int
    attention_heads: int
    max_chunks: int

    def __post_init__(self) -> None:
        LatentMemoryConfig(
            hidden_size=self.hidden_size,
            latent_slots=self.latent_slots,
            attention_heads=self.attention_heads,
        )
        if (
            type(self.max_chunks) is not int
            or isinstance(self.max_chunks, bool)
            or self.max_chunks < 1
        ):
            raise ValueError("max_chunks must be a positive exact int")

    @property
    def direct_target(self) -> int:
        return self.max_chunks

    @property
    def insufficient_target(self) -> int:
        return self.max_chunks + 1

    def chunk_target(self, chunk_index: int) -> int:
        if (
            type(chunk_index) is not int
            or isinstance(chunk_index, bool)
            or not 0 <= chunk_index < self.max_chunks
        ):
            raise ValueError("chunk_index is outside configured range")
        return chunk_index


@dataclass(frozen=True, slots=True)
class ResolutionRouterOutput:
    logits: torch.Tensor
    latent_state: torch.Tensor
    route_labels: tuple[str, ...]


class LatentResolutionRouter(nn.Module):
    """Learn which higher-resolution block to promote from compact latent state."""

    def __init__(self, config: ResolutionRouterConfig) -> None:
        super().__init__()
        if type(config) is not ResolutionRouterConfig:
            raise TypeError("config must be exact ResolutionRouterConfig")
        self.config = config
        self.memory = QueryConditionedLatentMemory(
            LatentMemoryConfig(
                hidden_size=config.hidden_size,
                latent_slots=config.latent_slots,
                attention_heads=config.attention_heads,
            )
        )
        self.route_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.max_chunks + 2),
        )

    @property
    def route_labels(self) -> tuple[str, ...]:
        return (
            *(f"chunk:{index}" for index in range(self.config.max_chunks)),
            DIRECT_ROUTE,
            INSUFFICIENT_ROUTE,
        )

    def forward(
        self,
        query_summary: torch.Tensor,
        chunks: tuple[torch.Tensor, ...],
        chunk_attention_masks: tuple[torch.Tensor | None, ...] | None = None,
    ) -> ResolutionRouterOutput:
        if type(chunks) is not tuple or not chunks:
            raise ValueError("chunks must be a non-empty exact tuple")
        if len(chunks) > self.config.max_chunks:
            raise ValueError("chunk count exceeds configured max_chunks")

        latent_state = self.memory.compress(
            query_summary,
            chunks,
            chunk_attention_masks,
        )
        pooled = latent_state.mean(dim=1)
        logits = self.route_head(pooled)

        if len(chunks) < self.config.max_chunks:
            valid = torch.ones(
                logits.shape[-1],
                dtype=torch.bool,
                device=logits.device,
            )
            valid[len(chunks): self.config.max_chunks] = False
            logits = logits.masked_fill(~valid.unsqueeze(0), -torch.inf)

        return ResolutionRouterOutput(
            logits=logits,
            latent_state=latent_state,
            route_labels=self.route_labels,
        )
