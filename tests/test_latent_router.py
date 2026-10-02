from __future__ import annotations

import torch

from spm_bench.latent_router import (
    DIRECT_ROUTE,
    INSUFFICIENT_ROUTE,
    LatentResolutionRouter,
    ResolutionRouterConfig,
)


def _router() -> LatentResolutionRouter:
    torch.manual_seed(1729)
    return LatentResolutionRouter(
        ResolutionRouterConfig(
            hidden_size=16,
            latent_slots=4,
            attention_heads=4,
            max_chunks=5,
        )
    )


def test_router_outputs_chunk_plus_direct_plus_insufficient_classes():
    router = _router()
    query = torch.randn(2, 16)
    chunks = (
        torch.randn(2, 7, 16),
        torch.randn(2, 5, 16),
        torch.randn(2, 3, 16),
    )

    result = router(query, chunks)

    assert result.logits.shape == (2, 7)
    assert result.latent_state.shape == (2, 4, 16)
    assert result.route_labels == (
        "chunk:0",
        "chunk:1",
        "chunk:2",
        "chunk:3",
        "chunk:4",
        DIRECT_ROUTE,
        INSUFFICIENT_ROUTE,
    )


def test_router_masks_nonexistent_chunk_routes():
    router = _router()
    query = torch.randn(1, 16)
    chunks = (
        torch.randn(1, 3, 16),
        torch.randn(1, 4, 16),
    )

    result = router(query, chunks)

    assert torch.isneginf(result.logits[0, 2:5]).all()
    assert torch.isfinite(result.logits[0, :2]).all()
    assert torch.isfinite(result.logits[0, 5:]).all()


def test_route_label_for_exact_case_is_decisive_chunk_index():
    config = ResolutionRouterConfig(
        hidden_size=16,
        latent_slots=4,
        attention_heads=4,
        max_chunks=5,
    )

    assert config.chunk_target(0) == 0
    assert config.chunk_target(4) == 4


def test_special_route_targets_are_stable():
    config = ResolutionRouterConfig(
        hidden_size=16,
        latent_slots=4,
        attention_heads=4,
        max_chunks=5,
    )

    assert config.direct_target == 5
    assert config.insufficient_target == 6


def test_router_keeps_state_budget_fixed_as_chunk_count_changes():
    router = _router()
    query = torch.randn(1, 16)
    few = (torch.randn(1, 2, 16),)
    many = tuple(torch.randn(1, 2 + i, 16) for i in range(5))

    first = router(query, few)
    second = router(query, many)

    assert first.latent_state.numel() == second.latent_state.numel() == 64
