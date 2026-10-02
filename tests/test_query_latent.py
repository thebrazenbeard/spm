from __future__ import annotations

import pytest
import torch

from spm_bench.query_latent import (
    LatentMemoryConfig,
    QueryConditionedLatentMemory,
    compose_latent_prefix,
)


def _module() -> QueryConditionedLatentMemory:
    torch.manual_seed(1729)
    return QueryConditionedLatentMemory(
        LatentMemoryConfig(
            hidden_size=16,
            latent_slots=4,
            attention_heads=4,
        )
    )


def test_config_rejects_invalid_slot_or_attention_geometry():
    with pytest.raises(ValueError, match="latent_slots"):
        LatentMemoryConfig(hidden_size=16, latent_slots=0, attention_heads=4)

    with pytest.raises(ValueError, match="divisible"):
        LatentMemoryConfig(hidden_size=15, latent_slots=4, attention_heads=4)


def test_initialize_is_query_conditioned_and_has_fixed_slot_budget():
    module = _module()
    query_a = torch.zeros(2, 16)
    query_b = torch.ones(2, 16)

    state_a = module.initialize(query_a)
    state_b = module.initialize(query_b)

    assert state_a.shape == (2, 4, 16)
    assert state_b.shape == (2, 4, 16)
    assert not torch.equal(state_a, state_b)


def test_update_preserves_fixed_state_shape_across_longer_chunks():
    module = _module()
    query = torch.randn(2, 16)
    state = module.initialize(query)

    short = torch.randn(2, 5, 16)
    long = torch.randn(2, 41, 16)

    state = module.update(state, short)
    state = module.update(state, long)

    assert state.shape == (2, 4, 16)
    assert torch.isfinite(state).all()


def test_masked_chunk_tokens_do_not_change_state():
    module = _module()
    query = torch.randn(1, 16)
    state = module.initialize(query)
    visible = torch.randn(1, 2, 16)
    junk = torch.randn(1, 3, 16) * 1000

    padded = torch.cat((visible, junk), dim=1)
    mask = torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.long)

    from_visible = module.update(state, visible)
    from_padded = module.update(state, padded, mask)

    torch.testing.assert_close(from_visible, from_padded)


def test_compress_updates_sequentially_without_growing_latent_slots():
    module = _module()
    query = torch.randn(1, 16)
    chunks = (
        torch.randn(1, 7, 16),
        torch.randn(1, 11, 16),
        torch.randn(1, 3, 16),
    )

    compressed = module.compress(query, chunks)

    assert compressed.shape == (1, 4, 16)
    assert compressed.numel() == 64


def test_compose_latent_prefix_prepends_slots_and_masks_loss_on_them():
    latent = torch.randn(2, 4, 16)
    tokens = torch.randn(2, 6, 16)
    token_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [10, 11, 12, 13, -100, -100],
            [20, 21, 22, 23, 24, 25],
        ],
        dtype=torch.long,
    )

    batch = compose_latent_prefix(
        latent,
        tokens,
        token_mask,
        labels=labels,
    )

    assert batch.inputs_embeds.shape == (2, 10, 16)
    assert batch.attention_mask.shape == (2, 10)
    assert torch.equal(batch.attention_mask[:, :4], torch.ones(2, 4, dtype=torch.long))
    assert batch.labels is not None
    assert torch.equal(batch.labels[:, :4], torch.full((2, 4), -100))
    assert torch.equal(batch.labels[:, 4:], labels)


def test_compose_latent_prefix_rejects_mismatched_hidden_width():
    with pytest.raises(ValueError, match="hidden"):
        compose_latent_prefix(
            torch.randn(1, 4, 16),
            torch.randn(1, 5, 15),
            torch.ones(1, 5, dtype=torch.long),
        )
