from __future__ import annotations

import torch

from spm_bench.vector_state import VectorStateModule


def make_module() -> VectorStateModule:
    torch.manual_seed(1729)
    return VectorStateModule(hidden_size=2048, state_size=128)


def test_vector_state_v1_has_frozen_trainable_parameter_budget():
    module = make_module()
    assert sum(parameter.numel() for parameter in module.parameters()) == 623_488
    assert module.hidden_size == 2048
    assert module.state_size == 128


def test_bstar_resets_prior_while_c_uses_prior():
    module = make_module()
    observation = torch.randn(2, 2048)
    prior = torch.randn(2, 128)
    zero = torch.zeros_like(prior)

    bstar_from_prior = module.update(observation, prior, persistent=False)
    bstar_from_zero = module.update(observation, zero, persistent=False)
    c_from_prior = module.update(observation, prior, persistent=True)
    c_from_zero = module.update(observation, zero, persistent=True)

    assert torch.equal(bstar_from_prior, bstar_from_zero)
    assert not torch.equal(c_from_prior, c_from_zero)


def test_reinject_changes_only_declared_last_token_positions():
    module = make_module()
    hidden = torch.zeros(2, 5, 2048)
    state = torch.randn(2, 128)
    positions = torch.tensor([1, 4])

    conditioned = module.reinject(hidden, state, positions)

    assert conditioned.shape == hidden.shape
    assert torch.count_nonzero(conditioned[0, 0]) == 0
    assert torch.count_nonzero(conditioned[0, 2:]) == 0
    assert torch.count_nonzero(conditioned[1, :4]) == 0
    assert torch.count_nonzero(conditioned[0, 1]) > 0
    assert torch.count_nonzero(conditioned[1, 4]) > 0
    assert torch.count_nonzero(hidden) == 0


def test_state_shape_validation_fails_closed():
    module = make_module()
    observation = torch.randn(2, 2048)
    bad_prior = torch.randn(2, 127)

    try:
        module.update(observation, bad_prior, persistent=True)
    except ValueError as error:
        assert "prior_state" in str(error)
    else:
        raise AssertionError("invalid prior state width was accepted")
