"""Small learned vector-state pathway for the first matched B*/C experiment."""

from __future__ import annotations

import torch
from torch import nn


class VectorStateModule(nn.Module):
    """Update and reinject an opaque fixed-width learned recurrent state."""

    def __init__(self, *, hidden_size: int, state_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or state_size <= 0:
            raise ValueError("hidden_size and state_size must be positive")
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.observation_projection = nn.Linear(hidden_size, state_size)
        self.updater = nn.GRUCell(state_size, state_size)
        self.state_to_hidden = nn.Linear(state_size, hidden_size, bias=False)

    def zero_state(self, batch_size: int, *, device=None, dtype=None) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = self.observation_projection.weight
        return torch.zeros(
            batch_size,
            self.state_size,
            device=device if device is not None else reference.device,
            dtype=dtype if dtype is not None else reference.dtype,
        )

    def update(
        self,
        observation_hidden: torch.Tensor,
        prior_state: torch.Tensor,
        *,
        persistent: bool,
    ) -> torch.Tensor:
        if observation_hidden.ndim != 2 or observation_hidden.shape[-1] != self.hidden_size:
            raise ValueError("observation_hidden must have shape [batch, hidden_size]")
        if prior_state.ndim != 2 or prior_state.shape != (
            observation_hidden.shape[0],
            self.state_size,
        ):
            raise ValueError("prior_state must have shape [batch, state_size]")
        projected = self.observation_projection(observation_hidden)
        effective_prior = prior_state if persistent else torch.zeros_like(prior_state)
        return self.updater(projected, effective_prior)

    def reinject(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        last_positions: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have shape [batch, seq, hidden_size]")
        batch_size = hidden_states.shape[0]
        if state.shape != (batch_size, self.state_size):
            raise ValueError("state must have shape [batch, state_size]")
        if last_positions.shape != (batch_size,):
            raise ValueError("last_positions must have shape [batch]")
        if last_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("last_positions must be an integer tensor")
        if torch.any(last_positions < 0) or torch.any(last_positions >= hidden_states.shape[1]):
            raise ValueError("last_positions contains an out-of-range index")

        delta = self.state_to_hidden(state).to(hidden_states.dtype)
        conditioned = hidden_states.clone()
        row_ids = torch.arange(batch_size, device=hidden_states.device)
        positions = last_positions.to(hidden_states.device)
        conditioned[row_ids, positions] = conditioned[row_ids, positions] + delta.to(
            hidden_states.device
        )
        return conditioned
