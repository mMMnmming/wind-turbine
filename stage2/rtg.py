"""Undiscounted event rewards and same-graph leave-one-out policy gradients."""

import torch
from torch import Tensor

from .train import TrainingConfig, _reward


def event_rewards(
    times: Tensor,
    routes: Tensor,
    counts: Tensor,
    hover: Tensor,
    endurance: Tensor,
    config: TrainingConfig,
) -> Tensor:
    """Customer events followed by one return/completion event (no policy action)."""
    batch, steps = routes.shape
    row = torch.arange(batch, device=routes.device)
    active = torch.arange(steps, device=routes.device)[None] < counts[:, None]
    previous = torch.cat(
        (torch.zeros(batch, 1, device=routes.device, dtype=torch.long), routes), 1
    )
    legs = times[row[:, None], previous[:, :steps], routes] * active
    last = previous.gather(1, counts[:, None]).squeeze(1)
    returning = times[row, last, 0]
    # Preserve the original reduction order, including the return at counts.
    # Regrouping a float32 sum could flip feasibility at an endurance boundary.
    all_legs = torch.cat((legs, times.new_zeros(batch, 1)), 1)
    all_legs.scatter_add_(1, counts[:, None], returning[:, None])
    flight = all_legs.sum(1)
    energy = flight + counts * hover
    scale = endurance.clamp_min(1e-6)
    events = times.new_zeros(batch, steps + 1)
    events[:, :steps] = (
        -config.distance_weight * legs / scale[:, None]
        - config.energy_weight * (legs + hover[:, None] * active) / scale[:, None]
    )
    total = _reward(flight, energy, energy <= endurance, endurance, config)
    # Subtract only customer costs: the residual is exactly return + terminal
    # completion/safety, using the same full-tour feasibility as the old reward.
    terminal = total - events.sum(1)
    events.scatter_add_(1, counts[:, None], terminal[:, None])
    return events


def future_returns(events: Tensor) -> Tensor:
    return events.flip(-1).cumsum(-1).flip(-1)


def step_advantage(returns: Tensor) -> Tensor:
    if returns.ndim != 3 or returns.size(1) < 2:
        raise ValueError("Step leave-one-out requires (graphs, rollouts>=2, steps).")
    baseline = (returns.sum(1, keepdim=True) - returns) / (returns.size(1) - 1)
    return (returns - baseline).detach()


def step_policy_loss(advantage: Tensor, logp: Tensor, mask: Tensor) -> Tensor:
    if advantage.shape != logp.shape or mask.shape != logp.shape:
        raise ValueError("Step tensors must have identical shapes.")
    return -(advantage.detach() * logp * mask).sum(-1).mean()
