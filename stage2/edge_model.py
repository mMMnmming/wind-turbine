"""Directed-cost encoder and remaining-task-aware attention policy, without labels."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DirectedEncoderLayer(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.edge_bias = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, heads))
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, width))
        self.norm2 = nn.LayerNorm(width)

    def forward(self, encoded: Tensor, edges: Tensor, valid: Tensor) -> Tensor:
        batch, nodes, _ = encoded.shape
        bias = self.edge_bias(edges).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = self.attention(encoded, encoded, encoded,
                                  attn_mask=bias.reshape(batch * self.heads, nodes, nodes),
                                  need_weights=False)[0]
        encoded = self.norm1(encoded + attended)
        return self.norm2(encoded + self.feedforward(encoded))


class EdgeAttentionPolicy(nn.Module):
    """A custom edge-aware policy, not a claim to reproduce the full MatNet."""

    feature_count = 7

    def __init__(self, hidden_dim: int = 128, encoder_layers: int = 3, heads: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Linear(7, hidden_dim)
        self.layers = nn.ModuleList([DirectedEncoderLayer(hidden_dim, heads) for _ in range(encoder_layers)])
        self.context = nn.Sequential(nn.Linear(4 * hidden_dim + 3, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.glimpse = nn.MultiheadAttention(hidden_dim, heads, dropout=0, batch_first=True)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.edge_score = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, features: Tensor, valid_node_mask: Tensor | None = None, *,
               time_matrix: Tensor, hover: Tensor, endurance: Tensor) -> dict[str, Tensor]:
        if features.ndim != 3 or features.size(-1) != 7 or features.size(1) < 1:
            raise ValueError("features must have shape (batch, nodes>=1, 7).")
        batch, nodes, _ = features.shape
        valid = (torch.ones(batch, nodes, dtype=torch.bool, device=features.device)
                 if valid_node_mask is None else valid_node_mask.to(device=features.device, dtype=torch.bool))
        if valid.shape != (batch, nodes) or not valid[:, 0].all():
            raise ValueError("Every graph needs a valid depot and a correctly shaped valid mask.")
        if time_matrix.shape != (batch, nodes, nodes):
            raise ValueError("Directed time matrix shape differs from nodes.")
        if not torch.isfinite(time_matrix).all() or (time_matrix < 0).any():
            raise ValueError("Directed times must be finite and nonnegative.")
        if (hover.shape != (batch,) or endurance.shape != (batch,) or not torch.isfinite(hover).all()
                or not torch.isfinite(endurance).all() or (hover < 0).any() or (endurance <= 0).any()):
            raise ValueError("Invalid hover/endurance vectors.")
        off_diagonal = ~torch.eye(nodes, dtype=torch.bool, device=features.device)[None]
        edge_valid = valid[:, :, None] & valid[:, None, :] & off_diagonal
        scale = (time_matrix.masked_fill(~edge_valid, 0).sum((1, 2)) /
                 edge_valid.sum((1, 2)).clamp_min(1)).clamp_min(1e-6)
        normalized = time_matrix / scale[:, None, None]
        edges = torch.stack((normalized, normalized.transpose(1, 2)), dim=-1)
        encoded = self.embedding(features)
        for layer in self.layers:
            encoded = layer(encoded, edges, valid)
        return {"encoded": encoded, "valid": valid, "times": time_matrix, "normalized": normalized,
                "scale": scale, "hover": hover, "endurance": endurance}

    def decode(self, cache: dict[str, Tensor], decode_type: str = "sample", rollouts: int = 1,
               generator: torch.Generator | None = None, return_details: bool = False
               ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Decode independent policy trajectories from one shared graph encoding."""
        if decode_type not in ("sample", "greedy") or rollouts < 1:
            raise ValueError("Invalid decode type or rollout count.")
        state = {name: value.repeat_interleave(rollouts, dim=0) for name, value in cache.items()}
        encoded, valid = state["encoded"], state["valid"]
        batch, nodes, width = encoded.shape
        row = torch.arange(batch, device=encoded.device)
        counts = valid.sum(1) - 1
        steps = int(counts.max().item())
        if not steps:
            result = (torch.empty(batch, 0, dtype=torch.long, device=encoded.device), encoded.sum((1, 2)) * 0)
            return (*result, {"entropy": result[1],
                              "step_log_probs": encoded.sum((1, 2))[:, None].expand(-1, 0),
                              "step_mask": torch.empty(batch, 0, dtype=torch.bool, device=encoded.device)}) if return_details else result
        visited = ~valid.clone()
        visited[:, 0] = True
        current = torch.zeros(batch, dtype=torch.long, device=encoded.device)
        spent = encoded.new_zeros(batch)
        global_context = (encoded * valid[:, :, None]).sum(1) / valid.sum(1)[:, None]
        keys = self.key(encoded)
        orders, logps, entropies = [], [], []
        for step in range(steps):
            active = step < counts
            remaining = ~visited
            remaining_context = (encoded * remaining[:, :, None]).sum(1) / remaining.sum(1).clamp_min(1)[:, None]
            scalars = torch.stack(((state["endurance"] - spent) / state["endurance"],
                                   remaining.sum(1) / counts.clamp_min(1), state["scale"] / state["endurance"]), 1)
            query = self.context(torch.cat((global_context, remaining_context, encoded[row, current],
                                            encoded[:, 0], scalars), 1))
            # Finished rows attend to a harmless depot, avoiding all-masked softmax.
            invalid = visited.clone()
            invalid[~active, 0] = False
            read = self.glimpse(query[:, None], encoded, encoded, key_padding_mask=invalid,
                                need_weights=False)[0][:, 0]
            query = self.context_norm(query + read)
            times = state["normalized"]
            edge_features = torch.stack((times[row, current], times[row, :, current],
                                          times[:, :, 0], times[:, 0, :]), -1)
            logits = (keys * self.query(query)[:, None]).sum(-1) / width**0.5
            logits = 10 * torch.tanh(logits + self.edge_score(edge_features).squeeze(-1))
            logits = logits.masked_fill(invalid, float("-inf"))
            distribution = torch.distributions.Categorical(logits=logits)
            action = (logits.argmax(1) if decode_type == "greedy" else
                      torch.multinomial(distribution.probs, 1, generator=generator).squeeze(1))
            action = torch.where(active, action, 0)
            orders.append(action)
            logps.append(torch.where(active, distribution.log_prob(action), 0))
            entropies.append(torch.where(active, distribution.entropy(), 0))
            spent = spent + torch.where(active, state["times"][row, current, action] + state["hover"], 0)
            visited = visited.scatter(1, action[:, None], True)
            current = action
        result = (torch.stack(orders, 1), torch.stack(logps, 1).sum(1))
        details = {"entropy": torch.stack(entropies, 1).sum(1) / counts.clamp_min(1),
                   "step_log_probs": torch.stack(logps, 1),
                   "step_mask": torch.arange(steps, device=encoded.device)[None] < counts[:, None]}
        return (*result, details) if return_details else result

    def forward(self, features: Tensor, decode_type: str = "sample", valid_node_mask: Tensor | None = None,
                *, time_matrix: Tensor, hover: Tensor, endurance: Tensor,
                rollouts: int = 1) -> tuple[Tensor, Tensor]:
        return self.decode(self.encode(features, valid_node_mask, time_matrix=time_matrix,
                                       hover=hover, endurance=endurance), decode_type, rollouts)
