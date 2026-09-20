"""Multi-head graph encoder and masked pointer decoder for routing."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphPointerNetwork(nn.Module):
    """Autoregressive graph pointer policy adapted from attention-learn-to-route concepts."""

    def __init__(self, hidden_dim: int = 128, encoder_layers: int = 3, heads: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Linear(6, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=512,
            batch_first=True,
            activation="relu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.query = nn.GRUCell(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        features: Tensor,
        decode_type: str = "sample",
        travel_time_seconds: Tensor | None = None,
        hover_equivalent_seconds: float = 0.0,
        max_endurance_seconds: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return a masked client order and its summed log-probability.

        When a directed time matrix and endurance are supplied, each decoding
        step additionally masks a client unless the UAV can inspect it and
        still return safely to its depot.  This keeps the learned policy from
        sampling an invalid action rather than merely rejecting its route
        after decoding.
        """
        if features.ndim != 3 or features.size(-1) != 6:
            raise ValueError("features must have shape (batch, nodes, 6).")
        if (travel_time_seconds is None) != (max_endurance_seconds is None):
            raise ValueError("travel_time_seconds and max_endurance_seconds must be supplied together.")
        expected_time_shape = (features.size(0), features.size(1), features.size(1))
        if travel_time_seconds is not None and tuple(travel_time_seconds.shape) != expected_time_shape:
            raise ValueError("travel_time_seconds must have shape (batch, nodes, nodes).")
        encoded = self.encoder(self.embedding(features))
        batch_size, node_count, hidden_dim = encoded.shape
        depot = encoded[:, 0]
        query = depot
        keys = self.key(encoded)
        visited = torch.zeros(batch_size, node_count, dtype=torch.bool, device=features.device)
        visited[:, 0] = True
        current = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        elapsed = torch.zeros(batch_size, device=features.device)
        selected: list[Tensor] = []
        log_probabilities: list[Tensor] = []
        scale = hidden_dim**-0.5
        for _ in range(node_count - 1):
            query = self.query(query, depot)
            logits = (keys * self.query_projection(query).unsqueeze(1)).sum(dim=-1) * scale
            invalid = visited
            if travel_time_seconds is not None:
                from_current = travel_time_seconds[
                    torch.arange(batch_size, device=features.device), current, :
                ]
                return_to_depot = travel_time_seconds[:, :, 0]
                safe = elapsed.unsqueeze(1) + from_current + hover_equivalent_seconds + return_to_depot <= max_endurance_seconds
                invalid = visited | ~safe
                if torch.any(invalid.all(dim=1)):
                    raise RuntimeError("No endurance-safe unvisited action is available during decoding.")
            logits = logits.masked_fill(invalid, float("-inf"))
            distribution = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if decode_type == "greedy" else distribution.sample()
            selected.append(action)
            log_probabilities.append(distribution.log_prob(action))
            visited.scatter_(1, action.unsqueeze(1), True)
            if travel_time_seconds is not None:
                elapsed = elapsed + travel_time_seconds[
                    torch.arange(batch_size, device=features.device), current, action
                ] + hover_equivalent_seconds
            current = action
            query = encoded[torch.arange(batch_size, device=features.device), action]
        return torch.stack(selected, dim=1), torch.stack(log_probabilities, dim=1).sum(dim=1)


def make_node_features(coordinates: Tensor, wind_speed_mps: float, wind_direction_degrees: float) -> Tensor:
    """Normalize coordinates and append depot flag plus global wind features."""
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    normalized = centered / scale
    batch_size, nodes, _ = coordinates.shape
    depot_flag = torch.zeros(batch_size, nodes, 1, device=coordinates.device)
    depot_flag[:, 0, 0] = 1.0
    direction = torch.deg2rad(torch.tensor(wind_direction_degrees, device=coordinates.device))
    wind = torch.tensor(
        [wind_speed_mps / 23.0, torch.sin(direction).item(), torch.cos(direction).item()],
        device=coordinates.device,
    ).view(1, 1, 3).expand(batch_size, nodes, 3)
    return torch.cat((normalized, depot_flag, wind), dim=-1)
