"""Transformer graph encoder and masked pointer decoder for routing."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GraphPointerNetwork(nn.Module):
    """Autoregressive policy with valid-node and visited-client masks."""

    feature_count = 7

    def __init__(self, hidden_dim: int = 128, encoder_layers: int = 3, heads: int = 8) -> None:
        super().__init__()
        self.embedding = nn.Linear(self.feature_count, hidden_dim)
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
        valid_node_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return each graph's unpadded client order and summed log probability.

        Depot is node zero and is an implicit start/end point. Padded nodes are
        excluded from both encoder attention and decoder actions. Each row stops
        decoding after visiting its own valid clients.
        """
        if features.ndim != 3 or features.size(-1) != self.feature_count:
            raise ValueError(f"features must have shape (batch, nodes, {self.feature_count}).")
        batch_size, node_count, _ = features.shape
        if node_count < 2:
            raise ValueError("Each graph must contain a depot and at least one client.")
        if valid_node_mask is None:
            valid_node_mask = torch.ones(batch_size, node_count, dtype=torch.bool, device=features.device)
        if tuple(valid_node_mask.shape) != (batch_size, node_count):
            raise ValueError("valid_node_mask must have shape (batch, nodes).")
        valid_node_mask = valid_node_mask.to(device=features.device, dtype=torch.bool)
        if not torch.all(valid_node_mask[:, 0]) or torch.any(valid_node_mask.sum(dim=1) < 2):
            raise ValueError("Every graph must have a valid depot and at least one client.")

        encoded = self.encoder(self.embedding(features), src_key_padding_mask=~valid_node_mask)
        depot = encoded[:, 0]
        query = depot
        keys = self.key(encoded)
        visited = torch.zeros(batch_size, node_count, dtype=torch.bool, device=features.device)
        visited[:, 0] = True
        client_counts = valid_node_mask.sum(dim=1) - 1
        selected: list[Tensor] = []
        log_probabilities: list[Tensor] = []
        # Dot products use the encoded width, not the seven input features.
        scale = encoded.size(-1)**-0.5
        for step in range(int(client_counts.max().item())):
            active = step < client_counts
            query = self.query(query, depot)
            logits = (keys * self.query_projection(query).unsqueeze(1)).sum(dim=-1) * scale
            invalid = visited | ~valid_node_mask
            logits = logits.masked_fill(invalid, float("-inf"))
            # Completed rows take a harmless depot placeholder with zero log-P.
            logits = torch.where(active.unsqueeze(1), logits, torch.zeros_like(logits))
            distribution = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if decode_type == "greedy" else distribution.sample()
            action = torch.where(active, action, torch.zeros_like(action))
            log_probability = torch.where(active, distribution.log_prob(action), torch.zeros_like(action, dtype=features.dtype))
            selected.append(action)
            log_probabilities.append(log_probability)
            visited.scatter_(1, action.unsqueeze(1), visited.gather(1, action.unsqueeze(1)) | active.unsqueeze(1))
            query = encoded[torch.arange(batch_size, device=features.device), action]

        return torch.stack(selected, dim=1), torch.stack(log_probabilities, dim=1).sum(dim=1)


def make_node_features(
    coordinates: Tensor,
    wind_speed_mps: float,
    wind_direction_degrees: float,
    gust_speed_mps: float = 12.0,
) -> Tensor:
    """Normalize XY and append depot, mean wind, compass direction and gust."""
    centered = coordinates - coordinates.mean(dim=1, keepdim=True)
    scale = centered.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    normalized = centered / scale
    batch_size, nodes, _ = coordinates.shape
    depot_flag = torch.zeros(batch_size, nodes, 1, device=coordinates.device, dtype=coordinates.dtype)
    depot_flag[:, 0, 0] = 1.0
    direction = torch.deg2rad(torch.tensor(wind_direction_degrees, device=coordinates.device))
    wind = torch.tensor(
        [wind_speed_mps / 23.0, torch.sin(direction).item(), torch.cos(direction).item(), gust_speed_mps / 23.0],
        device=coordinates.device,
        dtype=coordinates.dtype,
    ).view(1, 1, 4).expand(batch_size, nodes, 4)
    return torch.cat((normalized, depot_flag, wind), dim=-1)
