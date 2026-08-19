"""MODELf2 multi-task network with a learnable satellite embedding."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MTLEmbModel(nn.Module):
    """Shared MLP for joint DIR/DIF estimation with satellite-specific embedding."""

    def __init__(
        self,
        numeric_input_dim: int,
        num_sats: int,
        embedding_dim: int | None = None,
        hidden_shared: Sequence[int] = (128, 64),
        head_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if embedding_dim is None:
            embedding_dim = min(50, max(2, (num_sats + 1) // 2))

        self.embedding = nn.Embedding(
            num_embeddings=num_sats,
            embedding_dim=embedding_dim,
        )

        layers: list[nn.Module] = []
        input_dim = numeric_input_dim + embedding_dim
        for hidden_dim in hidden_shared:
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_dim = hidden_dim
        self.shared_net = nn.Sequential(*layers)

        self.dir_head = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )
        self.dif_head = nn.Sequential(
            nn.Linear(input_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        numeric_features: torch.Tensor,
        satellite_ids: torch.Tensor,
    ) -> torch.Tensor:
        satellite_embedding = self.embedding(satellite_ids)
        shared_input = torch.cat([numeric_features, satellite_embedding], dim=1)
        shared_features = self.shared_net(shared_input)
        dir_output = self.dir_head(shared_features)
        dif_output = self.dif_head(shared_features)
        return torch.cat([dir_output, dif_output], dim=1)


# Compatibility with the class name used in the original estimation script.
MTL_Emb_Model = MTLEmbModel

