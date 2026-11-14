"""PyTorch model definition for the dish classifier."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    hidden_dims: tuple[int, ...] = (384, 192)
    dropout: float = 0.35

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden_dims"] = list(self.hidden_dims)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        data = dict(data)
        data["hidden_dims"] = tuple(data["hidden_dims"])
        return cls(**data)


class DishClassifier(nn.Module):
    """A feed-forward classifier over TF-IDF features.

    Deliberately shallow: the input is already a high-dimensional sparse
    bag-of-ngrams, so depth buys very little while costing a lot of overfitting
    on a catalogue of a few hundred classes. LayerNorm rather than BatchNorm,
    because inference happens one sample at a time behind the API.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        prev = config.input_dim
        for width in config.hidden_dims:
            layers += [
                nn.Linear(prev, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ]
            prev = width
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev, config.num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation — useful for nearest-dish explanations."""
        return self.encoder(x)
