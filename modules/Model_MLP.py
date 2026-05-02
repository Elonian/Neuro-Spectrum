"""Simple feedforward model for current and past x samples."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, L: int, hidden: int, num_hidden_layers: int) -> None:
        super().__init__()
        if L < 0:
            raise ValueError("L must not be negative")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must not be negative")
        if num_hidden_layers > 0 and hidden <= 0:
            raise ValueError("hidden must be positive when num_hidden_layers > 0")

        self.L = L
        self.hidden = hidden
        self.num_hidden_layers = num_hidden_layers
        in_dim = L + 1

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU(inplace=True))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.L + 1:
            raise ValueError(
                f"expected x shape (batch, {self.L + 1}), got {tuple(x.shape)}"
            )
        return self.net(x).squeeze(-1)


def build_model(L: int, hidden: int = 64, num_hidden_layers: int = 2) -> SimpleMLP:
    return SimpleMLP(L=L, hidden=hidden, num_hidden_layers=num_hidden_layers)
