"""Simple LSTM model for current and past x samples."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, L: int, num_layers: int, hidden: int = 64) -> None:
        super().__init__()
        if L < 0:
            raise ValueError("L must not be negative")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.L = L
        self.hidden = hidden
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either:
        # - (batch, L+1) -> (batch,)
        # - (B, T, L+1)  -> (B, T)
        if x.dim() == 3:
            if x.size(-1) != self.L + 1:
                raise ValueError(
                    f"expected x shape (B,T,{self.L + 1}), got {tuple(x.shape)}"
                )
            B, T, _ = x.shape
            y = self.forward(x.reshape(B * T, self.L + 1))
            return y.reshape(B, T)

        if x.dim() != 2 or x.size(1) != self.L + 1:
            raise ValueError(f"expected x shape (batch, {self.L + 1}), got {tuple(x.shape)}")
        # (B, T, 1) time runs from current sample toward older samples
        seq = x.unsqueeze(-1)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


def build_model(L: int, num_layers: int, hidden: int = 64) -> SimpleLSTM:
    return SimpleLSTM(L=L, num_layers=num_layers, hidden=hidden)
