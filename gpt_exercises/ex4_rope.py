"""EX4: replace learned absolute positions with rotary position embeddings."""

import torch
from torch import nn
from torch.nn import functional as F


class RotaryAttention(nn.Module):
    def __init__(self, n_embed: int, n_head: int, block_size: int) -> None:
        super().__init__()
        if n_embed % n_head or (n_embed // n_head) % 2:
            raise ValueError("head dimension must be even and divisible")
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.projection = nn.Linear(n_embed, n_embed)
        positions = torch.arange(block_size).float().unsqueeze(1)
        frequencies = torch.arange(0, self.head_size, 2).float()
        frequencies = 1.0 / (10000 ** (frequencies / self.head_size))
        angles = positions * frequencies
        self.register_buffer("cos", angles.cos()[None, None, :, :])
        self.register_buffer("sin", angles.sin()[None, None, :, :])
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def rotate(self, values: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.cos, torch.Tensor)
        assert isinstance(self.sin, torch.Tensor)

        even = values[..., 0::2]
        odd = values[..., 1::2]
        cos = self.cos[:, :, : values.size(2)]
        sin = self.sin[:, :, : values.size(2)]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        query, key, value = self.qkv(x).split(channels, dim=-1)
        shape = (batch, time, self.n_head, self.head_size)
        query = self.rotate(query.view(shape).transpose(1, 2))
        key = self.rotate(key.view(shape).transpose(1, 2))
        value = value.view(shape).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) * self.head_size**-0.5
        scores = scores.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        output = (
            (weights @ value).transpose(1, 2).contiguous().view(batch, time, channels)
        )
        return self.projection(output)


if __name__ == "__main__":
    layer = RotaryAttention(n_embed=64, n_head=4, block_size=32)
    x = torch.randn(2, 32, 64)
    print(f"input={tuple(x.shape)} output={tuple(layer(x).shape)}")
