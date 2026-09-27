"""EX1: fused multi-head self-attention using one batched tensor operation."""

import torch
from torch import nn
from torch.nn import functional as F


class CombinedMultiHeadAttention(nn.Module):
    """Equivalent to separate Head objects, but heads are a tensor dimension."""

    def __init__(
        self, n_embed: int, n_head: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if n_embed % n_head != 0:
            raise ValueError("n_embed must be divisible by n_head")
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.query = nn.Linear(n_embed, n_embed, bias=False)
        self.key = nn.Linear(n_embed, n_embed, bias=False)
        self.value = nn.Linear(n_embed, n_embed, bias=False)
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        query = (
            self.query(x).view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        )
        key = self.key(x).view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        value = (
            self.value(x).view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        )
        scores = query @ key.transpose(-2, -1) * self.head_size**-0.5
        scores = scores.masked_fill(self.causal_mask[:time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.projection(output))


if __name__ == "__main__":
    torch.manual_seed(1337)
    attention = CombinedMultiHeadAttention(n_embed=64, n_head=4, block_size=32)
    sample = torch.randn(2, 32, 64)
    print(f"input={tuple(sample.shape)} output={tuple(attention(sample).shape)}")
