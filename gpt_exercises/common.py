"""Small reusable decoder-only Transformer components for the exercises."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        if n_embed % n_head:
            raise ValueError("n_embed must be divisible by n_head")
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.qkv = nn.Linear(n_embed, 3 * n_embed)
        self.projection = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape
        query, key, value = self.qkv(x).split(channels, dim=-1)
        query = query.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch, time, self.n_head, self.head_size).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) * self.head_size**-0.5
        scores = scores.masked_fill(self.mask[:time, :time] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        output = weights @ value
        output = output.transpose(1, 2).contiguous().view(batch, time, channels)
        return self.dropout(self.projection(output))


class Block(nn.Module):
    def __init__(
        self, n_embed: int, n_head: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(n_embed)
        self.attention = CausalSelfAttention(n_embed, n_head, block_size, dropout)
        self.norm2 = nn.LayerNorm(n_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_embed: int = 64,
        n_head: int = 4,
        n_layer: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *(Block(n_embed, n_head, block_size, dropout) for _ in range(n_layer))
        )
        self.norm = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = tokens.shape
        positions = torch.arange(time, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        logits = self.lm_head(self.norm(self.blocks(hidden)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens: torch.Tensor, count: int) -> torch.Tensor:
        for _ in range(count):
            context = tokens[:, -self.block_size :]
            logits, _ = self(context)
            next_token = torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
