"""Train a minimal character-level GPT from scratch."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

# Small defaults make the example practical on CPU.
BATCH_SIZE = 16
BLOCK_SIZE = 64
MAX_ITERS = 300
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-4
EVAL_ITERS = 10
N_EMBED = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
SEED = 1337


class Head(nn.Module):
    """One masked self-attention head."""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, time, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        weights = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5
        weights = weights.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights @ self.value(x)


class MultiHeadAttention(nn.Module):
    """Several attention heads working in parallel."""

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(head_size) for _ in range(num_heads))
        self.projection = nn.Linear(num_heads * head_size, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(output))


class FeedForward(nn.Module):
    """A simple projection block applied independently at each position."""

    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Block(nn.Module):
    """Transformer block with pre-layer normalization."""

    def __init__(self, n_embed: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embed)
        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norm_1(x))
        return x + self.feed_forward(self.layer_norm_2(x))


class MiniGPT(nn.Module):
    """A minimal decoder-only character-level language model."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*(Block(N_EMBED, N_HEAD) for _ in range(N_LAYER)))
        self.layer_norm = nn.LayerNorm(N_EMBED)
        self.language_head = nn.Linear(N_EMBED, vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, time = inputs.shape
        positions = torch.arange(time, device=inputs.device)
        x = self.token_embedding(inputs) + self.position_embedding(positions)
        logits = self.language_head(self.layer_norm(self.blocks(x)))

        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(
                logits.view(batch * time, channels), targets.view(batch * time)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            context = inputs[:, -BLOCK_SIZE:]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs


def get_batch(
    split: str,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else validation_data
    starts = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    inputs = torch.stack([data[start : start + BLOCK_SIZE] for start in starts])
    targets = torch.stack(
        [data[start + 1 : start + BLOCK_SIZE + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: MiniGPT,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    device: torch.device,
    eval_iters: int = EVAL_ITERS,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split in ("train", "validation"):
        values = torch.zeros(eval_iters)
        for index in range(eval_iters):
            inputs, targets = get_batch(split, train_data, validation_data, device)
            _, loss = model(inputs, targets)
            values[index] = loss.item()
        losses[split] = values.mean().item()
    model.train()
    return losses


def diagnose_fit(loss_history: list[dict[str, float]], vocab_size: int) -> str:
    if len(loss_history) < 3:
        return "Not enough evaluation points yet to diagnose fit."

    training_losses = [point["train"] for point in loss_history]
    validation_losses = [point["validation"] for point in loss_history]
    final_training_loss = training_losses[-1]
    final_validation_loss = validation_losses[-1]
    generalization_gap = final_validation_loss - final_training_loss
    if (
        generalization_gap > 0.2
        and final_validation_loss > min(validation_losses[:-1]) + 0.05
    ):
        return "Overfitting risk: validation loss is rising above training loss."

    random_baseline = math.log(vocab_size)
    training_improvement = training_losses[0] - final_training_loss
    if final_training_loss > 0.8 * random_baseline and training_improvement < 0.1:
        return "Underfitting risk: training loss remains near the random baseline."

    return "No strong overfitting or underfitting signal in the recorded losses."


def save_loss_curve(loss_history: list[dict[str, float]], path: Path) -> None:
    steps = [point["step"] for point in loss_history]
    plt.figure(figsize=(8, 5))
    plt.plot(steps, [point["train"] for point in loss_history], label="Train")
    plt.plot(
        steps,
        [point["validation"] for point in loss_history],
        label="Validation",
    )
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Character GPT training loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and sample the character GPT")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERS)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-iters", type=int, default=EVAL_ITERS)
    parser.add_argument("--sample-tokens", type=int, default=160)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.max_iters, args.eval_interval, args.eval_iters) < 1:
        parser.error("iteration and evaluation counts must all be positive")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_dir = Path(__file__).parent
    text = (project_dir / "corpus.txt").read_text(encoding="utf-8")
    characters = sorted(set(text))
    encode = lambda value: [characters.index(character) for character in value]
    decode = lambda values: "".join(characters[index] for index in values)
    data = torch.tensor(encode(text), dtype=torch.long)

    split_index = int(0.9 * len(data))
    train_data, validation_data = data[:split_index], data[split_index:]
    if min(len(train_data), len(validation_data)) <= BLOCK_SIZE:
        raise ValueError("corpus.txt must contain more than 10 * BLOCK_SIZE characters")

    model = MiniGPT(len(characters)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    output_dir = args.output_dir or project_dir / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} | parameters={sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    loss_history = []
    for iteration in range(args.max_iters + 1):
        if (
            iteration == 0
            or iteration % args.eval_interval == 0
            or iteration == args.max_iters
        ):
            losses = estimate_loss(
                model,
                train_data,
                validation_data,
                device,
                eval_iters=args.eval_iters,
            )
            loss_history.append({"step": iteration, **losses})
            print(
                f"step {iteration:04d}: train loss {losses['train']:.4f}, "
                f"validation loss {losses['validation']:.4f}"
            )
        if iteration == args.max_iters:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = get_batch("train", train_data, validation_data, device)
        _, loss = model(inputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    checkpoint_path = output_dir / "mini_gpt.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "characters": characters,
        "config": {
            "block_size": BLOCK_SIZE,
            "n_embed": N_EMBED,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    curve_path = output_dir / "training_loss.png"
    save_loss_curve(loss_history, curve_path)
    print(f"Fit diagnostic: {diagnose_fit(loss_history, len(characters))}")
    print(f"Loss curve: {curve_path}")
    print(f"Checkpoint: {checkpoint_path}")

    model.eval()
    prompt_starts = [0, len(text) // 2]
    for sample_index, start in enumerate(prompt_starts, start=1):
        prompt_text = text[start : start + min(24, BLOCK_SIZE)]
        prompt = torch.tensor([encode(prompt_text)], dtype=torch.long, device=device)
        sample = model.generate(prompt, max_new_tokens=args.sample_tokens)[0].tolist()
        print(f"\n--- generated sample {sample_index} ---")
        print(decode(sample))


if __name__ == "__main__":
    main()
