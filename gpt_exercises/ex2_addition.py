"""EX2: train a character GPT to add two numbers."""

import random

import torch
from torch.nn import functional as F

from common import TinyGPT

DIGITS = 3
BLOCK_SIZE = 4 * DIGITS + 1
VOCAB = "0123456789+=\n"
TO_ID = {character: index for index, character in enumerate(VOCAB)}
FROM_ID = {index: character for character, index in TO_ID.items()}


def make_problem() -> str:
    left = random.randint(0, 10**DIGITS - 1)
    right = random.randint(0, 10**DIGITS - 1)
    # Reversed output lets the causal model learn least-significant digits first.
    return (
        f"{left:0{DIGITS}d}+{right:0{DIGITS}d}="
        + f"{left + right:0{DIGITS + 1}d}"[::-1]
        + "\n"
    )


def encode(text: str) -> list[int]:
    return [TO_ID[character] for character in text]


def decode(tokens: list[int]) -> str:
    return "".join(FROM_ID[token] for token in tokens)


def batch(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [encode(make_problem()) for _ in range(batch_size)]
    values = torch.tensor(rows, dtype=torch.long, device=device)
    return values[:, :-1], values[:, 1:]


def main() -> None:
    torch.manual_seed(1337)
    random.seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyGPT(len(VOCAB), BLOCK_SIZE, n_embed=96, n_head=4, n_layer=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(2000):
        inputs, targets = batch(64, device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            print(f"step {step:04d} loss {loss.item():.4f}")

    model.eval()
    for _ in range(5):
        problem = make_problem().split("=")[0] + "="
        prompt = torch.tensor([encode(problem)], dtype=torch.long, device=device)
        result = decode(model.generate(prompt, DIGITS + 1)[0].tolist())
        print(f"{problem}{result[len(problem) :]}")


if __name__ == "__main__":
    main()
