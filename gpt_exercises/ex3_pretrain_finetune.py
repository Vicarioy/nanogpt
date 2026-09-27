"""EX3: configurable pretraining followed by fine-tuning on a smaller corpus.

Usage expects two text files. The defaults make the experiment runnable with the
provided sample files, while users can replace them with a large dataset.
"""

from pathlib import Path

import torch
from torch.nn import functional as F

from common import TinyGPT

ROOT = Path(__file__).parent
BLOCK_SIZE = 64
STEPS = 300


def load_text(
    path: Path, mapping: dict[str, int] | None = None
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    if mapping is None:
        characters = sorted(set(text))
        mapping = {character: index for index, character in enumerate(characters)}
    else:
        characters = list(mapping)
    return (
        torch.tensor([mapping[character] for character in text], dtype=torch.long),
        characters,
        mapping,
    )


def train(
    model: TinyGPT, data: torch.Tensor, steps: int, label: str, device: torch.device
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4 if label == "pretrain" else 1e-4
    )
    model.train()
    for step in range(steps):
        starts = torch.randint(len(data) - BLOCK_SIZE - 1, (32,))
        inputs = torch.stack([data[start : start + BLOCK_SIZE] for start in starts]).to(
            device
        )
        targets = torch.stack(
            [data[start + 1 : start + BLOCK_SIZE + 1] for start in starts]
        ).to(device)
        _, loss = model(inputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"{label} step {step:04d} loss {loss.item():.4f}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrain_text = (ROOT / "pretrain.txt").read_text(encoding="utf-8")
    finetune_text = (ROOT / "finetune.txt").read_text(encoding="utf-8")
    characters = sorted(set(pretrain_text) | set(finetune_text))
    mapping = {character: index for index, character in enumerate(characters)}
    pretrain_data, _, _ = load_text(ROOT / "pretrain.txt", mapping)
    finetune_data, _, _ = load_text(ROOT / "finetune.txt", mapping)

    model = TinyGPT(len(characters), block_size=BLOCK_SIZE).to(device)
    train(model, pretrain_data, STEPS, "pretrain", device)
    torch.save(model.state_dict(), ROOT / "pretrained.pt")
    train(model, finetune_data, STEPS // 3, "finetune", device)
    torch.save(model.state_dict(), ROOT / "finetuned.pt")
    print("Saved pretrained.pt and finetuned.pt")


if __name__ == "__main__":
    main()
