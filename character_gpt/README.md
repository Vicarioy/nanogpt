# Character GPT

A minimal character-level GPT built from scratch in PyTorch. The model learns to predict the next character from a small text corpus, then generates new text one character at a time.

## Project Files

- `train.py` - Transformer model, training loop, evaluation, fit diagnostics, loss-curve saving, and text generation.
- `corpus.txt` - Training and validation text.
- `mini_gpt.pt` - Checkpoint produced after training.

## Requirements

- Python 3.10 or newer
- PyTorch

Install PyTorch in the active environment if needed:

```powershell
python -m pip install torch
```

## Run Training

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe character_gpt\train.py
```

The script will:

1. Read `corpus.txt`.
2. Build a character-to-integer vocabulary.
3. Split the data into training and validation sets.
4. Train a small decoder-only Transformer.
5. Print training and validation loss with an overfitting/underfitting diagnostic.
6. Save the checkpoint and `training_loss.png` under `character_gpt/artifacts/`.
7. Generate and print two text samples.

For a short smoke run, use `--max-iters 5 --eval-interval 1 --eval-iters 2 --sample-tokens 20`. Training options also include `--output-dir` to choose where the checkpoint and loss plot are written.

## Model

The current model uses:

- Character-level tokenization
- A context window of 64 characters
- 64-dimensional token and position embeddings
- 2 Transformer blocks
- 4 masked self-attention heads per block
- AdamW optimization
- Cross-entropy language-modeling loss
- CPU or CUDA execution, selected automatically

The causal attention mask prevents each character from attending to future characters. This makes the model autoregressive: it can only use the current and previous context when predicting the next character.

## Configuration

Training and model settings are defined near the top of `train.py`:

```python
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
```

For better generations, increase the size and quality of `corpus.txt`, then increase `MAX_ITERS`. Larger values for `N_EMBED`, `N_LAYER`, and `BLOCK_SIZE` also increase model capacity, but require more memory and training time.

## Checkpoint Contents

`mini_gpt.pt` stores:

- The model state dictionary
- The character vocabulary
- The block size
- The embedding size
- The number of attention heads
- The number of Transformer layers

The checkpoint can be used later to restore the model without rebuilding the vocabulary manually.

## Limitations

This is an educational implementation rather than a production language model. The bundled corpus is very small, so generated text will be noisy and repetitive. The model has no subword tokenizer, no distributed training, no checkpoint-resume command, and no instruction-tuning data.

The project is intentionally compact so the complete training and generation process can be read in one file.
