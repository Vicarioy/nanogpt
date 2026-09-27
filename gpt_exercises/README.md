# GPT Exercises

This folder contains separate implementations of the four suggested GPT exercises. It is independent from `character_gpt`; no files in that folder are imported or modified.

## Run the examples

Run these commands from the repository root:

```powershell
.\.venv\Scripts\python.exe gpt_exercises\ex1_combined_attention.py
.\.venv\Scripts\python.exe gpt_exercises\ex2_addition.py
.\.venv\Scripts\python.exe gpt_exercises\ex3_pretrain_finetune.py
.\.venv\Scripts\python.exe gpt_exercises\ex4_rope.py
```

The addition and pretraining examples perform actual optimization and can take longer on CPU. Their step counts are constants near the top of each file, so they can be reduced for a smoke test.

## EX1: Combine attention heads into one operation

`ex1_combined_attention.py` implements `CombinedMultiHeadAttention` without creating one Python `Head` object per attention head.

Instead of processing each head separately, it projects all queries, keys, and values into one tensor and reshapes it to:

```text
(batch, heads, time, head_size)
```

Attention scores then have shape:

```text
(batch, heads, time, time)
```

This lets PyTorch process all heads in parallel using batched matrix multiplication. The causal mask still prevents a token from attending to future positions.

## EX2: Teach GPT addition

`ex2_addition.py` generates random problems such as:

```text
042+137=971\n
```

The answer is reversed in the training representation:

```text
042+137=179\n
```

The actual generated target is the reversed answer, so the model learns the least-significant digit first. This matches the direction used by ordinary carrying-based addition.

The model receives the problem prefix and predicts the answer character by character. It is trained on fresh random problems instead of a fixed dataset, which gives it a virtually unlimited stream of examples.

A production-quality version would mask the loss for the problem prefix and calculate loss only on answer characters. This example keeps the full next-character loss visible so the training path remains easy to inspect.

## EX3: Pretraining and fine-tuning

`ex3_pretrain_finetune.py` trains one model in two stages:

1. Pretraining on `pretrain.txt` with a relatively higher learning rate.
2. Fine-tuning on `finetune.txt` with a lower learning rate.

The script creates a shared character vocabulary from both files, trains the model on the broad corpus, saves `pretrained.pt`, then adapts the same weights to the focused corpus and saves `finetuned.pt`.

For a meaningful experiment, replace `pretrain.txt` with a much larger corpus and keep `finetune.txt` focused. Compare validation loss and generated samples against a model trained only on the smaller corpus. Pretraining should help when the broad corpus teaches patterns that transfer to the focused task.

The supplied files are intentionally tiny and demonstrate the pipeline only; they are not large enough to establish a strong pretraining result.

## EX4: Rotary position embeddings

`ex4_rope.py` implements rotary positional embeddings, commonly called RoPE.

The learned absolute position embedding used by the original character model is removed from the attention input. Instead, the query and key vectors are rotated by position-dependent angles before attention scores are calculated.

For every pair of feature dimensions, RoPE applies:

```text
q_even' = q_even * cos(theta) - q_odd * sin(theta)
q_odd'  = q_even * sin(theta) + q_odd * cos(theta)
```

The same operation is applied to keys. Because both queries and keys receive position-dependent rotations, their dot product contains relative-position information. This can generalize more naturally to sequence positions than learned absolute embeddings.

The example is a focused attention layer rather than a complete replacement model. To run a full comparison, place it inside the Transformer blocks in `common.py`, train one model with learned positions and one with RoPE, then compare validation loss and generation quality.

## Shared utility

`common.py` contains a compact decoder-only Transformer used by the addition and pretraining exercises:

- Character token embeddings
- Learned absolute position embeddings
- Causal self-attention
- Pre-layer normalization
- GELU feed-forward networks
- Residual connections
- Autoregressive generation

## Important limitation

These examples are educational experiments. The tiny bundled datasets, short training schedules, and small models are designed for readability and laptop execution, not for competitive language-model quality.
