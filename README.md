# NanoGPT Learning Project

This repository contains a character-level GPT, a small intent chatbot, and scratch CNN, RNN, and attention examples.

## Run the chatbot

The chatbot validates each intent against a JSON Schema before dispatch. Its supported tool call adds two integers:

```powershell
.\.venv\Scripts\python.exe .\src\nanogpt\chatbot.py --message "add 12 and 30"
```

Omit `--message` to start the interactive prompt. The installed `nanogpt-chat` command provides the same entry point.

## Train and sample the GPT

```powershell
.\.venv\Scripts\python.exe .\nanogpt_mine.py
```

The script uses a zero-grad, forward, backward, gradient-clip, optimizer-step loop; periodically evaluates train and validation cross-entropy; reports overfitting/underfitting diagnostics; saves a loss plot and checkpoint under `character_gpt/artifacts/`; and prints two generated samples. Use `--max-iters 5 --eval-interval 1 --eval-iters 2 --sample-tokens 20` for a short smoke run.

## Scratch examples

- `cnn-rnn/cnn_scratch.py` trains a PyTorch CIFAR-10 CNN and reports fit diagnostics.
- `cnn-rnn/rnn_scratch.py` demonstrates manual BPTT and checks its gradients against autograd.
- `cnn-rnn/attention_scratch.py` implements scaled dot-product attention with a toy test.

The CNN uses PyTorch convolution primitives; the RNN recurrence/BPTT and attention calculation are the educational scratch implementations.
