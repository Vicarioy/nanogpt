# CNN, RNN, and Attention Scratch Examples

This folder contains small educational implementations for image classification, recurrent sequence modeling, and scaled dot-product attention.

## Examples

- `cnn_scratch.py` defines and trains a CIFAR-10 convolutional classifier. It uses PyTorch's `Conv2d`, batch normalization, pooling, and Adam; it reports train/validation metrics and a simple overfitting/underfitting diagnosis.
- `rnn_scratch.py` implements a tanh RNN and manual backpropagation through time (BPTT), including global gradient clipping and an autograd gradient comparison.
- `attention_scratch.py` computes scaled dot-product attention using nested Python lists and includes a toy-tensor check.

Run the lightweight examples from the repository root:

```powershell
.\.venv\Scripts\python.exe .\cnn-rnn\rnn_scratch.py
.\.venv\Scripts\python.exe .\cnn-rnn\attention_scratch.py
```

Run the CNN training example with:

```powershell
.\.venv\Scripts\python.exe .\cnn-rnn\cnn_scratch.py
```

The CNN command downloads CIFAR-10 if it is not already cached and trains for 20 epochs on CPU. Its dataset path is `./data`, relative to the current working directory. The `cnn-rnn/data` subfolder is currently an unused placeholder.