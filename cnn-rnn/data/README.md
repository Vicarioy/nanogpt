# CNN Data Folder

This folder is reserved for CNN-related data. No dataset files are currently stored here.

At present, `cnn_scratch.py` asks TorchVision to download CIFAR-10 into `./data`, relative to the directory from which the script is run. When launched from the repository root as documented in `cnn-rnn/README.md`, that location is the repository-level `data/` folder, not this one.