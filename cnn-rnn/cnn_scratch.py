import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.dataset[self.indices[index]]
        return self.transform(image), label


class CIFAR10ConvBlock(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, images):
        images = self.pool1(self.block2(self.block1(images)))
        images = self.pool2(self.block3(images))
        images = self.global_pool(images)
        images = torch.flatten(images, start_dim=1)
        return self.fc(self.fc_dropout(images))


def evaluate(model, data_loader, criterion, device):
    model.eval()
    loss_total = 0.0
    correct_total = 0
    sample_total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item() * images.size(0)
            correct_total += (outputs.argmax(dim=1) == labels).sum().item()
            sample_total += labels.size(0)

    return loss_total / sample_total, correct_total / sample_total


def diagnose_fit(train_losses, validation_losses):
    if len(train_losses) < 3:
        return "Collect at least three epochs before diagnosing fit."

    train_loss = train_losses[-1]
    validation_loss = validation_losses[-1]
    validation_worsened = validation_loss > min(validation_losses[:-1]) + 0.15
    if validation_loss - train_loss > 0.2 and validation_worsened:
        return "Overfitting risk: training improves while validation loss worsens."

    random_baseline = math.log(10)
    training_improvement = train_losses[0] - train_loss
    if train_loss > 0.75 * random_baseline and training_improvement < 0.1:
        return "Underfitting risk: training loss remains near the random baseline."

    return "No strong overfitting or underfitting signal in the recorded losses."


def main():
    device = torch.device("cpu")
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )

    raw_trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=None
    )
    train_indices, val_indices = random_split(
        range(len(raw_trainset)),
        [45_000, 5_000],
        generator=torch.Generator().manual_seed(42),
    )
    trainset = TransformedSubset(raw_trainset, train_indices.indices, train_transform)
    valset = TransformedSubset(raw_trainset, val_indices.indices, eval_transform)
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=eval_transform
    )

    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    model = CIFAR10ConvBlock().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    train_losses = []
    validation_losses = []

    for epoch in range(epochs):
        model.train()
        loss_total = 0.0
        correct_total = 0
        sample_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_total += loss.item() * images.size(0)
            correct_total += (outputs.argmax(dim=1) == labels).sum().item()
            sample_total += labels.size(0)

        train_loss = loss_total / sample_total
        train_accuracy = correct_total / sample_total
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        validation_losses.append(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy * 100:.2f}%"
        )
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_accuracy * 100:.2f}%")

    print(diagnose_fit(train_losses, validation_losses))
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print("================ FINAL TEST RESULTS ================")
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    main()
