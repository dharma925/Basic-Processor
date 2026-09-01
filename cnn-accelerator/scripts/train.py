#!/usr/bin/env python3
"""
Train the small CNN this whole project models in hardware: Conv -> ReLU ->
MaxPool -> Flatten -> FC, on MNIST. Ordinary float32 PyTorch training --
nothing accelerator-specific happens here. Quantization (the step that
actually connects to the C++ INT8 model) is a separate, later step in
export_int8.py, deliberately kept apart from training so each script has one
job: this one produces the best float model it can, export_int8.py turns
that into something the accelerator model can run.

Usage:
    python3 scripts/train.py
Produces:
    models/mnist_fp32.pt   (float32 state_dict; not shipped to C++, input to
                             export_int8.py)
"""
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_DIR = "models/mnist_raw"
OUT_PATH = "models/mnist_fp32.pt"
EPOCHS = 3
BATCH_SIZE = 128
LR = 1e-3
SEED = 0


class SmallCNN(nn.Module):
    """Conv -> ReLU -> MaxPool -> FC: one of each op the C++ model implements
    (accel::conv2d, accel::relu_inplace, accel::maxpool2d, accel::fully_connected),
    so every layer here has a direct counterpart on the C++ side."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 28x28 -> 28x28, 8 channels
        self.fc = nn.Linear(8 * 14 * 14, 10)  # after 2x2 maxpool: 28x28 -> 14x14

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    # ToTensor() scales pixels to [0, 1] float -- matches the [0,1] input
    # range export_int8.py assumes when it picks the input quantization scale.
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_set)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}/{EPOCHS}  loss={train_loss:.4f}  test_acc={test_acc:.4f}  "
              f"({time.time() - start:.1f}s)")

    final_acc = evaluate(model, test_loader, device)
    print(f"final float32 test accuracy: {final_acc:.4f}")

    torch.save(model.state_dict(), OUT_PATH)
    print(f"saved float32 weights to {OUT_PATH}")


if __name__ == "__main__":
    main()
