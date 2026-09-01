#!/usr/bin/env python3
"""
Reports two real, measured numbers: float32 test accuracy vs INT8 quantized
test accuracy, over the full MNIST test set. This is the accuracy-impact
evidence for docs/ -- every number printed here comes from an actual run,
not an assumption about how much quantization "should" cost.

Usage (after train.py and export_int8.py have been run):
    python3 scripts/eval_quantized_accuracy.py
"""
import numpy as np
import torch
from torchvision import datasets, transforms

from model_io import load_tensor_container
from quantized_reference import quantized_forward
from train import SmallCNN, DATA_DIR, evaluate

FP32_PATH = "models/mnist_fp32.pt"
MODEL_PATH = "models/mnist_int8_model.bin"
SCALE_INPUT = 1.0 / 127.0


def main():
    transform = transforms.ToTensor()
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    model = SmallCNN()
    model.load_state_dict(torch.load(FP32_PATH, map_location="cpu"))
    model.eval()
    from torch.utils.data import DataLoader
    float_acc = evaluate(model, DataLoader(test_set, batch_size=256), torch.device("cpu"))

    tensors = load_tensor_container(MODEL_PATH)
    correct = 0
    n = len(test_set)
    for i in range(n):
        image, label = test_set[i]
        image_q = np.clip(np.round(image.numpy() / SCALE_INPUT), -128, 127).astype(np.int8)
        logits = quantized_forward(tensors, image_q)
        pred = int(np.argmax(logits))
        correct += int(pred == label)
    quant_acc = correct / n

    print(f"float32 test accuracy:  {float_acc:.4f}  ({n} images)")
    print(f"INT8    test accuracy:  {quant_acc:.4f}  ({n} images)")
    print(f"accuracy delta:         {float_acc - quant_acc:+.4f}")


if __name__ == "__main__":
    main()
