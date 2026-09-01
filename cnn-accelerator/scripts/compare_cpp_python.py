#!/usr/bin/env python3
"""
The actual "Python result vs C++ functional model" comparison the project
spec asks for: reruns the independent numpy reference (quantized_reference.py)
on the exported test images, reads results/cpp_predictions.csv (written by
the C++ mnist_infer binary), and diffs them logit-by-logit.

Two independently-written implementations of the same integer pipeline are
expected to match exactly here, not just closely -- there is no floating
point in the shared datapath (INT8 x INT8 -> INT32 accumulate; the one
scale multiply in requantize() uses matching round-half-away-from-zero on
both sides), so any mismatch is a real bug in one implementation or the
other, not rounding noise to shrug off.

Usage (after models/mnist_int8_model.bin exists and build/mnist_infer has
been run from the cnn-accelerator/ directory):
    python3 scripts/compare_cpp_python.py
"""
import csv
import sys

import numpy as np

from model_io import load_tensor_container
from quantized_reference import quantized_forward

MODEL_PATH = "models/mnist_int8_model.bin"
CPP_CSV_PATH = "results/cpp_predictions.csv"


def main():
    tensors = load_tensor_container(MODEL_PATH)
    images = tensors["test_images"]
    labels = tensors["test_labels"]

    with open(CPP_CSV_PATH, newline="") as f:
        cpp_rows = {int(row["image"]): row for row in csv.DictReader(f)}

    if len(cpp_rows) != images.shape[0]:
        print(f"WARNING: {CPP_CSV_PATH} has {len(cpp_rows)} rows, expected {images.shape[0]}")

    all_match = True
    max_abs_diff = 0
    for i in range(images.shape[0]):
        py_logits = quantized_forward(tensors, images[i])
        py_pred = int(np.argmax(py_logits))

        row = cpp_rows[i]
        cpp_logits = np.array([int(row[f"logit{j}"]) for j in range(10)], dtype=np.int64)
        cpp_pred = int(row["predicted"])

        diff = np.abs(py_logits.astype(np.int64) - cpp_logits)
        max_abs_diff = max(max_abs_diff, int(diff.max()))
        exact = bool(np.all(diff == 0))
        pred_match = py_pred == cpp_pred
        all_match = all_match and exact and pred_match

        status = "OK" if exact and pred_match else "MISMATCH"
        print(f"image {i:2d}: python_pred={py_pred} cpp_pred={cpp_pred} "
              f"true={int(labels[i])} max|diff|={int(diff.max())}  {status}")

    print()
    print(f"max |python_logit - cpp_logit| over all images/classes: {max_abs_diff}")
    if all_match:
        print("Python reference and C++ model agree EXACTLY on every image and every logit.")
        sys.exit(0)
    else:
        print("MISMATCH: Python reference and C++ model disagree -- see rows above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
