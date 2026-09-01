#!/usr/bin/env python3
"""
Pure-numpy INT8 quantized inference: the "Python reference implementation"
the project spec asks for, to compare against the C++ functional model.

Deliberately re-implements conv2d/relu/requantize/maxpool/fc from scratch in
numpy rather than calling back into PyTorch -- the point is an independent
implementation of the *same integer semantics* the C++ accelerator model
uses, so agreement between this file and the C++ output is a real check, the
same way a DV testbench's golden model is kept independently implemented
from the DUT rather than sharing code with it.

Only needs numpy + model_io -- no torch dependency, so it can run against an
exported model.bin on its own.
"""
import numpy as np

from model_io import load_tensor_container

MODEL_PATH = "models/mnist_int8_model.bin"


def conv2d_int8(x: np.ndarray, w: np.ndarray, bias: np.ndarray, stride: int = 1,
                 padding: int = 0) -> np.ndarray:
    """x: (C_in,H,W) int8, w: (C_out,C_in,KH,KW) int8, bias: (C_out,) int32.
    Returns (C_out,H_out,W_out) int32. Same zero-padding / accumulation
    semantics as accel::conv2d in ops.hpp -- vectorized here for speed
    (needed to evaluate accuracy over hundreds of images), not because the
    underlying math differs from the C++ model's explicit loops."""
    c_in, h, w_ = x.shape
    c_out, c_in_w, kh, kw = w.shape
    assert c_in_w == c_in, "conv2d_int8: weight C_in does not match input channels"

    x_pad = np.pad(x.astype(np.int64), ((0, 0), (padding, padding), (padding, padding)))
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w_ + 2 * padding - kw) // stride + 1

    # windows[c, oh, ow, i, j] = x_pad[c, oh*stride + i, ow*stride + j]
    windows = np.lib.stride_tricks.sliding_window_view(x_pad, (kh, kw), axis=(1, 2))
    windows = windows[:, ::stride, ::stride, :, :][:, :h_out, :w_out, :, :]

    out = np.einsum("chwij,ocij->ohw", windows, w.astype(np.int64))
    if bias is not None:
        out = out + bias.astype(np.int64).reshape(-1, 1, 1)
    return out.astype(np.int32)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def requantize(acc: np.ndarray, scale: float) -> np.ndarray:
    """INT32 -> INT8 via multiply-by-scale + round-half-away-from-zero +
    saturate. Round-half-away-from-zero (not numpy's default round-half-to-
    even) is used deliberately, to match C++'s std::lround exactly -- see
    accel::requantize's float-scale overload."""
    scaled = acc.astype(np.float64) * np.float64(scale)
    rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    return np.clip(rounded, -128, 127).astype(np.int8)


def maxpool2d_int8(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    c, h, w = x.shape
    h_out = (h - kernel) // stride + 1
    w_out = (w - kernel) // stride + 1
    windows = np.lib.stride_tricks.sliding_window_view(x, (kernel, kernel), axis=(1, 2))
    windows = windows[:, ::stride, ::stride, :, :][:, :h_out, :w_out, :, :]
    return windows.max(axis=(3, 4))


def fully_connected_int8(x: np.ndarray, w: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """x: (K,) int8, w: (N,K) int8, bias: (N,) int32 -> (N,) int32."""
    return (x.astype(np.int64) @ w.astype(np.int64).T + bias.astype(np.int64)).astype(np.int32)


def quantized_forward(tensors: dict, image_int8: np.ndarray) -> np.ndarray:
    """image_int8: (1,28,28) int8. Returns (10,) int32 logits."""
    acc = conv2d_int8(image_int8, tensors["conv1_weight"], tensors["conv1_bias"],
                       stride=1, padding=1)
    acc = relu(acc)
    activ = requantize(acc, float(tensors["requant_scale_conv1"][0]))
    pooled = maxpool2d_int8(activ, kernel=2, stride=2)
    flat = pooled.reshape(-1)  # (8,14,14) -> (1568,), same row-major order as Tensor<T>
    logits = fully_connected_int8(flat, tensors["fc_weight"], tensors["fc_bias"])
    return logits


def main():
    tensors = load_tensor_container(MODEL_PATH)
    images = tensors["test_images"]
    labels = tensors["test_labels"]

    correct = 0
    for i in range(images.shape[0]):
        logits = quantized_forward(tensors, images[i])
        pred = int(np.argmax(logits))
        true = int(labels[i])
        correct += int(pred == true)
        print(f"image {i:2d}: predicted={pred} true={true} "
              f"{'OK' if pred == true else 'WRONG'}  logits={logits.tolist()}")

    print(f"\n{correct}/{images.shape[0]} correct on exported test images")


if __name__ == "__main__":
    main()
