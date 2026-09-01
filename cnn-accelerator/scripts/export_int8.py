#!/usr/bin/env python3
"""
Post-training INT8 quantization + export of the trained SmallCNN.

Quantization scheme: per-tensor symmetric.
    real_value ~= int8_value * scale
    scale = max(abs(calibration values)) / 127

This is chosen to match exactly what the C++ accelerator model (and the
numpy reference in quantized_reference.py) actually compute:

  - conv2d / fully_connected consume INT8 activations and INT8 weights,
    accumulating in INT32 -- see accel::conv2d / MacArray::computeTile in
    the C++ model. Integer accumulation means this step introduces no
    quantization error beyond the INT8 rounding already applied to the
    operands.
  - ReLU is applied directly to the INT32 accumulator. Valid because ReLU
    commutes with a positive scale: relu(x) * s == relu(x * s) for s > 0,
    so doing it before or after converting units doesn't change the result.
  - requantize(acc, scale) narrows INT32 -> INT8 between layers using ONE
    combined scale factor, computed here and baked into the exported model.
  - the final FC layer's output is left as raw INT32 "logits": the same
    scale multiplies every output element identically, so argmax over the
    INT32 accumulator equals argmax over the true (float) logits -- no
    need to requantize the very last layer just to classify.

Bias handling: a real bias is added in the *accumulator's* implied units
(scale_input * scale_weight for that layer), not in real units, because
that's the domain the INT32 accumulator itself is in before requantization:
    bias_int32 = round(bias_real / (scale_in * scale_weight))

Usage:
    python3 scripts/train.py            # first, produces models/mnist_fp32.pt
    python3 scripts/export_int8.py      # this script
Produces:
    models/mnist_int8_model.bin
"""
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from model_io import save_tensor_container
from train import SmallCNN, DATA_DIR

FP32_PATH = "models/mnist_fp32.pt"
OUT_PATH = "models/mnist_int8_model.bin"
NUM_CALIBRATION_IMAGES = 200
NUM_TEST_EXPORT_IMAGES = 20

# MNIST pixels (via ToTensor) are in [0, 1] and never negative, so this scale
# maps them to [0, 127] -- using only the positive half of INT8's range, but
# keeping the mapping trivially simple and exactly reproducible on the C++ side.
SCALE_INPUT = 1.0 / 127.0


def quantize_symmetric(tensor: torch.Tensor):
    """Per-tensor symmetric INT8 quantization of a weight tensor."""
    max_abs = tensor.abs().max().item()
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    q = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return q.numpy(), scale


def main():
    model = SmallCNN()
    model.load_state_dict(torch.load(FP32_PATH, map_location="cpu"))
    model.eval()

    conv1_w_q, scale_conv1_w = quantize_symmetric(model.conv1.weight.data)
    fc_w_q, scale_fc_w = quantize_symmetric(model.fc.weight.data)

    # Calibrate scale_conv1_out from the FLOAT model's own post-ReLU conv1
    # activations over a small calibration set -- standard post-training-
    # quantization practice: size the next layer's INT8 range from the full-
    # precision model's real activation statistics.
    transform = transforms.ToTensor()
    calib_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    calib_images = torch.stack([calib_set[i][0] for i in range(NUM_CALIBRATION_IMAGES)])

    captured = {}
    def capture_conv1(_module, _input, output):
        captured["conv1_relu"] = F.relu(output)
    handle = model.conv1.register_forward_hook(capture_conv1)
    with torch.no_grad():
        model(calib_images)
    handle.remove()

    max_abs_conv1_out = captured["conv1_relu"].abs().max().item()
    scale_conv1_out = max_abs_conv1_out / 127.0

    # int32 accumulator (units of SCALE_INPUT*scale_conv1_w) -> int8 (units
    # of scale_conv1_out): this is the single scalar accel::requantize needs.
    requant_scale_conv1 = (SCALE_INPUT * scale_conv1_w) / scale_conv1_out

    conv1_bias_q = torch.round(model.conv1.bias.data / (SCALE_INPUT * scale_conv1_w))
    conv1_bias_q = conv1_bias_q.to(torch.int32).numpy()

    # After conv1 -> relu -> requantize -> maxpool, activations are INT8 in
    # units of scale_conv1_out (max() commutes with a positive scale just
    # like relu() does, so pooling doesn't change the effective scale).
    fc_bias_q = torch.round(model.fc.bias.data / (scale_conv1_out * scale_fc_w))
    fc_bias_q = fc_bias_q.to(torch.int32).numpy()

    # A handful of quantized test images + labels, exported alongside the
    # weights so the C++ model and the Python reference run inference on
    # exactly the same inputs later.
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    test_images = torch.stack([test_set[i][0] for i in range(NUM_TEST_EXPORT_IMAGES)])
    test_labels = torch.tensor([test_set[i][1] for i in range(NUM_TEST_EXPORT_IMAGES)])
    test_images_q = torch.clamp(torch.round(test_images / SCALE_INPUT), -128, 127).to(torch.int8)

    tensors = {
        "conv1_weight": conv1_w_q,
        "conv1_bias": conv1_bias_q,
        "fc_weight": fc_w_q,
        "fc_bias": fc_bias_q,
        "requant_scale_conv1": np.array([requant_scale_conv1], dtype=np.float32),
        "test_images": test_images_q.numpy(),
        "test_labels": test_labels.to(torch.int32).numpy(),
    }
    save_tensor_container(OUT_PATH, tensors)

    print(f"scale_input          = {SCALE_INPUT:.6f}")
    print(f"scale_conv1_weight    = {scale_conv1_w:.6f}")
    print(f"scale_conv1_out       = {scale_conv1_out:.6f}")
    print(f"scale_fc_weight       = {scale_fc_w:.6f}")
    print(f"requant_scale_conv1   = {requant_scale_conv1:.6f}")
    print(f"exported {OUT_PATH}:")
    for name, arr in tensors.items():
        print(f"  {name:24s} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
