"""
Minimal binary tensor-container format shared between the Python export
pipeline and the C++ accelerator model.

Deliberately not JSON/pickle/npz: the goal is a format simple enough that
the C++ *reader* (weight_loader in the accelerator model) is a plain
ifstream + struct-of-raw-bytes parser, no third-party library on either
side. This is the same kind of artifact a DV engineer already deals with
when loading a memory image (.mem/.hex) into a simulated SRAM -- a flat,
documented binary layout instead of a "black box" library format.

Layout (little-endian throughout):

    magic:        4 bytes, ASCII "CACM"
    version:      uint32
    num_tensors:  uint32
    for each tensor, in order:
        name_len: uint32
        name:     name_len bytes, UTF-8 (not NUL-terminated)
        dtype:    uint8   (0 = int8, 1 = int32, 2 = float32)
        rank:     uint32
        dims:     rank * uint64
        data:     prod(dims) * itemsize(dtype) bytes, raw values

A tensor is looked up by name on the C++ side, e.g. "conv1_weight",
"fc_bias", "requant_scale_conv1" (scalars are just rank-1, size-1 tensors).
"""

import struct
import numpy as np

MAGIC = b"CACM"
VERSION = 1

_DTYPE_TO_TAG = {np.dtype("int8"): 0, np.dtype("int32"): 1, np.dtype("float32"): 2}
_TAG_TO_DTYPE = {v: k for k, v in _DTYPE_TO_TAG.items()}


def save_tensor_container(path, tensors: dict):
    """tensors: dict[str, np.ndarray] with dtype in {int8, int32, float32}."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr)
            if arr.dtype not in _DTYPE_TO_TAG:
                raise ValueError(f"tensor '{name}': unsupported dtype {arr.dtype}")
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", _DTYPE_TO_TAG[arr.dtype]))
            f.write(struct.pack("<I", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<Q", d))
            f.write(arr.tobytes())


def load_tensor_container(path) -> dict:
    tensors = {}
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
        (version,) = struct.unpack("<I", f.read(4))
        if version != VERSION:
            raise ValueError(f"{path}: unsupported version {version}")
        (count,) = struct.unpack("<I", f.read(4))
        for _ in range(count):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")
            (tag,) = struct.unpack("<B", f.read(1))
            (rank,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{rank}Q", f.read(8 * rank)) if rank else ()
            dtype = _TAG_TO_DTYPE[tag]
            n = 1
            for d in dims:
                n *= d
            data = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype).reshape(dims)
            tensors[name] = data
    return tensors


if __name__ == "__main__":
    # Self-test: round-trip a small container through disk.
    import tempfile
    import os

    sample = {
        "conv1_weight": np.arange(24, dtype=np.int8).reshape(2, 1, 3, 4) - 12,
        "conv1_bias": np.array([10, -5], dtype=np.int32),
        "requant_scale_conv1": np.array([0.125], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "test.bin")
        save_tensor_container(path, sample)
        loaded = load_tensor_container(path)
        for k, v in sample.items():
            assert np.array_equal(loaded[k], v), f"round-trip mismatch on '{k}'"
            assert loaded[k].dtype == v.dtype, f"dtype mismatch on '{k}'"
        print(f"model_io self-test OK ({len(sample)} tensors round-tripped)")
