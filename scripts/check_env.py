"""Diagnostic script: prints the detected backend/device/memory/bf16 status.

Always exits 0 — this is a diagnostic, not a gate. Hard-fail-on-missing-GPU
behavior belongs to train.py when the config explicitly requests a
non-"auto" backend.
"""

from smrt.device import resolve_device

if __name__ == "__main__":
    info = resolve_device("auto")
    print(f"backend: {info.backend}")
    print(f"device: {info.device}")
    print(f"dtype: {info.dtype}")
    print(f"bf16_ok: {info.bf16_ok}")
    print(f"total_memory_bytes: {info.total_memory_bytes}")
    if info.backend in ("cuda", "rocm"):
        import torch

        print(f"device name: {torch.cuda.get_device_name(0)}")
