"""Backend detection and dtype resolution.

Every other module that needs to know "what hardware am I on" goes through
`resolve_device`. This is the single place that distinguishes ROCm from CUDA
(both report `torch.cuda.is_available() == True`; only `torch.version.hip`
tells them apart) so the distinction is made once, not re-derived per file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

Backend = Literal["rocm", "cuda", "mps", "cpu"]


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    backend: Backend
    dtype: torch.dtype
    total_memory_bytes: Optional[int]
    bf16_ok: bool


def _bf16_ok_for_cuda_like() -> bool:
    return hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()


def _detect_auto() -> DeviceInfo:
    if torch.cuda.is_available():
        backend: Backend = "rocm" if getattr(torch.version, "hip", None) is not None else "cuda"
        bf16_ok = _bf16_ok_for_cuda_like()
        dtype = torch.bfloat16 if bf16_ok else torch.float32
        total_mem = torch.cuda.get_device_properties(0).total_memory
        return DeviceInfo(
            device=torch.device("cuda:0"),
            backend=backend,
            dtype=dtype,
            total_memory_bytes=total_mem,
            bf16_ok=bf16_ok,
        )
    if torch.backends.mps.is_available():
        return DeviceInfo(
            device=torch.device("mps"),
            backend="mps",
            dtype=torch.float32,
            total_memory_bytes=None,
            bf16_ok=False,
        )
    return DeviceInfo(
        device=torch.device("cpu"),
        backend="cpu",
        dtype=torch.float32,
        total_memory_bytes=None,
        bf16_ok=False,
    )


def resolve_device(requested: str = "auto") -> DeviceInfo:
    """Resolve the backend to run on.

    Preconditions: `requested` is one of "auto", "cuda", "rocm", "cpu", "mps".
    Postconditions: "auto" always returns a valid DeviceInfo (falls back to
    cpu). An explicit backend that isn't actually available raises
    RuntimeError rather than silently downgrading to cpu.
    """
    if requested == "auto":
        return _detect_auto()

    if requested == "cpu":
        return DeviceInfo(
            device=torch.device("cpu"),
            backend="cpu",
            dtype=torch.float32,
            total_memory_bytes=None,
            bf16_ok=False,
        )

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested backend 'mps' but MPS is not available on this machine.")
        return DeviceInfo(
            device=torch.device("mps"),
            backend="mps",
            dtype=torch.float32,
            total_memory_bytes=None,
            bf16_ok=False,
        )

    if requested in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested backend '{requested}' but torch.cuda.is_available() is False on this machine."
            )
        actual: Backend = "rocm" if getattr(torch.version, "hip", None) is not None else "cuda"
        if actual != requested:
            raise RuntimeError(
                f"Requested backend '{requested}' but the available CUDA-like backend is actually '{actual}'."
            )
        bf16_ok = _bf16_ok_for_cuda_like()
        dtype = torch.bfloat16 if bf16_ok else torch.float32
        total_mem = torch.cuda.get_device_properties(0).total_memory
        return DeviceInfo(
            device=torch.device("cuda:0"),
            backend=actual,
            dtype=dtype,
            total_memory_bytes=total_mem,
            bf16_ok=bf16_ok,
        )

    raise RuntimeError(f"Unknown backend requested: {requested!r}")


def pick_batch_schedule(
    total_memory_bytes: Optional[int], base_micro_batch: int, base_grad_accum: int
) -> tuple[int, int]:
    """Pick (micro_batch, grad_accum) for the detected memory tier.

    Preconditions: base_micro_batch >= 1, base_grad_accum >= 1.
    Postconditions: micro_batch * grad_accum (effective batch size) is held
    constant across tiers relative to base_micro_batch * base_grad_accum.
    """
    if total_memory_bytes is None or total_memory_bytes < 6_000_000_000:
        return (1, base_grad_accum * base_micro_batch)
    if total_memory_bytes < 16_000_000_000:
        return (max(1, base_micro_batch // 4), base_grad_accum * 4)
    if total_memory_bytes < 28_000_000_000:
        return (base_micro_batch, base_grad_accum)
    return (base_micro_batch * 2, max(1, base_grad_accum // 2))
