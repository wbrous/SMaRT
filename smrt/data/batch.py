"""Batch-shape/needle-span invariant checks, called at the top of every
batch-producing generator in needle.py/pretrain.py.
"""

from __future__ import annotations

import torch

from smrt.config import TrainConfig


def validate_batch(
    input_ids: torch.Tensor, cfg: TrainConfig, needle_spans: list | None = None
) -> None:
    """Preconditions checked here (not asserted by callers individually):

    - input_ids is int64 (torch.long) — required by nn.Embedding.
    - input_ids' last dimension equals cfg.seq_len — every example is a
      fixed-length training chunk, never padded/truncated silently.
    - If needle_spans is given, every (start, end) span lies entirely
      within [0, cfg.seq_len) — a needle must never be truncated away by a
      later step that cuts the sequence down to seq_len.
    """
    assert input_ids.dtype == torch.long, f"expected torch.long, got {input_ids.dtype}"
    assert input_ids.shape[-1] == cfg.seq_len, (
        f"expected last dim {cfg.seq_len}, got {input_ids.shape[-1]}"
    )
    if needle_spans is not None:
        for start, end in needle_spans:
            assert end <= cfg.seq_len, f"needle span end {end} exceeds seq_len {cfg.seq_len}"
