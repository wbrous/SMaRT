"""Shared pad/truncate contract for every SFT data pipeline in this repo
(smrt.data.toolcalls, smrt.data.web_search_bias, smrt.data.code_repair,
smrt.data.general_instruction): each builds a (token_ids, loss_mask) pair
for one trajectory, then must fit it to a fixed seq_len before yielding.

Truncating a too-long trajectory from the end can remove its entire
assistant span (loss_mask all 1s), leaving cross_entropy's ignore_index
mask fully set for that row -- an empty reduction (0/0), which surfaces
as `loss: NaN` in training diagnostics. `pad_or_truncate` detects this
case and signals the caller to skip the row instead of yielding a
trajectory with no learnable target.
"""

from __future__ import annotations


def pad_or_truncate(ids: list[int], mask: list[int], seq_len: int, end_id: int) -> tuple[list[int], list[int]] | None:
    """Fits (ids, mask) to exactly seq_len entries.

    Shorter-than-seq_len trajectories are padded with `end_id` (mask 0).
    Longer trajectories are truncated from the end; if that truncation
    would remove every assistant (mask == 1) position, returns None so
    the caller skips this row rather than yielding a target-less example.
    """
    if len(ids) < seq_len:
        pad_len = seq_len - len(ids)
        return ids + [end_id] * pad_len, mask + [0] * pad_len
    truncated_mask = mask[:seq_len]
    if not any(m == 1 for m in truncated_mask):
        return None
    return ids[:seq_len], truncated_mask
