"""General instruction-following SFT data: streams HuggingFaceH4/no_robots
(10K human-annotated instruction examples) and formats each row's
`messages` list into a (token_ids, loss_mask) pair using the same
<|system|>/<|user|>/<|assistant|> special tokens as smrt.data.toolcalls.

Source schema (verified against the HF dataset card): each row has a
`messages` field: a list of {"role": "system"|"user"|"assistant",
"content": str} dicts, in turn order. Not every row has a leading system
message.
"""

from __future__ import annotations

from typing import Iterator

import torch

_ROLE_TOKEN = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


def format_messages(messages: list[dict], tokenizer) -> tuple[list[int], list[int]] | None:
    """Builds (token_ids, loss_mask) for one no_robots row. Returns None
    if `messages` is empty or contains a role outside {system, user,
    assistant} (caller must drop the row rather than silently
    mis-tokenize it as unwrapped plain text).
    """
    segments: list[str] = []
    masks: list[int] = []
    for msg in messages:
        role = msg.get("role")
        if role not in _ROLE_TOKEN:
            return None
        segments.append(f"{_ROLE_TOKEN[role]}{msg.get('content', '')}<|end|>")
        masks.append(1 if role == "assistant" else 0)
    if not segments:
        return None

    token_ids: list[int] = []
    loss_mask: list[int] = []
    for segment, mask_value in zip(segments, masks):
        ids = tokenizer.encode(segment)
        token_ids.extend(ids)
        loss_mask.extend([mask_value] * len(ids))
    assert len(token_ids) == len(loss_mask)
    return token_ids, loss_mask


def load_general_instruction_stream(
    seq_len: int, tokenizer, dataset_name: str = "HuggingFaceH4/no_robots"
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Streams no_robots, yielding (ids, mask) pairs of shape (seq_len,)
    each. Rows with no usable messages are skipped. Pad token is
    <|end|>'s id (pad region's mask is always 0); trajectories longer
    than seq_len are truncated from the end, and skipped entirely if
    that truncation would remove the whole assistant span (see
    smrt.data.sft_common), matching
    smrt.data.toolcalls.load_toolcall_sft_stream's pad/truncate contract.
    """
    from datasets import load_dataset

    from smrt.data.sft_common import pad_or_truncate

    end_id = tokenizer.encode("<|end|>")[0]
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for example in ds:
        trajectory = format_messages(example.get("messages", []), tokenizer)
        if trajectory is None:
            continue
        ids, mask = trajectory
        fitted = pad_or_truncate(ids, mask, seq_len, end_id)
        if fitted is None:
            continue
        ids, mask = fitted
        yield torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
