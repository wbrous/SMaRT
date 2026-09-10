"""Real pretraining data: HuggingFace streaming loader (used when
vocab_size != 256), plus a synthetic-random-bytes substitute used only by
tiny_cpu.yaml's offline CPU smoke path (vocab_size == 256).
"""

from __future__ import annotations

import random
from typing import Iterator

import torch


def load_pretrain_stream(
    seq_len: int,
    tokenizer,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    split: str = "train",
    token_budget: int = 2_000_000_000,
) -> Iterator[torch.Tensor]:
    """Stream-tokenize a HuggingFace dataset into fixed-seq_len chunks.

    Token budget: 2B tokens (see DESIGN.md for the Chinchilla-headroom
    rationale). Stops once token_budget tokens have been yielded.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split, streaming=True)
    buffer: list[int] = []
    yielded = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        buffer.extend(tokenizer.encode(text))
        while len(buffer) >= seq_len:
            chunk, buffer = buffer[:seq_len], buffer[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)
            yielded += seq_len
            if yielded >= token_budget:
                return


def load_nemotron_cc_stream(
    seq_len: int,
    tokenizer,
    dataset_name: str = "nvidia/Nemotron-CC-v2",
    config_name: str = "High-Quality",
    token_budget: float = float("inf"),
) -> Iterator[torch.Tensor]:
    """Stream-tokenize nvidia/Nemotron-CC-v2's High-Quality config into
    fixed-seq_len chunks. Requires an HF_TOKEN belonging to an account
    that has accepted this dataset's gated NVIDIA Data Agreement (see
    https://huggingface.co/datasets/nvidia/Nemotron-CC-v2) -- unlike
    codeparrot/github-code-clean, this dataset ships proper HF `configs:`
    metadata, so no dataset-script workaround is needed; plain
    `load_dataset(dataset_name, config_name, ...)` works directly.

    token_budget defaults to infinite: this stream is constructed once
    and persists for the entire training run (see smrt/train.py's
    `streams` dict), so the *outer* training loop's max_steps is what
    bounds duration, not this generator's own budget.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config_name, split="train", streaming=True)
    buffer: list[int] = []
    yielded = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        buffer.extend(tokenizer.encode(text))
        while len(buffer) >= seq_len:
            chunk, buffer = buffer[:seq_len], buffer[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)
            yielded += seq_len
            if yielded >= token_budget:
                return


def synthetic_random_stream(
    rng: random.Random, batch_size: int, seq_len: int, vocab_size: int
) -> torch.Tensor:
    """Uniform-random valid token ids, shape (batch_size, seq_len).

    CPU-smoke substitute for load_pretrain_stream — same (seq_len,)-chunk
    shape contract but with no HF dataset dependency, so tiny_cpu.yaml can
    run fully offline.
    """
    rows = [[rng.randrange(vocab_size) for _ in range(seq_len)] for _ in range(batch_size)]
    return torch.tensor(rows, dtype=torch.long)
