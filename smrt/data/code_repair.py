"""Synthetic Python syntax-repair SFT data (Vercel AutoFix-inspired):
streams real Python source from codeparrot/github-code-clean (same
dataset_name/config_name as smrt.data.code.load_code_stream), injects a
single syntax-breaking corruption verified via ast.parse to actually raise
SyntaxError, and trains the model to recover the original valid source
from the corrupted one.
"""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Iterator

import torch

_COLON_HEADER_RE = re.compile(
    r"^(\s*(?:def|if|for|while|class|elif|else|try|except|finally)\b[^\n]*?):(\s*)$",
    re.MULTILINE,
)
_CLOSING_BRACKETS = ")]}"


def _strip_trailing_colon(source: str, rng: random.Random) -> str | None:
    matches = list(_COLON_HEADER_RE.finditer(source))
    if not matches:
        return None
    m = rng.choice(matches)
    return source[: m.end(1)] + m.group(2) + source[m.end():]


def _drop_closing_bracket(source: str, rng: random.Random) -> str | None:
    positions = [i for i, ch in enumerate(source) if ch in _CLOSING_BRACKETS]
    if not positions:
        return None
    i = rng.choice(positions)
    return source[:i] + source[i + 1:]


_CORRUPTIONS = (_strip_trailing_colon, _drop_closing_bracket)


def corrupt_source(source: str, rng: random.Random, max_attempts: int = 10) -> str | None:
    """Returns a syntactically INVALID variant of `source` (verified via
    ast.parse raising SyntaxError), or None if no corruption attempt over
    max_attempts produced an invalid result (e.g. `source` has neither a
    colon-header line nor any closing bracket).
    """
    for _ in range(max_attempts):
        fn = rng.choice(_CORRUPTIONS)
        candidate = fn(source, rng)
        if candidate is None or candidate == source:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            return candidate
    return None


_INSTRUCTION = "Fix the syntax error in this Python code:\n\n"


def _build_trajectory(source: str, tokenizer, rng: random.Random) -> tuple[list[int], list[int]] | None:
    broken = corrupt_source(source, rng)
    if broken is None:
        return None
    segments = [
        (f"<|user|>{_INSTRUCTION}{broken}<|end|>", 0),
        (f"<|assistant|>{source}<|end|>", 1),
    ]
    token_ids: list[int] = []
    loss_mask: list[int] = []
    for segment, mask_value in segments:
        ids = tokenizer.encode(segment)
        token_ids.extend(ids)
        loss_mask.extend([mask_value] * len(ids))
    return token_ids, loss_mask

def load_code_repair_stream(
    seq_len: int,
    tokenizer,
    seed: int = 0,
    dataset_name: str = "codeparrot/github-code-clean",
    config_name: str = "all-all",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Streams codeparrot/github-code-clean filtered to language=="Python",
    yielding (ids, mask) pairs of shape (seq_len,) each. Rows whose source
    doesn't parse as valid Python, or that no corruption attempt can break,
    are skipped. Pad token is <|end|>'s id (pad region's mask is always
    0); trajectories longer than seq_len are truncated from the end, and
    skipped entirely if that truncation would remove the whole assistant
    span (see smrt.data.sft_common), matching
    smrt.data.toolcalls.load_toolcall_sft_stream's pad/truncate contract.

    Loaded via the `parquet` builder against an explicit `hf://` glob
    (matching smrt.data.code.load_code_stream's fix): this repo's HF
    `datasets` version refuses to run codeparrot/github-code-clean's
    bundled loading script ("Dataset scripts are no longer supported"),
    even though the repo also ships the same data as parquet shards
    under `data/train-*.parquet`. `config_name` is accepted for
    backward-compatible signature consistency but is unused by the
    parquet path.
    """
    from datasets import load_dataset

    from smrt.data.sft_common import pad_or_truncate

    rng = random.Random(seed)
    end_id = tokenizer.encode("<|end|>")[0]
    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{dataset_name}/data/train-*.parquet",
        split="train",
        streaming=True,
    )
    for example in ds:
        if example.get("language") != "Python":
            continue
        code = example.get("code", "")
        if not code:
            continue
        try:
            ast.parse(code)
        except SyntaxError:
            continue  # source itself isn't valid -- can't be a repair target
        trajectory = _build_trajectory(code, tokenizer, rng)
        if trajectory is None:
            continue
        ids, mask = trajectory
        fitted = pad_or_truncate(ids, mask, seq_len, end_id)
        if fitted is None:
            continue
        ids, mask = fitted
        yield torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
