"""Code pretraining data: HuggingFace streaming loader for
codeparrot/github-code-clean, filtered to 8 well-represented languages.

Mirrors smrt/data/pretrain.py::load_pretrain_stream's structure exactly.
"""

from __future__ import annotations

from typing import Iterator

import torch

CODE_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Go", "Rust", "Java", "C", "C++"}


def load_code_stream(
    seq_len: int,
    tokenizer,
    token_budget: int,
    dataset_name: str = "codeparrot/github-code-clean",
    config_name: str = "all-all",
) -> Iterator[torch.Tensor]:
    """Stream-tokenize codeparrot/github-code-clean into fixed-seq_len chunks.

    Only files whose `language` field is in CODE_LANGUAGES and whose `code`
    field is non-empty are tokenized; anything else is skipped, never
    yielded as a zero-length chunk. Stops once token_budget tokens have
    been yielded.

    Loaded via the `parquet` builder against an explicit `hf://` glob
    rather than `load_dataset(dataset_name, config_name, ...)` directly:
    this repo's HF `datasets` version refuses to run
    codeparrot/github-code-clean's bundled loading script ("Dataset
    scripts are no longer supported"), even though the repo also ships
    the same data as parquet shards under `data/train-*.parquet` — this
    loads those shards directly, sidestepping the script entirely.
    `config_name` is accepted for backward-compatible signature
    consistency but is unused by the parquet path (the parquet shards
    contain every language already; per-language filtering is done by
    the `CODE_LANGUAGES` check below).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=f"hf://datasets/{dataset_name}/data/train-*.parquet",
        split="train",
        streaming=True,
    )
    buffer: list[int] = []
    yielded = 0
    for example in ds:
        if example.get("language") not in CODE_LANGUAGES:
            continue
        code = example.get("code", "")
        if not code:
            continue
        buffer.extend(tokenizer.encode(code) + tokenizer.encode("\n\n"))
        while len(buffer) >= seq_len:
            chunk, buffer = buffer[:seq_len], buffer[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)
            yielded += seq_len
            if yielded >= token_budget:
                return


STARCODER_LANGUAGES = ("python", "javascript", "typescript", "go", "rust", "java", "c", "cpp")


def load_starcoder_stream(
    seq_len: int,
    tokenizer,
    dataset_name: str = "bigcode/starcoderdata",
    languages: tuple = STARCODER_LANGUAGES,
    token_budget: float = float("inf"),
) -> Iterator[torch.Tensor]:
    """Stream-tokenize bigcode/starcoderdata (250B tokens, 86 languages;
    this repo uses 8) into fixed-seq_len chunks via the same `parquet` +
    explicit `hf://` glob technique already proven in this file's
    load_code_stream and smrt/data/code_repair.py for
    codeparrot/github-code-clean -- one `data_files` list spanning all
    requested per-language directories gives a single unified stream.

    Requires an HF_TOKEN belonging to an account that has clicked
    "Agree" on https://huggingface.co/datasets/bigcode/starcoderdata
    (gated:auto -- immediate, no manual review, unlike the Nemotron
    datasets).

    token_budget defaults to infinite for the same persistent-stream
    reason as load_nemotron_cc_stream.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "parquet",
        data_files=[f"hf://datasets/{dataset_name}/{lang}/*.parquet" for lang in languages],
        split="train",
        streaming=True,
    )
    buffer: list[int] = []
    yielded = 0
    for example in ds:
        content = example.get("content", "")
        if not content:
            continue
        buffer.extend(tokenizer.encode(content) + tokenizer.encode("\n\n"))
        while len(buffer) >= seq_len:
            chunk, buffer = buffer[:seq_len], buffer[seq_len:]
            yield torch.tensor(chunk, dtype=torch.long)
            yielded += seq_len
            if yielded >= token_budget:
                return
