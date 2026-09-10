"""Tool-call SFT data pipeline: streams glaiveai/glaive-function-calling-v2
and formats each row into a (token_ids, loss_mask) pair using the
<|system|>/<|user|>/<|assistant|>/<|tool_call|>/<|tool_result|> special
tokens from smrt.data.tokenizer.

Source schema (verified against sampled rows): two string fields, `system`
and `chat`. `system` holds zero-or-more concatenated JSON tool schemas
after a literal marker; `chat` holds "\\n\\n\\n"-separated turns each
prefixed with "USER: " / "ASSISTANT: " / "FUNCTION RESPONSE: ". Assistant
turns that call a tool look like:
    ASSISTANT: <functioncall> {"name": "...", "arguments": '{...}'} <|endoftext|>
where the outer JSON blob wraps `arguments` in single quotes (making the
outer blob invalid JSON as-is) but the inner `arguments` string is valid
JSON on its own.
"""

from __future__ import annotations

import json
import re
from typing import Iterator

import torch

FUNCTIONCALL_RE = re.compile(r'\{\s*"name":\s*"([^"]+)",\s*"arguments":\s*\'(.*)\'\s*\}', re.S)

_skipped_segments = 0


def _extract_json_objects(text: str) -> list[dict]:
    """Repeatedly json.JSONDecoder().raw_decode from the current position,
    skipping whitespace between objects — handles N concatenated JSON
    blobs without needing a delimiter, robust to arbitrary internal
    nesting.
    """
    decoder = json.JSONDecoder()
    objs: list[dict] = []
    pos = 0
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        obj, end = decoder.raw_decode(text, pos)
        objs.append(obj)
        pos = end
    return objs


def parse_system(system_text: str) -> tuple[str, list[dict]]:
    marker = "functions. Use them if required -\n"
    if marker not in system_text:
        return system_text.replace("SYSTEM: ", "", 1).strip(), []
    intro, rest = system_text.split(marker, 1)
    intro = intro.replace("SYSTEM: ", "", 1).strip()
    return intro, _extract_json_objects(rest)


def parse_chat_turns(chat_text: str) -> list[tuple[str, str]]:
    """Returns [(role, body), ...] where role is 'user'|'assistant'|'tool_result'.

    Segments without a recognized role prefix are malformed and skipped
    silently (counted via the module-level _skipped_segments counter for
    eyeballing during data-pipeline debugging) — a single malformed row
    must not crash a multi-hundred-thousand-row stream.
    """
    global _skipped_segments
    turns: list[tuple[str, str]] = []
    for segment in chat_text.split("\n\n\n"):
        segment = segment.strip()
        if not segment:
            continue
        if segment.startswith("USER: "):
            turns.append(("user", segment[len("USER: "):].rstrip(" <|endoftext|>").strip()))
        elif segment.startswith("ASSISTANT: "):
            turns.append(("assistant", segment[len("ASSISTANT: "):].rstrip(" <|endoftext|>").strip()))
        elif segment.startswith("FUNCTION RESPONSE: "):
            turns.append(("tool_result", segment[len("FUNCTION RESPONSE: "):].strip()))
        else:
            _skipped_segments += 1
    return turns


def format_trajectory(system_text: str, chat_text: str, tokenizer) -> tuple[list[int], list[int]] | None:
    """Builds (token_ids, loss_mask) for one glaive-function-calling-v2 row.

    Postconditions: len(token_ids) == len(loss_mask). Returns None if any
    assistant turn starts with "<functioncall>" but doesn't match the
    expected single-quoted-arguments shape, or whose inner `arguments`
    string isn't valid JSON on its own, or if `system_text`'s tool-schema
    section isn't valid JSON (caller must drop the row rather than train
    on a malformed target, or crash a multi-hundred-thousand-row stream
    on one bad row).
    """
    segments: list[str] = []
    masks: list[int] = []

    try:
        intro, schemas = parse_system(system_text)
    except json.JSONDecodeError:
        return None
    system_block = "<|system|>" + intro
    for schema in schemas:
        system_block += "\nTool: " + json.dumps(schema, separators=(",", ":"))
    system_block += "<|end|>"
    segments.append(system_block)
    masks.append(0)

    for role, body in parse_chat_turns(chat_text):
        if role == "user":
            segments.append(f"<|user|>{body}<|end|>")
            masks.append(0)
        elif role == "tool_result":
            segments.append(f"<|tool_result|>{body}<|/tool_result|>")
            masks.append(0)
        else:  # assistant
            if body.startswith("<functioncall>"):
                m = FUNCTIONCALL_RE.search(body)
                if m is None:
                    return None
                name = m.group(1)
                try:
                    arguments = json.loads(m.group(2))
                except json.JSONDecodeError:
                    return None  # inner arguments string isn't valid JSON -- malformed row, drop it
                call = json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
                segments.append(f"<|assistant|><|tool_call|>{call}<|/tool_call|><|end|>")
            else:
                segments.append(f"<|assistant|>{body}<|end|>")
            masks.append(1)

    token_ids: list[int] = []
    loss_mask: list[int] = []
    for segment, mask_value in zip(segments, masks):
        ids = tokenizer.encode(segment)
        token_ids.extend(ids)
        loss_mask.extend([mask_value] * len(ids))

    assert len(token_ids) == len(loss_mask)
    return token_ids, loss_mask


def load_toolcall_sft_stream(
    seq_len: int,
    tokenizer,
    dataset_name: str = "glaiveai/glaive-function-calling-v2",
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Streams glaive-function-calling-v2, yielding (ids, mask) pairs of
    shape (seq_len,) each. Rows whose trajectory doesn't fit the expected
    single-quoted-arguments shape are skipped. Pad token is <|end|>'s id
    (pad region's mask is always 0); trajectories longer than seq_len are
    truncated from the end, and skipped entirely if that truncation would
    remove the whole assistant span (see smrt.data.sft_common).
    """
    from datasets import load_dataset

    from smrt.data.sft_common import pad_or_truncate

    end_id = tokenizer.encode("<|end|>")[0]
    ds = load_dataset(dataset_name, split="train", streaming=True)
    for example in ds:
        trajectory = format_trajectory(example.get("system", ""), example.get("chat", ""), tokenizer)
        if trajectory is None:
            continue
        ids, mask = trajectory
        fitted = pad_or_truncate(ids, mask, seq_len, end_id)
        if fitted is None:
            continue
        ids, mask = fitted
        yield torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
