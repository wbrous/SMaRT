"""Synthetic tool-context needle-in-haystack data: extends
smrt/data/needle.py's fact-recall mechanism into the tool-use domain — the
model must recall a value returned by a tool call issued many turns ago,
not just a fact sentence in prose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from smrt.config import TrainConfig
from smrt.data.batch import validate_batch
from smrt.data.needle import TOPICS, _sample_filler_tokens

import torch

FAKE_TOOLS: list[dict] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"], "additionalProperties": False},
    },
    {
        "name": "search_docs",
        "description": "Search internal documentation.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False},
    },
    {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"], "additionalProperties": False},
    },
    {
        "name": "write_file",
        "description": "Write content to a file on disk.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"], "additionalProperties": False},
    },
    {
        "name": "run_shell",
        "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"], "additionalProperties": False},
    },
    {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
    },
    {
        "name": "get_stock_price",
        "description": "Get the current stock price for a ticker symbol.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"], "additionalProperties": False},
    },
    {
        "name": "send_email",
        "description": "Send an email.",
        "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "subject", "body"], "additionalProperties": False},
    },
    {
        "name": "query_database",
        "description": "Run a SQL query against the database.",
        "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"], "additionalProperties": False},
    },
]

_NON_NEEDLE_TOOLS = [t for t in FAKE_TOOLS if t["name"] != "query_database"]


def _random_tool_call_text(rng: random.Random, tool: dict, tokenizer) -> str:
    import json

    props = tool["parameters"]["properties"]
    args = {}
    for key in props:
        if key in ("sql", "expression", "command", "path", "content", "body"):
            args[key] = rng.choice(TOPICS)
        else:
            args[key] = rng.choice(TOPICS)
    call = json.dumps({"name": tool["name"], "arguments": args}, separators=(",", ":"))
    result = json.dumps({"result": rng.choice(TOPICS)}, separators=(",", ":"))
    return f"<|assistant|><|tool_call|>{call}<|/tool_call|><|end|><|tool_result|>{result}<|/tool_result|>"


@dataclass
class ToolNeedleExample:
    token_ids: list
    needle_span: tuple
    question_span: tuple
    answer_span: tuple


def generate_tool_needle_example(
    rng: random.Random,
    context_len: int,
    filler_tokens: int,
    depth_bin: int,
    num_depth_bins: int,
    tokenizer,
) -> ToolNeedleExample:
    import json

    system_text = "<|system|>"
    for tool in FAKE_TOOLS:
        system_text += "\nTool: " + json.dumps(tool, separators=(",", ":"))
    system_text += "<|end|>"
    system_ids = tokenizer.encode(system_text)

    question_text = "<|user|>What record_id did the query_database call return earlier in this conversation?<|end|><|assistant|>"

    reserve_sample = tokenizer.encode(question_text) + tokenizer.encode(
        '<|assistant|><|tool_call|>{"name":"query_database","arguments":{"sql":"SELECT record_id FROM t"}}<|/tool_call|><|end|><|tool_result|>{"record_id":"000000"}<|/tool_result|>'
    ) + tokenizer.encode(" 000000")
    reserve = len(system_ids) + len(reserve_sample) + 10
    effective_filler_tokens = max(0, min(filler_tokens, context_len - reserve))

    filler_before_n = round(depth_bin / max(num_depth_bins - 1, 1) * effective_filler_tokens)
    filler_after_n = effective_filler_tokens - filler_before_n

    for _attempt in range(10):
        value = rng.randint(100000, 999999)
        value_str = str(value)

        before_ids: list[int] = []
        while len(before_ids) < filler_before_n:
            tool = rng.choice(_NON_NEEDLE_TOOLS)
            before_ids.extend(tokenizer.encode(_random_tool_call_text(rng, tool, tokenizer)))
        before_ids = before_ids[:filler_before_n]

        after_ids: list[int] = []
        while len(after_ids) < filler_after_n:
            tool = rng.choice(_NON_NEEDLE_TOOLS)
            after_ids.extend(tokenizer.encode(_random_tool_call_text(rng, tool, tokenizer)))
        after_ids = after_ids[:filler_after_n]

        before_text = tokenizer.decode(before_ids)
        after_text = tokenizer.decode(after_ids)
        if value_str in before_text or value_str in after_text:
            continue  # collision: resample value

        needle_call = json.dumps(
            {"name": "query_database", "arguments": {"sql": "SELECT record_id FROM records LIMIT 1"}},
            separators=(",", ":"),
        )
        needle_result = json.dumps({"record_id": value_str}, separators=(",", ":"))
        needle_text = f"<|assistant|><|tool_call|>{needle_call}<|/tool_call|><|end|><|tool_result|>{needle_result}<|/tool_result|>"

        needle_ids = tokenizer.encode(needle_text)
        question_ids = tokenizer.encode(question_text)
        answer_ids = tokenizer.encode(f" {value_str}")

        token_ids = system_ids + before_ids + needle_ids + after_ids + question_ids + answer_ids

        needle_start = len(system_ids) + len(before_ids)
        needle_end = needle_start + len(needle_ids)
        question_start = needle_end + len(after_ids)
        question_end = question_start + len(question_ids)
        answer_start = question_end
        answer_end = answer_start + len(answer_ids)

        if len(token_ids) < context_len:
            pad_ids = _sample_filler_tokens(rng, tokenizer, context_len - len(token_ids))
            token_ids = token_ids + pad_ids
        elif len(token_ids) > context_len:
            token_ids = token_ids[:context_len]
            answer_end = min(answer_end, context_len)
            answer_start = min(answer_start, answer_end)
            question_end = min(question_end, answer_start)
            question_start = min(question_start, question_end)

        return ToolNeedleExample(
            token_ids=token_ids,
            needle_span=(needle_start, needle_end),
            question_span=(question_start, question_end),
            answer_span=(answer_start, answer_end),
        )

    raise RuntimeError("Could not generate a tool-needle example without a filler collision after 10 tries.")


def generate_tool_needle_batch(
    rng: random.Random,
    batch_size: int,
    context_len: int,
    filler_tokens: int,
    depth_bin: int,
    num_depth_bins: int,
    tokenizer,
    train_cfg: TrainConfig,
) -> tuple:
    examples = [
        generate_tool_needle_example(rng, context_len, filler_tokens, depth_bin, num_depth_bins, tokenizer)
        for _ in range(batch_size)
    ]
    seq_len = train_cfg.seq_len
    rows = []
    spans = []
    for ex in examples:
        ids = ex.token_ids
        if len(ids) < seq_len:
            ids = ids + _sample_filler_tokens(rng, tokenizer, seq_len - len(ids))
        else:
            ids = ids[:seq_len]
        rows.append(ids)
        spans.append(ex.needle_span)
    input_ids = torch.tensor(rows, dtype=torch.long)
    validate_batch(input_ids, train_cfg, needle_spans=spans)
    return input_ids, spans
