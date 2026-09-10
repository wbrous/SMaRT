"""Synthetic web-search-bias SFT data: SMaRT is a small model with limited
built-in knowledge, so it must be EXTREMELY biased toward calling the
web_search tool instead of guessing whenever a question needs current,
specific, or obscure information. Every example in this stream has
exactly one correct target: a <|tool_call|> to web_search, never a direct
prose answer.
"""

from __future__ import annotations

import json
import random
from typing import Iterator

import torch

WEB_SEARCH_TOOL_SCHEMA = {
    "name": "web_search",
    "description": "Search the live web for current, specific, or obscure information not reliably known from training data.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}

WEB_SEARCH_SYSTEM_PROMPT = (
    "You are a helpful assistant with a small parameter count and limited "
    "built-in knowledge. You must NEVER guess at facts you are not certain "
    "of. Whenever a question needs current, specific, numeric, or obscure "
    "information (dates, prices, scores, versions, statistics, names of "
    "recent things, anything you were not near-certainly trained on), you "
    "MUST call the web_search tool before answering. When in doubt, search."
)

_PRICE_TEMPLATES = ["What is the current price of {topic}?", "What is the latest trading price of {topic}?"]
_PRICE_TOPICS = ["Bitcoin", "Ethereum", "Tesla stock", "Apple stock", "gold", "crude oil", "the S&P 500 index", "Nvidia stock"]

_AWARD_TEMPLATES = ["Who won the most recent {topic}?", "Who was awarded the {topic} this year?"]
_AWARD_TOPICS = ["Nobel Prize in Physics", "Academy Award for Best Picture", "Super Bowl", "World Cup", "Ballon d'Or", "Grammy Award for Album of the Year"]

_VERSION_TEMPLATES = ["What is the latest version of {topic}?", "What is the current release number of {topic}?"]
_VERSION_TOPICS = ["Python", "the Linux kernel", "Node.js", "React", "Windows", "macOS", "Kubernetes", "PostgreSQL"]

_WEATHER_TEMPLATES = ["What is today's weather in {topic}?", "What is the current temperature in {topic}?"]
_WEATHER_TOPICS = ["Tokyo", "New York City", "London", "Sydney", "Cairo", "Reykjavik", "Mumbai", "Sao Paulo"]

_NEWS_TEMPLATES = ["What is the latest news about {topic}?", "What happened with {topic} this week?"]
_NEWS_TOPICS = ["OpenAI", "the stock market", "the upcoming election", "the local sports team", "the ongoing trade negotiations", "the tech industry"]

_KNOWLEDGE_BOUNDARY_GROUPS: list[tuple[list[str], list[str]]] = [
    (_PRICE_TEMPLATES, _PRICE_TOPICS),
    (_AWARD_TEMPLATES, _AWARD_TOPICS),
    (_VERSION_TEMPLATES, _VERSION_TOPICS),
    (_WEATHER_TEMPLATES, _WEATHER_TOPICS),
    (_NEWS_TEMPLATES, _NEWS_TOPICS),
]


def _random_question(rng: random.Random) -> str:
    templates, topics = rng.choice(_KNOWLEDGE_BOUNDARY_GROUPS)
    template = rng.choice(templates)
    topic = rng.choice(topics)
    return template.format(topic=topic)


def _build_trajectory(rng: random.Random, tokenizer) -> tuple[list[int], list[int]]:
    question = _random_question(rng)

    system_block = "<|system|>" + WEB_SEARCH_SYSTEM_PROMPT
    system_block += "\nTool: " + json.dumps(WEB_SEARCH_TOOL_SCHEMA, separators=(",", ":"))
    system_block += "<|end|>"
    call = json.dumps({"name": "web_search", "arguments": {"query": question}}, separators=(",", ":"))

    segments = [
        (system_block, 0),
        (f"<|user|>{question}<|end|>", 0),
        (f"<|assistant|><|tool_call|>{call}<|/tool_call|><|end|>", 1),
    ]

    token_ids: list[int] = []
    loss_mask: list[int] = []
    for segment, mask_value in segments:
        ids = tokenizer.encode(segment)
        token_ids.extend(ids)
        loss_mask.extend([mask_value] * len(ids))
    assert len(token_ids) == len(loss_mask)
    return token_ids, loss_mask


def load_web_search_bias_stream(
    seq_len: int, tokenizer, seed: int = 0
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields (ids, mask) pairs of shape (seq_len,) forever. Pad token is
    <|end|>'s id (pad region's mask is always 0); trajectories longer than
    seq_len are truncated from the end, and skipped (re-drawn) entirely if
    that truncation would remove the whole assistant span (see
    smrt.data.sft_common), matching
    smrt.data.toolcalls.load_toolcall_sft_stream's pad/truncate contract.
    """
    from smrt.data.sft_common import pad_or_truncate

    rng = random.Random(seed)
    end_id = tokenizer.encode("<|end|>")[0]
    while True:
        ids, mask = _build_trajectory(rng, tokenizer)
        fitted = pad_or_truncate(ids, mask, seq_len, end_id)
        if fitted is None:
            continue
        ids, mask = fitted
        yield torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long)
