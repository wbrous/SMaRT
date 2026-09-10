"""Agent tokenizer: tiktoken's o200k_base encoding plus 8 chat/tool-call
special tokens, used whenever ModelConfig.vocab_size == AGENT_VOCAB_SIZE.
"""

from __future__ import annotations

import tiktoken

TOOL_SPECIAL_TOKENS = [
    "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    "<|tool_call|>", "<|/tool_call|>", "<|tool_result|>", "<|/tool_result|>",
]

_BASE_N_VOCAB = tiktoken.get_encoding("o200k_base").n_vocab
AGENT_VOCAB_SIZE = _BASE_N_VOCAB + len(TOOL_SPECIAL_TOKENS)


def build_agent_encoding() -> tiktoken.Encoding:
    """Extends o200k_base with TOOL_SPECIAL_TOKENS at consecutive ids
    immediately above the base encoding's vocab.

    Postconditions: the returned Encoding's n_vocab equals AGENT_VOCAB_SIZE.
    """
    base = tiktoken.get_encoding("o200k_base")
    specials = {tok: base.n_vocab + i for i, tok in enumerate(TOOL_SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="smrt_agent_o200k",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens={**base._special_tokens, **specials},
    )
    assert enc.n_vocab == AGENT_VOCAB_SIZE, (enc.n_vocab, AGENT_VOCAB_SIZE)
    return enc


class AgentTokenizer:
    """encode/decode adapter matching the ByteTokenizer/gpt2_tokenizer()
    interface used throughout smrt.data.*.

    encode() passes allowed_special="all" so text containing literal
    special-token substrings (e.g. from source data) round-trips through
    the special-token ids instead of raising tiktoken's default
    "disallowed special token" error.
    """

    def __init__(self, enc: tiktoken.Encoding):
        self._enc = enc
        self.n_vocab = enc.n_vocab

    def encode(self, s: str) -> list[int]:
        return self._enc.encode(s, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)


_AGENT_TOKENIZER: AgentTokenizer | None = None


def agent_tokenizer() -> AgentTokenizer:
    """Lazily builds and caches a single module-level AgentTokenizer
    instance, mirroring smrt.data.needle's _FILLER_TEXT cache pattern.
    """
    global _AGENT_TOKENIZER
    if _AGENT_TOKENIZER is None:
        _AGENT_TOKENIZER = AgentTokenizer(build_agent_encoding())
    return _AGENT_TOKENIZER
