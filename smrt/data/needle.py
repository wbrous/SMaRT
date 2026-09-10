"""Synthetic needle-in-haystack data generation for recall training/eval.

A "needle" is a single fact sentence ("The secret code for {topic} is
{value}.") spliced into a haystack of unrelated public-domain filler prose
at a controlled depth, followed by a question asking for the value back.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from smrt.config import TrainConfig
from smrt.data.batch import validate_batch

import torch

_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "filler_corpus.txt")

TOPICS: list[str] = [
    "the vault", "the archive", "the observatory", "the greenhouse", "the workshop",
    "the lighthouse", "the orchard", "the bakery", "the foundry", "the harbor",
    "the cellar", "the attic", "the garden", "the tower", "the bridge",
    "the mill", "the quarry", "the stable", "the chapel", "the market",
    "the library", "the theatre", "the museum", "the depot", "the cannery",
    "the shipyard", "the mine", "the reservoir", "the aviary", "the nursery",
    "the pantry", "the forge", "the tannery", "the brewery", "the distillery",
    "the abbey", "the cloister", "the granary", "the hive", "the kennel",
    "the aquarium", "the arboretum", "the conservatory", "the vineyard", "the orchestra pit",
    "the printing house", "the mint", "the armory", "the arsenal", "the shrine",
    "the crypt", "the sanctuary", "the cistern", "the aqueduct", "the courtyard",
    "the pavilion", "the rotunda", "the gallery", "the studio", "the atelier",
    "the parlor", "the drawing room", "the study", "the annex", "the wing",
    "the outpost", "the garrison", "the barracks", "the citadel", "the keep",
    "the bastion", "the rampart", "the moat", "the gatehouse", "the portcullis",
    "the belfry", "the spire", "the vestry", "the transept", "the nave",
    "the apiary", "the dovecote", "the piggery", "the sheepfold", "the paddock",
    "the corral", "the silo", "the smokehouse", "the icehouse", "the springhouse",
    "the washhouse", "the woodshed", "the boathouse", "the dockyard", "the wharf",
    "the pier", "the jetty", "the marina", "the causeway", "the ferry landing",
    "the depot office", "the signal box", "the roundhouse", "the switchyard", "the terminal",
    "the customs house", "the counting house", "the exchange", "the guildhall", "the almshouse",
    "the infirmary", "the sanatorium", "the asylum", "the orphanage", "the workhouse",
    "the poorhouse", "the tollhouse", "the tithe barn", "the manor", "the estate",
    "the lodge", "the cottage", "the farmstead", "the homestead", "the ranch",
    "the plantation", "the vineyard office", "the winery", "the cidery", "the malthouse",
    "the sugarhouse", "the potashery", "the sawmill", "the gristmill", "the fulling mill",
    "the tidewater mill", "the windmill", "the watermill", "the pumphouse", "the waterworks",
    "the gasworks", "the powerhouse", "the substation", "the generator room", "the boiler room",
    "the engine room", "the turbine hall", "the control room", "the signal room", "the wireless room",
    "the darkroom", "the printshop", "the bindery", "the composing room", "the pressroom",
    "the newsroom", "the editorial office", "the recording booth", "the broadcast tower", "the relay station",
    "the weather station", "the seismograph room", "the planetarium", "the herbarium", "the vivarium",
    "the terrarium", "the insectarium", "the menagerie", "the zoological garden", "the botanical garden",
    "the rose garden", "the kitchen garden", "the walled garden", "the sunken garden", "the topiary garden",
    "the fernery", "the orangery", "the palm house", "the glasshouse", "the coldframe",
    "the potting shed", "the tool shed", "the equipment shed", "the machine shop", "the tinsmith's shop",
    "the cobbler's shop", "the tailor's shop", "the milliner's shop", "the apothecary", "the herbalist's shop",
    "the chandlery", "the ironmonger's", "the haberdashery", "the confectionery", "the creamery",
    "the dairy", "the cheesehouse", "the smokery", "the fishmonger's", "the butcher's shop",
]

assert len(TOPICS) == 200, f"TOPICS must have exactly 200 entries, has {len(TOPICS)}"


class ByteTokenizer:
    """Trivial byte-level tokenizer used when vocab_size == 256."""

    def encode(self, s: str) -> list[int]:
        return [b for b in s.encode("utf-8") if b < 256]

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def gpt2_tokenizer():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")

    class _Adapter:
        def encode(self, s: str) -> list[int]:
            return enc.encode(s)

        def decode(self, ids: list[int]) -> str:
            return enc.decode(ids)

    return _Adapter()


_FILLER_TEXT: str | None = None


def _load_filler_text() -> str:
    global _FILLER_TEXT
    if _FILLER_TEXT is None:
        with open(_CORPUS_PATH, "r", encoding="utf-8", errors="replace") as f:
            _FILLER_TEXT = f.read()
    return _FILLER_TEXT


def _sample_filler_tokens(rng: random.Random, tokenizer, num_tokens: int) -> list[int]:
    text = _load_filler_text()
    start = rng.randint(0, max(1, len(text) - num_tokens * 8))
    chunk = text[start:]
    ids: list[int] = []
    pos = 0
    # Grow window until we have enough tokens (rough char->token ratio varies).
    window = num_tokens * 6
    while len(ids) < num_tokens and pos < len(chunk):
        pos = min(pos + window, len(chunk))
        ids = tokenizer.encode(chunk[:pos])
        window *= 2
        if pos >= len(chunk):
            break
    return ids[:num_tokens] if len(ids) >= num_tokens else (ids + ids * (num_tokens // max(len(ids), 1) + 1))[:num_tokens]


@dataclass
class NeedleExample:
    token_ids: list
    needle_span: tuple
    question_span: tuple
    answer_span: tuple


def generate_needle_example(
    rng: random.Random,
    context_len: int,
    filler_tokens: int,
    depth_bin: int,
    num_depth_bins: int,
    tokenizer,
) -> NeedleExample:
    topic = rng.choice(TOPICS)

    reserve_sample = tokenizer.encode(
        f"The secret code for {topic} is 000000.\nQuestion: What is the secret code for {topic}?\nAnswer: 000000"
    )
    reserve = len(reserve_sample) + 10  # small safety margin for tokenizer boundary effects
    effective_filler_tokens = max(0, min(filler_tokens, context_len - reserve))

    filler_before_n = round(depth_bin / max(num_depth_bins - 1, 1) * effective_filler_tokens)
    filler_after_n = effective_filler_tokens - filler_before_n

    for _attempt in range(10):
        value = rng.randint(100000, 999999)
        value_str = str(value)

        filler_before_ids = _sample_filler_tokens(rng, tokenizer, filler_before_n) if filler_before_n > 0 else []
        filler_after_ids = _sample_filler_tokens(rng, tokenizer, filler_after_n) if filler_after_n > 0 else []

        filler_before_text = tokenizer.decode(filler_before_ids)
        filler_after_text = tokenizer.decode(filler_after_ids)
        if value_str in filler_before_text or value_str in filler_after_text:
            continue  # collision: resample value

        needle_text = f"The secret code for {topic} is {value_str}."
        question_text = f"\nQuestion: What is the secret code for {topic}?\nAnswer:"
        answer_text = f" {value_str}"

        needle_ids = tokenizer.encode(needle_text)
        question_ids = tokenizer.encode(question_text)
        answer_ids = tokenizer.encode(answer_text)

        token_ids = filler_before_ids + needle_ids + filler_after_ids + question_ids + answer_ids

        needle_start = len(filler_before_ids)
        needle_end = needle_start + len(needle_ids)
        question_start = needle_end + len(filler_after_ids)
        question_end = question_start + len(question_ids)
        answer_start = question_end
        answer_end = answer_start + len(answer_ids)

        # Pad/trim to context_len with more filler if needed.
        if len(token_ids) < context_len:
            pad_ids = _sample_filler_tokens(rng, tokenizer, context_len - len(token_ids))
            token_ids = token_ids + pad_ids
        elif len(token_ids) > context_len:
            token_ids = token_ids[:context_len]
            answer_end = min(answer_end, context_len)
            answer_start = min(answer_start, answer_end)
            question_end = min(question_end, answer_start)
            question_start = min(question_start, question_end)

        return NeedleExample(
            token_ids=token_ids,
            needle_span=(needle_start, needle_end),
            question_span=(question_start, question_end),
            answer_span=(answer_start, answer_end),
        )

    raise RuntimeError(f"Could not generate a needle example for topic {topic!r} without a filler collision after 10 tries.")


def generate_needle_batch(
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
        generate_needle_example(rng, context_len, filler_tokens, depth_bin, num_depth_bins, tokenizer)
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
