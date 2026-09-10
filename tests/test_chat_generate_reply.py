"""generate_reply's decode loop tested against a fake deterministic model
(never a real trained SMaRT) -- these tests check the loop's stopping
logic and output shape, not model output quality."""

import torch

from smrt.chat import generate_reply

_VOCAB_SIZE = 300
_END_ID = 3


class _FakeTokenizer:
    def encode(self, s: str) -> list:
        assert s == "<|end|>"
        return [_END_ID]


class _AlwaysEndModel(torch.nn.Module):
    """Emits end_id at every position -- generate_reply must stop immediately."""

    def forward(self, x, mem_states=None):
        b, t = x.shape
        logits = torch.full((b, t, _VOCAB_SIZE), -1e9)
        logits[:, :, _END_ID] = 1e9
        return logits, None


class _NeverEndModel(torch.nn.Module):
    """Always emits a fixed non-end token -- generate_reply must run the
    full max_new_tokens budget without ever stopping early."""

    def forward(self, x, mem_states=None):
        b, t = x.shape
        logits = torch.full((b, t, _VOCAB_SIZE), -1e9)
        logits[:, :, 7] = 1e9
        return logits, None


# If this fails: generate_reply stopped breaking out of its loop the
# moment the model emits end_id -- it would instead keep decoding past
# the model's own turn-ending signal, appending end_id itself into the
# generated reply.
def test_stops_immediately_when_model_emits_end_token():
    model = _AlwaysEndModel()
    tokenizer = _FakeTokenizer()
    generated = generate_reply(model, tokenizer, history_ids=[1, 2], device=torch.device("cpu"), max_new_tokens=10)
    assert generated == []


# If this fails: generate_reply stopped respecting max_new_tokens as a
# hard cap -- a model that never emits end_id would generate forever
# (or the wrong number of tokens).
def test_runs_full_budget_when_model_never_emits_end_token():
    model = _NeverEndModel()
    tokenizer = _FakeTokenizer()
    generated = generate_reply(model, tokenizer, history_ids=[1, 2], device=torch.device("cpu"), max_new_tokens=5)
    assert generated == [7, 7, 7, 7, 7]


# If this fails: generate_reply stopped appending each newly generated
# token onto its running `ids` sequence before the next forward pass --
# every subsequent step would keep re-decoding from the same fixed
# prompt instead of an actually-growing context.
def test_history_ids_input_is_not_mutated():
    model = _NeverEndModel()
    tokenizer = _FakeTokenizer()
    history = [1, 2]
    generate_reply(model, tokenizer, history_ids=history, device=torch.device("cpu"), max_new_tokens=3)
    assert history == [1, 2]
