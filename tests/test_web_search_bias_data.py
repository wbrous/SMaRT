import random

from smrt.data.tokenizer import agent_tokenizer
from smrt.data.web_search_bias import _build_trajectory, _random_question


# If this fails: the knowledge-boundary question generator stopped
# producing a non-empty question for some template/topic group, silently
# breaking every downstream trajectory built from it.
def test_random_question_is_always_nonempty():
    rng = random.Random(0)
    for _ in range(20):
        assert _random_question(rng).strip() != ""


# If this fails: the emitted trajectory stopped always targeting a
# web_search tool call -- the entire point of this dataset is that the
# ONLY correct response to a knowledge-boundary question is a tool call,
# never a direct prose answer.
def test_trajectory_always_calls_web_search():
    tokenizer = agent_tokenizer()
    rng = random.Random(1)
    ids, mask = _build_trajectory(rng, tokenizer)
    text = tokenizer.decode(ids)
    assert '<|tool_call|>{"name":"web_search","arguments":{"query":' in text


# If this fails: the loss mask is not actually zero on system/user content
# and one on the assistant tool-call span -- the core contract the SFT
# loop's ignore_index masking depends on.
def test_trajectory_mask_only_covers_assistant_span():
    tokenizer = agent_tokenizer()
    rng = random.Random(2)
    ids, mask = _build_trajectory(rng, tokenizer)
    assert any(m == 1 for m in mask)
    assert any(m == 0 for m in mask)
    assistant_id = tokenizer.encode("<|assistant|>")[0]
    first_assistant_pos = ids.index(assistant_id)
    unmasked_positions = [i for i, m in enumerate(mask) if m == 1]
    assert min(unmasked_positions) >= first_assistant_pos
