import ast
import random

from smrt.data.code_repair import corrupt_source, _build_trajectory
from smrt.data.tokenizer import agent_tokenizer

_VALID_SOURCE = (
    "def add(a, b):\n"
    "    total = a + b\n"
    "    return total\n"
)


# If this fails: corrupt_source stopped producing a syntactically INVALID
# variant of valid source (e.g. a corruption that happens to still
# parse), breaking the guarantee that every training example's "broken"
# side is actually broken.
def test_corrupt_source_produces_syntax_error():
    rng = random.Random(0)
    broken = corrupt_source(_VALID_SOURCE, rng)
    assert broken is not None
    assert broken != _VALID_SOURCE
    raised = False
    try:
        ast.parse(broken)
    except SyntaxError:
        raised = True
    assert raised


# If this fails: corrupt_source stopped giving up cleanly (returning
# None) when no corruption can produce a syntax error, e.g. because the
# source has no colon-header line and no closing bracket at all.
def test_corrupt_source_returns_none_when_uncorruptible():
    rng = random.Random(0)
    assert corrupt_source("x = 1\n", rng) is None


# If this fails: the loss mask is not actually zero on the broken-code
# user turn and one on the fixed-code assistant turn -- the core contract
# the SFT loop's ignore_index masking depends on.
def test_trajectory_mask_only_covers_assistant_turn():
    tokenizer = agent_tokenizer()
    rng = random.Random(0)
    result = _build_trajectory(_VALID_SOURCE, tokenizer, rng)
    assert result is not None
    ids, mask = result
    assert any(m == 1 for m in mask)
    assert any(m == 0 for m in mask)
    assistant_id = tokenizer.encode("<|assistant|>")[0]
    first_assistant_pos = ids.index(assistant_id)
    unmasked_positions = [i for i, m in enumerate(mask) if m == 1]
    assert min(unmasked_positions) >= first_assistant_pos
