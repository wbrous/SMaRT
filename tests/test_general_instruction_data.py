from smrt.data.tokenizer import agent_tokenizer
from smrt.data.general_instruction import format_messages

_MESSAGES_WITH_SYSTEM = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

_MESSAGES_NO_SYSTEM = [
    {"role": "user", "content": "Say hi."},
    {"role": "assistant", "content": "Hi!"},
]

_MESSAGES_UNKNOWN_ROLE = [
    {"role": "tool", "content": "unexpected role"},
]


# If this fails: format_messages stopped correctly wrapping each role in
# its matching special token, or started dropping a turn.
def test_format_messages_wraps_every_turn():
    tokenizer = agent_tokenizer()
    ids, _mask = format_messages(_MESSAGES_WITH_SYSTEM, tokenizer)
    text = tokenizer.decode(ids)
    assert "<|system|>Be concise.<|end|>" in text
    assert "<|user|>What is 2+2?<|end|>" in text
    assert "<|assistant|>4<|end|>" in text


# If this fails: a row with no leading system message stopped being
# handled (e.g. an assumption that messages[0] is always
# role=="system"), even though the no_robots dataset has rows without
# one.
def test_format_messages_handles_missing_system_turn():
    tokenizer = agent_tokenizer()
    assert format_messages(_MESSAGES_NO_SYSTEM, tokenizer) is not None


# If this fails: the loss mask is not actually zero on system/user
# content and one on assistant content -- the core contract the SFT
# loop's ignore_index masking depends on.
def test_format_messages_mask_only_covers_assistant_turns():
    tokenizer = agent_tokenizer()
    ids, mask = format_messages(_MESSAGES_WITH_SYSTEM, tokenizer)
    assert any(m == 1 for m in mask)
    assert any(m == 0 for m in mask)
    assistant_id = tokenizer.encode("<|assistant|>")[0]
    first_assistant_pos = ids.index(assistant_id)
    unmasked_positions = [i for i, m in enumerate(mask) if m == 1]
    assert min(unmasked_positions) >= first_assistant_pos


# If this fails: an unrecognized role stopped causing the whole row to be
# dropped (returning None) and instead got silently mis-tokenized as
# plain text with no role wrapper.
def test_format_messages_returns_none_for_unknown_role():
    tokenizer = agent_tokenizer()
    assert format_messages(_MESSAGES_UNKNOWN_ROLE, tokenizer) is None
