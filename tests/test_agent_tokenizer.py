from smrt.data.tokenizer import AGENT_VOCAB_SIZE, TOOL_SPECIAL_TOKENS, agent_tokenizer


# If this fails: the agent encoding's advertised vocab size doesn't match
# the base o200k_base vocab plus the 8 added special tokens (e.g. a
# collision with an existing base token id, or a miscounted constant).
def test_vocab_size_matches_base_plus_specials():
    tokenizer = agent_tokenizer()
    assert tokenizer.n_vocab == AGENT_VOCAB_SIZE


# If this fails: a special token like <|tool_call|> is being split into
# multiple base-vocab byte-pair tokens instead of landing as one id, which
# would break every span-offset assumption the tool-call/tool-needle data
# pipelines make about special tokens being atomic.
def test_special_tokens_encode_as_single_ids():
    tokenizer = agent_tokenizer()
    for token in TOOL_SPECIAL_TOKENS:
        ids = tokenizer.encode(token)
        assert len(ids) == 1, f"{token!r} encoded to {len(ids)} ids, expected 1"


# If this fails: text mixing special tokens with ordinary content doesn't
# round-trip losslessly (encode then decode changes the string), which
# would corrupt any training example built via string concatenation.
def test_roundtrip_with_mixed_special_and_ordinary_text():
    tokenizer = agent_tokenizer()
    text = '<|system|>You are helpful.<|end|><|user|>hi<|end|><|assistant|><|tool_call|>{"a":1}<|/tool_call|><|end|>'
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


# If this fails: two distinct special tokens were assigned the same id
# (an off-by-one in the enumerate-based id assignment), which would make
# them indistinguishable to the model.
def test_special_token_ids_are_distinct():
    tokenizer = agent_tokenizer()
    ids = [tokenizer.encode(token)[0] for token in TOOL_SPECIAL_TOKENS]
    assert len(set(ids)) == len(TOOL_SPECIAL_TOKENS)
