import random

from smrt.data.tokenizer import agent_tokenizer
from smrt.data.tool_needle import generate_tool_needle_example


# If this fails: the needle span offsets don't actually point at the
# query_database tool-call/tool-result exchange (a span-offset bug in
# generate_tool_needle_example, mirroring needle.py's own contract).
def test_needle_span_contains_query_database_call():
    tokenizer = agent_tokenizer()
    ex = generate_tool_needle_example(random.Random(42), context_len=2048, filler_tokens=1500, depth_bin=1, num_depth_bins=3, tokenizer=tokenizer)
    needle_text = tokenizer.decode(ex.token_ids[ex.needle_span[0]:ex.needle_span[1]])
    assert "query_database" in needle_text
    assert "record_id" in needle_text


# If this fails: the answer span is empty or misplaced — the exact
# truncation-bug failure mode this generator was designed to avoid (per
# needle.py's own history).
def test_answer_span_is_non_empty_and_after_question():
    tokenizer = agent_tokenizer()
    ex = generate_tool_needle_example(random.Random(0), context_len=2048, filler_tokens=1800, depth_bin=2, num_depth_bins=5, tokenizer=tokenizer)
    assert ex.answer_span[1] > ex.answer_span[0]
    assert ex.answer_span[0] >= ex.question_span[1]


# If this fails: the collision-avoidance loop isn't actually preventing
# the true record_id digits from appearing elsewhere in the haystack
# (which would make the recall task trivially guessable by string search).
def test_record_id_does_not_leak_into_filler():
    tokenizer = agent_tokenizer()
    ex = generate_tool_needle_example(random.Random(7), context_len=2048, filler_tokens=1800, depth_bin=0, num_depth_bins=3, tokenizer=tokenizer)
    answer_text = tokenizer.decode(ex.token_ids[ex.answer_span[0]:ex.answer_span[1]])
    value_str = answer_text.strip()

    before = tokenizer.decode(ex.token_ids[:ex.needle_span[0]])
    after = tokenizer.decode(ex.token_ids[ex.needle_span[1]:ex.question_span[0]])
    assert value_str not in before
    assert value_str not in after


# If this fails: the depth->token-offset mapping formula is wrong (e.g.
# swapped, or not actually proportional to filler_tokens), so "depth"
# curriculum control over needle position is broken.
def test_depth_bin_controls_needle_position():
    tokenizer = agent_tokenizer()
    filler_tokens = 1500

    ex_low = generate_tool_needle_example(random.Random(1), context_len=4096, filler_tokens=filler_tokens, depth_bin=0, num_depth_bins=3, tokenizer=tokenizer)
    ex_high = generate_tool_needle_example(random.Random(2), context_len=4096, filler_tokens=filler_tokens, depth_bin=2, num_depth_bins=3, tokenizer=tokenizer)

    system_prefix_low = ex_low.needle_span[0]
    system_prefix_high = ex_high.needle_span[0]
    assert system_prefix_high > system_prefix_low


# If this fails: the emitted token stream isn't a fixed exact length of
# context_len (a downstream padding/truncation bug in generate_tool_needle_example).
def test_output_length_equals_context_len():
    tokenizer = agent_tokenizer()
    ex = generate_tool_needle_example(random.Random(3), context_len=1024, filler_tokens=800, depth_bin=1, num_depth_bins=3, tokenizer=tokenizer)
    assert len(ex.token_ids) == 1024
