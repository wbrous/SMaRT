import random

from smrt.data.needle import ByteTokenizer, generate_needle_example


# If this fails: the needle span offsets don't actually point at the
# spliced-in fact sentence (a span-offset bug in generate_needle_example).
def test_needle_span_contains_topic_and_value():
    rng = random.Random(42)
    tokenizer = ByteTokenizer()
    ex = generate_needle_example(rng, context_len=200, filler_tokens=150, depth_bin=1, num_depth_bins=3, tokenizer=tokenizer)
    needle_text = tokenizer.decode(ex.token_ids[ex.needle_span[0]:ex.needle_span[1]])
    assert "secret code" in needle_text


# If this fails: the filler/answer collision check isn't actually
# preventing the true value digits from appearing elsewhere in the
# haystack (which would make the recall task trivially guessable).
def test_value_does_not_leak_into_filler():
    rng = random.Random(7)
    tokenizer = ByteTokenizer()
    ex = generate_needle_example(rng, context_len=200, filler_tokens=150, depth_bin=0, num_depth_bins=3, tokenizer=tokenizer)
    needle_text = tokenizer.decode(ex.token_ids[ex.needle_span[0]:ex.needle_span[1]])
    value_str = needle_text.split(" is ")[1].rstrip(".")

    before = tokenizer.decode(ex.token_ids[:ex.needle_span[0]])
    after = tokenizer.decode(ex.token_ids[ex.needle_span[1]:ex.question_span[0]])
    assert value_str not in before
    assert value_str not in after


# If this fails: the depth->token-offset mapping formula is wrong (e.g.
# swapped, or not actually proportional to filler_tokens).
def test_depth_bin_controls_needle_position():
    tokenizer = ByteTokenizer()
    filler_tokens = 300

    rng_low = random.Random(1)
    ex_low = generate_needle_example(rng_low, context_len=1000, filler_tokens=filler_tokens, depth_bin=0, num_depth_bins=3, tokenizer=tokenizer)
    assert ex_low.needle_span[0] < 0.05 * filler_tokens

    rng_high = random.Random(2)
    ex_high = generate_needle_example(rng_high, context_len=1000, filler_tokens=filler_tokens, depth_bin=2, num_depth_bins=3, tokenizer=tokenizer)
    assert ex_high.needle_span[0] > 0.95 * filler_tokens
