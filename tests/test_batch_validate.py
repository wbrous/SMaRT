import pytest
import torch

from smrt.data.batch import validate_batch


# If this fails: validate_batch stopped rejecting a wrong dtype (e.g.
# int32/float) -- nn.Embedding requires torch.long indices, and a wrong
# dtype reaching the model would raise a much more confusing error deep
# inside the forward pass instead of at the batch-construction boundary.
def test_rejects_wrong_dtype(tiny_config):
    bad = torch.zeros(2, tiny_config.train.seq_len, dtype=torch.int32)
    with pytest.raises(AssertionError, match="torch.long"):
        validate_batch(bad, tiny_config.train)


# If this fails: validate_batch stopped rejecting a last-dim length that
# doesn't match cfg.seq_len -- every downstream consumer (model, loss)
# assumes every batch has exactly this fixed sequence length.
def test_rejects_wrong_seq_len(tiny_config):
    bad = torch.zeros(2, tiny_config.train.seq_len + 1, dtype=torch.long)
    with pytest.raises(AssertionError, match="last dim"):
        validate_batch(bad, tiny_config.train)


# If this fails: a correctly-shaped, correctly-typed batch with no
# needle_spans argument stopped passing validation -- every non-needle
# batch source (e.g. plain pretrain/code streams) would start failing.
def test_accepts_correct_batch_with_no_needle_spans(tiny_config):
    ids = torch.zeros(2, tiny_config.train.seq_len, dtype=torch.long)
    validate_batch(ids, tiny_config.train)  # must not raise


# If this fails: validate_batch stopped rejecting a needle span whose end
# exceeds cfg.seq_len -- the exact truncated-needle bug this check exists
# to catch (see smrt/data/needle.py's own history).
def test_rejects_needle_span_exceeding_seq_len(tiny_config):
    seq_len = tiny_config.train.seq_len
    ids = torch.zeros(1, seq_len, dtype=torch.long)
    with pytest.raises(AssertionError, match="exceeds seq_len"):
        validate_batch(ids, tiny_config.train, needle_spans=[(0, seq_len + 1)])


# If this fails: a needle span exactly at the sequence boundary (end ==
# seq_len, the maximum valid end-exclusive value) stopped being accepted
# -- an off-by-one that would reject every needle placed in the last
# position of the context.
def test_accepts_needle_span_exactly_at_seq_len_boundary(tiny_config):
    seq_len = tiny_config.train.seq_len
    ids = torch.zeros(1, seq_len, dtype=torch.long)
    validate_batch(ids, tiny_config.train, needle_spans=[(seq_len - 1, seq_len)])  # must not raise
