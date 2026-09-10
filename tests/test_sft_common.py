from smrt.data.sft_common import pad_or_truncate

_END_ID = 999


# If this fails: pad_or_truncate stopped padding short trajectories with
# the pad token id at mask 0 -- the core contract every SFT loader's
# fixed-length batching depends on.
def test_pads_short_trajectory_with_end_id_and_zero_mask():
    ids, mask = [1, 2, 3], [0, 0, 1]
    fitted = pad_or_truncate(ids, mask, seq_len=5, end_id=_END_ID)
    assert fitted is not None
    out_ids, out_mask = fitted
    assert out_ids == [1, 2, 3, _END_ID, _END_ID]
    assert out_mask == [0, 0, 1, 0, 0]


# If this fails: truncation stopped happening from the end, or a
# truncation that still retains part of the assistant span started being
# discarded unnecessarily.
def test_truncates_long_trajectory_when_assistant_span_survives():
    ids, mask = [1, 2, 3, 4, 5], [0, 0, 1, 1, 1]
    fitted = pad_or_truncate(ids, mask, seq_len=4, end_id=_END_ID)
    assert fitted is not None
    out_ids, out_mask = fitted
    assert out_ids == [1, 2, 3, 4]
    assert out_mask == [0, 0, 1, 1]


# If this fails: a truncation that removes the ENTIRE assistant span
# (loss_mask all 0s after cutting) stopped being rejected -- this is the
# exact bug that produced `loss: NaN` in the SFT smoke run (an
# ignore_index cross_entropy reduction over zero unmasked positions).
def test_returns_none_when_truncation_removes_entire_assistant_span():
    ids, mask = [1, 2, 3, 4, 5], [0, 0, 0, 0, 1]
    assert pad_or_truncate(ids, mask, seq_len=4, end_id=_END_ID) is None


# If this fails: an exact-length trajectory (no padding or truncation
# needed) stopped passing through unchanged.
def test_exact_length_trajectory_passes_through_unchanged():
    ids, mask = [1, 2, 3], [0, 1, 1]
    fitted = pad_or_truncate(ids, mask, seq_len=3, end_id=_END_ID)
    assert fitted == (ids, mask)
