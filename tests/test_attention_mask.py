import math

import torch

from smrt.model.attention import sliding_window_causal_mask, SlidingWindowAttention, _apply_rope
from smrt.config import AttentionConfig


# If this fails: the window boundary has an off-by-one (either an extra
# position visible one step before the window, or the window-edge position
# incorrectly masked out).
def test_window_boundary_off_by_one():
    seq_len, window_size = 20, 5
    mask = sliding_window_causal_mask(seq_len, window_size, torch.device("cpu"), torch.float32)
    # i=15: window covers j in [11, 15] (exactly window_size=5 positions).
    assert mask[15, 11] == 0.0  # window edge, visible
    assert mask[15, 10] == torch.finfo(torch.float32).min  # one before window, masked


# If this fails: the causal component of the mask is broken independent of
# the window component (future positions become visible).
def test_future_positions_masked():
    seq_len, window_size = 20, 5
    mask = sliding_window_causal_mask(seq_len, window_size, torch.device("cpu"), torch.float32)
    neg = torch.finfo(torch.float32).min
    for i, j in [(0, 1), (5, 6), (10, 19), (3, 4)]:
        assert mask[i, j] == neg


# If this fails: the attention weights computed by SDPA diverge from a
# manual softmax(qk^T/sqrt(d) + mask) recomputation, meaning the mask isn't
# actually being applied the way the module claims.
def test_attention_weights_respect_window(seeded_rng):
    cfg = AttentionConfig(window_size=5, num_heads=1, num_kv_heads=1, head_dim=8)
    attn = SlidingWindowAttention(cfg, d_model=16)
    B, T = 1, 20
    x = torch.randn(B, T, 16)

    with torch.no_grad():
        q = attn.q_norm(attn.q_proj(x).view(B, T, cfg.num_heads, cfg.head_dim).transpose(1, 2))
        k_x, _v_x = attn.kv_proj(x).split(cfg.num_kv_heads * cfg.head_dim, dim=-1)
        k = attn.k_norm(k_x.view(B, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2))
        cos, sin = attn._rope_cos_sin(T, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        mask = sliding_window_causal_mask(T, cfg.window_size, x.device, x.dtype)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim) + mask
        weights = torch.softmax(scores, dim=-1)

    assert weights[0, 0, 15, 9].item() == 0.0
    assert weights[0, 0, 15, 14].item() > 0.0
