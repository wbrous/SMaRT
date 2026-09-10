import math

import torch
import torch.nn.functional as F

from smrt.config import AttentionConfig
from smrt.model.attention import _apply_rope, SlidingWindowAttention
from smrt.model.block import GatedMLP
from smrt.model.norm import RMSNorm


# If this fails: RoPE's rotation is no longer purely a function of relative
# position (e.g. an absolute-position leak, or the cos/sin table is wrong),
# breaking the core property RoPE exists to provide.
def test_rope_dot_product_depends_only_on_relative_position():
    cfg = AttentionConfig(window_size=100, num_heads=1, num_kv_heads=1, head_dim=8)
    attn = SlidingWindowAttention(cfg, d_model=16)
    cos, sin = attn._rope_cos_sin(20, torch.device("cpu"), torch.float32)

    torch.manual_seed(0)
    q_vec = torch.randn(1, 1, 1, 8)
    k_vec = torch.randn(1, 1, 1, 8)

    def rotated_dot(qi: int, ki: int) -> float:
        q_rot = _apply_rope(q_vec, cos[qi:qi + 1], sin[qi:qi + 1])
        k_rot = _apply_rope(k_vec, cos[ki:ki + 1], sin[ki:ki + 1])
        return (q_rot * k_rot).sum().item()

    assert abs(rotated_dot(5, 5) - rotated_dot(15, 15)) < 1e-4  # both offset 0
    assert abs(rotated_dot(3, 0) - rotated_dot(13, 10)) < 1e-4  # both offset 3


# If this fails: RMSNorm is normalizing over the wrong dimension or omitting
# the sqrt, reintroducing the activation-scale instability RMSNorm exists to
# prevent at 2B-8B parameter scale.
def test_rmsnorm_produces_unit_rms_with_unit_weight():
    torch.manual_seed(0)
    norm = RMSNorm(dim=16)
    x = torch.randn(2, 5, 16) * 37.0
    with torch.no_grad():
        out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


# If this fails: GQA's kv_proj silently reverted to producing full
# num_heads worth of K/V, defeating the point of the reduced KV head count.
def test_gqa_kv_proj_produces_fewer_heads_than_q_proj():
    cfg = AttentionConfig(window_size=10, num_heads=4, num_kv_heads=2, head_dim=8)
    attn = SlidingWindowAttention(cfg, d_model=32)
    assert attn.q_proj.out_features == 4 * 8
    assert attn.kv_proj.out_features == 2 * 2 * 8
    x = torch.randn(1, 6, 32)
    out = attn(x)
    assert out.shape == (1, 6, 32)


# If this fails: a bias term was introduced somewhere in the gated MLP
# (violating the bias=False convention) or the SwiGLU gating formula no
# longer zeros out at zero input (silu(0) == 0, so the whole product must
# be exactly zero regardless of up_proj's value).
def test_gated_mlp_zero_input_yields_zero_output():
    torch.manual_seed(0)
    mlp = GatedMLP(d_model=4, hidden_dim=6)
    x = torch.zeros(1, 1, 4)
    out = mlp(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)
