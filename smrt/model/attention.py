"""Sliding-window causal attention with an optional fixed-size extra-KV prefix
(persistent tokens + memory read), per the MAC contract in DESIGN.md.

Adds RoPE, QK-norm, and GQA over plain MHA. All positions are LOCAL to the
current chunk (this module never receives cross-chunk raw K/V -- long-range
information flows through NeuralMemory instead, per block.py's chunking):
extra_kv (persistent tokens + memory read) is never rotated -- it has no
sequential position, so it stays in a fixed, position-invariant frame.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from smrt.config import AttentionConfig
from smrt.model.norm import RMSNorm

ROPE_BASE = 500_000.0  # Llama-3-style base, chosen for this repo's up-to-65536-token curriculum


def sliding_window_causal_mask(
    seq_len: int, window_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Additive mask of shape (seq_len, seq_len).

    mask[i, j] = 0 if j <= i and i - j < window_size, else -inf (finfo.min).
    """
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    visible = (cols <= rows) & (rows - cols < window_size)
    neg = torch.finfo(dtype).min
    mask = torch.where(
        visible,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), neg, dtype=dtype, device=device),
    )
    return mask.expand(seq_len, seq_len).clone()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, D) or broadcastable; cos/sin: (T, D) or a (1, D) slice of one."""
    return x * cos + _rotate_half(x) * sin


class SlidingWindowAttention(nn.Module):
    def __init__(self, cfg: AttentionConfig, d_model: int):
        super().__init__()
        assert cfg.head_dim % 2 == 0, f"head_dim must be even for RoPE, got {cfg.head_dim}"
        assert cfg.num_heads % cfg.num_kv_heads == 0, (
            f"num_heads ({cfg.num_heads}) must be divisible by num_kv_heads ({cfg.num_kv_heads})"
        )
        self.cfg = cfg
        self.d_model = d_model
        H, Hkv, D = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(d_model, H * D, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * Hkv * D, bias=False)
        self.out_proj = nn.Linear(H * D, d_model, bias=False)
        self.q_norm = RMSNorm(D)
        self.k_norm = RMSNorm(D)

        inv_freq = 1.0 / (ROPE_BASE ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(self, T: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(T, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, extra_kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        H, Hkv, D = self.cfg.num_heads, self.cfg.num_kv_heads, self.cfg.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k_x, v_x = self.kv_proj(x).split(Hkv * D, dim=-1)
        k_x = k_x.view(B, T, Hkv, D).transpose(1, 2)  # (B, Hkv, T, D)
        v_x = v_x.view(B, T, Hkv, D).transpose(1, 2)

        q = self.q_norm(q)
        k_x = self.k_norm(k_x)

        cos, sin = self._rope_cos_sin(T, x.device, x.dtype)
        q = _apply_rope(q, cos, sin)
        k_x = _apply_rope(k_x, cos, sin)

        if extra_kv is not None:
            M = extra_kv.shape[1]
            k_extra, v_extra = self.kv_proj(extra_kv).split(Hkv * D, dim=-1)
            k_extra = self.k_norm(k_extra.view(B, M, Hkv, D).transpose(1, 2))  # never rotated
            v_extra = v_extra.view(B, M, Hkv, D).transpose(1, 2)
            k = torch.cat([k_extra, k_x], dim=2)  # (B, Hkv, M+T, D)
            v = torch.cat([v_extra, v_x], dim=2)
            window_mask = sliding_window_causal_mask(T, self.cfg.window_size, x.device, x.dtype)
            extra_visible = torch.zeros((T, M), dtype=x.dtype, device=x.device)
            mask = torch.cat([extra_visible, window_mask], dim=1)  # (T, M+T)
        else:
            k, v = k_x, v_x
            mask = sliding_window_causal_mask(T, self.cfg.window_size, x.device, x.dtype)

        with sdpa_kernel(SDPBackend.MATH):
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False, enable_gqa=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.out_proj(attn_out)
