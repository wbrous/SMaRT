"""SMaRT transformer block: memory read before local attention, memory
update after — the literal "Memory as Context" (MAC) contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from smrt.config import ModelConfig
from smrt.model.attention import SlidingWindowAttention
from smrt.model.memory import MemoryState, NeuralMemory
from smrt.model.norm import RMSNorm


class GatedMLP(nn.Module):
    """SwiGLU-gated MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SMaRTBlock(nn.Module):
    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.memory = NeuralMemory(model_cfg.memory, model_cfg.d_model)
        self.attn = SlidingWindowAttention(model_cfg.attention, model_cfg.d_model)
        self.persistent = nn.Parameter(
            torch.randn(model_cfg.num_persistent_tokens, model_cfg.d_model) * 0.02
        )
        self.ln1 = RMSNorm(model_cfg.d_model)
        self.ln2 = RMSNorm(model_cfg.d_model)
        self.ln3 = RMSNorm(model_cfg.d_model)
        self.mem_out_proj = nn.Linear(model_cfg.memory.value_dim, model_cfg.d_model, bias=False)
        self.mlp = GatedMLP(model_cfg.d_model, model_cfg.mlp_hidden_dim)

    def forward(self, x: torch.Tensor, mem_state: MemoryState) -> tuple[torch.Tensor, MemoryState]:
        B = x.shape[0]
        chunk_size = self.model_cfg.memory.chunk_size
        chunks = torch.split(x, chunk_size, dim=1)

        out_chunks = []
        for chunk in chunks:
            normed = self.ln1(chunk)
            if self.model_cfg.memory.disabled:
                extra = self.persistent.unsqueeze(0).expand(B, -1, -1)
            else:
                read = self.memory.read(mem_state, normed)
                read_proj = self.mem_out_proj(read)
                persistent = self.persistent.unsqueeze(0).expand(B, -1, -1)
                extra = torch.cat([persistent, read_proj], dim=1)

            attn_out = self.attn(normed, extra_kv=extra)
            chunk = chunk + attn_out
            chunk = chunk + self.mlp(self.ln2(chunk))
            mem_state = self.memory.update(mem_state, self.ln3(chunk))
            out_chunks.append(chunk)

        return torch.cat(out_chunks, dim=1), mem_state
