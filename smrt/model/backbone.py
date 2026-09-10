"""SMaRT: full decoder-only model (embed -> blocks -> ln_f -> tied lm_head)."""

from __future__ import annotations

import torch
import torch.nn as nn

from smrt.config import ModelConfig
from smrt.model.block import SMaRTBlock
from smrt.model.memory import MemoryState
from smrt.model.norm import RMSNorm


class SMaRT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # GPT-2-convention init: nn.Embedding defaults to N(0, 1), but the
        # table is tied to lm_head, so std-1.0 rows scale every logit ~50x
        # and a fresh model starts at CE ~355 instead of ~ln(vocab). Std
        # 0.02 puts initial loss at chance level so training starts from a
        # sane gradient scale. Applied in-place to keep the tying below.
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([SMaRTBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

    def forward(
        self, input_ids: torch.Tensor, mem_states: list | None = None
    ) -> tuple[torch.Tensor, list]:
        B = input_ids.shape[0]
        device = input_ids.device
        dtype = self.embed.weight.dtype

        if mem_states is None:
            mem_states = [block.memory.init_state(B, device, dtype) for block in self.blocks]

        x = self.embed(input_ids)
        new_states: list[MemoryState] = []
        for block, state in zip(self.blocks, mem_states):
            x, state = block(x, state)
            new_states.append(state)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_states


def assert_constant_memory_footprint(model: SMaRT, cfg: ModelConfig) -> None:
    """Regression check: memory parameter count is fixed by config alone,
    never by sequence length (length-independence holds by construction —
    init_weights takes no sequence-length argument — this catches a future
    accidental change that ties memory shape to something length-related).
    """
    mem_cfg = cfg.memory
    theta_param_count = sum(w.numel() for w in model.blocks[0].memory.init_weights)

    if mem_cfg.num_mlp_layers == 1:
        expected = mem_cfg.key_dim * mem_cfg.value_dim
    else:
        expected = mem_cfg.key_dim * mem_cfg.hidden_dim
        expected += (mem_cfg.num_mlp_layers - 2) * mem_cfg.hidden_dim * mem_cfg.hidden_dim
        expected += mem_cfg.hidden_dim * mem_cfg.value_dim

    assert theta_param_count == expected, (
        f"Memory theta param count {theta_param_count} does not match analytically "
        f"derived expected count {expected} from MemoryConfig shape formula — memory "
        f"shape must depend only on key_dim/hidden_dim/value_dim/num_mlp_layers."
    )
