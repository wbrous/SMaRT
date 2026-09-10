"""Optimizer construction: Moonshot AI's Kimi K2 technical report
(arXiv:2507.20534) trains the model's 2D hidden-layer weight matrices with
the Muon optimizer (Newton-Schulz-orthogonalized momentum updates,
`torch.optim.Muon`) and everything else (embeddings, norm scales, biases,
degenerate single-output gate projections, the persistent-token table)
with AdamW -- matching Muon's own documented guidance that embeddings and
non-hidden-layer parameters should stay on a standard optimizer.

This repo intentionally does NOT layer Moonshot's separate QK-Clip
technique on top: QK-Clip rescales query/key weights post-step to bound
exploding attention logits, and smrt/model/attention.py already applies
RMSNorm to q/k before every attention call (see DESIGN.md), which bounds
attention logit magnitude architecturally. Adding QK-Clip on top of
QK-norm would target the same failure mode twice; Muon alone (without
QK-Clip) is used here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def split_muon_adamw_params(model: nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Splits `model`'s parameters into (muon_params, adamw_params).

    A parameter goes to the Muon group iff it is a genuine 2D hidden-layer
    weight matrix: ndim == 2, more than one output row (excludes the
    degenerate (1, d_model)-shaped momentum/forget/lr gate weights in
    NeuralMemory, which behave like scalar heads, not hidden mixing
    layers), and its name does not contain "embed" (the tied
    embed.weight/lm_head.weight parameter -- weight tying means
    named_parameters() yields it once, under the "embed.weight" name) or
    "persistent" (SMaRTBlock.persistent, an embedding-like learned token
    table, not a matmul weight). Every other parameter (1D norm/bias
    weights, the excluded ones above) goes to the AdamW group.

    Postcondition: every parameter in model.parameters() appears in
    exactly one of the two returned lists.
    """
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        is_hidden_matrix = p.ndim == 2 and p.shape[0] > 1 and "embed" not in name and "persistent" not in name
        (muon_params if is_hidden_matrix else adamw_params).append(p)
    return muon_params, adamw_params


def build_optimizers(model: nn.Module, lr: float, weight_decay: float) -> tuple[torch.optim.Muon, torch.optim.AdamW]:
    """Builds the (muon_optimizer, adamw_optimizer) pair for `model`, both
    driven by the same `lr`/`weight_decay` scalars. Muon is constructed
    with adjust_lr_fn="match_rms_adamw" (Moonshot's RMS-matching learning
    rate adjustment from the Kimi K2 technical report), which is
    specifically designed so a single lr/weight_decay pair already tuned
    for AdamW can be reused unchanged for Muon -- this is why no separate
    muon_lr config field exists.
    """
    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_opt = torch.optim.Muon(muon_params, lr=lr, weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
    adamw_opt = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay)
    return muon_opt, adamw_opt
