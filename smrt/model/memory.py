"""Titans-style ("Memory as Context") neural long-term memory.

See DESIGN.md for the full prose derivation. Summary of the contract:

- `theta` (the memory MLP's weights) is per-example state carried across a
  sequence, NOT an nn.Parameter trained directly by the outer optimizer. It
  starts from a learned initializer (`init_weights`) and evolves via a
  momentum-decayed gradient-descent rule on its own associative-recall loss.
- `read(state, x)` is a pure function: given a state and input, it computes
  the memory MLP's forward pass and returns a fixed-size (B, T, value_dim)
  tensor. It never mutates `state`.
- `update(state, x)` computes a NEW state functionally (never mutates the
  input state in place) via:
    ell = || M_theta(k) - v ||^2               (associative-recall loss)
    grad_theta = d(ell)/d(theta), create_graph=True (differentiable grad)
    surprise_new = eta * surprise - theta_lr * grad_theta
    theta_new    = (1 - alpha) * theta + surprise_new
  where eta (momentum gate), alpha (forget gate), theta_lr (update-rate
  gate) are all per-example scalars produced by learned Linear+sigmoid
  gates conditioned on the current chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from smrt.config import MemoryConfig


@dataclass
class MemoryState:
    theta: list  # list[Tensor], per-layer weights, shape (B, in, out)
    surprise: list  # list[Tensor], same shapes as theta


def _mlp_forward(theta: list, x: torch.Tensor) -> torch.Tensor:
    """Batched MLP forward: x is (B, T, in0); each theta[i] is (B, in_i, out_i)."""
    h = x
    n = len(theta)
    for i, w in enumerate(theta):
        h = torch.bmm(h, w)
        if i < n - 1:
            h = F.silu(h)
    return h


class NeuralMemory(nn.Module):
    def __init__(self, cfg: MemoryConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.to_key = nn.Linear(d_model, cfg.key_dim)
        self.to_value = nn.Linear(d_model, cfg.value_dim)
        self.to_query = nn.Linear(d_model, cfg.key_dim)

        shapes = self._layer_shapes()
        self.init_weights = nn.ParameterList(
            [nn.Parameter(self._xavier(in_d, out_d)) for in_d, out_d in shapes]
        )

        self.momentum_gate = nn.Linear(d_model, 1)
        self.forget_gate = nn.Linear(d_model, 1)
        self.lr_gate = nn.Linear(d_model, 1)
        nn.init.constant_(self.momentum_gate.bias, cfg.momentum_init)
        nn.init.constant_(self.forget_gate.bias, cfg.forget_init)
        nn.init.constant_(self.lr_gate.bias, cfg.lr_init)

    def _layer_shapes(self) -> list[tuple[int, int]]:
        cfg = self.cfg
        if cfg.num_mlp_layers == 1:
            return [(cfg.key_dim, cfg.value_dim)]
        shapes = [(cfg.key_dim, cfg.hidden_dim)]
        for _ in range(cfg.num_mlp_layers - 2):
            shapes.append((cfg.hidden_dim, cfg.hidden_dim))
        shapes.append((cfg.hidden_dim, cfg.value_dim))
        return shapes

    @staticmethod
    def _xavier(in_d: int, out_d: int) -> torch.Tensor:
        w = torch.empty(in_d, out_d)
        nn.init.xavier_uniform_(w)
        return w

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemoryState:
        theta = [w.to(device=device, dtype=dtype).expand(batch_size, -1, -1).clone() for w in self.init_weights]
        surprise = [torch.zeros_like(w) for w in theta]
        return MemoryState(theta=theta, surprise=surprise)

    def read(self, state: MemoryState, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.disabled:
            B, T, _ = x.shape
            return torch.zeros(B, T, self.cfg.value_dim, device=x.device, dtype=x.dtype)
        q = self.to_query(x)
        return _mlp_forward(state.theta, q)

    def update(self, state: MemoryState, x: torch.Tensor) -> MemoryState:
        if self.cfg.disabled:
            return state

        k = self.to_key(x)
        v = self.to_value(x)

        pred = _mlp_forward(state.theta, k)
        ell = ((pred - v) ** 2).mean()
        grads = torch.autograd.grad(ell, state.theta, create_graph=True)

        eta = torch.sigmoid(self.momentum_gate(x).mean(dim=1)).unsqueeze(-1)  # (B,1,1)
        alpha = torch.sigmoid(self.forget_gate(x).mean(dim=1)).unsqueeze(-1)
        theta_lr = torch.sigmoid(self.lr_gate(x).mean(dim=1)).unsqueeze(-1)

        new_theta = []
        new_surprise = []
        for w, s, g in zip(state.theta, state.surprise, grads):
            s_new = eta * s - theta_lr * g
            w_new = (1 - alpha) * w + s_new
            new_theta.append(w_new)
            new_surprise.append(s_new)

        return MemoryState(theta=new_theta, surprise=new_surprise)

    def surprise_magnitude(self, state: MemoryState) -> torch.Tensor:
        norms = torch.stack([s.detach().flatten(1).norm(dim=1) for s in state.surprise])
        return norms.mean(dim=0)
