import torch

from smrt.model.backbone import SMaRT
from smrt.optim import build_optimizers, split_muon_adamw_params


# If this fails: a genuine hidden-layer weight matrix (e.g. an attention
# q_proj) stopped being routed to the Muon optimizer group, or a
# non-hidden parameter (embedding, norm weight, degenerate gate) leaked
# into it -- the exact split this optimizer setup depends on.
def test_split_routes_known_parameters_correctly(tiny_config):
    model = SMaRT(tiny_config.model)
    muon_params, adamw_params = split_muon_adamw_params(model)
    muon_ids = {id(p) for p in muon_params}
    adamw_ids = {id(p) for p in adamw_params}

    assert id(model.blocks[0].attn.q_proj.weight) in muon_ids
    assert id(model.blocks[0].mlp.gate_proj.weight) in muon_ids
    assert id(model.blocks[0].memory.to_key.weight) in muon_ids

    assert id(model.embed.weight) in adamw_ids
    assert id(model.blocks[0].ln1.weight) in adamw_ids
    assert id(model.blocks[0].persistent) in adamw_ids
    assert id(model.blocks[0].memory.momentum_gate.weight) in adamw_ids


# If this fails: the split is dropping or double-counting parameters --
# every parameter in the model must land in exactly one optimizer group,
# or some weights would silently never receive a gradient update.
def test_split_partitions_every_parameter_exactly_once(tiny_config):
    model = SMaRT(tiny_config.model)
    muon_params, adamw_params = split_muon_adamw_params(model)
    all_ids = sorted(id(p) for p in muon_params) + sorted(id(p) for p in adamw_params)
    model_ids = sorted(id(p) for p in model.parameters())
    assert sorted(all_ids) == model_ids


# If this fails: build_optimizers wired the Muon/AdamW groups to
# optimizers that don't actually update weights on .step() -- a dead
# optimizer that silently leaves the model at its random init.
def test_build_optimizers_step_changes_muon_group_weights(tiny_config):
    model = SMaRT(tiny_config.model)
    muon_opt, adamw_opt = build_optimizers(model, lr=0.1, weight_decay=0.0)
    target_weight = model.blocks[0].attn.q_proj.weight
    before = target_weight.detach().clone()

    x = torch.randint(0, tiny_config.model.vocab_size, (1, 32))
    logits, _ = model(x, mem_states=None)
    loss = logits.float().pow(2).mean()
    muon_opt.zero_grad()
    adamw_opt.zero_grad()
    loss.backward()
    muon_opt.step()
    adamw_opt.step()

    assert not torch.allclose(before, target_weight.detach())
