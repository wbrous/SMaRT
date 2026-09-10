import torch

from smrt.model.memory import NeuralMemory


def _make_memory(cfg):
    return NeuralMemory(cfg.model.memory, cfg.model.d_model)


# If this fails: NeuralMemory.read has nondeterministic ops (e.g. an
# uncontrolled dropout or a batched-matmul path with nondeterministic
# reduction order) breaking reproducibility.
def test_read_is_deterministic(tiny_config, seeded_rng):
    memory = _make_memory(tiny_config)
    B, T = 2, 8
    state = memory.init_state(B, torch.device("cpu"), torch.float32)
    x = torch.randn(B, T, tiny_config.model.d_model)

    out1 = memory.read(state, x)
    out2 = memory.read(state, x)
    assert torch.equal(out1, out2)


# If this fails: a batched-einsum/bmm axis mixup produced the wrong shape
# (e.g. batch and feature dims swapped).
def test_read_output_shape(tiny_config):
    memory = _make_memory(tiny_config)
    B, T = 3, 5
    state = memory.init_state(B, torch.device("cpu"), torch.float32)
    x = torch.randn(B, T, tiny_config.model.d_model)
    out = memory.read(state, x)
    assert out.shape == (B, T, tiny_config.model.memory.value_dim)


# If this fails: the surprise/gate update wiring is broken (e.g. gates
# always emit ~0 so weights never move, or the associative-recall loss
# isn't actually connected to theta's gradient).
def test_update_changes_theta(tiny_config, seeded_rng):
    memory = _make_memory(tiny_config)
    B, T = 2, tiny_config.model.memory.chunk_size
    state = memory.init_state(B, torch.device("cpu"), torch.float32)
    x = torch.randn(B, T, tiny_config.model.d_model)
    new_state = memory.update(state, x)
    assert not torch.equal(new_state.theta[0], state.theta[0])


# If this fails: the surprise-update path silently detaches theta from the
# computation graph, which would break the meta-learning mechanism (the
# initializer / gates would never receive a gradient signal).
def test_update_theta_requires_grad_to_init_weights(tiny_config, seeded_rng):
    memory = _make_memory(tiny_config)
    B, T = 2, tiny_config.model.memory.chunk_size
    state = memory.init_state(B, torch.device("cpu"), torch.float32)
    x = torch.randn(B, T, tiny_config.model.d_model)
    new_state = memory.update(state, x)
    assert new_state.theta[0].requires_grad
    grads = torch.autograd.grad(new_state.theta[0].sum(), memory.init_weights[0], retain_graph=True)
    assert grads[0] is not None


# If this fails: read() mutates the state object in place (e.g. an
# in-place op on state.theta), so calling read twice with the same state
# returns different results.
def test_read_does_not_mutate_state(tiny_config, seeded_rng):
    memory = _make_memory(tiny_config)
    B, T = 2, 8
    state = memory.init_state(B, torch.device("cpu"), torch.float32)
    x = torch.randn(B, T, tiny_config.model.d_model)
    out1 = memory.read(state, x)
    out2 = memory.read(state, x)
    assert torch.equal(out1, out2)
