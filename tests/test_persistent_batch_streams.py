import torch

from smrt.config import AttentionConfig, Config, CurriculumStage, MemoryConfig, ModelConfig, TrainConfig
from smrt.data.tokenizer import AGENT_VOCAB_SIZE
from smrt.train import _get_agent_batch


def _agent_cfg(seq_len: int = 8) -> Config:
    model = ModelConfig(
        vocab_size=AGENT_VOCAB_SIZE, d_model=16, num_layers=1, num_persistent_tokens=2,
        mlp_hidden_dim=64,
        attention=AttentionConfig(window_size=16, num_heads=2, num_kv_heads=2, head_dim=8),
        memory=MemoryConfig(key_dim=4, value_dim=4, hidden_dim=8, num_mlp_layers=2,
                             momentum_init=2.0, forget_init=-2.0, lr_init=-2.0, chunk_size=8),
        dropout=0.0,
    )
    train = TrainConfig(
        backend="cpu", base_micro_batch=1, base_grad_accum=1, lr=1e-3, weight_decay=0.0,
        warmup_steps=1, max_steps=10, seq_len=seq_len, checkpoint_every=100,
        checkpoint_dir="/tmp/unused", seed=0,
        batch_source_schedule=["code", "code", "code"],
    )
    curriculum = [CurriculumStage(min_step=0, context_len=seq_len, haystack_filler_tokens=0, needle_depth_bins=1)]
    return Config(model=model, train=train, curriculum=curriculum)


# If this fails: _get_agent_batch is constructing a brand-new stream on
# every call instead of reusing the persistent one passed in via
# `streams` -- the exact bug (found and fixed this session) that made
# every "code"/"text" training step re-read the same tokens from the
# start of the corpus forever, regardless of dataset size.
def test_get_agent_batch_advances_through_persistent_stream_across_calls():
    cfg = _agent_cfg(seq_len=8)
    fake_chunks = iter([torch.full((8,), i, dtype=torch.long) for i in range(5)])
    streams = {"code": fake_chunks}

    batch_1 = _get_agent_batch(cfg, step=0, rng=None, tokenizer=None, micro_batch=1, streams=streams)
    batch_2 = _get_agent_batch(cfg, step=1, rng=None, tokenizer=None, micro_batch=1, streams=streams)

    assert batch_1[0, 0].item() == 0
    assert batch_2[0, 0].item() == 1  # must be the NEXT chunk, not the same one again
