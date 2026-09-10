import random

from smrt.config import Config, CurriculumStage, ModelConfig, AttentionConfig, MemoryConfig, TrainConfig
from smrt.data.tokenizer import AGENT_VOCAB_SIZE, AgentTokenizer
from smrt.train import AGENT_BATCH_SOURCE_SCHEDULE, _get_batch, _tokenizer_for


def _agent_cfg(seq_len: int = 2048) -> Config:
    model = ModelConfig(
        vocab_size=AGENT_VOCAB_SIZE, d_model=16, num_layers=1, num_persistent_tokens=2,
        mlp_hidden_dim=64,
        attention=AttentionConfig(window_size=16, num_heads=2, num_kv_heads=2, head_dim=8),
        memory=MemoryConfig(key_dim=4, value_dim=4, hidden_dim=8, num_mlp_layers=2,
                             momentum_init=2.0, forget_init=-2.0, lr_init=-2.0, chunk_size=8),
        dropout=0.0,
    )
    train = TrainConfig(
        backend="cpu", base_micro_batch=2, base_grad_accum=1, lr=1e-3, weight_decay=0.0,
        warmup_steps=1, max_steps=10, seq_len=seq_len, checkpoint_every=100,
        checkpoint_dir="/tmp/unused", seed=0,
    )
    curriculum = [CurriculumStage(min_step=0, context_len=seq_len, haystack_filler_tokens=int(seq_len * 0.8), needle_depth_bins=3)]
    return Config(model=model, train=train, curriculum=curriculum)


# If this fails: a model config with the agent vocab size stopped getting
# routed to the AgentTokenizer (e.g. falling through to the gpt2 branch),
# which would desync every special-token id the tool/code pipelines rely on.
def test_tokenizer_for_agent_vocab_returns_agent_tokenizer():
    cfg = _agent_cfg()
    tokenizer = _tokenizer_for(cfg)
    assert isinstance(tokenizer, AgentTokenizer)
    assert tokenizer.n_vocab == AGENT_VOCAB_SIZE


# If this fails: the offline "needle" source in AGENT_BATCH_SOURCE_SCHEDULE
# (step 0) stopped producing a (base_micro_batch, seq_len)-shaped batch of
# valid token ids.
def test_get_batch_needle_source_shape():
    cfg = _agent_cfg()
    tokenizer = _tokenizer_for(cfg)
    rng = random.Random(0)
    assert AGENT_BATCH_SOURCE_SCHEDULE[0] == "needle"
    batch = _get_batch(cfg, step=0, rng=rng, tokenizer=tokenizer, micro_batch=cfg.train.base_micro_batch, streams={})
    assert batch.shape == (2, 2048)
    assert batch.dtype.__str__() == "torch.int64"


# If this fails: the offline "tool_needle" source (step 1) stopped
# producing a correctly shaped batch — this is the path that specifically
# exercises the tool-call recall curriculum, not just prose recall.
def test_get_batch_tool_needle_source_shape():
    cfg = _agent_cfg()
    tokenizer = _tokenizer_for(cfg)
    rng = random.Random(0)
    assert AGENT_BATCH_SOURCE_SCHEDULE[1] == "tool_needle"
    batch = _get_batch(cfg, step=1, rng=rng, tokenizer=tokenizer, micro_batch=cfg.train.base_micro_batch, streams={})
    assert batch.shape == (2, 2048)


# If this fails: the step-to-source schedule no longer cycles with period
# 10 (e.g. an off-by-one in the modulo indexing), silently changing the
# mix ratio between needle/tool_needle/code/text sources.
def test_batch_source_schedule_cycles_with_period_ten():
    assert len(AGENT_BATCH_SOURCE_SCHEDULE) == 10
    assert AGENT_BATCH_SOURCE_SCHEDULE[0] == AGENT_BATCH_SOURCE_SCHEDULE[10 % 10]
    assert AGENT_BATCH_SOURCE_SCHEDULE.count("code") == 2
    assert AGENT_BATCH_SOURCE_SCHEDULE.count("text") == 6
