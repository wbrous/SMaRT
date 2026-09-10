import pytest

from smrt.config import config_from_dict

_MODEL_DICT = {
    "vocab_size": 256, "d_model": 8, "num_layers": 1, "num_persistent_tokens": 2,
    "mlp_hidden_dim": 64, "dropout": 0.0,
    "attention": {"window_size": 8, "num_heads": 2, "num_kv_heads": 2, "head_dim": 4},
    "memory": {"key_dim": 4, "value_dim": 4, "hidden_dim": 8, "num_mlp_layers": 2,
               "momentum_init": 2.0, "forget_init": -2.0, "lr_init": -2.0, "chunk_size": 8},
}
_TRAIN_DICT = {
    "backend": "cpu", "base_micro_batch": 1, "base_grad_accum": 1, "lr": 1e-3,
    "weight_decay": 0.0, "warmup_steps": 1, "max_steps": 1, "seq_len": 16,
    "checkpoint_every": 1, "checkpoint_dir": "/tmp/x", "seed": 0,
}


# If this fails: a config with no batch_source_schedule key stopped
# defaulting to None, breaking every pre-existing agent config that
# relies on smrt.train.AGENT_BATCH_SOURCE_SCHEDULE as the fallback.
def test_missing_batch_source_schedule_defaults_to_none():
    cfg = config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT})
    assert cfg.train.batch_source_schedule is None


# If this fails: an explicit batch_source_schedule list stopped being
# parsed into TrainConfig -- the whole point of making the per-model-size
# code ratio config-driven instead of hardcoded.
def test_explicit_batch_source_schedule_parses():
    train = dict(_TRAIN_DICT, batch_source_schedule=["needle", "code", "code", "text"])
    cfg = config_from_dict({"model": _MODEL_DICT, "train": train})
    assert cfg.train.batch_source_schedule == ["needle", "code", "code", "text"]


# If this fails: an unrecognized source name in batch_source_schedule
# (e.g. a typo) is silently accepted instead of raising a named error,
# which would surface only as a confusing KeyError deep inside
# _get_agent_batch at training time.
def test_invalid_batch_source_name_raises_named_error():
    train = dict(_TRAIN_DICT, batch_source_schedule=["needle", "not_a_real_source"])
    with pytest.raises(ValueError, match="not_a_real_source"):
        config_from_dict({"model": _MODEL_DICT, "train": train})


# If this fails: an empty batch_source_schedule list is silently
# accepted, which would cause a ZeroDivisionError/IndexError the moment
# _get_agent_batch computes `step % len(schedule)`.
def test_empty_batch_source_schedule_raises_named_error():
    train = dict(_TRAIN_DICT, batch_source_schedule=[])
    with pytest.raises(ValueError, match="non-empty"):
        config_from_dict({"model": _MODEL_DICT, "train": train})
