import pytest

from smrt.config import config_from_dict

_BASE_MODEL = {
    "vocab_size": 256, "d_model": 16, "num_layers": 1, "num_persistent_tokens": 2,
    "mlp_hidden_dim": 64, "dropout": 0.0,
    "attention": {"window_size": 8, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8},
    "memory": {"key_dim": 4, "value_dim": 4, "hidden_dim": 8, "num_mlp_layers": 2,
               "momentum_init": 2.0, "forget_init": -2.0, "lr_init": -2.0, "chunk_size": 8},
}
_TRAIN = {"backend": "cpu", "base_micro_batch": 1, "base_grad_accum": 1, "lr": 1e-3,
          "weight_decay": 0.0, "warmup_steps": 1, "max_steps": 1, "seq_len": 16,
          "checkpoint_every": 1, "checkpoint_dir": "/tmp/x", "seed": 0}


# If this fails: a num_heads/num_kv_heads combination that doesn't evenly
# divide is being silently accepted instead of raising, which would let an
# un-runnable GQA config (enable_gqa requires integer head-group size) reach
# model construction before failing with a confusing error.
def test_num_heads_not_divisible_by_num_kv_heads_raises():
    bad = dict(_BASE_MODEL)
    bad["attention"] = dict(bad["attention"], num_heads=5, num_kv_heads=2)
    with pytest.raises(ValueError, match="num_kv_heads"):
        config_from_dict({"model": bad, "train": _TRAIN})


# If this fails: an odd head_dim (incompatible with RoPE's pairwise
# rotation) is being silently accepted instead of raising at config time.
def test_odd_head_dim_raises():
    bad = dict(_BASE_MODEL)
    bad["attention"] = dict(bad["attention"], head_dim=7)
    with pytest.raises(ValueError, match="head_dim"):
        config_from_dict({"model": bad, "train": _TRAIN})


# If this fails: a valid GQA config (num_kv_heads < num_heads, dividing
# evenly) is being rejected, blocking every legitimate GQA config from
# loading at all.
def test_valid_gqa_config_builds():
    cfg = config_from_dict({"model": _BASE_MODEL, "train": _TRAIN})
    assert cfg.model.attention.num_kv_heads == 2
    assert cfg.model.mlp_hidden_dim == 64
