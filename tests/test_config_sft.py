import dataclasses

import pytest

from smrt.config import config_from_dict, load_config


_MODEL_DICT = {
    "vocab_size": 256,
    "d_model": 8,
    "num_layers": 1,
    "num_persistent_tokens": 2,
    "mlp_hidden_dim": 64,
    "dropout": 0.0,
    "attention": {"window_size": 8, "num_heads": 2, "num_kv_heads": 2, "head_dim": 4},
    "memory": {
        "key_dim": 4, "value_dim": 4, "hidden_dim": 8, "num_mlp_layers": 2,
        "momentum_init": 2.0, "forget_init": -2.0, "lr_init": -2.0, "chunk_size": 8,
    },
}

_TRAIN_DICT = {
    "backend": "cpu", "base_micro_batch": 1, "base_grad_accum": 1,
    "lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 1, "max_steps": 1,
    "seq_len": 16, "checkpoint_every": 1, "checkpoint_dir": "/tmp/x", "seed": 0,
}

_SFT_DICT = {
    "backend": "cpu", "base_micro_batch": 1, "base_grad_accum": 1,
    "lr": 1e-4, "weight_decay": 0.0, "warmup_steps": 1, "max_steps": 1,
    "seq_len": 16, "checkpoint_every": 1, "checkpoint_dir": "/tmp/y", "seed": 0,
    "init_checkpoint": "/tmp/does_not_need_to_exist.pt",
}


# If this fails: a config file with no sft: key stops defaulting to
# sft=None (e.g. a required-field regression), breaking every pre-existing
# config (tiny_cpu.yaml, base_50m.yaml) that doesn't set one.
def test_config_without_sft_key_defaults_to_none():
    cfg = config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT})
    assert cfg.sft is None


# If this fails: a present sft: block isn't actually being parsed into an
# SFTConfig (e.g. silently dropped, or the wrong dict handed to the
# builder).
def test_config_with_sft_key_builds_sft_config():
    cfg = config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "sft": _SFT_DICT})
    assert cfg.sft is not None
    assert cfg.sft.init_checkpoint == "/tmp/does_not_need_to_exist.pt"
    assert cfg.sft.seq_len == 16


# If this fails: a missing required sft field (e.g. init_checkpoint) is
# silently defaulted instead of raising a named error, hiding a
# misconfigured SFT run until it crashes deep inside sft_loop.
def test_sft_missing_required_key_raises_named_error():
    bad = dict(_SFT_DICT)
    del bad["init_checkpoint"]
    with pytest.raises(ValueError, match="init_checkpoint"):
        config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "sft": bad})


# If this fails: an unexpected key in the sft: block is silently ignored
# instead of raising, letting a typo'd config field (e.g. "seq_lne") pass
# through unnoticed.
def test_sft_extra_key_raises_named_error():
    bad = dict(_SFT_DICT)
    bad["not_a_real_field"] = 1
    with pytest.raises(ValueError, match="not_a_real_field"):
        config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "sft": bad})


# If this fails: a Config with sft=None can't be round-tripped through
# dataclasses.asdict -> config_from_dict (the exact path smrt.train's
# checkpoint resume takes) — a regression of the "sft": None-in-saved-dict
# bug this session found and fixed.
def test_config_with_none_sft_roundtrips_through_asdict():
    cfg = config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT})
    rebuilt = config_from_dict(dataclasses.asdict(cfg))
    assert rebuilt.sft is None
    assert rebuilt == cfg


# If this fails: pre-existing production configs (which predate SFTConfig)
# stopped loading after the schema change.
@pytest.mark.parametrize("path", ["configs/tiny_cpu.yaml", "configs/base_50m.yaml"])
def test_existing_configs_still_load_with_no_sft(path):
    cfg = load_config(path)
    assert cfg.sft is None
