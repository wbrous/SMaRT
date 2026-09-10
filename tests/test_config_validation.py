import pytest

from smrt.config import config_from_dict

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


def _curriculum_stage(min_step, context_len=8, filler_tokens=0, depth_bins=1):
    return {
        "min_step": min_step,
        "context_len": context_len,
        "haystack_filler_tokens": filler_tokens,
        "needle_depth_bins": depth_bins,
    }


# If this fails: curriculum stages out of ascending min_step order stopped
# being rejected -- a mis-ordered curriculum would silently apply the
# wrong stage at a given step (smrt.train._active_stage assumes ascending
# order and returns the last stage whose min_step <= step).
def test_curriculum_out_of_order_min_step_raises():
    bad_curriculum = [_curriculum_stage(min_step=10), _curriculum_stage(min_step=5)]
    with pytest.raises(ValueError, match="ascending"):
        config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "curriculum": bad_curriculum})


# If this fails: a curriculum stage whose context_len exceeds
# train.seq_len stopped being rejected -- such a stage would generate
# batches longer than the fixed seq_len every downstream shape assertion
# (smrt.data.batch.validate_batch) depends on.
def test_curriculum_context_len_exceeding_seq_len_raises():
    bad_curriculum = [_curriculum_stage(min_step=0, context_len=_TRAIN_DICT["seq_len"] + 1)]
    with pytest.raises(ValueError, match="context_len"):
        config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "curriculum": bad_curriculum})


# If this fails: a valid ascending curriculum with context_len within
# train.seq_len stopped being accepted -- every legitimate curriculum
# would be rejected.
def test_valid_ascending_curriculum_builds():
    good_curriculum = [_curriculum_stage(min_step=0, context_len=8), _curriculum_stage(min_step=5, context_len=16)]
    cfg = config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "curriculum": good_curriculum})
    assert [s.min_step for s in cfg.curriculum] == [0, 5]


# If this fails: model.memory.chunk_size exceeding
# model.attention.window_size stopped being rejected -- NeuralMemory's
# chunking contract (block.py) assumes a memory chunk never spans more
# than one attention window.
def test_memory_chunk_size_exceeding_window_size_raises():
    bad_model = dict(_MODEL_DICT)
    bad_model["memory"] = dict(bad_model["memory"], chunk_size=bad_model["attention"]["window_size"] + 1)
    with pytest.raises(ValueError, match="chunk_size"):
        config_from_dict({"model": bad_model, "train": _TRAIN_DICT})


# If this fails: a config dict missing the required top-level "model" key
# stopped raising a named error and instead raised a generic KeyError (or
# silently built a half-populated Config) deep inside _build_model.
def test_missing_root_model_key_raises_named_error():
    with pytest.raises(ValueError, match="model"):
        config_from_dict({"train": _TRAIN_DICT})


# If this fails: a config dict missing the required top-level "train" key
# stopped raising a named error.
def test_missing_root_train_key_raises_named_error():
    with pytest.raises(ValueError, match="train"):
        config_from_dict({"model": _MODEL_DICT})


# If this fails: an unexpected top-level key (e.g. a typo'd section name)
# stopped being rejected, letting a misconfigured YAML file load with a
# silently-ignored section.
def test_extra_root_key_raises_named_error():
    with pytest.raises(ValueError, match="not_a_real_section"):
        config_from_dict({"model": _MODEL_DICT, "train": _TRAIN_DICT, "not_a_real_section": {}})
