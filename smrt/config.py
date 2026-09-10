"""Config schema: frozen dataclasses loaded/validated from YAML.

Explicit field-by-field construction (not a generic recursive dataclass-
from-dict library) so a bad/missing key error names the exact offending
field instead of a generic type-mismatch traceback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import yaml


@dataclass(frozen=True)
class MemoryConfig:
    key_dim: int
    value_dim: int
    hidden_dim: int
    num_mlp_layers: int
    momentum_init: float
    forget_init: float
    lr_init: float
    chunk_size: int
    disabled: bool = False


@dataclass(frozen=True)
class AttentionConfig:
    window_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    num_layers: int
    num_persistent_tokens: int
    mlp_hidden_dim: int
    attention: AttentionConfig
    memory: MemoryConfig
    dropout: float


@dataclass(frozen=True)
class TrainConfig:
    backend: Literal["auto", "cuda", "rocm", "cpu", "mps"]
    base_micro_batch: int
    base_grad_accum: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    seq_len: int
    checkpoint_every: int
    checkpoint_dir: str
    seed: int
    batch_source_schedule: list[str] | None = None


@dataclass(frozen=True)
class SFTConfig:
    backend: Literal["auto", "cuda", "rocm", "cpu", "mps"]
    base_micro_batch: int
    base_grad_accum: int
    lr: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    seq_len: int
    checkpoint_every: int
    checkpoint_dir: str
    seed: int
    init_checkpoint: str  # path to a pretrained SMaRT checkpoint (from smrt.train) to fine-tune from


@dataclass(frozen=True)
class CurriculumStage:
    min_step: int
    context_len: int
    haystack_filler_tokens: int
    needle_depth_bins: int


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    train: TrainConfig
    curriculum: list = field(default_factory=list)
    sft: SFTConfig | None = None


def _require(d: dict, key: str, section: str) -> object:
    if key not in d:
        raise ValueError(f"Missing required config key '{key}' in section '{section}'")
    return d[key]


def _check_no_extra(d: dict, allowed: set, section: str) -> None:
    extra = set(d.keys()) - allowed
    if extra:
        raise ValueError(f"Unexpected config key(s) {sorted(extra)} in section '{section}'")


def _build_memory(d: dict) -> MemoryConfig:
    allowed = {
        "key_dim",
        "value_dim",
        "hidden_dim",
        "num_mlp_layers",
        "momentum_init",
        "forget_init",
        "lr_init",
        "chunk_size",
        "disabled",
    }
    _check_no_extra(d, allowed, "model.memory")
    return MemoryConfig(
        key_dim=_require(d, "key_dim", "model.memory"),
        value_dim=_require(d, "value_dim", "model.memory"),
        hidden_dim=_require(d, "hidden_dim", "model.memory"),
        num_mlp_layers=_require(d, "num_mlp_layers", "model.memory"),
        momentum_init=_require(d, "momentum_init", "model.memory"),
        forget_init=_require(d, "forget_init", "model.memory"),
        lr_init=_require(d, "lr_init", "model.memory"),
        chunk_size=_require(d, "chunk_size", "model.memory"),
        disabled=d.get("disabled", False),
    )


def _build_attention(d: dict) -> AttentionConfig:
    allowed = {"window_size", "num_heads", "num_kv_heads", "head_dim"}
    _check_no_extra(d, allowed, "model.attention")
    return AttentionConfig(
        window_size=_require(d, "window_size", "model.attention"),
        num_heads=_require(d, "num_heads", "model.attention"),
        num_kv_heads=_require(d, "num_kv_heads", "model.attention"),
        head_dim=_require(d, "head_dim", "model.attention"),
    )


def _build_model(d: dict) -> ModelConfig:
    allowed = {
        "vocab_size",
        "d_model",
        "num_layers",
        "num_persistent_tokens",
        "mlp_hidden_dim",
        "attention",
        "memory",
        "dropout",
    }
    _check_no_extra(d, allowed, "model")
    return ModelConfig(
        vocab_size=_require(d, "vocab_size", "model"),
        d_model=_require(d, "d_model", "model"),
        num_layers=_require(d, "num_layers", "model"),
        num_persistent_tokens=_require(d, "num_persistent_tokens", "model"),
        mlp_hidden_dim=_require(d, "mlp_hidden_dim", "model"),
        attention=_build_attention(_require(d, "attention", "model")),
        memory=_build_memory(_require(d, "memory", "model")),
        dropout=_require(d, "dropout", "model"),
    )


_VALID_BATCH_SOURCES = {"needle", "tool_needle", "code", "text"}


def _build_batch_source_schedule(items: list) -> list[str]:
    if not items:
        raise ValueError("train.batch_source_schedule must be a non-empty list when provided")
    for i, item in enumerate(items):
        if item not in _VALID_BATCH_SOURCES:
            raise ValueError(
                f"train.batch_source_schedule[{i}] = {item!r} is not one of {sorted(_VALID_BATCH_SOURCES)}"
            )
    return list(items)


def _build_train(d: dict) -> TrainConfig:
    allowed = {
        "backend",
        "base_micro_batch",
        "base_grad_accum",
        "lr",
        "weight_decay",
        "warmup_steps",
        "max_steps",
        "seq_len",
        "checkpoint_every",
        "checkpoint_dir",
        "seed",
        "batch_source_schedule",
    }
    _check_no_extra(d, allowed, "train")
    raw_schedule = d.get("batch_source_schedule")
    return TrainConfig(
        backend=d.get("backend", "auto"),
        base_micro_batch=_require(d, "base_micro_batch", "train"),
        base_grad_accum=_require(d, "base_grad_accum", "train"),
        lr=_require(d, "lr", "train"),
        weight_decay=_require(d, "weight_decay", "train"),
        warmup_steps=_require(d, "warmup_steps", "train"),
        max_steps=_require(d, "max_steps", "train"),
        seq_len=_require(d, "seq_len", "train"),
        checkpoint_every=_require(d, "checkpoint_every", "train"),
        checkpoint_dir=_require(d, "checkpoint_dir", "train"),
        seed=_require(d, "seed", "train"),
        batch_source_schedule=_build_batch_source_schedule(raw_schedule) if raw_schedule is not None else None,
    )


def _build_sft(d: dict) -> SFTConfig:
    allowed = {
        "backend",
        "base_micro_batch",
        "base_grad_accum",
        "lr",
        "weight_decay",
        "warmup_steps",
        "max_steps",
        "seq_len",
        "checkpoint_every",
        "checkpoint_dir",
        "seed",
        "init_checkpoint",
    }
    _check_no_extra(d, allowed, "sft")
    return SFTConfig(
        backend=d.get("backend", "auto"),
        base_micro_batch=_require(d, "base_micro_batch", "sft"),
        base_grad_accum=_require(d, "base_grad_accum", "sft"),
        lr=_require(d, "lr", "sft"),
        weight_decay=_require(d, "weight_decay", "sft"),
        warmup_steps=_require(d, "warmup_steps", "sft"),
        max_steps=_require(d, "max_steps", "sft"),
        seq_len=_require(d, "seq_len", "sft"),
        checkpoint_every=_require(d, "checkpoint_every", "sft"),
        checkpoint_dir=_require(d, "checkpoint_dir", "sft"),
        seed=_require(d, "seed", "sft"),
        init_checkpoint=_require(d, "init_checkpoint", "sft"),
    )


def _build_curriculum(items: list) -> list[CurriculumStage]:
    stages = []
    allowed = {"min_step", "context_len", "haystack_filler_tokens", "needle_depth_bins"}
    for i, item in enumerate(items):
        _check_no_extra(item, allowed, f"curriculum[{i}]")
        stages.append(
            CurriculumStage(
                min_step=_require(item, "min_step", f"curriculum[{i}]"),
                context_len=_require(item, "context_len", f"curriculum[{i}]"),
                haystack_filler_tokens=_require(item, "haystack_filler_tokens", f"curriculum[{i}]"),
                needle_depth_bins=_require(item, "needle_depth_bins", f"curriculum[{i}]"),
            )
        )
    return stages


def _validate(cfg: Config) -> None:
    steps = [s.min_step for s in cfg.curriculum]
    if steps != sorted(steps):
        raise ValueError("curriculum stages must be sorted ascending by min_step")
    for i, stage in enumerate(cfg.curriculum):
        if stage.context_len > cfg.train.seq_len:
            raise ValueError(
                f"curriculum[{i}].context_len ({stage.context_len}) exceeds train.seq_len ({cfg.train.seq_len})"
            )
    if cfg.model.memory.chunk_size > cfg.model.attention.window_size:
        raise ValueError(
            f"model.memory.chunk_size ({cfg.model.memory.chunk_size}) exceeds "
            f"model.attention.window_size ({cfg.model.attention.window_size})"
        )
    if cfg.model.attention.num_heads % cfg.model.attention.num_kv_heads != 0:
        raise ValueError(
            f"model.attention.num_heads ({cfg.model.attention.num_heads}) must be divisible by "
            f"model.attention.num_kv_heads ({cfg.model.attention.num_kv_heads})"
        )
    if cfg.model.attention.head_dim % 2 != 0:
        raise ValueError(f"model.attention.head_dim ({cfg.model.attention.head_dim}) must be even for RoPE")


def config_from_dict(d: dict) -> Config:
    """Build a Config from a plain nested dict (e.g. a checkpoint's saved config)."""
    allowed = {"model", "train", "curriculum", "sft"}
    _check_no_extra(d, allowed, "root")
    cfg = Config(
        model=_build_model(_require(d, "model", "root")),
        train=_build_train(_require(d, "train", "root")),
        curriculum=_build_curriculum(d.get("curriculum", [])),
        sft=_build_sft(d["sft"]) if d.get("sft") is not None else None,
    )
    _validate(cfg)
    return cfg


def load_config(path: str) -> Config:
    """Load and validate a Config from a YAML file.

    Preconditions: `path` points to a YAML file matching the Config schema.
    Postconditions: raises ValueError naming the offending key on any
    missing/extra field or cross-field validation failure; otherwise returns
    a fully validated, immutable Config.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return config_from_dict(raw)
