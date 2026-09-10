"""SMaRT training loop: config-driven, resumable, curriculum-mixed
(needle recall examples + pretrain-stream continuation).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from smrt.config import Config, TrainConfig, config_from_dict, load_config
from smrt.data.needle import ByteTokenizer, gpt2_tokenizer, generate_needle_batch
from smrt.data.pretrain import load_pretrain_stream, load_nemotron_cc_stream, synthetic_random_stream
from smrt.data.tokenizer import AGENT_VOCAB_SIZE, agent_tokenizer
from smrt.data.code import load_starcoder_stream
from smrt.data.tool_needle import generate_tool_needle_batch
from smrt.device import resolve_device, pick_batch_schedule
from smrt.model.backbone import SMaRT, assert_constant_memory_footprint
from smrt.optim import build_optimizers

CURRICULUM_NEEDLE_RATIO = 4  # 1 needle-batch per 4 pretrain-batches

# Agent-vocab batch source schedule: index 0 = fact-needle (existing
# needle.py), 1 = tool-needle, 2-3 = code, 4-9 = text.
AGENT_BATCH_SOURCE_SCHEDULE = ("needle", "tool_needle", "code", "code", "text", "text", "text", "text", "text", "text")


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    remaining = max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min((step - cfg.warmup_steps) / remaining, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * 0.1 + (cfg.lr - cfg.lr * 0.1) * cosine


def _active_stage(cfg: Config, step: int):
    stage = cfg.curriculum[0]
    for s in cfg.curriculum:
        if s.min_step <= step:
            stage = s
        else:
            break
    return stage


def _tokenizer_for(cfg: Config):
    if cfg.model.vocab_size == 256:
        return ByteTokenizer()
    if cfg.model.vocab_size == AGENT_VOCAB_SIZE:
        return agent_tokenizer()
    return gpt2_tokenizer()


def _get_agent_batch(cfg: Config, step: int, rng: random.Random, tokenizer, micro_batch: int, streams: dict):
    stage = _active_stage(cfg, step)
    schedule = cfg.train.batch_source_schedule or AGENT_BATCH_SOURCE_SCHEDULE
    source = schedule[step % len(schedule)]
    if source == "needle":
        depth_bin = rng.randrange(stage.needle_depth_bins)
        input_ids, _spans = generate_needle_batch(
            rng,
            micro_batch,
            stage.context_len,
            stage.haystack_filler_tokens,
            depth_bin,
            stage.needle_depth_bins,
            tokenizer,
            cfg.train,
        )
        return input_ids
    if source == "tool_needle":
        depth_bin = rng.randrange(stage.needle_depth_bins)
        input_ids, _spans = generate_tool_needle_batch(
            rng,
            micro_batch,
            stage.context_len,
            stage.haystack_filler_tokens,
            depth_bin,
            stage.needle_depth_bins,
            tokenizer,
            cfg.train,
        )
        return input_ids
    if source == "code":
        rows = [next(streams["code"]) for _ in range(micro_batch)]
        return torch.stack(rows, dim=0)
    # source == "text"
    rows = [next(streams["text"]) for _ in range(micro_batch)]
    return torch.stack(rows, dim=0)


def _get_batch(cfg: Config, step: int, rng: random.Random, tokenizer, micro_batch: int, streams: dict):
    if cfg.model.vocab_size == AGENT_VOCAB_SIZE:
        return _get_agent_batch(cfg, step, rng, tokenizer, micro_batch, streams)
    stage = _active_stage(cfg, step)
    if step % (CURRICULUM_NEEDLE_RATIO + 1) == 0:
        depth_bin = rng.randrange(stage.needle_depth_bins)
        input_ids, _spans = generate_needle_batch(
            rng,
            micro_batch,
            stage.context_len,
            stage.haystack_filler_tokens,
            depth_bin,
            stage.needle_depth_bins,
            tokenizer,
            cfg.train,
        )
        return input_ids
    if cfg.model.vocab_size == 256:
        return synthetic_random_stream(rng, micro_batch, cfg.train.seq_len, cfg.model.vocab_size)
    # Real pretrain stream: pull one batch worth of seq_len chunks.
    rows = [next(streams["text"]) for _ in range(micro_batch)]
    return torch.stack(rows, dim=0)


def train_loop(cfg: Config, resume_path: str | None = None, stop_step: int | None = None) -> dict:
    """Runs the full training loop in-process. Returns a dict of final
    diagnostics (used directly by tests, not just the CLI).

    `stop_step` (exclusive upper bound on the step loop) defaults to
    cfg.train.max_steps; it exists so callers (e.g. the checkpoint-resume
    test) can truncate a run early without changing cfg.train.max_steps,
    which stays fixed for LR-schedule purposes and resume validation.
    """
    device_info = resolve_device(cfg.train.backend)
    micro_batch, grad_accum = pick_batch_schedule(
        device_info.total_memory_bytes, cfg.train.base_micro_batch, cfg.train.base_grad_accum
    )
    torch.manual_seed(cfg.train.seed)
    rng = random.Random(cfg.train.seed)

    model = SMaRT(cfg.model).to(device_info.device)
    assert_constant_memory_footprint(model, cfg.model)

    muon_opt, adamw_opt = build_optimizers(model, cfg.train.lr, cfg.train.weight_decay)

    start_step = 0
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device_info.device)
        saved_cfg = config_from_dict(ckpt["config"])
        if dataclasses.asdict(saved_cfg) != dataclasses.asdict(cfg):
            for section in ("model", "train", "curriculum"):
                if getattr(saved_cfg, section, None) != getattr(cfg, section, None):
                    raise ValueError(f"Resume config mismatch in section '{section}'")
            raise ValueError("Resume config mismatch")
        model.load_state_dict(ckpt["model"])
        muon_opt.load_state_dict(ckpt["muon_optimizer"])
        adamw_opt.load_state_dict(ckpt["adamw_optimizer"])
        rng.setstate(ckpt["rng_state"])
        torch.random.set_rng_state(ckpt["torch_rng_state"])
        start_step = ckpt["step"] + 1

    tokenizer = _tokenizer_for(cfg)
    if cfg.model.vocab_size == AGENT_VOCAB_SIZE:
        assert tokenizer.n_vocab == AGENT_VOCAB_SIZE
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    diag_path = os.path.join(cfg.train.checkpoint_dir, "memory_diagnostics.jsonl")

    autocast_device_type = "cuda" if device_info.backend in ("cuda", "rocm") else (
        "cpu" if device_info.backend in ("cpu", "mps") else device_info.backend
    )

    streams: dict = {}
    if cfg.model.vocab_size == AGENT_VOCAB_SIZE:
        streams["code"] = load_starcoder_stream(cfg.train.seq_len, tokenizer)
        streams["text"] = load_nemotron_cc_stream(cfg.train.seq_len, tokenizer)
    elif cfg.model.vocab_size != 256:
        streams["text"] = load_pretrain_stream(cfg.train.seq_len, tokenizer, token_budget=float("inf"))

    last_loss = None
    end_step = cfg.train.max_steps if stop_step is None else stop_step
    for step in range(start_step, end_step):
        lr = lr_at_step(step, cfg.train)
        for g in muon_opt.param_groups:
            g["lr"] = lr
        for g in adamw_opt.param_groups:
            g["lr"] = lr

        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        step_loss_sum = 0.0
        surprise_vals = []
        for _ in range(grad_accum):
            input_ids = _get_batch(cfg, step, rng, tokenizer, micro_batch, streams).to(device_info.device)
            with torch.autocast(
                device_type=autocast_device_type,
                dtype=device_info.dtype,
                enabled=device_info.dtype == torch.bfloat16,
            ):
                logits, mem_states = model(input_ids, mem_states=None)
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                    input_ids[:, 1:].reshape(-1),
                ) / grad_accum
            loss.backward()
            step_loss_sum += loss.item()
            if not cfg.model.memory.disabled:
                for block, state in zip(model.blocks, mem_states):
                    surprise_vals.append(block.memory.surprise_magnitude(state))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step()
        adamw_opt.step()

        if surprise_vals:
            all_surprise = torch.stack(surprise_vals)
            surprise_mean = all_surprise.mean().item()
            surprise_std = all_surprise.std().item() if all_surprise.numel() > 1 else 0.0
        else:
            surprise_mean = 0.0
            surprise_std = 0.0

        last_loss = step_loss_sum
        with open(diag_path, "a") as f:
            f.write(json.dumps({
                "step": step,
                "loss": last_loss,
                "surprise_mean": surprise_mean,
                "surprise_std": surprise_std,
            }) + "\n")

        is_last = step == cfg.train.max_steps - 1
        if (step + 1) % cfg.train.checkpoint_every == 0 or is_last:
            ckpt_path = os.path.join(cfg.train.checkpoint_dir, f"step_{step + 1}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "muon_optimizer": muon_opt.state_dict(),
                    "adamw_optimizer": adamw_opt.state_dict(),
                    "step": step,
                    "config": dataclasses.asdict(cfg),
                    "rng_state": rng.getstate(),
                    "torch_rng_state": torch.random.get_rng_state(),
                },
                ckpt_path,
            )

    return {"final_loss": last_loss, "final_step": end_step - 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--calibration-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.calibration_steps is not None:
        start = time.time()
        train_loop(cfg, resume_path=args.resume, stop_step=args.calibration_steps)
        elapsed = time.time() - start
        device_info = resolve_device(cfg.train.backend)
        micro_batch, grad_accum = pick_batch_schedule(
            device_info.total_memory_bytes, cfg.train.base_micro_batch, cfg.train.base_grad_accum
        )
        tokens_per_step = micro_batch * grad_accum * cfg.train.seq_len
        steps_per_second = args.calibration_steps / elapsed
        seven_day_seconds = 7 * 24 * 60 * 60 * 0.9  # 10% reserved for restarts/checkpointing overhead
        recommended_max_steps = int(steps_per_second * seven_day_seconds)
        print(
            f"Calibration: {args.calibration_steps} steps in {elapsed:.1f}s "
            f"({tokens_per_step} tokens/step, {steps_per_second:.4f} steps/s). "
            f"Recommended max_steps for a 7-day run: {recommended_max_steps}"
        )
        return

    result = train_loop(cfg, resume_path=args.resume)
    print(f"Training complete. final_step={result['final_step']} final_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
