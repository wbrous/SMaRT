"""SMaRT SFT loop: fine-tunes a pretrained checkpoint on tool-call
trajectories with per-token loss masking (loss only on assistant turns).

Non-goal: resuming an interrupted SFT run is out of scope for this plan —
sft_loop always initializes fresh optimizer state from cfg.sft.init_checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import torch
import torch.nn.functional as F

from smrt.config import Config, SFTConfig, load_config
from smrt.data.code_repair import load_code_repair_stream
from smrt.data.general_instruction import load_general_instruction_stream
from smrt.data.tokenizer import agent_tokenizer
from smrt.data.toolcalls import load_toolcall_sft_stream
from smrt.data.web_search_bias import load_web_search_bias_stream
from smrt.device import resolve_device
from smrt.model.backbone import SMaRT, assert_constant_memory_footprint
from smrt.optim import build_optimizers
from smrt.train import lr_at_step

# SFT source schedule: 3/10 tool-call trajectories (glaive), 2/10
# web-search-bias (knowledge-boundary questions -> forced web_search
# call), 2/10 Python syntax repair, 3/10 general instruction-following.
SFT_SOURCE_SCHEDULE = (
    "toolcall", "toolcall", "toolcall",
    "web_search_bias", "web_search_bias",
    "code_repair", "code_repair",
    "general_instruction", "general_instruction", "general_instruction",
)


def sft_loop(cfg: Config, stop_step: int | None = None) -> dict:
    """Runs the full SFT loop in-process. Returns a dict of final
    diagnostics, mirroring smrt.train.train_loop's return shape.
    """
    assert cfg.sft is not None, "sft_loop requires cfg.sft to be set"
    device_info = resolve_device(cfg.sft.backend)
    torch.manual_seed(cfg.sft.seed)
    tokenizer = agent_tokenizer()

    model = SMaRT(cfg.model).to(device_info.device)
    assert_constant_memory_footprint(model, cfg.model)

    ckpt = torch.load(cfg.sft.init_checkpoint, map_location=device_info.device)
    model.load_state_dict(ckpt["model"])

    muon_opt, adamw_opt = build_optimizers(model, cfg.sft.lr, cfg.sft.weight_decay)
    streams = {
        "toolcall": load_toolcall_sft_stream(cfg.sft.seq_len, tokenizer),
        "web_search_bias": load_web_search_bias_stream(cfg.sft.seq_len, tokenizer, seed=cfg.sft.seed),
        "code_repair": load_code_repair_stream(cfg.sft.seq_len, tokenizer, seed=cfg.sft.seed),
        "general_instruction": load_general_instruction_stream(cfg.sft.seq_len, tokenizer),
    }

    os.makedirs(cfg.sft.checkpoint_dir, exist_ok=True)
    diag_path = os.path.join(cfg.sft.checkpoint_dir, "sft_diagnostics.jsonl")

    autocast_device_type = "cuda" if device_info.backend in ("cuda", "rocm") else (
        "cpu" if device_info.backend in ("cpu", "mps") else device_info.backend
    )

    last_loss = None
    end_step = cfg.sft.max_steps if stop_step is None else stop_step
    for step in range(end_step):
        lr = lr_at_step(step, cfg.sft)
        for g in muon_opt.param_groups:
            g["lr"] = lr
        for g in adamw_opt.param_groups:
            g["lr"] = lr

        ids_rows = []
        mask_rows = []
        source = SFT_SOURCE_SCHEDULE[step % len(SFT_SOURCE_SCHEDULE)]
        for _ in range(cfg.sft.base_micro_batch):
            ids, mask = next(streams[source])
            ids_rows.append(ids)
            mask_rows.append(mask)
        ids = torch.stack(ids_rows, dim=0).to(device_info.device)
        mask = torch.stack(mask_rows, dim=0).to(device_info.device)

        with torch.autocast(
            device_type=autocast_device_type,
            dtype=device_info.dtype,
            enabled=device_info.dtype == torch.bfloat16,
        ):
            logits, _ = model(ids, mem_states=None)
            target = ids[:, 1:].clone()
            target[mask[:, 1:] == 0] = -100  # ignore_index — never train on masked positions
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=-100,
            )

        muon_opt.zero_grad()
        adamw_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        muon_opt.step()
        adamw_opt.step()

        last_loss = loss.item()
        with open(diag_path, "a") as f:
            f.write(json.dumps({"step": step, "loss": last_loss}) + "\n")

        is_last = step == cfg.sft.max_steps - 1
        if (step + 1) % cfg.sft.checkpoint_every == 0 or is_last:
            ckpt_path = os.path.join(cfg.sft.checkpoint_dir, f"step_{step + 1}.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "muon_optimizer": muon_opt.state_dict(),
                    "adamw_optimizer": adamw_opt.state_dict(),
                    "step": step,
                    "config": dataclasses.asdict(cfg),
                },
                ckpt_path,
            )

    return {"final_loss": last_loss, "final_step": end_step - 1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = sft_loop(cfg)
    print(f"SFT complete. final_step={result['final_step']} final_loss={result['final_loss']:.4f}")


if __name__ == "__main__":
    main()
