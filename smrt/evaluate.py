"""Recall-vs-depth-vs-length evaluation, plus a --dry-run-1m footprint check."""

from __future__ import annotations

import argparse
import csv
import random

import torch

from smrt.config import config_from_dict
from smrt.data.needle import ByteTokenizer, gpt2_tokenizer, generate_needle_example
from smrt.device import resolve_device
from smrt.model.backbone import SMaRT, assert_constant_memory_footprint


def _tokenizer_for(cfg):
    if cfg.model.vocab_size == 256:
        return ByteTokenizer()
    return gpt2_tokenizer()


def _load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_dict(ckpt["config"])
    model = SMaRT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _decode_answer(model, token_ids: list, question_end: int, answer_len: int, device) -> list:
    # Note: memory.update() always builds a differentiable graph
    # (torch.autograd.grad(..., create_graph=True)) as part of the forward
    # pass, per the MAC contract — it cannot run under torch.no_grad(), so
    # we detach only the final prediction we read out.
    ids = list(token_ids[:question_end])
    predicted = []
    for _ in range(answer_len):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x, mem_states=None)
        next_id = int(logits[0, -1, :].argmax().item())
        predicted.append(next_id)
        ids.append(next_id)
    return predicted


def run_eval_grid(model, cfg, context_lengths: list, depths: list, examples_per_cell: int, seed: int = 0):
    tokenizer = _tokenizer_for(cfg)
    rng = random.Random(seed)
    device = next(model.parameters()).device
    grid = {}
    for context_len in context_lengths:
        for depth in depths:
            depth_bin = round(depth / 100 * 4)
            matches = 0
            for _ in range(examples_per_cell):
                ex = generate_needle_example(rng, context_len, int(context_len * 0.9), depth_bin, 5, tokenizer)
                answer_len = ex.answer_span[1] - ex.answer_span[0]
                predicted = _decode_answer(model, ex.token_ids, ex.question_span[1], answer_len, device)
                true = ex.token_ids[ex.answer_span[0]:ex.answer_span[1]]
                if predicted == true:
                    matches += 1
            grid[(context_len, depth)] = matches / examples_per_cell
    return grid


def _write_results(
    output_path: str,
    context_lengths: list,
    depths: list,
    grid: dict,
    baseline_grid: dict | None = None,
) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        if baseline_grid is not None:
            header = ["context_len"] + [f"memory_depth_{d}" for d in depths] + [f"baseline_depth_{d}" for d in depths]
        else:
            header = ["context_len"] + [f"depth_{d}" for d in depths]
        writer.writerow(header)
        for context_len in context_lengths:
            row = [context_len] + [grid[(context_len, d)] for d in depths]
            if baseline_grid is not None:
                row += [baseline_grid[(context_len, d)] for d in depths]
            writer.writerow(row)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import re

        heatmap_path = re.sub(r"\.csv$", "", output_path) + "_heatmap.png" if output_path.endswith(".csv") else output_path + "_heatmap.png"
        data = [[grid[(c, d)] for d in depths] for c in context_lengths]
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(depths)))
        ax.set_xticklabels(depths)
        ax.set_yticks(range(len(context_lengths)))
        ax.set_yticklabels(context_lengths)
        ax.set_xlabel("depth (%)")
        ax.set_ylabel("context length")
        fig.colorbar(im, ax=ax, label="accuracy")
        fig.tight_layout()
        fig.savefig(heatmap_path)
        plt.close(fig)
    except ImportError:
        pass


def _dry_run_1m(cfg, model, device_info) -> None:
    assert_constant_memory_footprint(model, cfg.model)
    chunk_size = cfg.model.memory.chunk_size
    total_tokens = 1_000_000
    chunks = total_tokens // chunk_size
    print(f"--dry-run-1m: {chunks} chunks of {chunk_size} tokens would be processed sequentially.")

    x = torch.randint(0, cfg.model.vocab_size, (1, chunk_size), device=device_info.device)
    if device_info.backend in ("cuda", "rocm"):
        torch.cuda.reset_peak_memory_stats()
        model(x, mem_states=None)
        peak = torch.cuda.max_memory_allocated()
        print(f"measured peak activation bytes (one chunk) = {peak}")
        print(f"estimated peak activation bytes (steady-state) ≈ {peak * 3} (small constant × one-chunk footprint)")
    else:
        d_model = cfg.model.d_model
        num_layers = cfg.model.num_layers
        estimate = chunk_size * d_model * num_layers * 4
        print(f"analytical estimate, not measured (backend={device_info.backend}): "
              f"activation_bytes ≈ chunk_size * d_model * num_layers * 4 bytes = {estimate}")
        print(f"estimated peak activation bytes ≈ {estimate * 3} (small constant × one-chunk footprint)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-lengths", default="128000,512000,1000000")
    parser.add_argument("--depths", default="0,25,50,75,100")
    parser.add_argument("--examples-per-cell", type=int, default=20)
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--dry-run-1m", action="store_true")
    args = parser.parse_args()

    device_info = resolve_device("auto")
    model, cfg = _load_model(args.checkpoint, device_info.device)

    if getattr(args, "dry_run_1m"):
        _dry_run_1m(cfg, model, device_info)
        return

    context_lengths = [int(x) for x in args.context_lengths.split(",")]
    depths = [int(x) for x in args.depths.split(",")]

    grid = run_eval_grid(model, cfg, context_lengths, depths, args.examples_per_cell)

    baseline_grid = None
    if args.baseline_checkpoint:
        baseline_model, baseline_cfg = _load_model(args.baseline_checkpoint, device_info.device)
        baseline_grid = run_eval_grid(baseline_model, baseline_cfg, context_lengths, depths, args.examples_per_cell)

    _write_results(args.output, context_lengths, depths, grid, baseline_grid)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
