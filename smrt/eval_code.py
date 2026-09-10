"""Code pass@1 eval on openai_humaneval: greedy-decode a completion for
each task, score by running prompt + completion + test in a *separate*
subprocess (never exec() in-process — untrusted model-generated code must
not run inside the eval script's own process).

Reports pass@1 only — sampling-based pass@k for k > 1 requires
temperature-sampled generation, which is out of scope for this plan.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile

import torch

from smrt.config import config_from_dict
from smrt.data.tokenizer import agent_tokenizer
from smrt.device import resolve_device
from smrt.model.backbone import SMaRT


def _load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_dict(ckpt["config"])
    model = SMaRT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _decode_completion(model, prompt_ids: list, tokenizer, device, max_new_tokens: int) -> str:
    # Note: memory.update() always builds a differentiable graph as part of
    # the forward pass; it cannot run under torch.no_grad(), matching
    # smrt.evaluate._decode_answer.
    end_id = tokenizer.encode("<|end|>")[0]
    ids = list(prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x, mem_states=None)
        next_id = int(logits[0, -1, :].argmax().item())
        if next_id == end_id:
            break
        generated.append(next_id)
        ids.append(next_id)
    return tokenizer.decode(generated)


def _score_task(prompt: str, completion: str, test: str, entry_point: str) -> bool:
    program = f"{prompt}{completion}\n{test}\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        tmp_path = f.name
    try:
        result = subprocess.run(["python3", tmp_path], timeout=10, capture_output=True)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def run_humaneval(model, tokenizer, device, max_new_tokens: int = 512) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for example in ds:
        prompt_text = f"<|system|>Complete the following Python function.<|end|><|user|>{example['prompt']}<|end|><|assistant|>"
        prompt_ids = tokenizer.encode(prompt_text)
        completion = _decode_completion(model, prompt_ids, tokenizer, device, max_new_tokens)
        passed = _score_task(example["prompt"], completion, example["test"], example["entry_point"])
        rows.append({"task_id": example["task_id"], "passed": passed})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="code_eval_results.csv")
    args = parser.parse_args()

    device_info = resolve_device("auto")
    model, cfg = _load_model(args.checkpoint, device_info.device)
    tokenizer = agent_tokenizer()

    rows = run_humaneval(model, tokenizer, device_info.device)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "passed"])
        for row in rows:
            writer.writerow([row["task_id"], row["passed"]])

    passed_count = sum(1 for row in rows if row["passed"])
    total = len(rows)
    pass_at_1 = passed_count / total if total > 0 else 0.0
    print(f"Wrote {args.output}. pass@1 = {passed_count}/{total} = {pass_at_1:.4f}")


if __name__ == "__main__":
    main()
