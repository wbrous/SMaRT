"""Tool-call correctness eval: two sources, both scored by exact
structural match (parsed tool `name` equals expected `name`, and
`arguments` dict equals expected `arguments` dict after json.loads on
both sides).

1. Held-out glaive-function-calling-v2 slice: greedy-decode the model's
   own tool call from a real trajectory prefix, compare to ground truth.
2. Synthetic tool-needle recall: decode the answer span, exact-match the
   digit string.
"""

from __future__ import annotations

import argparse
import csv
import json
import random

import jsonschema
import torch


from smrt.config import config_from_dict
from smrt.data.tokenizer import agent_tokenizer
from smrt.data.tool_needle import generate_tool_needle_example
from smrt.data.toolcalls import FUNCTIONCALL_RE, _extract_json_objects, parse_chat_turns, parse_system
from smrt.device import resolve_device
from smrt.evaluate import _write_results
from smrt.model.backbone import SMaRT


def _load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_dict(ckpt["config"])
    model = SMaRT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def _decode_until_end(model, prompt_ids: list, tokenizer, device, max_new_tokens: int) -> list:
    # Note: memory.update() always builds a differentiable graph as part of
    # the forward pass (torch.autograd.grad(..., create_graph=True)); it
    # cannot run under torch.no_grad(), matching smrt.evaluate._decode_answer.
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
    return generated


def _extract_model_tool_call(text: str) -> tuple[str, dict] | None:
    if "<|tool_call|>" not in text:
        return None
    body = text.split("<|tool_call|>", 1)[1].split("<|/tool_call|>", 1)[0]
    try:
        objs = _extract_json_objects(body)
    except json.JSONDecodeError:
        return None
    if not objs:
        return None
    obj = objs[0]
    if "name" not in obj or "arguments" not in obj:
        return None
    return obj["name"], obj["arguments"]


def _schema_valid(name: str, arguments: dict, schemas: list) -> bool:
    """True if `arguments` validates against the named tool's `parameters`
    JSON Schema in `schemas`, or if no schema for that name is known
    (nothing to check against — treated as vacuously valid).
    """
    for schema in schemas:
        if schema.get("name") == name:
            try:
                jsonschema.validate(instance=arguments, schema=schema.get("parameters", {}))
            except jsonschema.ValidationError:
                return False
            return True
    return True


def eval_glaive(model, cfg, tokenizer, num_examples: int, seed: int, max_new_tokens: int = 256) -> float:
    from datasets import load_dataset

    device = next(model.parameters()).device
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)

    matches = 0
    total = 0
    rng = random.Random(seed)
    for example in ds:
        if total >= num_examples:
            break
        turns = parse_chat_turns(example.get("chat", ""))
        intro, schemas = parse_system(example.get("system", ""))

        # Build prefix up to (and including) the first assistant turn that
        # calls a tool; treat that as the eval prompt and ground truth.
        prefix_text = "<|system|>" + intro
        for schema in schemas:
            prefix_text += "\nTool: " + json.dumps(schema, separators=(",", ":"))
        prefix_text += "<|end|>"

        found_target = None
        for role, body in turns:
            if role == "user":
                prefix_text += f"<|user|>{body}<|end|>"
            elif role == "tool_result":
                prefix_text += f"<|tool_result|>{body}<|/tool_result|>"
            else:  # assistant
                m = FUNCTIONCALL_RE.search(body) if body.startswith("<functioncall>") else None
                if m is not None:
                    found_target = (m.group(1), json.loads(m.group(2)))
                    prefix_text += "<|assistant|>"
                    break
                prefix_text += f"<|assistant|>{body}<|end|>"
        if found_target is None:
            continue

        total += 1
        prompt_ids = tokenizer.encode(prefix_text)
        generated_ids = _decode_until_end(model, prompt_ids, tokenizer, device, max_new_tokens)
        generated_text = tokenizer.decode(generated_ids)
        predicted = _extract_model_tool_call(generated_text)
        if (
            predicted is not None
            and predicted[0] == found_target[0]
            and predicted[1] == found_target[1]
            and _schema_valid(predicted[0], predicted[1], schemas)
        ):
            matches += 1

    return matches / total if total > 0 else 0.0


def eval_tool_needle_grid(model, cfg, tokenizer, context_lengths: list, depths: list, examples_per_cell: int, seed: int = 0) -> dict:
    device = next(model.parameters()).device
    rng = random.Random(seed)
    grid = {}
    for context_len in context_lengths:
        for depth in depths:
            depth_bin = round(depth / 100 * 4)
            matches = 0
            for _ in range(examples_per_cell):
                ex = generate_tool_needle_example(rng, context_len, int(context_len * 0.9), depth_bin, 5, tokenizer)
                answer_len = ex.answer_span[1] - ex.answer_span[0]
                prompt_ids = list(ex.token_ids[:ex.question_span[1]])
                generated_ids = _decode_until_end(model, prompt_ids, tokenizer, device, answer_len)
                true = ex.token_ids[ex.answer_span[0]:ex.answer_span[1]]
                if generated_ids == true:
                    matches += 1
            grid[(context_len, depth)] = matches / examples_per_cell
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--glaive-examples", type=int, default=200)
    parser.add_argument("--tool-needle-context-lengths", default="2048,8192,32768,65536")
    parser.add_argument("--tool-needle-depths", default="0,25,50,75,100")
    parser.add_argument("--tool-needle-examples-per-cell", type=int, default=10)
    parser.add_argument("--output", default="tool_eval_results.csv")
    args = parser.parse_args()

    device_info = resolve_device("auto")
    model, cfg = _load_model(args.checkpoint, device_info.device)
    tokenizer = agent_tokenizer()

    context_lengths = [int(x) for x in args.tool_needle_context_lengths.split(",")]
    depths = [int(x) for x in args.tool_needle_depths.split(",")]

    grid = eval_tool_needle_grid(model, cfg, tokenizer, context_lengths, depths, args.tool_needle_examples_per_cell)
    glaive_rate = eval_glaive(model, cfg, tokenizer, args.glaive_examples, seed=0)

    _write_results(args.output, context_lengths, depths, grid)
    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["glaive_exact_match_rate", glaive_rate])

    print(f"Wrote {args.output}. glaive_exact_match_rate={glaive_rate:.4f}")


if __name__ == "__main__":
    main()
