"""Interactive chat REPL against a SMaRT checkpoint.

Wraps each turn in the <|user|>...<|end|><|assistant|> chat template (same
special tokens as smrt/data/toolcalls.py) and greedy-decodes a response
until the model emits <|end|> or hits --max-new-tokens.

Maintains conversation history by growing the same token sequence across
turns (bounded by --max-context; oldest turns are dropped once exceeded,
never by dropping only the current turn's prompt).
"""

from __future__ import annotations

import argparse

import torch

from smrt.config import config_from_dict
from smrt.data.needle import gpt2_tokenizer
from smrt.data.tokenizer import AGENT_VOCAB_SIZE, agent_tokenizer
from smrt.data.web_search_bias import WEB_SEARCH_SYSTEM_PROMPT
from smrt.device import resolve_device
from smrt.model.backbone import SMaRT


def _load_model(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = config_from_dict(ckpt["config"])
    model = SMaRT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def generate_reply(model, tokenizer, history_ids: list, device, max_new_tokens: int) -> list:
    # Note: memory.update() always builds a differentiable graph as part of
    # the forward pass (torch.autograd.grad(..., create_graph=True)); it
    # cannot run under torch.no_grad(), matching smrt.evaluate._decode_answer.
    end_id = tokenizer.encode("<|end|>")[0]
    ids = list(history_ids)
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x, mem_states=None)
        next_id = int(logits[0, -1, :].argmax().item())
        if next_id == end_id:
            break
        generated.append(next_id)
    return generated


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    """Sample one token id from last-position logits.

    Preconditions: temperature >= 0, top_k >= 1. temperature == 0 means
    greedy (argmax); otherwise temperature-scaled multinomial over the
    top-k logits.
    """
    last = logits[0, -1, :]
    if temperature == 0:
        return int(last.argmax().item())
    scaled = last / temperature
    if top_k < scaled.numel():
        cutoff = scaled.topk(top_k).values.min()
        scaled = torch.where(scaled < cutoff, torch.tensor(float("-inf")), scaled)
    return int(torch.multinomial(torch.softmax(scaled, dim=-1), 1).item())


def generate_continuation(model, tokenizer, prompt_ids: list, device, max_new_tokens: int,
                          temperature: float, top_k: int) -> list:
    # Same constraint as generate_reply above: memory.update() builds a
    # differentiable graph, so this must not run under torch.no_grad().
    ids = list(prompt_ids)
    generated = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x, mem_states=None)
        generated.append(sample_next(logits, temperature, top_k))
        ids.append(generated[-1])
    return generated


def run_completion_repl(model, tokenizer, device, max_new_tokens: int, max_context: int,
                         temperature: float, top_k: int) -> None:
    print("Completing prompts with this checkpoint. Type 'exit' to quit.")
    print(f"(temperature={temperature}, top_k={top_k}; pass --temperature 0 for greedy)")
    while True:
        try:
            prompt = input("prompt> ")
        except EOFError:
            break
        if prompt.strip().lower() == "exit":
            break
        if not prompt.strip():
            continue
        ids = tok_encode(prompt, tokenizer)[-max_context:]
        out = generate_continuation(model, tokenizer, ids, device, max_new_tokens, temperature, top_k)
        print("model>", tokenizer.decode(out))


def tok_encode(prompt: str, tokenizer) -> list:
    return list(tokenizer.encode(prompt))


def run_repl(model, cfg, tokenizer, device, max_new_tokens: int, max_context: int, system_prompt: str) -> None:
    history_ids = tokenizer.encode(f"<|system|>{system_prompt}<|end|>")
    print("Chatting with the checkpoint. Type 'exit' to quit, 'reset' to clear history.")
    while True:
        try:
            user_text = input("you> ")
        except EOFError:
            break
        if user_text.strip().lower() == "exit":
            break
        if user_text.strip().lower() == "reset":
            history_ids = tokenizer.encode(f"<|system|>{system_prompt}<|end|>")
            continue

        history_ids += tokenizer.encode(f"<|user|>{user_text}<|end|><|assistant|>")
        if len(history_ids) > max_context:
            history_ids = history_ids[-max_context:]

        reply_ids = generate_reply(model, tokenizer, history_ids, device, max_new_tokens)
        history_ids += reply_ids + tokenizer.encode("<|end|>")
        print("model>", tokenizer.decode(reply_ids))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--system-prompt", default=WEB_SEARCH_SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    device_info = resolve_device("auto")
    model, cfg = _load_model(args.checkpoint, device_info.device)
    if cfg.model.vocab_size == AGENT_VOCAB_SIZE:
        run_repl(model, cfg, agent_tokenizer(), device_info.device,
                 args.max_new_tokens, args.max_context, args.system_prompt)
        return
    # Plain-LM checkpoint (e.g. GPT-2 vocab): no chat special tokens, so
    # complete raw prompts instead of wrapping them in a chat template.
    run_completion_repl(model, gpt2_tokenizer(), device_info.device,
                         args.max_new_tokens, args.max_context,
                         args.temperature, args.top_k)


if __name__ == "__main__":
    main()
