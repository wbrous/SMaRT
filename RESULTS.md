# SMaRT — Results

This session ran **CPU-only smoke tests** (correctness/wiring proof), not a
full pretraining run and not a real recall evaluation at scale. The
recall-vs-length table is **not yet populated** — it is pending an actual
GPU training run the user executes later using `smrt/train.py`/
`smrt/evaluate.py` unmodified, targeting `configs/base_50m.yaml`'s 2B-token
budget on their ROCm/CUDA machine.

## Environment validated this session

- PyTorch 2.14.0+cpu, Python 3.11.15, CPU-only (no ROCm/CUDA hardware
  available on this development machine).
- `scripts/check_env.py` correctly reported `backend: cpu`, `bf16_ok: False`.

## Measured non-embedding parameter count (`configs/base_50m.yaml`)

```
49,916,269
```

Measured directly from the constructed `SMaRT` model
(`d_model=512, num_layers=15, attention.window_size=1024`), via:

```python
n = sum(p.numel() for name, p in m.named_parameters()
        if 'embed' not in name and 'lm_head' not in name)
```

Within the target 40M–60M non-embedding parameter range. (`num_layers` was
swept from 8 to 18 to find this; see `DESIGN.md`/plan for the sweep.)

## Test suite

```
pytest tests/ -v
============================== 15 passed in 2.05s ==============================
```

All 15 tests pass:
`test_device.py` (3), `test_memory.py` (5), `test_attention_mask.py` (3),
`test_needle_data.py` (3), `test_checkpoint_resume.py` (1).

## CPU training smoke run (`configs/tiny_cpu.yaml`, 20 steps)

Completed in ~3 seconds. Wrote `checkpoints/step_10.pt`,
`checkpoints/step_20.pt`, and `checkpoints/memory_diagnostics.jsonl`.

`surprise_mean` (the neural memory's per-step diagnostic — see
`DESIGN.md`'s "surprise" section) ranged from **0.00518 to 0.00940** across
the 20 steps and visibly varied step-to-step rather than sitting at a
constant/zero value — a concrete observable-output confirmation that the
memory mechanism is actively participating in the forward/update cycle,
not being silently bypassed. Sample rows:

```
{"step": 0,  "loss": 58.13, "surprise_mean": 0.009402, "surprise_std": 0.000526}
{"step": 1,  "loss": 60.51, "surprise_mean": 0.005688, "surprise_std": 0.001079}
{"step": 18, "loss": 56.51, "surprise_mean": 0.005401, "surprise_std": 0.001303}
{"step": 19, "loss": 56.81, "surprise_mean": 0.005298, "surprise_std": 0.001336}
```

Training loss did not show a clean monotonic decrease over only 20 steps on
a byte-level, randomly initialized, tiny model — expected at this scale;
not a claim of trained-model quality.

## Evaluation pipeline smoke test

```
python -m smrt.evaluate --checkpoint checkpoints/step_20.pt \
  --context-lengths 512 --depths 0,50,100 --examples-per-cell 5
```

Completed without error; wrote `results.csv` and `results_heatmap.png`.
Accuracy was 0/5 at every (context_len, depth) cell — expected for a
20-step, randomly-initialized tiny model; this only proves the eval
pipeline (needle generation → chunked forward → greedy decode → exact-match
scoring → CSV/heatmap output) is wired correctly end to end, not that the
model has learned to recall.

```
python -m smrt.evaluate --dry-run-1m --checkpoint checkpoints/step_20.pt
```

Completed without attempting a real 1M-token forward pass. Output:

```
--dry-run-1m: 31250 chunks of 32 tokens would be processed sequentially.
analytical estimate, not measured (backend=cpu): activation_bytes ≈ chunk_size * d_model * num_layers * 4 bytes = 16384
estimated peak activation bytes ≈ 49152 (small constant × one-chunk footprint)
```

(On a real CUDA/ROCm machine this path measures actual peak allocated
bytes via `torch.cuda.max_memory_allocated()` instead of the analytical
estimate shown here.)

## Not yet done (pending a real GPU run)

- Full 2B-token pretraining run on `configs/base_50m.yaml`.
- Real recall-vs-depth-vs-length evaluation at 128K/512K/1M-token context
  (`--dry-run-1m` only sanity-checks the fixed-footprint architecture claim
  and estimates memory; it does not run an actual long-context forward
  pass).
- Baseline (`memory.disabled: true`) comparison run.
- ROCm/CUDA-specific validation (see `DESIGN.md`'s "ROCm gotchas" section —
  intentionally left for whoever runs `scripts/check_env.py` on the target
  GPU machine).

## Agent extension (tool-calling, code, recall at 2B/4B/8B) — this session

Extends the base recall model into an agentic model family per the plan in
`local://agent-tool-code-recall-plan.md`. Same CPU-only, wiring-correctness
scope as the base work above — no GPU hardware available on this
development machine, so every claim below is a small/CPU-scale smoke test,
never a real training curve.

### New tokenizer

`smrt/data/tokenizer.py::agent_tokenizer()` extends `tiktoken`'s
`o200k_base` (200019 base tokens) with 8 chat/tool special tokens for
`AGENT_VOCAB_SIZE = 200027`. Verified round-trip:
`<|tool_call|>{"a":1}<|/tool_call|>` encodes and decodes losslessly with
the special tokens as single ids (not fragmented).

### Model configs at 2B/4B/8B (`configs/agent_{2b,4b,8b}.yaml`)

Measured non-embedding parameter counts (same
`'embed' not in name and 'lm_head' not in name` formula as `base_50m.yaml`):

```
agent_2b.yaml  (d_model=2048, num_layers=38): 2,018,710,130  (+0.9% vs 2.0e9 target)
agent_4b.yaml  (d_model=2560, num_layers=48): 3,983,698,064  (-0.4% vs 4.0e9 target)
agent_8b.yaml  (d_model=3072, num_layers=67): 8,006,431,305  (+0.08% vs 8.0e9 target)
```

All three within the ±5% tolerance on the first sizing attempt — no
`num_layers` sweep needed this time.

### New data pipelines

- `smrt/data/code.py::load_code_stream` — streams
  `codeparrot/github-code-clean`, filtered to 8 languages. Structurally
  mirrors the already-proven `smrt/data/pretrain.py::load_pretrain_stream`;
  not independently smoke-run against live HF (network-dependent, same
  class of dependency as `load_pretrain_stream` itself).
- `smrt/data/toolcalls.py::load_toolcall_sft_stream` — streams
  `glaiveai/glaive-function-calling-v2` and formats `(token_ids, loss_mask)`
  pairs. **Verified against the real HF Hub this session**: pulled one live
  row, confirmed `ids.shape == mask.shape == (2048,)`, `mask.sum() > 0`, and
  the decoded prefix contains recognizable `<|system|>`/`<|user|>`/
  `<|assistant|>` structure with a real tool schema and conversation.
- `smrt/data/tool_needle.py::generate_tool_needle_example` — synthetic
  tool-call/tool-result needle-in-haystack recall. **Verified offline**: a
  2048-token example at depth-bin 2/5 produced a non-empty, correctly
  positioned `answer_span` (6-digit `record_id` value), with no truncation
  bug (the exact failure mode this test was designed to catch, per
  `smrt/data/needle.py`'s own history).

### Training loop extension (`smrt/train.py`)

`_get_batch` now dispatches to `_get_agent_batch` when
`cfg.model.vocab_size == AGENT_VOCAB_SIZE`, cycling through
`AGENT_BATCH_SOURCE_SCHEDULE` (needle / tool_needle / code / code / text×6).
The two pre-existing vocab-size branches (256, 50257) are untouched — full
existing test suite still passes (15/15, see below). Smoke-tested the two
offline sources (`needle`, `tool_needle`) directly against a tiny
agent-vocab model (`configs/agent_tiny_cpu.yaml`, `d_model=32`,
`num_layers=1`), producing correctly shaped `(2, 2048)` batches for both.

### SFT script (`smrt/sft.py`)

Loss-masking correctness verified with a throwaway script on
`configs/tiny_cpu.yaml`'s dimensions: constructed a synthetic
`(ids, mask)` pair with only the last 2 positions unmasked, ran one
forward + `ignore_index=-100` cross-entropy + `backward()`, and confirmed
`model.lm_head.weight.grad` is non-zero (16,384 nonzero elements) — proving
the masking doesn't silently zero the entire loss.

### Eval scripts

- `smrt/eval_tools.py` — tool-call correctness (glaive held-out slice +
  synthetic tool-needle grid), plus `jsonschema`-based argument validation
  against each tool's `parameters` schema. Smoke-tested the tool-needle
  grid path offline against the tiny agent checkpoint: pipeline runs
  end-to-end (0% accuracy at every cell, expected for an untrained
  1-layer/32-dim model trained for 2 steps).
- `smrt/eval_code.py` — HumanEval pass@1, scored via a **separate
  subprocess** (never `exec()` in-process). Smoke-tested the full
  `run_humaneval` flow (dataset → prompt → greedy decode → subprocess
  scoring → rows) end-to-end on 3 real HumanEval tasks against the tiny
  agent checkpoint: completed without error, all 3 failed (expected for
  the untrained tiny model — this only proves the harness is wired
  correctly, not code-generation quality). A full 164-task run was
  attempted but exceeded this session's time budget at `max_new_tokens=512`
  per task on CPU; the 3-task reduced-token-budget run above is the
  wiring-correctness proof instead.

### Not yet done (pending real GPU hardware + multi-day wall clock)

- Full pretraining runs at 2B/4B/8B on the new curriculum (needle /
  tool-needle / code / text mix), and the SFT stage on top of any of them.
- Real tool-call-correctness numbers (glaive exact-match rate, tool-needle
  recall-vs-depth-vs-length grid) and real HumanEval pass@1 — all current
  numbers are 0%/untrained-tiny-model smoke tests, not model quality
  claims.
- `load_code_stream` was not independently smoke-run against the live HF
  Hub this session (only `load_toolcall_sft_stream` was, since it's the
  pipeline step 8's loss-masking correctness proof most directly depends
  on); it shares `load_pretrain_stream`'s exact, already-proven streaming
  pattern.

## Post-training expansion (Muon optimizer, agentic SFT mixture, web-search bias)

Extends the agent post-training stage per the plan in
`local://smrt-post-training-web-search-plan.md`: a Muon+AdamW split
optimizer (Moonshot Kimi K2), strict-mode tool schemas (OpenAI
convention), and a 4-source SFT mixture (tool calls, web-search-bias,
Python syntax repair, general instruction-following). Same CPU-only,
wiring-correctness scope as the sections above — no GPU hardware
available on this development machine.

### Optimizer split (`smrt/optim.py`)

`torch.optim.Muon` (with `adjust_lr_fn="match_rms_adamw"`) exists natively
in this session's validated PyTorch build (`2.14.0+cpu`). New
`tests/test_optim.py` (3 tests) verifies: known hidden-layer weights
(`attn.q_proj`, `mlp.gate_proj`, `memory.to_key`) route to the Muon group
and known non-hidden parameters (`embed.weight`, `ln1.weight`,
`persistent`, `memory.momentum_gate.weight`) route to the AdamW group;
every model parameter lands in exactly one group; and a real
forward+backward+`step()` on both optimizers actually changes a Muon-group
weight (not a dead optimizer). `smrt/train.py` and `smrt/sft.py` both
checkpoint the two optimizers under separate `muon_optimizer`/
`adamw_optimizer` keys (clean cutover — no `optimizer` key remains).

```
pytest tests/test_optim.py -v
============================== 3 passed in 0.96s ===============================
```

### Full test suite (74 tests, up from 15)

```
pytest tests/ -v
============================== 74 passed in 4.83s ===============================
```

The 11 new tests (3 files: `test_web_search_bias_data.py`,
`test_code_repair_data.py`, `test_general_instruction_data.py`) plus
`test_optim.py`'s 3 all pass alongside the full pre-existing suite.

### New data pipelines

- `smrt/data/web_search_bias.py` — synthetic knowledge-boundary questions
  (prices, awards, versions, weather, news) whose only correct target is
  a `web_search` tool call; ships `WEB_SEARCH_SYSTEM_PROMPT` (used as
  `smrt/chat.py`'s new default `--system-prompt`) and the strict-mode
  `WEB_SEARCH_TOOL_SCHEMA`.
- `smrt/data/code_repair.py` — streams real Python source, injects a
  single verified-`SyntaxError` corruption (dropped colon or closing
  bracket), trains repair of the corrupted source back to the original
  (Vercel AutoFix-inspired, done via SFT rather than RL per this repo's
  SFT-only post-training stage).
- `smrt/data/general_instruction.py` — streams `HuggingFaceH4/no_robots`,
  formatting each row's real `messages` schema (verified this session,
  see below) into the same `<|system|>`/`<|user|>`/`<|assistant|>` special
  tokens as `smrt/data/toolcalls.py`.

**Dataset-loading gotcha found and fixed this session**: `codeparrot/
github-code-clean` (used by both the pre-existing `smrt/data/code.py` and
the new `smrt/data/code_repair.py`) ships a `.py` loading script that this
session's `datasets` library version refuses to run
(`RuntimeError: Dataset scripts are no longer supported`), regardless of
`config_name`. Fixed in both files by loading the same data directly via
the `parquet` builder against an explicit `hf://datasets/<name>/data/
train-*.parquet` glob, sidestepping the script entirely — verified this
session by pulling real rows (`code`/`language` fields confirmed present,
matching the pre-existing `language`/`code` field-name assumption).

### SFT mixture end-to-end smoke run (`configs/agent_tiny_cpu.yaml`)

```
PYTHONPATH=. python -m smrt.train --config configs/agent_tiny_cpu.yaml
Training complete. final_step=1 final_loss=32.5393

PYTHONPATH=. python -m smrt.sft --config configs/agent_tiny_cpu.yaml
SFT complete. final_step=9 final_loss=31.1562
```

Both commands completed without error. The SFT run wrote
`/tmp/agent_tiny_sft_ckpt/step_10.pt` and `sft_diagnostics.jsonl` with one
line per step, all 4 `SFT_SOURCE_SCHEDULE` sources exercised (toolcall,
web_search_bias, code_repair, general_instruction) with no `KeyError`/
`StopIteration` — this is the first time `python -m smrt.sft` has run
successfully end-to-end via the CLI in this repo's history (previously
only smoke-verified with a throwaway script per the "SFT script" section
above).

**Bug found and fixed this session**: the first smoke run (before this
fix) had two of the ten steps (the `code_repair` source, whose
corrupted+original Python source can be long) log `loss: NaN`. This
happened when truncate-from-the-end (the pad/truncate contract every SFT
data pipeline in this repo shares, previously duplicated verbatim in
each of `smrt/data/toolcalls.py`, `smrt/data/web_search_bias.py`,
`smrt/data/code_repair.py`, `smrt/data/general_instruction.py`) removed
an example's entire assistant span, leaving `cross_entropy`'s
`ignore_index` mask fully set and the reduction undefined (0/0). Fixed
by extracting the shared logic into `smrt/data/sft_common.py::
pad_or_truncate`, which now returns `None` (row skipped by the caller)
whenever truncation would remove every assistant-masked position, and
wiring it into all 4 loaders in place of their duplicated pad/truncate
blocks. New `tests/test_sft_common.py` (4 tests) covers padding,
truncation-with-surviving-assistant-span, truncation-that-would-empty-
the-assistant-span (the exact NaN-producing case, now rejected), and the
exact-length passthrough. Re-ran the SFT smoke run after the fix: all 10
steps now have finite losses (`31.15`-`35.50` range, no `NaN`).

### `smrt/chat.py`'s new web-search-biased default system prompt

```
PYTHONPATH=. python -m smrt.chat --checkpoint /tmp/agent_tiny_sft_ckpt/step_10.pt --max-new-tokens 20
```

Loaded the SFT checkpoint without a `state_dict` key mismatch (confirming
the two-key `muon_optimizer`/`adamw_optimizer` checkpoint change
round-trips — `smrt.chat` only loads `ckpt["model"]`, unaffected by the
optimizer-key split) and generated a reply with no `--system-prompt`
override, exercising the new `WEB_SEARCH_SYSTEM_PROMPT` default. Output
was repetitive `<|assistant|>` tokens — expected for a 10-SFT-step,
2-pretrain-step tiny model (this only proves the wiring, not response
quality, matching this repo's established smoke-test scope).

### Not yet done (pending real GPU hardware + multi-day wall clock)

- A real training run long enough to observe the model's tool-call rate
  actually shift toward `web_search` on knowledge-boundary questions, or
  a real repair accuracy on corrupted Python source — all current numbers
  are wiring-correctness smoke tests on an untrained tiny model.
- Confirming the `pad_or_truncate` fix behaves the same at real (large)
  `seq_len` values, not just this session's `seq_len=256` smoke config.

## Pretraining mixture overhaul (grad accumulation, persistent streams, size-scaled code bias)

Fixes two release-blocking pretraining-loop bugs and makes the code/text
mixture ratio size-scaled and config-driven — see DESIGN.md's new
"Pretraining mixture overhaul" section for the full rationale.

- Full test suite: **103 passed** (`python -m pytest tests/ -q`), including
  4 new files (`test_persistent_batch_streams.py`,
  `test_grad_accumulation.py`, `test_config_batch_source_schedule.py`, and
  the updated `test_agent_batch_dispatch.py`).
- `configs/agent_1b.yaml` (new, `d_model=1536`, `num_layers=28`,
  `num_heads=24`, `num_kv_heads=6`, `mlp_hidden_dim=4088`) measures at
  **754,358,356** non-embedding parameters — inside the approved
  500M–1B range — via this repo's established
  `sum(p.numel() for name, p in m.named_parameters() if 'embed' not in name and 'lm_head' not in name)`
  formula.
- CPU smoke run (`PYTHONPATH=. python -m smrt.train --config configs/agent_tiny_cpu.yaml`)
  still completes end-to-end after the grad-accumulation restructuring:
  `Training complete. final_step=1 final_loss=32.5393`.

### Not yet done (pending real GPU hardware + HF gate approval)

- The 7900 XT calibration run (`--calibration-steps`) and the live
  persistent-stream regression proof against `nvidia/Nemotron-CC-v2` and
  `bigcode/starcoderdata` both require ROCm hardware this development
  machine does not have, plus (for Nemotron-CC-v2) a manual HF gated-access
  approval with unpredictable turnaround.
- `nvidia/Nemotron-CC-v2`'s exact `"text"` field name is inferred from its
  confirmed-identical-lineage predecessor (`spyysalo/nemotron-cc-1M-sample`)
  since its own schema-preview endpoint 401s without an approved token — see
  DESIGN.md's fallback/verification note if `load_nemotron_cc_stream` needs
  a field-name correction once real access is granted.
