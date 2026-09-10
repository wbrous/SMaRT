# SMaRT — Sliding-window Memory and Recall Transformer

## Context

Build, from scratch, a ~50M-parameter decoder-only LM combining sliding-window
local attention with a Titans-style ("Memory as Context", MAC) neural
long-term memory module whose parameter count is fixed regardless of sequence
length, so it can in principle recall facts from contexts far longer than any
window ever attended to densely. Ship: `DESIGN.md`, model code, a synthetic
needle-in-haystack data pipeline + real pretraining data loader, a
config-driven resumable training loop, a length-generalization recall eval,
a contract-focused test suite, and `RESULTS.md`.

This machine (`wils` laptop) has only an integrated Radeon 890M iGPU — no
discrete 7900 XT, no ROCm installed, no configured remote GPU host. Per user
instruction, the codebase must be **device-modular**: it runs identically in
source form on a ROCm 7900 XT box, a CUDA box, or this CPU-only laptop; the
`README.md` tells the user which `pip`/`torch` install command to run for
their machine, and the code auto-detects the resulting backend at runtime —
no code edits required to switch machines. **This session's own verification
is CPU-only** (tiny synthetic smoke configs); full-scale pretraining is left
to run later on whichever GPU machine the user targets — the training/eval
scripts must work unmodified there.

## Repository layout

Everything lives under `smart/` (existing empty directory).

```
smart/
  README.md
  DESIGN.md
  RESULTS.md
  requirements.txt
  pyproject.toml
  smrt/
    __init__.py
    device.py          # backend detection, dtype resolution
    config.py           # dataclasses + YAML load/merge
    model/
      __init__.py
      attention.py       # sliding-window causal attention
      memory.py          # Titans MAC neural memory module
      block.py           # transformer block wiring memory + attention
      backbone.py         # full SMaRT model
    data/
      __init__.py
      needle.py          # synthetic needle-in-haystack generator
      pretrain.py         # FineWeb-Edu/SlimPajama streaming loader
      batch.py            # batch validation/assertions
    train.py             # training loop, checkpoint/resume
    evaluate.py          # recall-vs-depth-vs-length eval
  configs/
    tiny_cpu.yaml         # smoke-test config used by this session
    base_50m.yaml          # the ~50M-param architecture (device-independent)
    schedule_curriculum.yaml
  scripts/
    check_env.py          # device/bf16/ROCm-CUDA-CPU precondition check
  tests/
    test_device.py
    test_memory.py
    test_attention_mask.py
    test_needle_data.py
    test_checkpoint_resume.py
    conftest.py
```

## Approach

### 1. `smrt/device.py` — backend abstraction (no hardcoded CUDA/ROCm assumption)

- `resolve_device(requested: str = "auto") -> DeviceInfo` where
  `DeviceInfo` is a frozen dataclass: `device: torch.device`, `backend:
  Literal["rocm","cuda","cpu"]`, `dtype: torch.dtype`, `total_memory_bytes:
  int | None`, `bf16_ok: bool`.
- Detection order when `requested == "auto"`:
  1. `torch.cuda.is_available()` True → it's either ROCm or CUDA. Distinguish
     via `torch.version.hip is not None` → `backend="rocm"`, else
     `backend="cuda"`.
  2. Else `torch.backends.mps.is_available()` (Apple) → `backend="mps"` (not
     in the three targets above but trivially free to support — include it
     so "everything" genuinely means everything; dtype falls back to fp32
     since MPS bf16 support is inconsistent).
  3. Else `backend="cpu"`.
- `requested` may instead be an explicit `"cuda"`/`"rocm"`/`"cpu"`/`"mps"` —
  if the requested backend is unavailable, raise `RuntimeError` with a clear
  message (hard fail only when the user *asked* for a specific backend and
  it's missing; `"auto"` never hard-fails, it always resolves to at least
  `"cpu"`).
- `bf16_ok` computed only when backend is `cuda`/`rocm`: `hasattr(torch.cuda,
  "is_bf16_supported") and torch.cuda.is_bf16_supported()`. On `cpu`/`mps`,
  `bf16_ok = False` and `dtype = torch.float32` unconditionally — do not
  attempt bf16 autocast on backends where it isn't validated.
- `dtype = torch.bfloat16 if bf16_ok else torch.float32`.
- `resolve_device` never silently returns a CPU device when a GPU was
  detected but unusable for some other reason — any `torch.cuda.*` call
  raising is allowed to propagate (crash loud, per project convention), not
  caught and downgraded.
- `scripts/check_env.py`: imports `resolve_device("auto")`, prints backend,
  device name (`torch.cuda.get_device_name(0)` when applicable), total
  memory, `bf16_ok`; exits 0 always (it's a diagnostic, not a gate) — actual
  hard-fail-on-missing-GPU behavior belongs to `train.py` when the config
  explicitly requests a non-cpu backend (see Config below), not to this
  script.
- Config-driven memory-tiered batch/accumulation scaling (architecture is
  identical across machines; only the training *schedule* adapts):
  `pick_batch_schedule(total_memory_bytes: int | None, base_micro_batch:
  int, base_grad_accum: int) -> tuple[int, int]` — tiers: `None` or `<
  6_000_000_000` → `(1, base_grad_accum * base_micro_batch)` (CPU/tiny-VRAM:
  micro-batch 1, accumulate the rest); `6e9 <= mem < 16e9` →
  `(max(1, base_micro_batch // 4), base_grad_accum * 4)`; `16e9 <= mem <
  28e9` → `(base_micro_batch, base_grad_accum)` (this is the 7900 XT 20GB
  tier — the config's literal numbers target this tier); `>= 28e9` →
  `(base_micro_batch * 2, max(1, base_grad_accum // 2))`. Effective batch
  size (`micro_batch * grad_accum`) is held constant across tiers.

### 2. `smrt/config.py` — config schema

- `@dataclass(frozen=True) class MemoryConfig`: `key_dim: int`,
  `value_dim: int`, `hidden_dim: int`, `num_mlp_layers: int` (≥1; the
  memory MLP is `num_mlp_layers` `Linear`+`SiLU` layers, key_dim→hidden_dim→
  …→value_dim), `momentum_init: float` (init value for the η gate bias),
  `forget_init: float` (init value for the α gate bias), `lr_init: float`
  (init value for the θ' gate bias), `chunk_size: int` (tokens per
  surprise-update step within a window, ≤ `AttentionConfig.window_size`).
- `@dataclass(frozen=True) class AttentionConfig`: `window_size: int`,
  `num_heads: int`, `head_dim: int`.
- `@dataclass(frozen=True) class ModelConfig`: `vocab_size: int`,
  `d_model: int`, `num_layers: int`, `num_persistent_tokens: int`,
  `attention: AttentionConfig`, `memory: MemoryConfig`, `dropout: float`.
- `@dataclass(frozen=True) class TrainConfig`: `backend: Literal["auto",
  "cuda","rocm","cpu","mps"]`, `base_micro_batch: int`, `base_grad_accum:
  int`, `lr: float`, `weight_decay: float`, `warmup_steps: int`,
  `max_steps: int`, `seq_len: int`, `checkpoint_every: int`,
  `checkpoint_dir: str`, `seed: int`.
- `@dataclass(frozen=True) class CurriculumStage`: `min_step: int`,
  `context_len: int`, `haystack_filler_tokens: int`, `needle_depth_bins:
  int`.
- `@dataclass(frozen=True) class Config`: `model: ModelConfig`, `train:
  TrainConfig`, `curriculum: list[CurriculumStage]`.
- `load_config(path: str) -> Config`: reads YAML via `yaml.safe_load`
  (`pyyaml` dependency), constructs nested dataclasses field-by-field
  (explicit, not a generic recursive-dataclass-from-dict library — keeps
  error messages pointing at the exact bad field), raises `ValueError`
  naming the offending key on any missing/extra field (no silent defaults
  for architecture fields; `TrainConfig.backend` defaults to `"auto"` if
  absent — this is the one field allowed a default, since it's the
  machine-portability knob).
- `configs/base_50m.yaml`: the parameter budget must land at ~50M
  *non-embedding* params. Use `d_model=512`, `num_layers=8`,
  `attention.num_heads=8`, `attention.head_dim=64`, `attention.window_size=1024`,
  `memory.key_dim=64`, `memory.value_dim=64`, `memory.hidden_dim=256`,
  `memory.num_mlp_layers=2`, `memory.chunk_size=256`,
  `num_persistent_tokens=16`, `vocab_size=32000` (GPT-NeoX-style BPE via
  `tiktoken`'s `cl100k_base` truncated is NOT available offline — use
  `tiktoken.get_encoding("gpt2")`, vocab 50257; set `vocab_size: 50257` and
  drop the "32000" figure above), `dropout=0.0`. After writing
  `backbone.py`, run `python -c "from smrt.config import load_config;
  from smrt.model.backbone import SMaRT; c=load_config('configs/base_50m.yaml');
  m=SMaRT(c.model); print(sum(p.numel() for n,p in m.named_parameters() if
  'embed' not in n and 'lm_head' not in n))"` and if the count is outside
  40M–60M, adjust `d_model`/`num_layers` (prefer `num_layers` first) and
  re-check — do not hardcode a specific number without measuring, the
  9-layer-vs-8-layer/512-vs-576-dim tradeoff is not analytically obvious
  with the memory module's extra params included.
- `configs/tiny_cpu.yaml`: same shape but `d_model=64, num_layers=2,
  attention.num_heads=2, attention.head_dim=32, attention.window_size=64,
  memory.key_dim=32, memory.value_dim=32, memory.hidden_dim=64,
  memory.chunk_size=32, num_persistent_tokens=4, vocab_size=256` (byte-level
  vocab for the smoke test — avoids needing a tokenizer download in this
  offline dev session), `train.backend="cpu"`, `train.seq_len=256`,
  `train.max_steps=20`, `train.checkpoint_every=10`. This is the config
  every test and this session's verification runs against.

### 3. `smrt/model/attention.py` — sliding-window causal attention

- `sliding_window_causal_mask(seq_len: int, window_size: int, device,
  dtype) -> Tensor` returns an additive mask of shape `(seq_len, seq_len)`:
  `mask[i, j] = 0` if `j <= i and i - j < window_size`, else
  `-inf` (`torch.finfo(dtype).min`, not literal `float("-inf")`, to avoid
  NaN from an all-`-inf` row — impossible here since `mask[i,i]=0` always,
  but use `finfo.min` anyway as the project-wide convention for additive
  masks). Build once via `torch.full` + `torch.triu`/index arithmetic, not a
  Python double loop (O(seq_len²) Python loop is the "rival pattern" to
  avoid — use vectorized `torch.arange` row/col broadcasting).
- `class SlidingWindowAttention(nn.Module)`: `__init__(self, cfg:
  AttentionConfig, d_model: int)` builds `qkv_proj: Linear(d_model,
  3*num_heads*head_dim)`, `out_proj: Linear(num_heads*head_dim, d_model)`.
  `forward(self, x: Tensor, extra_kv: Tensor | None = None) -> Tensor` — `x`
  is `(B, T, d_model)`; when `extra_kv` (the memory-read + persistent
  tokens, shape `(B, M, d_model)`) is given, keys/values are computed from
  `torch.cat([extra_kv, x], dim=1)` while queries stay `x`-only, and the
  additive mask is widened to `(T, M+T)` with the first `M` columns always
  `0` (extra tokens are always visible — they are the fixed-size memory
  context, not part of the sliding window) and the remaining `T` columns
  using `sliding_window_causal_mask`. Uses
  `torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask,
  is_causal=False)` (mask already encodes causality; passing both
  `attn_mask` and `is_causal=True` is invalid) — this is the SDPA call the
  spec requires; no custom CUDA/HIP kernel.

### 4. `smrt/model/memory.py` — Titans MAC neural long-term memory

Implements the re-derived update rule (state this exact math verbatim in
`DESIGN.md`, see step 8):

- Memory state is the weight tensors of a small MLP `M_θ: R^key_dim →
  R^value_dim` (`num_mlp_layers` Linear+SiLU layers per `MemoryConfig`).
  `θ` is **not** a `nn.Parameter` trained by the outer optimizer — it is a
  plain `torch.Tensor` (per weight matrix) carried as running state across
  the sequence, updated by the surprise rule below. It is initialized once
  per forward pass (or once per training example) from a small learned
  `nn.Parameter` initializer (`init_weights: ParameterList`, one entry per
  MLP layer) so the *initialization* is learned via ordinary backprop while
  the *trajectory* of updates during the sequence is not.
- `class NeuralMemory(nn.Module)`:
  - `__init__(self, cfg: MemoryConfig, d_model: int)`: builds
    `to_key/to_value/to_query: Linear(d_model, cfg.key_dim or value_dim)`,
    `init_weights: ParameterList` (one `(in,out)` tensor per MLP layer,
    Xavier-initialized), and three gate projections `momentum_gate,
    forget_gate, lr_gate: Linear(d_model, 1)` each bias-initialized to
    `cfg.momentum_init/forget_init/lr_init` respectively (pre-sigmoid;
    `sigmoid(0.0) = 0.5`, so an init value of e.g. `2.0` biases toward
    "retain more" at the start of training — pick `momentum_init=2.0,
    forget_init=-2.0, lr_init=-2.0` in `base_50m.yaml`'s `MemoryConfig`,
    i.e. start with high momentum, low forgetting, low update rate, and let
    training move these).
  - `init_state(self, batch_size: int, device, dtype) -> MemoryState` where
    `MemoryState` is a small dataclass holding `theta: list[Tensor]`
    (per-layer weights, batched: shape `(B, in, out)`, expanded from
    `init_weights` via `.expand(B, -1, -1).clone()` — clone is required,
    `expand` is a view and the surprise update below must not write through
    it into the shared `nn.Parameter`) and `surprise: list[Tensor]` (same
    shapes, initialized to zeros — this is `S_0`).
  - `read(self, state: MemoryState, x: Tensor) -> Tensor`: `q =
    self.to_query(x)` (shape `(B, T, key_dim)`); apply the batched MLP
    defined by `state.theta` to `q` (per-example weights via
    `torch.bmm`/`einsum`, since each batch element has its own evolving
    `theta` — this cannot be a single shared `nn.Linear` call). Returns
    `(B, T, value_dim)`. This is a pure function of `state` and `x` — it
    performs no update and mutates nothing (verified by a test in step 9).
  - `update(self, state: MemoryState, x: Tensor) -> MemoryState`: computes
    `k = self.to_key(x)`, `v = self.to_value(x)` (both `(B, T, ·)`, `T =
    cfg.chunk_size`); computes the associative-recall loss `ell =
    ((M_theta(k) - v) ** 2).mean()` using the batched-MLP forward with
    `state.theta` (recomputed forward, not reusing `read`'s output, since
    `read` uses queries not keys); computes `grad_theta = torch.autograd.grad(
    ell, state.theta, create_graph=True)` (`create_graph=True` is required —
    gradients must themselves remain differentiable so backprop through the
    *outer* training loss can reach the gate projections and
    `init_weights`, which is the mechanism that makes the memory
    "meta-learned"); computes per-example gate scalars `eta =
    sigmoid(self.momentum_gate(x).mean(dim=1))`, `alpha =
    sigmoid(self.forget_gate(x).mean(dim=1))`, `theta_lr =
    sigmoid(self.lr_gate(x).mean(dim=1))` (each `(B, 1, 1)` broadcastable
    against `theta`'s `(B, in, out)`); computes new surprise per layer
    `surprise_new = eta * state.surprise - theta_lr * grad_theta` (momentum
    term decayed by `eta`, minus the instantaneous gradient scaled by
    `theta_lr` — gradient *descent*, hence the minus sign); computes new
    weights `theta_new = (1 - alpha) * state.theta + surprise_new` (weighted
    decay toward zero by `alpha`, plus the surprise step — this is the
    exact Titans "deep memory module" update, adaptive-forgetting
    generalization of a delta rule). Returns a new `MemoryState` (never
    mutates the input `state` in place — functional update, required for
    the "state before != state after, no aliasing" test in step 9).
  - `surprise_magnitude(self, state: MemoryState) -> Tensor`: returns
    `torch.stack([s.detach().flatten(1).norm(dim=1) for s in
    state.surprise]).mean(dim=0)` (per-example scalar) — this is the
    diagnostic logged separately from loss per the Training-loop
    requirement (step 6).

### 5. `smrt/model/block.py` + `smrt/model/backbone.py`

- `class SMaRTBlock(nn.Module)`: `__init__(self, model_cfg: ModelConfig)`
  builds `self.memory = NeuralMemory(model_cfg.memory, model_cfg.d_model)`,
  `self.attn = SlidingWindowAttention(model_cfg.attention,
  model_cfg.d_model)`, `self.persistent = nn.Parameter(torch.randn(
  model_cfg.num_persistent_tokens, model_cfg.d_model) * 0.02)`, `self.ln1,
  self.ln2, self.ln3: LayerNorm(d_model)`, `self.mlp:` standard 4×
  gated-MLP (`Linear(d,4d) -> SiLU -> Linear(4d,d)`).
- `forward(self, x: Tensor, mem_state: MemoryState) -> tuple[Tensor,
  MemoryState]`: chunk `x` along the sequence dim into pieces of
  `memory.chunk_size` (`torch.split`); for each chunk: `read = self.memory
  .read(mem_state, self.ln1(chunk))`; `extra = torch.cat([self.persistent
  .expand(B,-1,-1), read], dim=1)`; `attn_out = self.attn(self.ln1(chunk),
  extra_kv=extra)`; `chunk = chunk + attn_out`; `chunk = chunk +
  self.mlp(self.ln2(chunk))`; `mem_state = self.memory.update(mem_state,
  self.ln3(chunk))` — **read happens before local attention, update happens
  after**, per MAC (this ordering is the literal "Memory as Context"
  contract from the spec's Hard Architecture Requirements: "each block can
  read from it before local attention and write to it after"). Concatenate
  processed chunks back along the sequence dim; return `(x_out, mem_state)`.
  Chunking exists so the sliding-window attention mask (bounded by
  `attention.window_size`) and the memory update (bounded by
  `memory.chunk_size`) both operate on bounded-length pieces — no full-
  sequence tensor of shape `(T,T)` is ever materialized for `T` larger than
  `chunk_size` (this is what keeps activation memory independent of total
  sequence length; assert `attention.window_size >= memory.chunk_size` is
  not required, but `memory.chunk_size <= attention.window_size` should hold
  so a chunk fits in one window — enforce this in `load_config` as a
  `ValueError` if violated).
- `class SMaRT(nn.Module)`: `embed: Embedding(vocab_size, d_model)`,
  `blocks: ModuleList[SMaRTBlock] * num_layers`, `ln_f: LayerNorm(d_model)`,
  `lm_head: Linear(d_model, vocab_size, bias=False)` weight-tied to `embed
  .weight`. `forward(self, input_ids: Tensor, mem_states: list[MemoryState]
  | None = None) -> tuple[Tensor, list[MemoryState]]` — one `MemoryState`
  per block (each block has its own memory); if `mem_states is None`, call
  `block.memory.init_state(...)` per block. Returns `(logits, mem_states)`
  so `mem_states` can be threaded across chunks/training steps if ever
  needed (training loop in step 6 re-inits fresh state per example — see
  that step for why).
- Fixed-state-size assertion (Hard Architecture Requirement): add
  `smrt/model/backbone.py::assert_constant_memory_footprint(model: SMaRT,
  cfg: ModelConfig)` — computes `theta_param_count = sum(w.numel() for w in
  model.blocks[0].memory.init_weights)` and asserts this number is a
  function of `cfg.memory` fields only, independent of any sequence-length
  argument (there is no sequence-length argument to `init_weights`, so this
  is true by construction — the assertion exists as an executable
  regression check, not a live measurement across two runs: assert
  `theta_param_count == sum of cfg.memory dims as computed from the MLP
  shape formula`, i.e. re-derive the expected count from `key_dim,
  hidden_dim, value_dim, num_mlp_layers` and compare to the actual
  `numel()` sum). Call this once in `train.py` right after model
  construction, before the training loop starts, and fail loudly if it
  doesn't match (catches an accidental future change that makes memory
  shape depend on something sequence-length-related).

### 6. `smrt/train.py` — training loop

- CLI: `python -m smrt.train --config configs/base_50m.yaml` (or
  `tiny_cpu.yaml`). Loads `Config` via `load_config`; calls
  `resolve_device(cfg.train.backend)`; if `cfg.train.backend != "auto"` and
  detection disagrees, `resolve_device` already raised — no extra handling
  needed here.
- Builds `SMaRT(cfg.model)`, moves to device, casts to `device_info.dtype`
  for the *parameters that support it* — LayerNorm/gate biases stay fp32
  via `torch.autocast(device_type=..., dtype=device_info.dtype,
  enabled=device_info.dtype==torch.bfloat16)` context wrapping the forward+
  loss computation (standard mixed-precision pattern — do not hard-cast the
  whole model to bf16, which breaks LayerNorm numerics).
- Optimizer: `torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
  weight_decay=cfg.train.weight_decay)` — note `NeuralMemory.init_weights`
  and the three gate `Linear`s are ordinary `nn.Parameter`s and are included
  here; the per-example `MemoryState.theta`/`.surprise` tensors are **not**
  optimizer parameters (they're plain tensors produced fresh each forward
  pass) — they receive gradient *through* `create_graph=True` back into
  `init_weights` and the gates, not directly.
- LR schedule: linear warmup over `cfg.train.warmup_steps` then cosine decay
  to `0.1 * cfg.train.lr` over the remaining `max_steps - warmup_steps` —
  implement as a plain function `lr_at_step(step, cfg) -> float`, no
  external scheduler library dependency.
- Per-step: pull a batch from the curriculum-driven data loader (step 7),
  `mem_states = None` (fresh memory per training example — training
  examples are independent documents/needle-tasks, carrying memory state
  across unrelated examples would leak information the model was never
  meant to retain; this is a deliberate scope decision, not an oversight —
  state it in `DESIGN.md`), forward pass, cross-entropy loss on
  next-token prediction (standard causal LM loss, ignore last-position
  target), `loss.backward()`, gradient clip
  (`torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`), optimizer
  step, zero grad.
- Diagnostics logging (separate from loss, per spec): every step, log to a
  `jsonl` file at `{checkpoint_dir}/memory_diagnostics.jsonl`: `{"step":
  int, "loss": float, "surprise_mean": float, "surprise_std": float}` where
  `surprise_mean/std` come from `block.memory.surprise_magnitude(state)`
  averaged across all blocks and the batch — a plain `dict` written with
  `json.dumps` + newline via Python's builtin `json`, no extra logging
  framework dependency (matches "config-driven, not scattered constants"
  spirit without adding a new tool).
- Checkpointing: every `cfg.train.checkpoint_every` steps and at
  `max_steps`, `torch.save({"model": model.state_dict(), "optimizer":
  optimizer.state_dict(), "step": step, "config": dataclasses.asdict(cfg)},
  f"{checkpoint_dir}/step_{step}.pt")`. `--resume PATH` CLI flag: loads the
  dict, restores `model`/`optimizer` state, sets the starting step counter
  to the saved `step`, and validates `dataclasses.asdict(cfg) ==
  saved["config"]` (raise `ValueError` naming the mismatched field on
  divergence — resuming with a different architecture config is a bug, not
  a silently-tolerated case).

### 7. `smrt/data/` — pipeline

- `smrt/data/batch.py::validate_batch(input_ids: Tensor, cfg: TrainConfig,
  needle_spans: list[tuple[int,int]] | None) -> None`: asserts
  `input_ids.dtype == torch.long`, `input_ids.shape == (batch_size,
  cfg.seq_len)`, `input_ids.min() >= 0 and input_ids.max() <
  cfg.model.vocab_size` is checked by the caller (this function only checks
  shape/dtype — vocab-range checking belongs where the tokenizer is chosen,
  to keep this function tokenizer-agnostic), and when `needle_spans` is
  given, asserts every `(start,end)` span lies fully within one
  `seq_len`-sized example (`end <= cfg.seq_len`) — this is the "no example
  truncated in a way that separates a needle from its filler" precondition;
  call this at the top of every batch-producing generator in
  `needle.py`/`pretrain.py`, not just once at pipeline construction.
- `smrt/data/needle.py`:
  - `generate_needle_example(rng: random.Random, context_len: int,
    filler_tokens: int, depth_bin: int, num_depth_bins: int, tokenizer)
    -> NeedleExample` where `NeedleExample` is a dataclass:
    `token_ids: list[int]`, `needle_span: tuple[int,int]`, `question_span:
    tuple[int,int]`, `answer_span: tuple[int,int]`.
  - Fact template: `"The secret code for {topic} is {value}."` with `topic`
    drawn from a fixed list of 200 nouns and `value` a random 6-digit
    number rendered as digit tokens — this keeps the answer span
    unambiguous (a 6-digit number is very unlikely to appear verbatim
    elsewhere in randomly sampled filler text, satisfying "filler doesn't
    accidentally contain a copy of the answer" without needing an explicit
    scan — but scan anyway, see below).
  - Filler: sampled sentences from a small bundled corpus of Project
    Gutenberg-style public-domain filler text shipped as
    `smrt/data/filler_corpus.txt` (implementer must source ~2MB of
    plain-text public-domain prose, e.g. concatenated Gutenberg excerpts,
    and vendor it into the repo — this is a one-time data file, not
    generated code) concatenated until `filler_tokens` tokens reached, with
    the needle sentence spliced in at token position
    `round(depth_bin / (num_depth_bins-1) * filler_tokens)` (depth_bin
    0..num_depth_bins-1 maps to 0%..100% depth, matching the eval's
    required depth buckets in step 8).
  - Question appended at the very end: `"\nQuestion: What is the secret
    code for {topic}?\nAnswer:"`, and `answer_span` is the position range
    where the target digits should be predicted (used both for eval scoring
    and, during training, to optionally upweight loss there — do NOT
    upweight for this pass, plain uniform next-token loss over the whole
    sequence is sufficient and simpler; note this explicitly so no future
    reader wonders why there's no loss mask).
  - Explicit filler/answer collision check: after building `token_ids`,
    assert the exact digit substring of `value` does not appear anywhere in
    the filler region outside `needle_span` (`str(value) not in
    detokenize(filler_before) and str(value) not in
    detokenize(filler_after)`); on collision, resample `value` (loop, max 10
    tries, then raise — collisions should be astronomically rare with
    6-digit values against a few-MB corpus).
  - `generate_needle_batch(...) -> Tensor` calls `validate_batch` before
    returning.
- `smrt/data/pretrain.py`: `load_pretrain_stream(dataset_name: str =
  "HuggingFaceFW/fineweb-edu", split: str = "train", token_budget: int,
  tokenizer) -> Iterator[list[int]]` using HuggingFace `datasets`
  `load_dataset(..., streaming=True)`, tokenizing on the fly, yielding
  fixed-`seq_len` chunks, stopping once `token_budget` tokens have been
  yielded (a running counter, not a dataset-level slice — streaming
  datasets don't support cheap slicing). **Token budget: 2B tokens.**
  Rationale to record in `DESIGN.md` verbatim: at ~50M params, Chinchilla-
  optimal is ~1B tokens (20×params); 2B gives headroom for the curriculum's
  longer-context stages (which see fewer *examples* per token, since each
  example is longer) without exceeding a few-day single-GPU 7900 XT budget
  at the throughput measured by `scripts/check_env.py`'s SDPA benchmark —
  this is a target to record, not something achievable/verifiable on the
  CPU-only dev machine this session; the actual pretraining run is executed
  later, off this session, on the user's GPU machine.
- Curriculum: `smrt/train.py` reads `cfg.curriculum: list[CurriculumStage]`
  (ordered by `min_step` ascending — `load_config` must validate this
  ordering, raising `ValueError` on an out-of-order list) and at each step
  picks the last stage whose `min_step <= current_step`, using that stage's
  `context_len`/`haystack_filler_tokens`/`needle_depth_bins` to build that
  step's batch (alternating needle-data steps and plain-pretrain-data steps
  at a fixed ratio — 1 needle-batch per 4 pretrain-batches, hardcode this
  `4` as `CURRICULUM_NEEDLE_RATIO = 4` module constant in `train.py`, not
  config-exposed, since it's a fixed design choice not meant to be tuned
  per-run). `configs/schedule_curriculum.yaml` stages (all `context_len` ≤
  `cfg.train.seq_len`'s max across the schedule — the training config's
  `seq_len` must equal the *largest* stage's `context_len`, and shorter
  stages left-pad... no: right-truncate is wrong for needle tasks, so
  shorter-stage examples are generated at their own `context_len` and then
  **placed at the start of** a `seq_len`-sized buffer with the remainder
  filled by plain pretrain filler continuation, never padding tokens — this
  keeps every training example fully "real" text, no `[PAD]` handling
  needed anywhere in the model): `[{min_step: 0, context_len: 2048,
  haystack_filler_tokens: 1800, needle_depth_bins: 5}, {min_step: 2000,
  context_len: 8192, haystack_filler_tokens: 7900, needle_depth_bins: 5},
  {min_step: 6000, context_len: 32768, haystack_filler_tokens: 32500,
  needle_depth_bins: 5}]`, and `base_50m.yaml`'s `train.seq_len: 32768`.

### 8. `smrt/evaluate.py` — recall eval

- CLI: `python -m smrt.evaluate --checkpoint PATH --context-lengths
  128000,512000,1000000 --depths 0,25,50,75,100 --examples-per-cell 20`.
- For each `(context_len, depth)` cell: generate `examples_per_cell` needle
  examples via `generate_needle_example` at that exact `context_len`/depth
  (mapping the 0/25/50/75/100 percent depths to `depth_bin`/`num_depth_bins`
  via `depth_bin = round(depth/100 * 4)`, `num_depth_bins=5`, matching
  training's depth-bin convention); run the model forward (feeding the full
  `context_len` sequence through the chunked block forward from step 5 —
  this is the one place a genuinely long sequence is actually processed,
  bounded only by wall-clock, since activation memory per chunk is fixed);
  greedily decode the answer span token-by-token (`argmax` over `lm_head`
  logits at each position starting from `question_span[1]`, up to
  `len(answer_span)` tokens); score exact-match against the true 6 digits.
  Report `accuracy = matches / examples_per_cell` per cell.
- Output: `RESULTS.md`-ready table (`context_len × depth` grid of accuracy
  percentages) written as both a `.csv` (`smrt/evaluate.py` writes to
  `--output results.csv`) and a matplotlib heatmap PNG
  (`results_heatmap.png`) — `matplotlib` is an added dependency for this
  script only.
- Baseline comparison: `--baseline-checkpoint PATH` optional flag — a
  second checkpoint trained with `MemoryConfig` zeroed out to a no-op (add
  a `MemoryConfig.disabled: bool = False` field; when `True`,
  `NeuralMemory.read` returns `torch.zeros(B,T,value_dim)` and `.update` is
  a no-op returning `state` unchanged, and `SMaRTBlock.forward` skips
  `extra_kv` memory concat but keeps the `persistent` tokens — this makes
  the baseline a pure local-attention-only model at the same param count
  *except* the memory MLP's own parameters, which is an acceptable minor
  deviation to note in `RESULTS.md` rather than engineering exact param
  parity, since the spec only requires "same parameter count" as a fairness
  target, not bit-exact equality) — when given, the eval runs both
  checkpoints over the same generated examples and reports both accuracy
  grids side by side in the CSV.
- **1M-token-forward validation**: per the spec's explicit non-goal, do NOT
  run an actual 1M-token forward pass as a smoke test in this pass. Instead
  `smrt/evaluate.py` includes a `--dry-run-1m` flag that constructs a
  `1_000_000`-token *synthetic* `input_ids` tensor of random valid token
  ids (no real needle data needed) and calls only
  `assert_constant_memory_footprint` (step 5) plus a peak-memory estimate:
  `chunks = 1_000_000 // cfg.model.memory.chunk_size`; estimated peak
  activation bytes = `(one chunk's forward activation footprint, measured
  by actually running one real chunk-sized forward pass and reading
  `torch.cuda.max_memory_allocated()` when on a CUDA/ROCm device, or a
  rough analytical estimate on CPU) `× a small constant` (not `× chunks`,
  since chunks are processed sequentially and prior chunks' activations are
  freed) — printed as a report, not run end-to-end. This satisfies
  "validate the fixed-state-size property... synthetically/analytically."

### 9. `smrt/DESIGN.md`, `RESULTS.md`, `README.md`, `requirements.txt`

- `DESIGN.md` must contain, verbatim in plain sentences (transcribe/expand
  the math from step 4, do not just paste equations without prose): what
  `read` computes and returns (a forward pass of the current per-example
  memory MLP weights `θ_t` applied to a query projection of the input — a
  fixed-size vector per token regardless of how much text produced `θ_t`),
  what `update` computes/depends on/mutates (depends on the current chunk's
  key/value projections and the three learned gate signals; produces a new
  `θ_{t+1}` via momentum-decayed gradient descent on the chunk's
  associative-recall loss; mutates nothing in place — returns a new state,
  old state remains valid for anyone still holding a reference), what
  "surprise" means (the momentum-accumulated negative gradient of the
  memory's own reconstruction loss — large when the current chunk's
  key→value mapping is poorly predicted by the existing memory weights,
  small when the memory already "knows" this pattern; gates how much the
  weights move), and why fixed-size state approximates long recall (the
  memory's parameter count is set by `key_dim/hidden_dim/value_dim/
  num_mlp_layers` alone, never by sequence length — one gradient step
  compresses each chunk's information into a bounded-size weight update, so
  arbitrarily many chunks can be folded into the same fixed-size `θ` over
  time, trading exactness for a lossy but *persistent* compressed trace,
  unlike a KV cache which is exact but grows linearly and unlike pure local
  attention which forgets anything outside the window entirely). Also
  record: ROCm/PyTorch versions actually validated (see Verification),
  the 2B-token budget rationale (already stated in step 7, copy verbatim),
  the curriculum schedule table (copy from `configs/schedule_curriculum
  .yaml`), and a "ROCm gotchas" section seeded with whatever
  `scripts/check_env.py` reports on this machine (CPU-only — note plainly
  that ROCm-specific gotchas are unvalidated pending a real ROCm run and
  must be appended by whoever runs `check_env.py` on the target GPU
  machine — do not fabricate ROCm-specific claims from this CPU-only
  session).
- `README.md`: per-machine install instructions as three explicit blocks —
  NVIDIA CUDA: `pip install torch --index-url https://download.pytorch.org/
  whl/cu124` (or current stable cu12x tag — implementer checks
  pytorch.org's current recommended index URL at write time and uses that
  exact one, do not guess a version number); AMD ROCm (Linux only):
  `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`
  (again, verify the current recommended ROCm tag on pytorch.org at write
  time rather than assuming 6.2 — record whichever is actually current);
  CPU-only / Apple: `pip install torch`. Then `pip install -r
  requirements.txt` for the rest (which must NOT list `torch` — it is
  installed separately per the block above precisely so the same
  `requirements.txt` works everywhere). Followed by "run
  `python scripts/check_env.py` to confirm your backend was detected
  correctly" and a "quickstart" showing the `tiny_cpu.yaml` smoke run.
- `requirements.txt`: `pyyaml`, `datasets`, `tiktoken`, `matplotlib`,
  `pytest`, `numpy` — pinned to specific versions available on PyPI as of
  write time (implementer resolves exact current versions, does not use
  bare unpinned names, so a future `pip install` doesn't silently pull a
  breaking major version).
- `RESULTS.md`: written only after step 10's test suite and this session's
  CPU smoke run complete; it must honestly state that this session ran
  CPU-only smoke tests (correctness/wiring), not a full pretraining run or
  a real recall eval at scale, and that the recall-vs-length table is
  **not yet populated** — pending an actual GPU training run the user
  executes later using `smrt/train.py`/`smrt/evaluate.py` unmodified. Do
  not fabricate placeholder accuracy numbers.

### 10. Tests (`tests/`, run via `pytest`, all against `tiny_cpu.yaml`
    dimensions so they run in seconds on CPU)

Each test file's tests carry the required "what bug does this catch"
one-line comment directly above the test function, matching the exact bugs
named below.

- `test_device.py`: `resolve_device("cpu")` returns `backend="cpu",
  dtype==torch.float32` (catches accidental bf16-on-CPU); `resolve_device
  ("rocm")` raises `RuntimeError` on this CPU-only machine (catches a
  silent-fallback regression — requesting a specific unavailable backend
  must fail loud, not degrade to CPU quietly); `pick_batch_schedule(None,
  8, 4) == (1, 32)` and `pick_batch_schedule(20_000_000_000, 8, 4) ==
  (8, 4)` (catches a tier-boundary off-by-one).
- `test_memory.py`: forward-pass determinism — same seed, same input, two
  `NeuralMemory.read` calls on a freshly `init_state`'d state produce
  bit-identical output (catches nondeterministic ops like uninitialized-
  memory reads or unseeded dropout leaking into the memory path); output
  shape is exactly `(B, T, value_dim)` (catches a batched-`einsum` axis
  mixup, a very plausible bug given untested per-example weight batching);
  `update` changes `state.theta` (`not torch.equal(new.theta[0],
  old.theta[0])` for a non-degenerate random input — catches a broken
  gradient/gate wiring that leaves memory static); `update`'s output
  `theta`/`surprise` require grad and `torch.autograd.grad(new.theta[0]
  .sum(), memory.init_weights[0])` does not raise/return `None` (catches a
  silently detached tensor breaking the meta-learning path — this is
  exactly the "gradients flow through it" contract named in the spec);
  calling `read` twice with the same `state` object afterward still
  returns the same value both times (catches `read` accidentally mutating
  `state` in place, which would violate the read/write separation MAC
  depends on).
- `test_attention_mask.py`: build `sliding_window_causal_mask(seq_len=20,
  window_size=5, ...)`, run a controlled `SlidingWindowAttention` forward
  with random `q,k,v` where `extra_kv=None`, and assert the returned
  attention *weights* (recomputed manually via
  `torch.softmax(q@k.T/sqrt(d) + mask, dim=-1)` outside SDPA, since SDPA
  doesn't expose weights directly) are exactly `0.0` for every `(i,j)` with
  `j >= i - window_size + 1 + window_size` i.e. concretely: pick `i=15`,
  assert weight at `j=9` (== `15-5-1`, one before the window) is `0.0` and
  weight at `j=10` (`15-5`, at the window edge) is nonzero (catches an
  off-by-one in the window boundary — the single most likely bug in this
  file, per the spec's own example of `i+window_size+1`); also assert
  `j > i` (future positions) are `0.0` for several `(i,j)` pairs (catches a
  broken causal component independent of the window component).
- `test_needle_data.py`: `generate_needle_example` roundtrip — detokenize
  `token_ids[needle_span[0]:needle_span[1]]` and assert it contains the
  generated `topic`/`value` (catches a span-offset bug); assert `str(value)`
  does not appear in the detokenized filler outside `needle_span` (catches
  the collision case going unhandled); generate at `depth_bin=0` and
  `depth_bin=num_depth_bins-1` and assert the needle's token offset is
  `< 5%` / `> 95%` of `filler_tokens` respectively (catches a depth-mapping
  formula bug — the single most likely bug in this file, since it's `round`
  arithmetic).
- `test_checkpoint_resume.py`: train `tiny_cpu.yaml` for 10 steps, save,
  record `loss_at_step_10`; start a fresh process-equivalent (new model/
  optimizer instances) `--resume` from that checkpoint, run 1 more step,
  record its loss; separately, train a *fresh* (non-resumed) model for 11
  steps straight through, record its step-11 loss; assert the resumed
  run's step-11 loss matches the straight-through run's step-11 loss within
  `1e-4` (catches a discontinuity — optimizer momentum/step-count not
  actually restored — which is exactly the failure mode named in the spec;
  a looser-but-real check than exact bit-equality, since dropout is 0 in
  `tiny_cpu.yaml` so the only remaining nondeterminism is none — set
  `torch.manual_seed(cfg.train.seed)` identically at the start of both runs
  so data order matches).
- `conftest.py`: fixture `tiny_config()` loading `configs/tiny_cpu.yaml`
  once per test session; fixture `seeded_rng()` setting
  `torch.manual_seed(0)` before each test (isolation per project
  convention — restore global RNG state in teardown via
  `torch.random.get_rng_state()`/`set_rng_state()` save/restore around each
  test that touches global seeding, so tests don't leak seed state into
  each other).

## Critical files & anchors

- `smrt/model/memory.py` — the entire novel mechanism; every other file is
  comparatively conventional wiring around it. Get the surprise/forget/
  update formula (step 4) and the `create_graph=True` requirement right
  first; everything downstream depends on it.
- `smrt/model/block.py` — the read-before-attention / update-after-
  attention ordering is the literal MAC contract; easy to accidentally
  invert or fuse into one call.
- `smrt/train.py` — `mem_states = None` per-example reset decision (step 6)
  is easy to "improve" into cross-example state carry-over, which would
  silently break the training-time independence assumption the whole
  curriculum/eval design relies on.
- `configs/base_50m.yaml` — the ~50M non-embedding param count must be
  *measured*, not assumed; the measurement command is spelled out in step 2.
- `smrt/device.py` — the one file every other script imports for backend
  selection; get the ROCm-vs-CUDA distinguishing check (`torch.version.hip`)
  right here once rather than duplicating it elsewhere.

## Verification

All commands run from `smart/` on this CPU-only machine (`cd smart`).

1. `pip install -r requirements.txt` (torch installed separately first per
   README's CPU-only block: `pip install torch`) — inside a fresh venv
   (`python3 -m venv .venv && source .venv/bin/activate`) to avoid touching
   system Python.
2. `python scripts/check_env.py` — expected output shows `backend: cpu`,
   `bf16_ok: False`; confirms the device module runs without crashing on a
   GPU-less machine (this is the "ROCm device-availability precondition
   check... as an early integration test" requirement, exercised here in
   its CPU branch since ROCm hardware isn't present).
3. `pytest tests/ -v` — every test in step 10 passes; this is the primary
   proof-of-correctness for this session (memory update math, attention
   mask boundary, needle data integrity, checkpoint resume continuity).
4. `python -c "from smrt.config import load_config; from smrt.model.backbone
   import SMaRT; c=load_config('configs/base_50m.yaml'); m=SMaRT(c.model);
   n=sum(p.numel() for name,p in m.named_parameters() if 'embed' not in name
   and 'lm_head' not in name); print(n); assert 40_000_000 <= n <=
   60_000_000, n"` — confirms the ~50M non-embedding param target from the
   Hard Architecture Requirements section, using the actual constructed
   model, not a hand estimate.
5. `python -m smrt.train --config configs/tiny_cpu.yaml` — runs 20 steps to
   completion on CPU in well under a minute, writes
   `checkpoints/step_10.pt`, `checkpoints/step_20.pt`, and
   `memory_diagnostics.jsonl`; inspect the jsonl and confirm
   `surprise_mean` is nonzero and varies across steps (concrete
   observable-output check: this is the "tell whether the memory mechanism
   is doing anything" diagnostic actually producing signal, not the model
   silently ignoring the memory path).
6. `python -m smrt.evaluate --checkpoint checkpoints/step_20.pt
   --context-lengths 512 --depths 0,50,100 --examples-per-cell 3` (tiny
   sizes appropriate to the smoke-trained checkpoint, not the real 128K–1M
   targets, which require the real GPU training run) — completes without
   error and writes `results.csv`/`results_heatmap.png`; accuracy numbers
   will be near-random (model barely trained) and that is expected and
   fine — this step only proves the eval pipeline itself is wired
   correctly end to end, matching this session's CPU-only scope.
7. `python -m smrt.evaluate --dry-run-1m --checkpoint
   checkpoints/step_20.pt` — completes without attempting a real 1M-token
   forward pass, prints the fixed-footprint assertion result and the
   estimated peak-memory report.

## Assumptions & contingencies

- Exact current PyTorch ROCm/CUDA wheel index-tag (e.g. `rocm6.2` vs a
  newer tag) is looked up live at write time from pytorch.org rather than
  guessed — if pytorch.org is unreachable when writing `README.md`, use
  `rocm6.2` and `cu124` as the fallback tags and mark them
  `# verify current tag at https://pytorch.org/get-started/locally/` inline
  in the README rather than silently shipping a possibly-stale pin.
- `datasets`/`tiktoken` package downloads (for `pretrain.py`'s streaming
  loader and any non-byte-level tokenizer use) are not exercised by this
  session's verification (offline/CPU dev machine, `tiny_cpu.yaml` uses a
  byte-level vocab specifically to avoid needing them) — if a network check
  is later needed, `pretrain.py`'s streaming loader is only invoked by
  `train.py` when a config other than `tiny_cpu.yaml` is used, so this
  session's Verification steps never hit it.
- If measuring `base_50m.yaml`'s param count (Verification step 4) lands
  outside 40M–60M, adjust `num_layers` first (keeping `d_model=512`) since
  layer count scales param count roughly linearly and is the easiest knob
  to reason about; only touch `d_model` if adjusting `num_layers` alone
  can't land in range without going below 4 or above 16 layers.
- If `torch.autograd.grad(..., create_graph=True)` proves too slow/memory-
  heavy even at `tiny_cpu.yaml` scale during Verification step 3's test
  run, do not silently drop `create_graph=True` (that breaks the meta-
  learning contract tested explicitly in `test_memory.py`) — instead reduce
  `tiny_cpu.yaml`'s `memory.chunk_size`/`train.seq_len` further (e.g.
  chunk_size 16, seq_len 128) until it's fast enough; the architecture
  contract is non-negotiable, the smoke-test size is not.
