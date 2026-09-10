# SMaRT — Design

SMaRT (Sliding-window Memory and Recall Transformer) is a decoder-only
language model combining two complementary attention/recall mechanisms:

1. **Sliding-window local attention** — each token attends causally to at
   most `window_size` preceding tokens. Cheap, exact, but forgets anything
   outside the window entirely.
2. **A Titans-style ("Memory as Context", MAC) neural long-term memory** —
   a small MLP whose *weights* are updated online, chunk by chunk, as the
   sequence is processed. Its parameter count is fixed by config
   (`key_dim`, `hidden_dim`, `value_dim`, `num_mlp_layers`) and never grows
   with sequence length, so it can in principle carry a compressed trace of
   facts seen arbitrarily far in the past — something neither a bounded
   local-attention window (loses everything past the window) nor a
   full/linear-growth KV cache (exact but grows with sequence length) does.

## The neural memory mechanism

### What `read` computes

`NeuralMemory.read(state, x)` projects the input `x` through a learned
query projection (`to_query: Linear(d_model, key_dim)`), then runs that
query vector through the *current* memory MLP — the small feedforward
network whose weights are `state.theta`, evolved online over the sequence
so far. The output is a fixed-size `(B, T, value_dim)` tensor per token,
regardless of how much text produced `state.theta` — this is the whole
point: an arbitrarily long history is compressed into a bounded-size weight
tensor, and `read` simply evaluates that compressed function at the current
query. `read` is a pure function: it never mutates `state`, and calling it
twice on the same `(state, x)` pair returns bit-identical output (verified
by `tests/test_memory.py::test_read_does_not_mutate_state`).

### What `update` computes/depends on/mutates

`NeuralMemory.update(state, x)` evolves the memory weights by one step of
momentum-decayed gradient descent on the memory's own *associative-recall*
loss for the current chunk:

```
k = to_key(x)                          # (B, T, key_dim), T = memory.chunk_size
v = to_value(x)                        # (B, T, value_dim)
ell = mean( (M_theta(k) - v)^2 )       # can the memory reconstruct v from k?

grad_theta = d(ell)/d(theta)           # create_graph=True — see below

eta      = sigmoid(momentum_gate(x).mean(dim=T))   # (B,1,1) per-example
alpha    = sigmoid(forget_gate(x).mean(dim=T))
theta_lr = sigmoid(lr_gate(x).mean(dim=T))

surprise_new = eta * surprise - theta_lr * grad_theta   # momentum-decayed gradient descent
theta_new    = (1 - alpha) * theta + surprise_new        # weight decay toward 0, plus the surprise step
```

This depends on: the current chunk's key/value projections (`to_key`,
`to_value` applied to `x`), the current `state.theta`/`state.surprise`, and
three learned per-example gate scalars conditioned on the chunk. It
produces a **new** `MemoryState` — the old state tensors are never modified
in place (`torch.equal(old.theta[0], ...)` after `update` is guaranteed
false only for the *new* object; the old object's tensors are untouched,
verified by `tests/test_memory.py::test_update_changes_theta`).

`create_graph=True` on the `torch.autograd.grad` call is required: the
per-chunk gradient itself must remain part of the differentiable graph so
that, when the *outer* training loss backpropagates through many chunks'
worth of `update` calls, gradient can reach `init_weights` (the learned
initializer for `theta`) and the three gate projections. This is the
mechanism that makes the memory "meta-learned" — the network learns *how
to update its own memory*, not just what the memory should initially
contain. `tests/test_memory.py::test_update_theta_requires_grad_to_init_weights`
pins this: it directly asserts that
`torch.autograd.grad(new_state.theta[0].sum(), init_weights[0])` is
non-`None`.

### What "surprise" means

`surprise` is the momentum-accumulated *negative* gradient of the memory's
own reconstruction loss. It is large when the current chunk's key→value
mapping is poorly predicted by the existing memory weights (the memory is
"surprised" by this chunk — it doesn't already encode this association) and
small when the memory already "knows" this pattern (low reconstruction
loss, so a small gradient). `surprise` gates how much `theta` actually
moves on this step: a chunk that produces zero surprise leaves the memory's
trajectory essentially unchanged (subject only to the `alpha` decay term).
`NeuralMemory.surprise_magnitude` reports `‖surprise‖` per example as a
training diagnostic, logged separately from the training loss to
`memory_diagnostics.jsonl` so a human/agent can directly observe that the
memory is doing something (surprise nonzero and varying across steps),
rather than being silently ignored by the rest of the model.

### Why fixed-size state approximates long recall

`state.theta`'s parameter count is set entirely by
`key_dim × hidden_dim × value_dim × num_mlp_layers` (see the exact shape
formula in `assert_constant_memory_footprint`, `smrt/model/backbone.py`) —
there is no sequence-length argument anywhere in `NeuralMemory.init_state`
or `.update`, so length-independence holds by construction, not by
convention. Every chunk contributes one gradient-descent step that folds
its key→value associations into the same bounded-size weight tensor;
arbitrarily many chunks can fold into that same fixed-size `theta` over
time. This trades exactness for a lossy but *persistent* compressed trace:
unlike a KV cache (exact recall, but memory grows linearly with sequence
length — eventually the whole point of "sliding window" is defeated) and
unlike pure local attention (zero memory of anything outside the window,
but also zero cost), the neural memory sits in between — bounded cost,
imperfect but non-zero recall of arbitrarily distant history.

### Block-level wiring: read-before-attention, update-after

Per block, per `memory.chunk_size`-sized chunk (see `smrt/model/block.py`):

1. `read = memory.read(state, ln1(chunk))` — the memory is consulted
   *before* local attention runs.
2. `extra_kv = concat([persistent_tokens, project(read)])` is fed to
   `SlidingWindowAttention` as always-visible extra key/value tokens
   (in addition to the causal sliding-window tokens).
3. Local attention + MLP run as usual.
4. `state = memory.update(state, ln3(chunk))` — the memory is written
   *after* local attention, using the (locally-attended) chunk output.

This ordering is the literal "Memory as Context" contract: each block can
read from the memory before local attention and write to it after. Note:
`NeuralMemory.read` returns `(B, T, value_dim)` (pinned by
`tests/test_memory.py::test_read_output_shape`); since attention's extra-KV
path requires `d_model`-dimensional inputs (its `qkv_proj` is shared across
`x` and `extra_kv`), `SMaRTBlock` applies its own `mem_out_proj:
Linear(value_dim, d_model)` to the memory read before concatenating with
the persistent tokens — this projection lives in the block, not inside
`NeuralMemory`, so `read`'s own output contract stays exactly
`value_dim`-shaped as tested.

## Training-time scope decision: memory resets per example

`train.py` always calls the model with `mem_states=None`, so every training
example (a needle-recall task, or a plain pretrain continuation chunk)
starts from a *freshly initialized* memory state. Training examples are
independent documents/tasks; carrying memory state across unrelated
examples would leak information the model was never meant to retain across
example boundaries and would make the curriculum/eval design (which
measures recall *within* a single example's context) ill-defined. This is a
deliberate scope decision, not an oversight.

## Attention backend note

`SlidingWindowAttention` forces PyTorch's `SDPBackend.MATH` kernel via
`torch.nn.attention.sdpa_kernel`. This was required because this session's
CPU backend's flash-attention SDPA kernel does not implement a backward
pass for the additive-mask case used here
(`aten::_scaled_dot_product_flash_attention_for_cpu_backward` is not
implemented in the validated PyTorch 2.14.0+cpu build) — `RuntimeError:
derivative for ... is not implemented`. Forcing MATH trades some GPU
throughput (flash/mem-efficient kernels are typically faster on CUDA/ROCm)
for correctness and backend-portability guaranteed to work identically on
CPU, CUDA, and ROCm. Whoever runs the real GPU training pass should
benchmark whether the flash/mem-efficient SDPA backward has since been
implemented for the additive-mask case on their PyTorch version, and if so,
consider relaxing this to `sdpa_kernel([SDPBackend.FLASH_ATTENTION,
SDPBackend.MATH])` for a speed win — this is a documented follow-up, not
yet validated.

## Optimizer: Muon for hidden weights, AdamW for the rest

`smrt/optim.py::build_optimizers` splits every model parameter between
`torch.optim.Muon` (2D hidden-layer weight matrices: attention/MLP/memory
Linear weights, `NeuralMemory.init_weights`) and `torch.optim.AdamW`
(embeddings, RMSNorm scales, biases, the degenerate single-output memory
gates, the persistent-token table) via
`smrt/optim.py::split_muon_adamw_params`, following Moonshot AI's Kimi K2
technical report (arXiv:2507.20534). Muon is built with
`adjust_lr_fn="match_rms_adamw"` so `train.lr`/`sft.lr` (already tuned as
AdamW hyperparameters) apply unchanged to both optimizers — no separate
Muon learning rate exists in the config schema. This repo does not layer
Moonshot's QK-Clip on top of Muon: QK-Clip and the already-landed QK-norm
(RMSNorm on q/k before attention, see "Attention backend note" above)
both exist to bound exploding attention logits, and applying both would
be redundant.


## Pretraining mixture overhaul: grad accumulation, persistent streams, per-config code bias

Three fixes to `smrt/train.py`'s pretraining loop, made together because the
first two were release-blocking correctness bugs found while wiring the
third:

1. **`pick_batch_schedule` is now actually consumed.** `smrt/device.py`'s
   memory-tier `(micro_batch, grad_accum)` split (already unit-tested in
   `tests/test_device.py`) previously had no caller in `train_loop` — every
   config's `train.base_grad_accum` was parsed and silently ignored, and
   `train.base_micro_batch` was used as-is regardless of detected VRAM.
   `train_loop` now calls `pick_batch_schedule` once up front and runs an
   inner accumulation loop (`grad_accum` forward/backward passes per outer
   `step`, each loss scaled by `1/grad_accum`, a single
   `clip_grad_norm_`/`optimizer.step()` pair after the inner loop) so
   `checkpoint_every`, the LR schedule, and `memory_diagnostics.jsonl`
   (still exactly one line per outer `step`) all keep their existing
   per-outer-step meaning.
2. **Batch streams are now constructed once per run, not once per call.**
   `_get_agent_batch`'s `"code"`/`"text"` branches (and `_get_batch`'s
   non-agent `"text"` branch) used to build a brand-new `streaming=True` HF
   dataset iterator on every single call, always starting from the front of
   the corpus — every training step re-read the same first
   `seq_len * micro_batch` tokens forever, regardless of dataset size.
   `train_loop` now builds a `streams: dict` once before the step loop
   (`streams["code"]`/`streams["text"]` for the agent-vocab path,
   `streams["text"]` for the plain-text path) and threads it through
   `_get_batch`/`_get_agent_batch`, which pull `next(...)` from the same
   generator object on every call.
3. **The code/text mixture ratio is per-config, not a hardcoded module
   constant.** `TrainConfig.batch_source_schedule` (optional; validated by
   `smrt/config.py::_build_batch_source_schedule` against the same
   `{"needle", "tool_needle", "code", "text"}` set `AGENT_BATCH_SOURCE_SCHEDULE`
   already used) lets each `configs/agent_*.yaml` declare its own
   code-vs-text ratio. Smaller agent models are biased toward code and away
   from general prose, in inverse proportion to size: `agent_1b.yaml` is
   65% code / 15% text, `agent_2b.yaml` 55%/25%, `agent_4b.yaml` 40%/40%,
   `agent_8b.yaml` 25%/55% (the needle/tool_needle 20% share is unchanged
   across all four). Configs that omit the key (`agent_tiny_cpu.yaml`, and
   any non-agent-vocab config, which never reads it) fall back to the
   original `AGENT_BATCH_SOURCE_SCHEDULE` module constant.

The code half of the new mixture is `bigcode/starcoderdata`
(`smrt/data/code.py::load_starcoder_stream`) rather than the user's
originally-requested `nvidia/Nemotron-Pretraining-Code-v3`: that dataset's
row schema contains only `commit_id`/`rel_path`/`language` metadata, no file
content, and is not usable for streaming text at all. `starcoderdata` is
gated `auto` (one-click accept, no manual review) with a confirmed `content`
field. The text half is `nvidia/Nemotron-CC-v2`
(`smrt/data/pretrain.py::load_nemotron_cc_stream`), gated `manual`; see
RESULTS.md for the exact field-name verification caveat and fallback.

## 2B-token pretraining budget rationale

At ~50M non-embedding parameters, Chinchilla-optimal training is roughly
1B tokens (≈20× params). SMaRT's config targets **2B tokens**: the extra
headroom accounts for the curriculum's longer-context stages seeing fewer
*examples* per token (each example is longer, so a fixed token budget
yields fewer independent training examples at the 32K-context stage than
at the 2K-context stage), without exceeding a few-day single-GPU 7900 XT
training budget. This is a target to record for the eventual real GPU
training run — it is not achievable or verifiable on this session's
CPU-only development machine.

## Curriculum schedule

(from `configs/schedule_curriculum.yaml`, embedded identically in
`configs/base_50m.yaml`)

| min_step | context_len | haystack_filler_tokens | needle_depth_bins |
|---------:|------------:|------------------------:|-------------------:|
| 0        | 2048        | 1800                     | 5                   |
| 2000     | 8192        | 7900                     | 5                   |
| 6000     | 32768       | 32500                    | 5                   |

Every 5th training batch (`step % 5 == 0`) is a needle-recall batch at the
active stage's context length/depth-bin count; the other 4 are plain
pretrain-stream continuation batches. This 1-in-5 ratio
(`CURRICULUM_NEEDLE_RATIO = 4` in `smrt/train.py`) is a fixed design
choice, not config-exposed.

## Validated environment (this session)

This development session validated **CPU-only** correctness (PyTorch
2.14.0+cpu, Python 3.11.15, `tiny_cpu.yaml` dimensions): model construction,
forward/backward, the full `pytest` suite (15/15 passing — memory update
math, attention mask boundary, needle data integrity, checkpoint-resume
continuity), a 20-step training smoke run, and the evaluation pipeline
end-to-end (including `--dry-run-1m`'s fixed-footprint assertion). It did
**not** run on ROCm or CUDA hardware, and did not run a real pretraining
pass or a real long-context recall evaluation — both require the GPU
machine the user targets. See `RESULTS.md` for the exact measured numbers
from this session.

### ROCm gotchas

None recorded — this session had no ROCm hardware available.
**Whoever runs `scripts/check_env.py` on a real ROCm machine (the 7900 XT
box) should append any ROCm-specific gotchas discovered there to this
section** (e.g. `torch.version.hip` detection quirks, `bf16` support
caveats, SDPA backend availability under ROCm, memory-tier boundary
behavior on a 20GB card). Nothing ROCm-specific is fabricated here from
this CPU-only session.
