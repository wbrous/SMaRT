# SMaRT — Sliding-window Memory and Recall Transformer

A ~50M-parameter decoder-only language model combining sliding-window local
attention with a Titans-style ("Memory as Context") neural long-term
memory module. See `DESIGN.md` for the architecture and math, and
`RESULTS.md` for this session's validation results.

The codebase is **device-modular**: the exact same source runs on a ROCm
GPU, a CUDA GPU, or a CPU-only machine — `smrt/device.py` auto-detects the
backend at runtime. No code edits are required to switch machines; only
the `torch` install command below differs.

## Install

Install `torch` first, using the block for your machine
(current tags as of this writing — verify against
<https://pytorch.org/get-started/locally/> if these are stale):

**NVIDIA GPU (CUDA):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
# verify current tag at https://pytorch.org/get-started/locally/
```

**AMD GPU (ROCm):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
# verify current tag at https://pytorch.org/get-started/locally/
```

**CPU-only / Apple Silicon (MPS):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest of the dependencies (this file intentionally does
**not** list `torch`):

```bash
pip install -r requirements.txt
```

## Confirm your backend was detected correctly

```bash
PYTHONPATH=. python scripts/check_env.py
```

Prints the detected backend (`rocm`/`cuda`/`mps`/`cpu`), device, dtype,
and `bf16_ok`. Always exits 0 — it's a diagnostic, not a gate.

## Quickstart: CPU smoke run

Runs a tiny (`d_model=64`, 2 layers) config end-to-end in well under a
minute on any machine, byte-level vocab, no dataset download required:

```bash
python -m smrt.train --config configs/tiny_cpu.yaml
```

Writes checkpoints (`checkpoints/step_10.pt`, `checkpoints/step_20.pt`) and
`checkpoints/memory_diagnostics.jsonl` (per-step loss + neural-memory
"surprise" magnitude — a nonzero, varying trend confirms the memory
mechanism is active, not silently bypassed).

## Real training

```bash
python -m smrt.train --config configs/base_50m.yaml [--resume checkpoints/step_N.pt]
```

`configs/base_50m.yaml` is the ~50M-non-embedding-parameter architecture
(measured, not assumed — see `RESULTS.md`), 2B-token pretrain budget,
32768-token curriculum ceiling. It streams `HuggingFaceFW/fineweb-edu` via
`datasets` (requires network access) tokenized with `tiktoken`'s `gpt2`
encoding. Batch size / gradient-accumulation are chosen automatically per
`smrt/device.py::pick_batch_schedule` based on detected GPU memory, so the
same config runs unmodified on a 20GB 7900 XT, a larger CUDA card, or CPU
(where it'll just be very slow — intended for GPU use).

## Evaluation

```bash
python -m smrt.evaluate --checkpoint checkpoints/step_N.pt \
  --context-lengths 128000,512000,1000000 --depths 0,25,50,75,100 \
  --examples-per-cell 20 --output results.csv \
  [--baseline-checkpoint baseline_step_N.pt]
```

Runs a needle-in-haystack recall grid (accuracy vs. context length vs.
needle depth), writes `results.csv` and a `results_heatmap.png`. Pass
`--dry-run-1m` to sanity-check the fixed-memory-footprint architecture
guarantee at 1M-token scale without running a real 1M-token forward pass.

## Tests

```bash
pytest tests/ -v
```
