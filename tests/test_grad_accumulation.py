import dataclasses
import json

from smrt.train import train_loop


# If this fails: pick_batch_schedule's (micro_batch, grad_accum) split is
# not actually being consumed by train_loop -- optimizer.step() and the
# diagnostics-file write would either run grad_accum times per outer
# step (multiplying wall-clock cost with no batch-size benefit) or not
# run gradient accumulation at all (the original bug: base_grad_accum
# silently ignored).
def test_diagnostics_has_exactly_one_line_per_outer_step_regardless_of_grad_accum(tiny_config, tmp_path):
    cfg = dataclasses.replace(
        tiny_config,
        train=dataclasses.replace(
            tiny_config.train,
            base_micro_batch=4, base_grad_accum=2, max_steps=3, checkpoint_every=100,
            checkpoint_dir=str(tmp_path),
        ),
    )
    train_loop(cfg)
    with open(tmp_path / "memory_diagnostics.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 3
    for line in lines:
        assert "loss" in json.loads(line)
