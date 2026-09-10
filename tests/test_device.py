import pytest

from smrt.device import resolve_device, pick_batch_schedule


# If this fails: bf16 autocast is being silently enabled on a CPU backend
# where it was never validated (device.py's "cpu/mps always fp32" contract).
def test_resolve_device_cpu_is_fp32():
    info = resolve_device("cpu")
    assert info.backend == "cpu"
    assert info.dtype.__str__() == "torch.float32"
    assert info.bf16_ok is False


# If this fails: an explicit backend request silently falls back to cpu
# instead of raising when the requested backend is unavailable on this
# machine (this dev machine has no ROCm/CUDA).
def test_resolve_device_rocm_raises_on_cpu_only_machine():
    with pytest.raises(RuntimeError):
        resolve_device("rocm")


# If this fails: the batch-schedule tier boundaries have an off-by-one,
# or the effective batch size (micro_batch * grad_accum) isn't held
# constant across tiers.
def test_pick_batch_schedule_tiers():
    assert pick_batch_schedule(None, 8, 4) == (1, 32)
    assert pick_batch_schedule(20_000_000_000, 8, 4) == (8, 4)
