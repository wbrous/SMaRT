import torch
import pytest

from smrt.config import load_config


@pytest.fixture(scope="session")
def tiny_config():
    return load_config("configs/tiny_cpu.yaml")


@pytest.fixture
def seeded_rng():
    prior_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    yield
    torch.random.set_rng_state(prior_state)
