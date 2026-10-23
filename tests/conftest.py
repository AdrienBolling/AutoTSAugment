"""Test configuration: runtime shape checks on, CPU device, seeded generator."""

import os

os.environ.setdefault("AUTOTSAUGMENT_TYPECHECK", "1")

import pytest
import torch

from autotsaugment.core.seeding import make_generator


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def generator(device: torch.device) -> torch.Generator:
    return make_generator(device, 0)
