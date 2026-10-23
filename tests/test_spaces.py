import pytest
import torch

from autotsaugment.core.spaces import BoxSpace


def test_sample_inside_and_clip(generator: torch.Generator):
    space = BoxSpace(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]), ("x", "y"))
    sample = space.sample(100, generator=generator)
    assert sample.shape == (100, 2)
    assert space.contains(sample)
    clipped = space.clip(torch.tensor([[5.0, -5.0]]))
    torch.testing.assert_close(clipped, torch.tensor([[1.0, 0.0]]))
    assert space.dim == 2


def test_validation():
    with pytest.raises(ValueError, match="same shape"):
        BoxSpace(torch.zeros(2), torch.zeros(3), ("a", "b"))
    with pytest.raises(ValueError, match="names"):
        BoxSpace(torch.zeros(2), torch.ones(2), ("a",))
    with pytest.raises(ValueError, match="low must be"):
        BoxSpace(torch.ones(2), torch.zeros(2), ("a", "b"))
