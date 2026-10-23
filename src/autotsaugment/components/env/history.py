"""A short window of recent values per lockstep environment, with their slope and variance.

The featurizer and the reward both read the training loss of the last epochs. The values stay
on their device, and the number of values is a host-side count, so no method synchronizes
with the host.
"""

from collections import deque

import torch
from torch import Tensor

from autotsaugment.core.typechecking import PerEnv, checked

MIN_SIZE = 2


class RecentValues:
    """The last ``size`` values of one quantity, one value per lockstep environment each time.

    Args:
        size: Number of values kept, at least 2.

    Raises:
        ValueError: If ``size`` is less than 2.
    """

    def __init__(self, size: int) -> None:
        if size < MIN_SIZE:
            msg = f"size must be at least {MIN_SIZE}, got {size}"
            raise ValueError(msg)
        self._values: deque[Tensor] = deque(maxlen=size)

    def __len__(self) -> int:
        """Return the number of values kept now, at most ``size``."""
        return len(self._values)

    def clear(self) -> None:
        """Forget every value."""
        self._values.clear()

    @checked
    def append(self, value: PerEnv) -> None:
        """Add the newest value, shape ``(k,)``. The oldest value goes when the window is full."""
        self._values.append(value)

    def _stacked(self) -> Tensor:
        if not self._values:
            msg = "the window holds no value"
            raise RuntimeError(msg)
        return torch.stack(tuple(self._values), dim=-1)

    @checked
    def slope(self) -> PerEnv:
        """Return the least-squares slope per step of the kept values, shape ``(k,)``.

        A decreasing quantity has a negative slope. With one value, the slope is zero.

        Raises:
            RuntimeError: If the window holds no value.
        """
        values = self._stacked()
        n = values.shape[-1]
        if n < MIN_SIZE:
            return torch.zeros_like(values[..., 0])
        centered = torch.arange(n, device=values.device, dtype=values.dtype) - (n - 1) / 2.0
        return (values * centered).sum(dim=-1) / (n * (n * n - 1) / 12.0)

    @checked
    def variance(self) -> PerEnv:
        """Return the population variance of the kept values, shape ``(k,)``.

        With one value, the variance is zero.

        Raises:
            RuntimeError: If the window holds no value.
        """
        return self._stacked().var(dim=-1, correction=0)
