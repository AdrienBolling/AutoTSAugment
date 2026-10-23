"""Action spaces. v1 has one kind: a bounded box of continuous values."""

from dataclasses import dataclass

import torch
from torch import Tensor

from autotsaugment.core.typechecking import ActionBatch


@dataclass(frozen=True)
class BoxSpace:
    """A box of continuous actions, ``low <= action <= high`` element-wise.

    Attributes:
        low: Lower bound per dimension, shape ``(a,)``.
        high: Upper bound per dimension, shape ``(a,)``.
        names: One label per dimension, for logs and plots.
    """

    low: Tensor
    high: Tensor
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the bounds."""
        if self.low.shape != self.high.shape or self.low.ndim != 1:
            msg = f"low and high must be 1-D with the same shape, got {self.low.shape} and {self.high.shape}"
            raise ValueError(msg)
        if len(self.names) != self.low.shape[0]:
            msg = f"expected {self.low.shape[0]} names, got {len(self.names)}"
            raise ValueError(msg)
        if bool((self.low > self.high).any()):
            msg = "low must be <= high in every dimension"
            raise ValueError(msg)

    @property
    def dim(self) -> int:
        """Return the number of action dimensions."""
        return int(self.low.shape[0])

    @property
    def device(self) -> torch.device:
        """Return the device of the bounds."""
        return self.low.device

    def to(self, device: torch.device | str) -> "BoxSpace":
        """Return a copy of the space with its bounds on ``device``."""
        return BoxSpace(self.low.to(device), self.high.to(device), self.names)

    def sample(self, n: int, *, generator: torch.Generator) -> ActionBatch:
        """Return ``n`` actions drawn uniformly inside the box."""
        unit = torch.rand((n, self.dim), generator=generator, device=self.low.device)
        return self.low + unit * (self.high - self.low)

    def clip(self, action: ActionBatch) -> ActionBatch:
        """Return ``action`` clipped to the box."""
        return torch.maximum(torch.minimum(action, self.high), self.low)

    def contains(self, action: ActionBatch) -> bool:
        """Return True when every action lies inside the box."""
        return bool(((action >= self.low) & (action <= self.high)).all())
