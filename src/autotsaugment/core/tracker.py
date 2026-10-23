"""Metric tracking contract.

Trackers buffer step metrics on the device and convert them to host values once per
episode, so a tracker never adds a synchronization point inside the RL loop.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

Metrics = Mapping[str, Tensor | float | int]
"""Metric values. Tensors stay on their device until :meth:`Tracker.flush`."""


class Tracker(ABC):
    """Collect metrics and parameters of one run."""

    @abstractmethod
    def log_params(self, params: Mapping[str, object]) -> None:
        """Record the run parameters once, usually the resolved config."""

    @abstractmethod
    def log_step(self, step: int, metrics: Metrics) -> None:
        """Buffer the metrics of one RL step. Must not synchronize with the host."""

    @abstractmethod
    def log_episode(self, episode: int, metrics: Metrics) -> None:
        """Buffer the metrics of one episode, then flush the buffer."""

    @abstractmethod
    def log_artifact(self, path: Path) -> None:
        """Record a file produced by the run, for example a plot or a checkpoint."""

    @abstractmethod
    def flush(self) -> None:
        """Write every buffered metric to the backend. This is the one synchronization point."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release the backend."""


def to_host(metrics: Metrics) -> dict[str, float]:
    """Return ``metrics`` as plain floats, moving tensors to the host in one batch.

    Args:
        metrics: Scalar values or 0-d tensors. Tensors on the same device are stacked and
            transferred together, which costs one synchronization for the whole mapping.
    """
    scalars: dict[str, float] = {}
    tensors: dict[str, Tensor] = {}
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            tensors[key] = value.detach().reshape(())
        else:
            scalars[key] = float(value)
    if tensors:
        stacked = torch.stack(list(tensors.values())).to("cpu", torch.float64)
        scalars.update(zip(tensors.keys(), stacked.tolist(), strict=True))
    return scalars
