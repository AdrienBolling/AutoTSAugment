"""Forecaster contract: the model the RL agent trains, one epoch per step."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.typechecking import Context, QuantileLevels, Quantiles, SeriesBatch


@dataclass(frozen=True)
class TrainStats:
    """Statistics of one training epoch. Every field is a 0-d tensor on the model device.

    Attributes:
        loss: Mean training loss over the epoch.
        loss_last: Mean loss of the last optimizer step of the epoch.
        grad_norm: Mean global gradient norm over the epoch, before clipping.
        lr: Learning rate at the end of the epoch.
        n_steps: Optimizer steps in the epoch.
    """

    loss: Tensor
    loss_last: Tensor
    grad_norm: Tensor
    lr: Tensor
    n_steps: Tensor

    def as_dict(self) -> dict[str, Tensor]:
        """Return the fields as a mapping, in declaration order."""
        return {
            "loss": self.loss,
            "loss_last": self.loss_last,
            "grad_norm": self.grad_norm,
            "lr": self.lr,
            "n_steps": self.n_steps,
        }


class TSModelConfig(ComponentConfig):
    """Config shared by all tsmodels.

    Attributes:
        context_length: Context steps of a training window.
        horizon_length: Horizon steps of a training window.
        batch_size: Series per optimizer step.
        learning_rate: Peak learning rate of the episode schedule.
        weight_decay: AdamW weight decay.
        grad_clip: Global gradient-norm clip, or ``None``.
        mixed_precision: Use bf16 autocast in the forward pass.
        compile: Wrap the model with ``torch.compile``.
    """

    context_length: int = 768
    horizon_length: int = 256
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float | None = 1.0
    mixed_precision: bool = True
    compile: bool = False


class TSModel[ConfigT: TSModelConfig](ABC):
    """A probabilistic forecaster trained on synthetic series.

    The environment calls :meth:`reset` at the start of each episode, :meth:`train_epoch` once
    per step, and :meth:`forecast` to score the reward and comparison windows.

    Implementations take ``__init__(self, *, config, device, episode_epochs)``, where
    ``episode_epochs`` lets the learning-rate schedule span the episode.
    """

    config: ConfigT

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device of the parameters."""

    @property
    @abstractmethod
    def quantile_levels(self) -> QuantileLevels:
        """Return the quantile levels of :meth:`forecast`, increasing, shape ``(q,)``."""

    @abstractmethod
    def reset(self, *, seed: int) -> None:
        """Return the model to its initial weights and reset the optimizer and schedule.

        Bare models re-initialize from ``seed``. Pretrained models reload the checkpoint and
        re-initialize the adapters from ``seed``.
        """

    @abstractmethod
    def train_epoch(self, series: SeriesBatch, *, generator: torch.Generator) -> TrainStats:
        """Train for one epoch on ``series`` and return the epoch statistics.

        Args:
            series: Synthetic series, shape ``(n, t)``, on :attr:`device`.
            generator: Device generator for window positions and shuffling.
        """

    @abstractmethod
    def forecast(self, context: Context, horizon: int) -> Quantiles:
        """Return quantile forecasts without gradient, shape ``(n, horizon, q)``.

        Args:
            context: Observed context, shape ``(n, c)``. NaN marks a missing value.
            horizon: Number of steps to forecast.
        """

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the trainable state, enough to restore the model with :meth:`load_state_dict`."""

    @abstractmethod
    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the state returned by :meth:`state_dict`."""

    @abstractmethod
    def n_trainable_parameters(self) -> int:
        """Return the number of parameters the optimizer updates."""


TSModelFactory = Callable[[], TSModel[Any]]
"""Builds one fresh tsmodel. The environment calls it K times for K lockstep environments."""
