"""Augmentation method contract: a synthetic series generator with an action space."""

from abc import ABC, abstractmethod

import torch
from pydantic import Field

from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.typechecking import Action, ActionBatch, SeriesBatchK


class AugmentationConfig(ComponentConfig):
    """Config shared by all augmentation methods.

    Attributes:
        series_length: Length of every generated series, in time steps.
    """

    series_length: int = Field(default=1024, ge=2)


class AugmentationMethod[ConfigT: AugmentationConfig](ABC):
    """A generator of synthetic series whose behaviour is set by a continuous action.

    One action describes the generation distribution of one epoch. The method samples ``n``
    series per lockstep environment from that distribution, on the device, with no host
    synchronization.

    Implementations take ``__init__(self, *, config, device)``.
    """

    config: ConfigT

    @property
    @abstractmethod
    def action_space(self) -> BoxSpace:
        """Return the box of valid actions, with one name per dimension."""

    @property
    def series_length(self) -> int:
        """Return the length of every generated series."""
        return self.config.series_length

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device the series are generated on."""

    @abstractmethod
    def uniform_action(self) -> Action:
        """Return the action that reproduces the method's original random recipe.

        For KernelSynth this is the Chronos recipe: uniform kernels, uniform count, equal
        operator probability. It is the fixed baseline of every experiment.
        """

    @abstractmethod
    def sample(self, action: ActionBatch, n: int, *, generator: torch.Generator) -> SeriesBatchK:
        """Return ``n`` series per action.

        Args:
            action: One action per lockstep environment, shape ``(k, a)``.
            n: Number of series per environment.
            generator: Device generator that owns all randomness of the call.

        Returns:
            Series of shape ``(k, n, series_length)``, finite, on :attr:`device`.
        """

    @abstractmethod
    def describe(self, action: Action) -> dict[str, float]:
        """Return human-readable statistics of one action, for episode logs.

        This method may synchronize with the host. Call it once per episode, not per step.
        """
