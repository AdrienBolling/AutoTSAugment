"""Reward contract: a scalar per lockstep environment, computed on the device."""

from abc import ABC, abstractmethod

from torch import Tensor

from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.featurizer import EpochRecord


class RewardConfig(ComponentConfig):
    """Config shared by all reward functions."""


class RewardFunction[ConfigT: RewardConfig](ABC):
    """Maps the previous and the current epoch record to a reward of shape ``(k,)``.

    Implementations take ``__init__(self, *, config)``.
    """

    config: ConfigT

    @abstractmethod
    def reset(self, initial_aggregate: Tensor) -> None:
        """Start an episode from the aggregate score of the untrained model, shape ``(k,)``."""

    @abstractmethod
    def __call__(self, record: EpochRecord) -> Tensor:
        """Return the reward of the step described by ``record``, shape ``(k,)``."""
