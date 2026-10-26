"""Featurizer contract: turns the state of each lockstep environment into an observation."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from autotsaugment.core.config import ComponentConfig

Observation = dict[str, Tensor]
"""Named float tensors with a leading dimension of size k, one row per lockstep environment."""

ObservationSpec = dict[str, tuple[int, ...]]
"""Shape of each observation entry, without the leading k."""


@dataclass(frozen=True)
class EpochRecord:
    """What the environment knows after one step, for all k environments at once.

    Attributes:
        epoch: Index of the step just completed, from 1.
        epochs: Steps per episode, E.
        train: Training statistics stacked over environments, each of shape ``(k,)``.
        scores: Skill score per dataset on the reward subset, shape ``(k, d)``.
        aggregate: Geometric mean of ``scores`` over datasets, shape ``(k,)``.
        action: The action that produced the step, shape ``(k, a)``.
    """

    epoch: int
    epochs: int
    train: dict[str, Tensor]
    scores: Tensor
    aggregate: Tensor
    action: Tensor


class FeaturizerConfig(ComponentConfig):
    """Config shared by all featurizers."""


class Featurizer[ConfigT: FeaturizerConfig](ABC):
    """Builds the observation dict from the epoch records of an episode.

    Implementations take ``__init__(self, *, config, n_datasets, action_dim, batch_envs, device)``.

    :meth:`state_dict` and :meth:`load_state_dict` carry the cross-episode state only, for
    example the running normalization moments. The per-episode buffers are not part of the
    state: :meth:`reset` clears them. A featurizer without cross-episode state keeps the
    defaults: an empty state.
    """

    config: ConfigT

    @property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        """Return the shape of each observation entry."""

    @abstractmethod
    def reset(self) -> Observation:
        """Forget the previous episode and return the observation before the first step."""

    @abstractmethod
    def __call__(self, record: EpochRecord) -> Observation:
        """Return the observation after ``record``, updating running statistics."""

    def state_dict(self) -> dict[str, Tensor]:
        """Return the cross-episode state, for a checkpoint. The default is empty.

        The state holds what persists across episodes, for example normalization moments. It
        does not hold the buffers of the current episode.
        """
        return {}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore the state that :meth:`state_dict` returned, on the featurizer's device.

        The default accepts only an empty state.

        Raises:
            ValueError: If ``state`` does not match the state of this featurizer.
        """
        if state:
            msg = f"{type(self).__name__} has no state, got the keys {sorted(state)}"
            raise ValueError(msg)
