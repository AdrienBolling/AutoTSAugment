"""Environment contract: batched, device-resident, library-agnostic.

The environment owns K tsmodels trained in lockstep, one augmentation method, one benchmark,
one featurizer and one reward function. Adapters in ``components/agents`` expose it to TorchRL
and gymnasium.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from autotsaugment.core.config import ComponentConfig, EpisodeConfig
from autotsaugment.core.featurizer import Observation, ObservationSpec
from autotsaugment.core.spaces import BoxSpace


@dataclass(frozen=True)
class StepResult:
    """Outcome of one step for all k environments.

    Attributes:
        observation: The next observation.
        reward: Shape ``(k,)``.
        done: Shape ``(k,)``, True when the episode ended at this step.
        info: Diagnostics with a leading k, for logs: per-dataset scores, train stats,
            action statistics. Never read by agents.
    """

    observation: Observation
    reward: Tensor
    done: Tensor
    info: dict[str, Tensor] = field(default_factory=dict)


class EnvironmentConfig(ComponentConfig):
    """Config shared by all environments.

    Attributes:
        episode: The episode budget E, N, K.
        reward_subset_per_dataset: Reward windows scored per step and dataset; the subset is
            redrawn every step. ``None`` scores the full reward set.
        score_batch_size: Windows per forward pass when scoring.
    """

    episode: EpisodeConfig = EpisodeConfig()
    reward_subset_per_dataset: int | None = 128
    score_batch_size: int = 1024


class Environment[ConfigT: EnvironmentConfig](ABC):
    """K lockstep training runs seen as one batched environment.

    All tensors live on :attr:`device`. Every episode has exactly
    ``config.episode.epochs`` steps, so ``done`` is True for all k at the last step.

    :meth:`uniform_action` gives the reference action of the augmentation method. For
    KernelSynth it is all zeros, the Chronos recipe. The fixed baseline of every experiment
    uses it.

    :meth:`state_dict` and :meth:`load_state_dict` carry the state a saved policy needs to
    reproduce its observations, for example the normalization moments of the featurizer. The
    experiments save it next to the agent checkpoint. An environment without such state keeps
    the defaults: an empty state.
    """

    config: ConfigT

    @property
    def batch_size(self) -> int:
        """Return K, the number of lockstep environments."""
        return self.config.episode.batch_envs

    @property
    def episode_length(self) -> int:
        """Return E, the number of steps per episode."""
        return self.config.episode.epochs

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device of all tensors."""

    @property
    @abstractmethod
    def action_space(self) -> BoxSpace:
        """Return the action space of the augmentation method."""

    @abstractmethod
    def uniform_action(self) -> Tensor:
        """Return the reference action of the augmentation method, shape ``(a,)``.

        For KernelSynth this is all zeros, the Chronos recipe. The action lives on
        :attr:`device` and lies inside :attr:`action_space`.
        """

    @property
    @abstractmethod
    def observation_spec(self) -> ObservationSpec:
        """Return the shape of each observation entry, without the leading k."""

    @property
    @abstractmethod
    def dataset_names(self) -> tuple[str, ...]:
        """Return the benchmark dataset names, the order of the score vectors."""

    @abstractmethod
    def reset(self, *, seed: int) -> Observation:
        """Start a new episode: reset the K tsmodels, score them once, return the observation."""

    @abstractmethod
    def step(self, action: Tensor) -> StepResult:
        """Generate, train one epoch, score, and return the result.

        Args:
            action: Shape ``(k, a)``, inside :attr:`action_space`.
        """

    @abstractmethod
    def evaluate(self) -> Tensor:
        """Return the skill per dataset on the full comparison set, shape ``(k, d)``.

        Called once at the end of an episode. Never used for the reward.
        """

    def state_dict(self) -> dict[str, Tensor]:
        """Return the state a saved policy needs to reproduce its observations.

        The experiments save it next to the agent checkpoint. The default is empty.
        """
        return {}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore the state that :meth:`state_dict` returned, on :attr:`device`.

        The default accepts only an empty state.

        Raises:
            ValueError: If ``state`` does not match the state of this environment.
        """
        if state:
            msg = f"{type(self).__name__} has no state, got the keys {sorted(state)}"
            raise ValueError(msg)
