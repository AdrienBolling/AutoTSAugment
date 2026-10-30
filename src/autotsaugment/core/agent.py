"""Agent contract: anything that turns an environment into a policy."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.environment import Environment
from autotsaugment.core.featurizer import Observation
from autotsaugment.core.tracker import Tracker


class Policy(Protocol):
    """Maps a batched observation to a batched action."""

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return one action per environment, shape ``(k, a)``, inside the action space."""


@runtime_checkable
class StatefulPolicy(Policy, Protocol):
    """A policy with an episode memory.

    The episode loop calls :meth:`reset` once per episode, right after ``env.reset``, so the
    policy starts each episode with an empty memory.
    """

    def reset(self) -> None:
        """Forget the current episode. Called right after ``env.reset``."""


class AgentConfig(ComponentConfig):
    """Config shared by all agents.

    Attributes:
        episodes: Training episodes the agent may consume. Baselines may need zero.
    """

    episodes: int = 100


class Agent[ConfigT: AgentConfig](ABC):
    """Produces a :class:`Policy` for an environment.

    RL agents train on the environment inside :meth:`fit`. Baselines (fixed action, random,
    CMA-ES) implement :meth:`fit` with their own loop or with no training at all.

    Implementations take ``__init__(self, *, config, device)``.
    """

    config: ConfigT

    @abstractmethod
    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Train on ``env`` within ``config.episodes`` and return the resulting policy."""

    @abstractmethod
    def policy(self) -> Policy:
        """Return the current policy without training."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Write the agent state under ``path``."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore the state written by :meth:`save`."""
