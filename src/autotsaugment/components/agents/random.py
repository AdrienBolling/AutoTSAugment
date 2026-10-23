"""Random baseline: a new uniform action inside the box at every step."""

from pathlib import Path
from typing import Any

import torch
from pydantic import Field

from autotsaugment.components.agents.policies import (
    RandomPolicy,
    load_state,
    not_fitted,
    save_state,
    space_from_state,
    space_state,
)
from autotsaugment.core.agent import Agent, AgentConfig, Policy
from autotsaugment.core.environment import Environment
from autotsaugment.core.seeding import derive_seed, make_generator
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tracker import Tracker


class RandomAgentConfig(AgentConfig):
    """Config of :class:`RandomAgent`.

    Attributes:
        episodes: Training episodes. The agent does not train, so the default is zero.
    """

    episodes: int = Field(default=0, ge=0)


class RandomAgent(Agent[RandomAgentConfig]):
    """Returns a policy that draws each action uniformly inside the action box.

    Args:
        config: The agent config.
        device: Device of the generator and of the actions.
    """

    _space: BoxSpace | None
    _generator: torch.Generator | None

    def __init__(self, *, config: RandomAgentConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._space = None
        self._generator = None

    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Read the action box of ``env``, seed the generator from ``seed``. Runs no episode."""
        del tracker
        self._space = env.action_space.to(self.device)
        self._generator = make_generator(self.device, derive_seed(seed, "random_agent"))
        return self.policy()

    def policy(self) -> Policy:
        """Return the random policy. It shares the generator of the agent.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._space is None or self._generator is None:
            raise not_fitted(self)
        return RandomPolicy(self._space, self._generator)

    def save(self, path: Path) -> None:
        """Write the box and the generator state to ``path/agent.pt``.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._space is None or self._generator is None:
            raise not_fitted(self)
        save_state(path, {**space_state(self._space), "generator": self._generator.get_state()})

    def load(self, path: Path) -> None:
        """Restore the box and the generator state that :meth:`save` wrote under ``path``."""
        state = load_state(path)
        self._space = space_from_state(state, self.device)
        self._generator = torch.Generator(device=self.device)
        self._generator.set_state(state["generator"])
