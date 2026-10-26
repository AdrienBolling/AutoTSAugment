"""Fixed-action baseline: the same action at every step of every episode.

With no configured action, the agent uses ``env.uniform_action()``, the uniform action of the
augmentation method. For KernelSynth it is the Chronos recipe, the baseline of experiment 1.
"""

from pathlib import Path
from typing import Any

import torch
from pydantic import Field

from autotsaugment.components.agents.policies import (
    ConstantActionPolicy,
    load_state,
    not_fitted,
    save_state,
    uniform_action,
)
from autotsaugment.core.agent import Agent, AgentConfig, Policy
from autotsaugment.core.environment import Environment
from autotsaugment.core.tracker import Tracker


class FixedActionAgentConfig(AgentConfig):
    """Config of :class:`FixedActionAgent`.

    Attributes:
        episodes: Training episodes. The agent does not train, so the default is zero.
        action: The action, one value per action dimension. ``None`` selects the uniform
            action of the environment's augmentation method when :meth:`FixedActionAgent.fit`
            runs.
    """

    episodes: int = Field(default=0, ge=0)
    action: list[float] | None = None


class FixedActionAgent(Agent[FixedActionAgentConfig]):
    """Returns a constant policy: the configured action, or the uniform action.

    Args:
        config: The agent config.
        device: Device of the action.
    """

    def __init__(self, *, config: FixedActionAgentConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._action: torch.Tensor | None = None
        if config.action is not None:
            self._action = torch.tensor(config.action, dtype=torch.float32, device=self.device)

    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Bind the action to ``env`` and return the policy. Runs no episode.

        The action is clipped to the action box of ``env``.

        Raises:
            ValueError: If the configured action does not have one value per dimension.
        """
        del tracker, seed
        space = env.action_space.to(self.device)
        if self.config.action is None:
            action = uniform_action(env).to(self.device)
        else:
            action = torch.tensor(self.config.action, dtype=torch.float32, device=self.device)
            if action.shape != (space.dim,):
                msg = f"the action has {action.shape[0]} values, the action space has {space.dim}"
                raise ValueError(msg)
        self._action = space.clip(action.unsqueeze(0)).squeeze(0)
        return self.policy()

    @property
    def action(self) -> torch.Tensor | None:
        """Return the bound action, shape ``(a,)``, or ``None`` before :meth:`fit`."""
        return self._action

    def policy(self) -> Policy:
        """Return the constant policy.

        Raises:
            RuntimeError: If the config has no action and :meth:`fit` did not run yet.
        """
        if self._action is None:
            raise not_fitted(self)
        return ConstantActionPolicy(self._action)

    def save(self, path: Path) -> None:
        """Write the action to ``path/agent.pt``.

        Raises:
            RuntimeError: If no action is bound yet.
        """
        if self._action is None:
            raise not_fitted(self)
        save_state(path, {"action": self._action})

    def load(self, path: Path) -> None:
        """Restore the action that :meth:`save` wrote under ``path``."""
        self._action = load_state(path)["action"].to(self.device)
