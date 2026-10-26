"""Policies without learned parameters, and helpers that every agent shares."""

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from autotsaugment.core.environment import Environment
from autotsaugment.core.featurizer import Observation
from autotsaugment.core.spaces import BoxSpace

CHECKPOINT_NAME = "agent.pt"
"""File name of the agent state inside the directory given to ``save`` and ``load``."""

ACTION_BATCH_NDIM = 2
"""Dimensions of a batch of actions, ``(k, a)``."""


def observation_batch_size(observation: Observation) -> int:
    """Return k, the leading dimension of the observation entries.

    Raises:
        ValueError: If the observation has no entry.
    """
    for value in observation.values():
        return int(value.shape[0])
    msg = "the observation has no entry, so its batch size is unknown"
    raise ValueError(msg)


class ConstantActionPolicy:
    """Returns the same action for every environment and every step.

    Args:
        action: The action, shape ``(a,)``.
    """

    def __init__(self, action: Tensor) -> None:
        if action.ndim != 1:
            msg = f"expected an action of shape (a,), got {tuple(action.shape)}"
            raise ValueError(msg)
        self.action = action

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return the action repeated k times, shape ``(k, a)``."""
        del deterministic
        k = observation_batch_size(observation)
        return self.action.unsqueeze(0).expand(k, -1).clone()


class SlotActionPolicy:
    """Returns a fixed action per lockstep environment: row j goes to environment j.

    Args:
        actions: One action per environment, shape ``(k, a)``.
    """

    def __init__(self, actions: Tensor) -> None:
        if actions.ndim != ACTION_BATCH_NDIM:
            msg = f"expected actions of shape (k, a), got {tuple(actions.shape)}"
            raise ValueError(msg)
        self.actions = actions

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return the actions, shape ``(k, a)``."""
        del deterministic
        k = observation_batch_size(observation)
        if k != self.actions.shape[0]:
            msg = f"the policy holds {self.actions.shape[0]} actions, the observation has {k} rows"
            raise ValueError(msg)
        return self.actions.clone()


class RandomPolicy:
    """Returns actions drawn uniformly inside the box, from its own generator.

    Args:
        space: The action box.
        generator: Generator on the device of the box. It owns all randomness of the policy.
    """

    def __init__(self, space: BoxSpace, generator: torch.Generator) -> None:
        self.space = space
        self.generator = generator

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return k new uniform actions. The policy is random also when ``deterministic``."""
        del deterministic
        return self.space.sample(observation_batch_size(observation), generator=self.generator)


def uniform_action(env: Environment[Any]) -> Tensor:
    """Return the uniform action of the augmentation method of ``env``, shape ``(a,)``.

    The function calls ``env.uniform_action()`` and clips the result to the action box. For
    KernelSynth the uniform action is all zeros, the Chronos recipe.
    """
    space = env.action_space
    action = env.uniform_action().to(device=space.device, dtype=space.low.dtype)
    return space.clip(action.unsqueeze(0)).squeeze(0)


def not_fitted(agent: object) -> RuntimeError:
    """Return the error for a call to ``policy`` or ``save`` before ``fit`` or ``load``."""
    name = type(agent).__name__
    return RuntimeError(f"{name} has no policy yet: call fit(env, ...) or load(path) first")


def checkpoint_file(path: Path) -> Path:
    """Return the checkpoint file inside the agent directory ``path``."""
    return path / CHECKPOINT_NAME


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write ``state`` with ``torch.save`` to ``path/agent.pt``, and create ``path``."""
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_file(path))


def load_state(path: Path) -> dict[str, Any]:
    """Read the state that :func:`save_state` wrote, with every tensor on the CPU.

    Generator states must stay on the CPU, so the callers move the other tensors to the device.
    """
    return torch.load(checkpoint_file(path), map_location="cpu", weights_only=True)


def space_state(space: BoxSpace) -> dict[str, Any]:
    """Return the box as plain tensors and strings, for a checkpoint."""
    return {"low": space.low, "high": space.high, "names": list(space.names)}


def space_from_state(state: dict[str, Any], device: torch.device) -> BoxSpace:
    """Rebuild the box that :func:`space_state` stored, on ``device``."""
    return BoxSpace(
        low=state["low"].to(device),
        high=state["high"].to(device),
        names=tuple(state["names"]),
    )
