"""Gymnasium view of an :class:`~autotsaugment.core.environment.Environment` with K = 1.

:class:`GymEnv` takes and returns NumPy arrays, so other RL libraries can run on the
environment contract for cross-checks. Each call moves data between the device and the host, so
this adapter is not for the v1 experiments.
"""

from typing import Any, SupportsFloat

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from numpy.typing import NDArray

from autotsaugment.components.agents.loop import episode_seed
from autotsaugment.core.environment import Environment
from autotsaugment.core.featurizer import Observation

GymObservation = dict[str, Any]
"""One float32 array per observation entry; ``Any`` matches the type of ``spaces.Dict``."""
GymAction = NDArray[np.float32]
FLOAT32_MAX = float(np.finfo(np.float32).max)
"""Bound of the observation boxes: the observations are unbounded but finite."""


class GymEnv(gymnasium.Env[GymObservation, GymAction]):
    """A gymnasium environment over one environment of an ``Environment`` with K = 1.

    The observation space is a ``Dict`` with one ``Box`` over the float32 range per entry. The
    action space is the ``Box`` of the action space. ``terminated`` is True at step E;
    ``truncated`` is always False.

    ``reset(seed=s)`` restarts the episode seeds from ``s``: the episode i after it uses
    ``derive_seed(s, "episode", i)``, the same seeds as the agents of this package.

    Args:
        env: The environment to wrap, with ``batch_size == 1``.
        seed: Root seed of the episode seeds until the first seeded reset.

    Raises:
        ValueError: If ``env`` has more than one lockstep environment.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: Environment[Any], *, seed: int = 0) -> None:
        if env.batch_size != 1:
            msg = f"GymEnv needs an environment with batch_size 1, got {env.batch_size}"
            raise ValueError(msg)
        self.environment = env
        self.render_mode = None
        self._root_seed = seed
        self._episode = 0
        box = env.action_space
        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(-FLOAT32_MAX, FLOAT32_MAX, shape=shape, dtype=np.float32)
                for name, shape in sorted(env.observation_spec.items())
            }
        )
        self.action_space = spaces.Box(
            low=box.low.detach().cpu().numpy().astype(np.float32),
            high=box.high.detach().cpu().numpy().astype(np.float32),
            dtype=np.float32,
        )

    @staticmethod
    def _to_numpy(observation: Observation) -> GymObservation:
        """Return the single row of each observation entry as a float32 array."""
        return {
            name: observation[name][0].detach().cpu().numpy().astype(np.float32)
            for name in sorted(observation)
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[GymObservation, dict[str, Any]]:
        """Start the next episode; a seed restarts the episode seeds from it."""
        del options
        super().reset(seed=seed)
        if seed is not None:
            self._root_seed = seed
            self._episode = 0
        observation = self.environment.reset(seed=episode_seed(self._root_seed, self._episode))
        self._episode += 1
        return self._to_numpy(observation), {}

    def step(
        self, action: GymAction
    ) -> tuple[GymObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one step with ``action`` of shape ``(a,)``."""
        values = np.asarray(action, dtype=np.float32).reshape(1, -1)
        tensor = torch.from_numpy(values).to(self.environment.device)
        result = self.environment.step(tensor)
        info = {key: value[0].detach().cpu().numpy() for key, value in result.info.items()}
        return (
            self._to_numpy(result.observation),
            float(result.reward[0]),
            bool(result.done[0]),
            False,
            info,
        )
