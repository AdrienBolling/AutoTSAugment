"""TorchRL view of an :class:`~autotsaugment.core.environment.Environment`.

:class:`TorchRLEnv` exposes the K lockstep environments as one TorchRL environment with
``batch_size = [K]``. It lets TorchRL tools (``check_env_specs``, ``rollout``, collectors) run
on the environment contract. The agents of this package do not need it for training: they drive
the environment through :func:`~autotsaugment.components.agents.loop.run_episode`, which adds
no host synchronization. TorchRL's own reset checks read the ``done`` flags on the host.

This module also holds the tensordict keys and the one typed constructor of tensordicts that
the TorchRL agents share, so the loose typing of ``tensordict`` stays in one place.
"""

from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from autotsaugment.components.agents.loop import episode_seed
from autotsaugment.core.environment import Environment
from autotsaugment.core.featurizer import Observation, ObservationSpec

OBSERVATION_KEY = "observation"
ACTION_KEY = "action"
REWARD_KEY = "reward"
DONE_KEY = "done"
TERMINATED_KEY = "terminated"
NEXT_KEY = "next"


def make_tensordict(
    values: Mapping[str, Tensor | Mapping[str, Tensor]],
    batch_size: torch.Size,
    device: torch.device,
) -> TensorDict:
    """Return a tensordict with ``values``, one level of nesting allowed.

    Args:
        values: Tensors by key; a mapping value becomes a nested tensordict, for example
            ``"next"``.
        batch_size: The leading dimensions shared by every tensor.
        device: Device of the tensordict.
    """
    data = TensorDict(batch_size=batch_size, device=device)
    for key, value in values.items():
        if isinstance(value, Tensor):
            data.set(key, value)
        else:
            for inner, tensor in value.items():
                data.set((key, inner), tensor)
    return data


def observation_composite(
    spec: ObservationSpec, batch_envs: int, device: torch.device
) -> Composite:
    """Return the TorchRL spec of the observation dict, one ``Unbounded`` entry per key.

    Args:
        spec: Shape of each observation entry, without the leading k.
        batch_envs: The number of lockstep environments, k.
        device: Device of the observation.
    """
    entries = {
        name: Unbounded(shape=torch.Size([batch_envs, *shape]), device=device, dtype=torch.float32)
        for name, shape in sorted(spec.items())
    }
    return Composite(entries, shape=torch.Size([batch_envs]), device=device)


class TorchRLEnv(EnvBase):
    """A TorchRL environment over the K lockstep environments of an ``Environment``.

    Every episode has exactly E steps and all K environments end together, so ``done`` and
    ``terminated`` are equal and True for all K at step E. The observation includes the
    progress of the episode, so the end of an episode is a true terminal state. A reset always
    resets all K environments: partial resets do not exist in this contract.

    Episode i after :meth:`set_seed` (or after construction) uses the seed
    ``derive_seed(seed, "episode", i)``, the same seeds as the agents of this package.

    Args:
        env: The environment to wrap.
        seed: Root seed of the episode seeds.
    """

    def __init__(self, env: Environment[Any], *, seed: int = 0) -> None:
        k = env.batch_size
        device = env.device
        super().__init__(device=device, batch_size=torch.Size([k]))
        self.environment = env
        self._root_seed = seed
        self._episode = 0
        space = env.action_space.to(device)
        flag_shape = torch.Size([k, 1])
        self.observation_spec = observation_composite(env.observation_spec, k, device)
        self.action_spec = Bounded(
            low=space.low.expand(k, -1).clone(),
            high=space.high.expand(k, -1).clone(),
            shape=torch.Size([k, space.dim]),
            device=device,
            dtype=torch.float32,
        )
        self.reward_spec = Unbounded(shape=flag_shape, device=device, dtype=torch.float32)
        self.done_spec = Composite(
            {
                DONE_KEY: Categorical(2, shape=flag_shape, dtype=torch.bool, device=device),
                TERMINATED_KEY: Categorical(2, shape=flag_shape, dtype=torch.bool, device=device),
            },
            shape=torch.Size([k]),
            device=device,
        )

    @property
    def episode(self) -> int:
        """Return the index of the next episode since the last seed."""
        return self._episode

    def _output(self, observation: Observation, done: Tensor) -> TensorDict:
        """Return the observation and the done flags as one tensordict of batch ``[K]``."""
        flags = done.reshape(*self.batch_size, 1).to(torch.bool)
        values: dict[str, Tensor | Mapping[str, Tensor]] = {
            name: observation[name] for name in sorted(observation)
        }
        values[DONE_KEY] = flags
        values[TERMINATED_KEY] = flags.clone()
        return make_tensordict(values, self.batch_size, self.environment.device)

    def _reset(self, tensordict: TensorDictBase | None, **kwargs: Any) -> TensorDictBase:
        """Reset all K environments with the next episode seed."""
        del tensordict, kwargs
        seed = episode_seed(self._root_seed, self._episode)
        self._episode += 1
        observation = self.environment.reset(seed=seed)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.environment.device)
        return self._output(observation, done)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Step all K environments with the action of ``tensordict``."""
        action: Tensor = tensordict.get(ACTION_KEY)
        result = self.environment.step(action)
        output = self._output(result.observation, result.done)
        output.set(REWARD_KEY, result.reward.reshape(*self.batch_size, 1).to(torch.float32))
        return output

    def _set_seed(self, seed: int | None) -> None:
        """Restart the episode seeds from ``seed``. ``None`` keeps the current root seed."""
        if seed is not None:
            self._root_seed = seed
        self._episode = 0
