"""Networks of the TorchRL agents, the squashed Gaussian policy and the transition layout.

The networks read one flat observation vector: :func:`flatten_observation` concatenates the
entries of the observation dict in sorted key order. The actor outputs the location and scale
of a Gaussian over pre-squash actions. ``tanh`` and an affine map squash a sample into the
action box, as :class:`torchrl.modules.TanhNormal` does in the losses.

The acting path (:class:`GaussianPolicy`) does not build a ``TanhNormal``: its constructor
checks its bounds on the host. The policy computes the squashed action directly, with noise from
its own generator.
"""

import math
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager

import torch
from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import Tensor, nn
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

from autotsaugment.components.agents.torchrl.adapter import (
    ACTION_KEY,
    DONE_KEY,
    NEXT_KEY,
    OBSERVATION_KEY,
    REWARD_KEY,
    TERMINATED_KEY,
    make_tensordict,
)
from autotsaugment.core.environment import StepResult
from autotsaugment.core.featurizer import Observation, ObservationSpec
from autotsaugment.core.spaces import BoxSpace

LOC_KEY = "loc"
SCALE_KEY = "scale"


def observation_dim(spec: ObservationSpec) -> int:
    """Return the length of the flat observation vector of ``spec``."""
    return sum(math.prod(shape) for shape in spec.values())


def flatten_observation(observation: Mapping[str, Tensor]) -> Tensor:
    """Concatenate the observation entries in sorted key order into one vector per row.

    Args:
        observation: Entries of shape ``(k, ...)``.

    Returns:
        Shape ``(k, D)``, float32, with ``D`` the sum of the entry sizes.

    Raises:
        ValueError: If the observation has no entry.
    """
    keys = sorted(observation)
    if not keys:
        msg = "the observation has no entry"
        raise ValueError(msg)
    k = observation[keys[0]].shape[0]
    return torch.cat([observation[key].reshape(k, -1).float() for key in keys], dim=-1)


def squash(pre_squash: Tensor, low: Tensor, high: Tensor) -> Tensor:
    """Map a real value into ``[low, high]`` with ``tanh``, as ``TanhNormal`` does."""
    return low + 0.5 * (torch.tanh(pre_squash) + 1.0) * (high - low)


def mlp(
    in_features: int, out_features: int, hidden_sizes: Sequence[int], device: torch.device
) -> MLP:
    """Return a ReLU MLP with the given hidden layer sizes."""
    return MLP(
        in_features=in_features,
        out_features=out_features,
        num_cells=list(hidden_sizes),
        activation_class=nn.ReLU,
        device=device,
    )


def make_gaussian_head(
    in_features: int, action_dim: int, hidden_sizes: Sequence[int], device: torch.device
) -> nn.Sequential:
    """Return the actor head: flat observation to ``(loc, scale)`` of the pre-squash Gaussian.

    The scale is positive (biased softplus of ``NormalParamExtractor``). Both outputs have
    shape ``(..., a)``.

    Args:
        in_features: Length of the flat observation.
        action_dim: Action dimensions, a.
        hidden_sizes: Sizes of the hidden layers.
        device: Device of the parameters.
    """
    return nn.Sequential(
        mlp(in_features, 2 * action_dim, hidden_sizes, device), NormalParamExtractor()
    )


def split_head_output(output: object) -> tuple[Tensor, Tensor]:
    """Return the ``(loc, scale)`` pair of an actor head output, with its types checked.

    Raises:
        TypeError: If ``output`` is not a pair of tensors.
    """
    match output:
        case (Tensor() as loc, Tensor() as scale):
            return loc, scale
        case _:
            msg = f"expected (loc, scale) tensors from the actor head, got {type(output).__name__}"
            raise TypeError(msg)


def make_actor(head: nn.Module, space: BoxSpace) -> ProbabilisticActor:
    """Return the TorchRL actor over ``head``: a ``TanhNormal`` mapped to the box."""
    module = TensorDictModule(head, in_keys=[OBSERVATION_KEY], out_keys=[LOC_KEY, SCALE_KEY])
    return ProbabilisticActor(
        module,
        in_keys=[LOC_KEY, SCALE_KEY],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": space.low, "high": space.high},
        return_log_prob=True,
    )


def make_qvalue(
    in_features: int, action_dim: int, hidden_sizes: Sequence[int], device: torch.device
) -> ValueOperator:
    """Return a Q-network over the flat observation and the action."""
    module = mlp(in_features + action_dim, 1, hidden_sizes, device)
    return ValueOperator(module, in_keys=[OBSERVATION_KEY, ACTION_KEY])


def make_value(
    in_features: int, hidden_sizes: Sequence[int], device: torch.device
) -> ValueOperator:
    """Return a state-value network over the flat observation."""
    return ValueOperator(mlp(in_features, 1, hidden_sizes, device), in_keys=[OBSERVATION_KEY])


class GaussianPolicy:
    """The policy of a trained actor head.

    Deterministic calls return the squashed location, the mean action of the agents. Stochastic
    calls add Gaussian noise from the policy's generator before the squash. The action is
    clipped to the box in both cases.

    Args:
        head: The actor head.
        space: The action box, on the device of the head.
        generator: Generator of the exploration noise, on the same device.
    """

    def __init__(self, head: nn.Module, space: BoxSpace, generator: torch.Generator) -> None:
        self.head = head
        self.space = space
        self.generator = generator

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return one action per environment, shape ``(k, a)``."""
        with torch.no_grad():
            loc, scale = split_head_output(self.head(flatten_observation(observation)))
            if not deterministic:
                noise = torch.randn(
                    loc.shape, generator=self.generator, device=loc.device, dtype=loc.dtype
                )
                loc = loc + scale * noise
            return self.space.clip(squash(loc, self.space.low, self.space.high))


def transition(observation: Observation, action: Tensor, result: StepResult) -> TensorDict:
    """Return the K transitions of one step in the TorchRL layout, batch ``[K]``.

    Keys: ``observation`` (flat), ``action``, and under ``next``: ``observation``,
    ``reward`` ``(K, 1)``, ``done`` and ``terminated`` ``(K, 1)``. Every episode ends at step E
    in a true terminal state, so ``terminated`` equals ``done``.
    """
    k = action.shape[0]
    done = result.done.reshape(k, 1).to(torch.bool)
    return make_tensordict(
        {
            OBSERVATION_KEY: flatten_observation(observation),
            ACTION_KEY: action.detach(),
            NEXT_KEY: {
                OBSERVATION_KEY: flatten_observation(result.observation),
                REWARD_KEY: result.reward.detach().reshape(k, 1).float(),
                DONE_KEY: done,
                TERMINATED_KEY: done.clone(),
            },
        },
        torch.Size([k]),
        action.device,
    )


@contextmanager
def forked_rng(device: torch.device, seed: int) -> Generator[None]:
    """Seed the global torch streams inside the block and restore them after it.

    TorchRL losses and ``nn`` initializers draw from the global stream. This block makes them
    reproducible per seed without a change to the caller's global state.
    """
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        yield
