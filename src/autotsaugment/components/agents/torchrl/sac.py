"""Soft Actor-Critic with TorchRL losses, driven by the shared episode loop.

Design choice: the agent drives the environment with
:func:`~autotsaugment.components.agents.loop.run_episode` and a step callback, not with a
TorchRL collector over :class:`~autotsaugment.components.agents.torchrl.adapter.TorchRLEnv`.
The loop adds no host synchronization and logs the same metrics as every other agent and
experiment. The callback stores the K transitions of each step in a replay buffer on the device
and runs ``updates_per_step`` gradient updates after each environment step.

For the first ``warmup_steps`` environment steps (calls to ``env.step``, each one K
transitions), the actions are uniform in the box. Learning starts when the warm-up is over and
the buffer holds one batch.
"""

from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field
from torch import Tensor
from torchrl.data import Bounded, LazyTensorStorage, ReplayBuffer
from torchrl.objectives import SACLoss, SoftUpdate

from autotsaugment.components.agents.loop import episode_seed, run_episode
from autotsaugment.components.agents.policies import (
    RandomPolicy,
    load_state,
    not_fitted,
    save_state,
    space_from_state,
    space_state,
)
from autotsaugment.components.agents.torchrl.networks import (
    GaussianPolicy,
    forked_rng,
    make_actor,
    make_gaussian_head,
    make_qvalue,
    observation_dim,
    transition,
)
from autotsaugment.core.agent import Agent, AgentConfig, Policy
from autotsaugment.core.environment import Environment, StepResult
from autotsaugment.core.featurizer import Observation, ObservationSpec
from autotsaugment.core.seeding import derive_seed, make_generator
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tracker import Tracker

LOSS_KEYS = ("loss_actor", "loss_qvalue", "loss_alpha")
LOG_KEYS = (*LOSS_KEYS, "alpha", "entropy")


class SACAgentConfig(AgentConfig):
    """Config of :class:`SACAgent`.

    Attributes:
        episodes: Training episodes.
        hidden_sizes: Hidden layer sizes of the actor and of both Q-networks.
        learning_rate: Adam learning rate of the actor, the Q-networks and the temperature.
        gamma: Discount factor.
        tau: Soft-update rate of the target Q-networks.
        batch_size: Transitions per gradient update.
        replay_capacity: Transitions the replay buffer keeps.
        warmup_steps: Environment steps with uniform random actions before learning starts.
        updates_per_step: Gradient updates after each environment step.
        target_entropy: Target entropy of the temperature loss; ``"auto"`` is ``-a``.
    """

    episodes: int = Field(default=100, ge=0)
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    tau: float = Field(default=0.005, gt=0.0, le=1.0)
    batch_size: int = Field(default=256, ge=1)
    replay_capacity: int = Field(default=100_000, ge=1)
    warmup_steps: int = Field(default=200, ge=0)
    updates_per_step: int = Field(default=1, ge=0)
    target_entropy: float | Literal["auto"] = "auto"


class _SACLearner:
    """Acts in the environment, stores transitions and updates the SAC networks."""

    def __init__(
        self,
        config: SACAgentConfig,
        loss: SACLoss,
        policy: GaussianPolicy,
        *,
        seed: int,
    ) -> None:
        space = policy.space
        device = space.device
        self.config = config
        self.loss = loss
        self.policy = policy
        self.explore = RandomPolicy(space, make_generator(device, derive_seed(seed, "warmup")))
        self.optimizer = torch.optim.Adam(loss.parameters(), lr=config.learning_rate)
        self.target_update = SoftUpdate(loss, tau=config.tau)
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(config.replay_capacity, device=device),
            batch_size=config.batch_size,
            generator=make_generator(device, derive_seed(seed, "replay")),
        )
        self.env_steps = 0

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return uniform actions during the warm-up, then actions of the actor."""
        if self.env_steps < self.config.warmup_steps:
            return self.explore(observation)
        return self.policy(observation, deterministic=deterministic)

    def on_step(
        self, observation: Observation, action: Tensor, result: StepResult
    ) -> dict[str, Tensor] | None:
        """Store the transitions of the step, then train. Return the last update metrics."""
        self.buffer.extend(transition(observation, action, result))
        self.env_steps += 1
        if self.env_steps < self.config.warmup_steps or len(self.buffer) < self.config.batch_size:
            return None
        metrics: dict[str, Tensor] | None = None
        for _ in range(self.config.updates_per_step):
            metrics = self.update()
        return metrics

    def update(self) -> dict[str, Tensor]:
        """Run one gradient update on a replay batch and soft-update the target networks."""
        batch = self.buffer.sample()
        losses = self.loss(batch)
        values: dict[str, Tensor] = {key: losses.get(key) for key in LOG_KEYS}
        total = torch.stack([values[key] for key in LOSS_KEYS]).sum()
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.optimizer.step()
        self.target_update.step()
        return {f"sac/{key}": value.detach() for key, value in values.items()}


class SACAgent(Agent[SACAgentConfig]):
    """Soft Actor-Critic over the flat observation, with a squashed Gaussian actor.

    The policy that :meth:`fit` returns is deterministic by default: the mean action, the
    squashed location of the actor, clipped to the box. A call with ``deterministic=False``
    samples from the actor.

    Args:
        config: The agent config.
        device: Device of the networks, the replay buffer and the actions.
    """

    _space: BoxSpace | None
    _observation_spec: ObservationSpec | None
    _loss: SACLoss | None
    _policy: GaussianPolicy | None

    def __init__(self, *, config: SACAgentConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._space = None
        self._observation_spec = None
        self._loss = None
        self._policy = None

    def _build(self, observation_spec: ObservationSpec, space: BoxSpace, seed: int) -> SACLoss:
        """Create the networks and the loss module for the spaces of the environment."""
        in_features = observation_dim(observation_spec)
        hidden = self.config.hidden_sizes
        head = make_gaussian_head(in_features, space.dim, hidden, self.device)
        qvalue = make_qvalue(in_features, space.dim, hidden, self.device)
        action_spec = Bounded(
            low=space.low, high=space.high, shape=torch.Size([space.dim]), device=self.device
        )
        loss = SACLoss(
            make_actor(head, space),
            qvalue,
            num_qvalue_nets=2,
            action_spec=action_spec,
            target_entropy=self.config.target_entropy,
        )
        loss.make_value_estimator(gamma=self.config.gamma)
        generator = make_generator(self.device, derive_seed(seed, "sac", "policy"))
        self._space = space
        self._observation_spec = dict(observation_spec)
        self._loss = loss
        self._policy = GaussianPolicy(head, space, generator)
        return loss

    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Train for ``config.episodes`` episodes and return the deterministic policy."""
        space = env.action_space.to(self.device)
        with forked_rng(self.device, derive_seed(seed, "sac", "torch")):
            loss = self._build(env.observation_spec, space, seed)
            if self._policy is None:
                raise not_fitted(self)
            learner = _SACLearner(self.config, loss, self._policy, seed=derive_seed(seed, "sac"))
            for episode in range(self.config.episodes):
                run_episode(
                    env,
                    learner,
                    seed=episode_seed(seed, episode),
                    tracker=tracker,
                    step_offset=episode * env.episode_length,
                    episode=episode,
                    deterministic=False,
                    on_step=learner.on_step,
                )
        return self.policy()

    def policy(self) -> Policy:
        """Return the actor policy.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._policy is None:
            raise not_fitted(self)
        return self._policy

    def save(self, path: Path) -> None:
        """Write the spaces and the state of the loss module (all networks) to ``path``.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._loss is None or self._space is None or self._observation_spec is None:
            raise not_fitted(self)
        spec = {name: list(shape) for name, shape in self._observation_spec.items()}
        state = {"observation_spec": spec, "loss": self._loss.state_dict()}
        save_state(path, {**space_state(self._space), **state})

    def load(self, path: Path) -> None:
        """Rebuild the networks and restore the state that :meth:`save` wrote under ``path``."""
        state = load_state(path)
        spec = {name: tuple(shape) for name, shape in state["observation_spec"].items()}
        loss = self._build(spec, space_from_state(state, self.device), seed=0)
        loss.load_state_dict(state["loss"])
