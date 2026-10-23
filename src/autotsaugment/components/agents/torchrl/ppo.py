"""Proximal Policy Optimization with TorchRL losses, driven by the shared episode loop.

Each update collects ``episodes_per_update`` episodes with the stochastic actor through
:func:`~autotsaugment.components.agents.loop.run_episode` (K x E transitions per episode),
computes GAE advantages once, then runs ``ppo_epochs`` passes over shuffled minibatches with
``ClipPPOLoss`` and its entropy bonus.

The log-probabilities of the behaviour policy are computed after the collection, in one batch,
with the same ``TanhNormal`` that the loss uses. The actor does not change during a collection,
so these values are the behaviour log-probabilities, and the first importance ratio is exactly 1.
"""

import math
from pathlib import Path
from typing import Any

import torch
from pydantic import Field
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from autotsaugment.components.agents.loop import episode_seed, run_episode
from autotsaugment.components.agents.policies import (
    load_state,
    not_fitted,
    save_state,
    space_from_state,
    space_state,
)
from autotsaugment.components.agents.torchrl.adapter import (
    ACTION_KEY,
    DONE_KEY,
    NEXT_KEY,
    OBSERVATION_KEY,
    REWARD_KEY,
    TERMINATED_KEY,
    make_tensordict,
)
from autotsaugment.components.agents.torchrl.networks import (
    GaussianPolicy,
    flatten_observation,
    forked_rng,
    make_actor,
    make_gaussian_head,
    make_value,
    observation_dim,
)
from autotsaugment.core.agent import Agent, AgentConfig, Policy
from autotsaugment.core.environment import Environment, StepResult
from autotsaugment.core.featurizer import Observation, ObservationSpec
from autotsaugment.core.seeding import derive_seed, make_generator
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tracker import Tracker

LOSS_KEYS = ("loss_objective", "loss_critic", "loss_entropy")
LOG_KEYS = (*LOSS_KEYS, "entropy", "kl_approx", "clip_fraction")
LOG_PROB_KEY = "action_log_prob"


class PPOAgentConfig(AgentConfig):
    """Config of :class:`PPOAgent`.

    Attributes:
        episodes: Training episodes.
        hidden_sizes: Hidden layer sizes of the actor and of the value network.
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        gae_lambda: Lambda of the generalized advantage estimate.
        clip_epsilon: Clip range of the importance ratio.
        entropy_coef: Weight of the entropy bonus.
        episodes_per_update: Episodes collected before each update.
        ppo_epochs: Passes over the collected transitions per update.
        minibatch_size: Transitions per gradient step.
        max_grad_norm: Global gradient-norm clip per gradient step, or ``None``.
    """

    episodes: int = Field(default=200, ge=0)
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    clip_epsilon: float = Field(default=0.2, gt=0.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    episodes_per_update: int = Field(default=4, ge=1)
    ppo_epochs: int = Field(default=10, ge=1)
    minibatch_size: int = Field(default=64, ge=1)
    max_grad_norm: float | None = Field(default=1.0, gt=0.0)


class _Rollout:
    """Collects the transitions of one episode, step by step, on the device."""

    def __init__(self) -> None:
        self.observations: list[Tensor] = []
        self.actions: list[Tensor] = []
        self.next_observations: list[Tensor] = []
        self.rewards: list[Tensor] = []
        self.dones: list[Tensor] = []

    def on_step(self, observation: Observation, action: Tensor, result: StepResult) -> None:
        """Store the K transitions of one step."""
        self.observations.append(flatten_observation(observation))
        self.actions.append(action.detach())
        self.next_observations.append(flatten_observation(result.observation))
        self.rewards.append(result.reward.detach().float().unsqueeze(-1))
        self.dones.append(result.done.to(torch.bool).unsqueeze(-1))

    def tensordict(self) -> TensorDict:
        """Return the episode as one tensordict of batch ``[K, E]``, time last."""
        done = torch.stack(self.dones, dim=1)
        actions = torch.stack(self.actions, dim=1)
        return make_tensordict(
            {
                OBSERVATION_KEY: torch.stack(self.observations, dim=1),
                ACTION_KEY: actions,
                NEXT_KEY: {
                    OBSERVATION_KEY: torch.stack(self.next_observations, dim=1),
                    REWARD_KEY: torch.stack(self.rewards, dim=1),
                    DONE_KEY: done,
                    TERMINATED_KEY: done.clone(),
                },
            },
            torch.Size(actions.shape[:2]),
            actions.device,
        )


class _PPOLearner:
    """Runs the PPO update on collected episodes."""

    def __init__(
        self, config: PPOAgentConfig, loss: ClipPPOLoss, *, device: torch.device, seed: int
    ) -> None:
        self.config = config
        self.loss = loss
        self.device = device
        self.advantage = GAE(
            gamma=config.gamma,
            lmbda=config.gae_lambda,
            value_network=loss.critic_network,
            average_gae=False,
        )
        self.optimizer = torch.optim.Adam(loss.parameters(), lr=config.learning_rate)
        self.generator = make_generator(self.device, derive_seed(seed, "minibatch"))

    def update(self, batch: TensorDict) -> dict[str, Tensor]:
        """Run the PPO epochs on ``batch`` of shape ``[B, E]``; return the last metrics."""
        with torch.no_grad():
            distribution = self.loss.actor_network.get_dist(batch)
            log_prob = distribution.log_prob(batch.get(ACTION_KEY))
            batch.set(LOG_PROB_KEY, log_prob)
            self.advantage(batch)
        flat = batch.reshape(-1)
        n = flat.batch_size[0]
        metrics: dict[str, Tensor] = {}
        for _ in range(self.config.ppo_epochs):
            order = torch.randperm(n, generator=self.generator, device=self.device)
            for start in range(0, n, self.config.minibatch_size):
                minibatch = flat[order[start : start + self.config.minibatch_size]]
                if not isinstance(minibatch, TensorDictBase):
                    msg = f"indexing a tensordict returned {type(minibatch).__name__}"
                    raise TypeError(msg)
                metrics = self._step(minibatch)
        return metrics

    def _step(self, minibatch: TensorDictBase) -> dict[str, Tensor]:
        """Run one gradient step on ``minibatch``."""
        losses = self.loss(minibatch)
        values: dict[str, Tensor] = {key: losses.get(key) for key in LOG_KEYS}
        total = torch.stack([values[key] for key in LOSS_KEYS]).sum()
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.loss.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return {f"ppo/{key}": value.detach().mean() for key, value in values.items()}


class PPOAgent(Agent[PPOAgentConfig]):
    """PPO over the flat observation, with a squashed Gaussian actor and a value network.

    The policy that :meth:`fit` returns is deterministic by default: the mean action, the
    squashed location of the actor, clipped to the box. A call with ``deterministic=False``
    samples from the actor.

    Args:
        config: The agent config.
        device: Device of the networks, the rollouts and the actions.
    """

    _space: BoxSpace | None
    _observation_spec: ObservationSpec | None
    _loss: ClipPPOLoss | None
    _policy: GaussianPolicy | None

    def __init__(self, *, config: PPOAgentConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._space = None
        self._observation_spec = None
        self._loss = None
        self._policy = None

    def _build(self, observation_spec: ObservationSpec, space: BoxSpace, seed: int) -> ClipPPOLoss:
        """Create the networks and the loss module for the spaces of the environment."""
        in_features = observation_dim(observation_spec)
        hidden = self.config.hidden_sizes
        head = make_gaussian_head(in_features, space.dim, hidden, self.device)
        loss = ClipPPOLoss(
            make_actor(head, space),
            make_value(in_features, hidden, self.device),
            clip_epsilon=self.config.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=self.config.entropy_coef,
            normalize_advantage=True,
        )
        loss.set_keys(sample_log_prob=LOG_PROB_KEY)
        generator = make_generator(self.device, derive_seed(seed, "ppo", "policy"))
        self._space = space
        self._observation_spec = dict(observation_spec)
        self._loss = loss
        self._policy = GaussianPolicy(head, space, generator)
        return loss

    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Train for ``config.episodes`` episodes and return the deterministic policy.

        The update metrics go to the tracker as one extra step log, at the index of the last
        step of the update. The tracker is flushed at the end, so no metric stays buffered.
        """
        space = env.action_space.to(self.device)
        steps = env.episode_length
        with forked_rng(self.device, derive_seed(seed, "ppo", "torch")):
            loss = self._build(env.observation_spec, space, seed)
            policy = self.policy()
            learner = _PPOLearner(
                self.config, loss, device=self.device, seed=derive_seed(seed, "ppo")
            )
            per_update = self.config.episodes_per_update
            for update in range(math.ceil(self.config.episodes / per_update)):
                first = update * per_update
                last = min(first + per_update, self.config.episodes)
                episodes: list[TensorDict] = []
                for episode in range(first, last):
                    rollout = _Rollout()
                    run_episode(
                        env,
                        policy,
                        seed=episode_seed(seed, episode),
                        tracker=tracker,
                        step_offset=episode * steps,
                        episode=episode,
                        deterministic=False,
                        on_step=rollout.on_step,
                    )
                    episodes.append(rollout.tensordict())
                metrics = learner.update(TensorDict.cat(episodes, dim=0))
                tracker.log_step(last * steps - 1, metrics)
        tracker.flush()
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
        """Write the spaces and the state of the loss module (actor and critic) to ``path``.

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
