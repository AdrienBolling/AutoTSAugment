"""Group Relative Policy Optimization (GRPO) with a causal transformer policy.

GRPO replaces the critic of PPO with a group baseline. A group is every trajectory collected
with the same episode seed: the K lockstep environments of one episode, times
``group_episodes`` repeated episodes with that seed. The environment derives every random
stream from the seed, so the members of a group start from the same state and differ only
through the samples of the policy. The advantage of a trajectory is its return, standardized
over its group.

Each update collects ``groups_per_update`` groups through
:func:`~autotsaugment.components.agents.loop.run_episode` with the stochastic
:class:`~autotsaugment.components.agents.transformer.HistoryPolicy`. Then it computes the
advantages, the log-probabilities of the behaviour policy in one forward pass, and runs
``epochs`` passes of the PPO clipped surrogate over the trajectories. A minibatch holds whole
trajectories, because the policy at step t reads the whole prefix of its episode. There is no
critic and no value loss.

The update makes no host synchronization: the group statistics are tensor operations.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import torch
from pydantic import Field, model_validator
from torch import Tensor

from autotsaugment.components.agents.distributions import gaussian_entropy, log_prob
from autotsaugment.components.agents.loop import episode_seed, run_episode
from autotsaugment.components.agents.policies import (
    load_state,
    not_fitted,
    save_state,
    space_from_state,
    space_state,
)
from autotsaugment.components.agents.torchrl.networks import forked_rng, observation_dim
from autotsaugment.components.agents.transformer import HistoryNetwork, HistoryPolicy
from autotsaugment.core.agent import Agent, AgentConfig, Policy
from autotsaugment.core.environment import Environment, StepResult
from autotsaugment.core.featurizer import Observation, ObservationSpec
from autotsaugment.core.seeding import derive_seed, make_generator
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tracker import Tracker

logger = logging.getLogger(__name__)

ADVANTAGE_EPS = 1e-6
"""Added to the group standard deviation before the division."""

NETWORK_FIELDS = ("d_model", "n_layers", "n_heads", "init_scale", "min_scale")
"""The config fields that shape the network. The checkpoint stores them."""

MIN_GROUP_SIZE = 2
"""A group needs two trajectories for a baseline and a standard deviation."""

MATRIX_NDIM = 2
"""Parameters with at least this many dimensions get weight decay."""


class GRPOAgentConfig(AgentConfig):
    """Config of :class:`GRPOAgent`.

    Attributes:
        episodes: Training episodes. The agent runs ``episodes // (groups_per_update *
            group_episodes)`` updates and drops the remainder with a warning.
        group_episodes: Episodes with the same seed in one group. A group has
            ``group_episodes * K`` trajectories.
        groups_per_update: Groups collected before each update.
        advantage: ``"step"``: the return to go of each step, standardized over the group at
            the same step. ``"outcome"``: the episode return, standardized over the group and
            given to every step.
        normalize_std: Divide the centered return by the group standard deviation. False
            keeps the mean baseline only.
        gamma: Discount of the return to go (``"step"`` only). 1 makes the return the total
            improvement of the remaining episode.
        clip_epsilon: Clip range of the importance ratio.
        kl_coef: Weight of the k3 estimate of ``KL(new || old)``, old being the policy at the
            start of the update.
        entropy_coef: Weight of the entropy bonus of the pre-squash Gaussian.
        epochs: Passes over the trajectories of an update.
        trajectories_per_minibatch: Whole trajectories per gradient step. ``None`` puts all the
            trajectories of the update in one minibatch.
        learning_rate: AdamW learning rate. A probe on the fake bandit with a 39-dimensional
            action gave a clip fraction of 0.02 to 0.30 at the last epoch with 3e-5, against
            0.42 to 0.80 with 1e-4.
        weight_decay: AdamW weight decay, on the weight matrices only.
        max_grad_norm: Global gradient-norm clip per gradient step, or ``None``.
        d_model: Width of the transformer.
        n_layers: Encoder layers.
        n_heads: Attention heads per layer.
        init_scale: Scale of the pre-squash Gaussian at initialization.
        min_scale: Lower bound of the scale.
    """

    episodes: int = Field(default=200, ge=0)
    group_episodes: int = Field(default=2, ge=1)
    groups_per_update: int = Field(default=1, ge=1)
    advantage: Literal["step", "outcome"] = "step"
    normalize_std: bool = True
    gamma: float = Field(default=1.0, ge=0.0, le=1.0)
    clip_epsilon: float = Field(default=0.2, gt=0.0)
    kl_coef: float = Field(default=0.0, ge=0.0)
    entropy_coef: float = Field(default=0.0, ge=0.0)
    epochs: int = Field(default=4, ge=1)
    trajectories_per_minibatch: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=3e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float | None = Field(default=1.0, gt=0.0)
    d_model: int = Field(default=256, ge=1)
    n_layers: int = Field(default=4, ge=1)
    n_heads: int = Field(default=8, ge=1)
    init_scale: float = Field(default=0.3, gt=0.0)
    min_scale: float = Field(default=0.01, ge=0.0)

    @model_validator(mode="after")
    def _check_network(self) -> Self:
        """Require heads that divide the width, and a start scale above the minimum."""
        if self.d_model % self.n_heads != 0:
            msg = f"d_model ({self.d_model}) must be a multiple of n_heads ({self.n_heads})"
            raise ValueError(msg)
        if self.init_scale <= self.min_scale:
            msg = f"init_scale ({self.init_scale}) must be above min_scale ({self.min_scale})"
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class Trajectories:
    """Whole trajectories on the device, one row per trajectory.

    Attributes:
        tokens: The tokens that the policy read, shape ``(N, E, D_tok)``.
        samples: The pre-squash samples of the actions, shape ``(N, E, a)``.
        rewards: Shape ``(N, E)``.
        group: The group of each trajectory inside its update, shape ``(N,)``, int64.
    """

    tokens: Tensor
    samples: Tensor
    rewards: Tensor
    group: Tensor

    @staticmethod
    def cat(parts: list["Trajectories"]) -> "Trajectories":
        """Concatenate ``parts`` along the trajectories."""
        return Trajectories(
            tokens=torch.cat([part.tokens for part in parts]),
            samples=torch.cat([part.samples for part in parts]),
            rewards=torch.cat([part.rewards for part in parts]),
            group=torch.cat([part.group for part in parts]),
        )


class _Rollout:
    """Collects the K trajectories of one episode, step by step, on the device."""

    def __init__(self, policy: HistoryPolicy) -> None:
        self.policy = policy
        self.samples: list[Tensor] = []
        self.rewards: list[Tensor] = []

    def on_step(self, observation: Observation, action: Tensor, result: StepResult) -> None:
        """Store the pre-squash sample of the policy and the reward of one step."""
        del observation, action
        if self.policy.last_sample is None:
            msg = "the policy has no last sample: the rollout needs the policy of the episode"
            raise RuntimeError(msg)
        self.samples.append(self.policy.last_sample)
        self.rewards.append(result.reward.detach().float())

    def trajectories(self, group: int) -> Trajectories:
        """Return the episode as K trajectories of group ``group``."""
        rewards = torch.stack(self.rewards, dim=1)
        return Trajectories(
            tokens=self.policy.tokens(),
            samples=torch.stack(self.samples, dim=1),
            rewards=rewards,
            group=torch.full((rewards.shape[0],), group, dtype=torch.int64, device=rewards.device),
        )


def returns_to_go(rewards: Tensor, gamma: float) -> Tensor:
    """Return ``G_t = sum_{s >= t} gamma^(s - t) r_s`` per trajectory, shape ``(N, E)``.

    A loop over the E steps, in float32 on the device: a matrix product could run in TF32.
    """
    returns = torch.empty_like(rewards)
    running = torch.zeros_like(rewards[:, 0])
    for step in reversed(range(rewards.shape[1])):
        running = rewards[:, step] + gamma * running
        returns[:, step] = running
    return returns


def group_moments(values: Tensor, group: Tensor, n_groups: int) -> tuple[Tensor, Tensor]:
    """Return the mean and the sample standard deviation of ``values`` per group.

    Args:
        values: Shape ``(N, ...)``, one row per trajectory.
        group: The group of each row, shape ``(N,)``, in ``0 .. n_groups - 1``. Each group
            has at least two rows.
        n_groups: Number of groups.

    Returns:
        ``(mean, std)``, each of shape ``(n_groups, ...)``. The standard deviation divides by
        the group size minus one.
    """
    labels = torch.arange(n_groups, device=group.device)
    members = (group.unsqueeze(0) == labels.unsqueeze(1)).to(values.dtype)
    members = members.view(n_groups, -1, *([1] * (values.ndim - 1)))
    count = members.sum(dim=1)
    mean = (members * values.unsqueeze(0)).sum(dim=1) / count
    squares = (members * (values.unsqueeze(0) - mean.unsqueeze(1)) ** 2).sum(dim=1)
    return mean, torch.sqrt(squares / (count - 1.0))


def group_advantages(
    batch: Trajectories,
    n_groups: int,
    *,
    mode: Literal["step", "outcome"],
    gamma: float,
    normalize_std: bool,
) -> Tensor:
    """Return the group-relative advantage of every step, shape ``(N, E)``.

    ``"step"``: ``(G_t - mean_g G_t) / (std_g G_t + eps)`` with the return to go ``G_t``. The
    mean and the standard deviation run over the members of the group at the same step t.
    ``"outcome"``: the same with the episode return, given to every step. Without
    ``normalize_std``, the centered value.
    """
    values = returns_to_go(batch.rewards, gamma) if mode == "step" else batch.rewards.sum(dim=1)
    mean, std = group_moments(values, batch.group, n_groups)
    advantage = values - mean[batch.group]
    if normalize_std:
        advantage = advantage / (std[batch.group] + ADVANTAGE_EPS)
    if mode == "outcome":
        advantage = advantage.unsqueeze(1).expand_as(batch.rewards)
    return advantage


class _GRPOLearner:
    """Runs the GRPO update on collected trajectories."""

    def __init__(
        self,
        config: GRPOAgentConfig,
        network: HistoryNetwork,
        space: BoxSpace,
        *,
        generator: torch.Generator,
    ) -> None:
        self.config = config
        self.network = network
        self.space = space
        self.generator = generator
        matrices = [p for p in network.parameters() if p.ndim >= MATRIX_NDIM]
        others = [p for p in network.parameters() if p.ndim < MATRIX_NDIM]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": matrices, "weight_decay": config.weight_decay},
                {"params": others, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
        )

    def update(self, batch: Trajectories, n_groups: int) -> dict[str, Tensor]:
        """Run the epochs on ``batch``; return the metrics of the last gradient step."""
        config = self.config
        advantages = group_advantages(
            batch,
            n_groups,
            mode=config.advantage,
            gamma=config.gamma,
            normalize_std=config.normalize_std,
        )
        _, return_std = group_moments(batch.rewards.sum(dim=1), batch.group, n_groups)
        with torch.no_grad():
            loc, scale = self.network(batch.tokens)
            old_log_prob = log_prob(batch.samples, loc, scale, self.space.low, self.space.high)
        n = batch.tokens.shape[0]
        size = config.trajectories_per_minibatch or n
        metrics: dict[str, Tensor] = {}
        for _ in range(config.epochs):
            order = torch.randperm(n, generator=self.generator, device=batch.tokens.device)
            for start in range(0, n, size):
                rows = order[start : start + size]
                metrics = self._step(
                    batch.tokens[rows], batch.samples[rows], old_log_prob[rows], advantages[rows]
                )
        metrics["grpo/group_return_std"] = return_std.mean()
        return metrics

    def _step(
        self, tokens: Tensor, samples: Tensor, old_log_prob: Tensor, advantages: Tensor
    ) -> dict[str, Tensor]:
        """Run one gradient step on a minibatch of whole trajectories."""
        config = self.config
        loc, scale = self.network(tokens)
        new_log_prob = log_prob(samples, loc, scale, self.space.low, self.space.high)
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        clipped = ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon)
        loss_policy = -torch.minimum(ratio * advantages, clipped * advantages).mean()
        kl = torch.exp(-log_ratio) + log_ratio - 1.0
        entropy = gaussian_entropy(scale)
        loss = loss_policy
        if config.kl_coef > 0.0:
            loss = loss + config.kl_coef * kl.mean()
        if config.entropy_coef > 0.0:
            loss = loss - config.entropy_coef * entropy.mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.max_grad_norm)
        self.optimizer.step()
        clip_fraction = ((ratio - 1.0).abs() > config.clip_epsilon).float().mean()
        return {
            "grpo/loss_policy": loss_policy.detach(),
            "grpo/kl_approx": kl.detach().mean(),
            "grpo/clip_fraction": clip_fraction,
            "grpo/entropy": entropy.detach().mean(),
            "grpo/scale_mean": scale.detach().mean(),
        }


class GRPOAgent(Agent[GRPOAgentConfig]):
    """GRPO with a causal transformer policy that reads the whole episode so far.

    The policy that :meth:`fit` returns is deterministic by default: the squashed location at
    the last step, with its own memory of the episode. A call with ``deterministic=False``
    samples from the squashed Gaussian.

    Args:
        config: The agent config.
        device: Device of the network, the trajectories and the actions.
    """

    _observation_spec: ObservationSpec | None
    _network_shape: dict[str, Any]
    _policy: HistoryPolicy | None

    def __init__(self, *, config: GRPOAgentConfig, device: torch.device | str) -> None:
        self.config = config
        self.device = torch.device(device)
        self._observation_spec = None
        self._network_shape = {}
        self._policy = None

    def _build(
        self,
        observation_spec: ObservationSpec,
        space: BoxSpace,
        *,
        max_steps: int,
        shape: dict[str, Any],
        seed: int,
    ) -> HistoryPolicy:
        """Create the network and the policy for the spaces of the environment.

        The network stays in evaluation mode. It has no dropout, so the acting pass, the pass
        of the behaviour log-probabilities and the gradient passes compute the same function.
        """
        network = HistoryNetwork(
            token_dim=observation_dim(observation_spec) + space.dim,
            action_dim=space.dim,
            max_steps=max_steps,
            device=self.device,
            **shape,
        )
        network.eval()
        generator = make_generator(self.device, derive_seed(seed, "grpo", "policy"))
        self._observation_spec = dict(observation_spec)
        self._network_shape = dict(shape)
        self._policy = HistoryPolicy(network, space, generator)
        return self._policy

    def fit(self, env: Environment[Any], *, tracker: Tracker, seed: int) -> Policy:
        """Train for ``config.episodes`` episodes and return the deterministic policy.

        The update metrics go to the tracker as one extra step log, at the index of the last
        step of the update. The tracker is flushed at the end, so no metric stays buffered. The
        returned policy has an empty memory.

        Raises:
            ValueError: If a group has fewer than two trajectories (``group_episodes * K``).
        """
        config = self.config
        group_size = config.group_episodes * env.batch_size
        if group_size < MIN_GROUP_SIZE:
            msg = (
                f"a group needs at least two trajectories, got group_episodes * K = {group_size}: "
                "raise group_episodes"
            )
            raise ValueError(msg)
        per_update = config.groups_per_update * config.group_episodes
        updates, dropped = divmod(config.episodes, per_update)
        if dropped:
            logger.warning(
                "GRPO runs %d updates of %d episodes and drops the last %d of %d episodes",
                updates,
                per_update,
                dropped,
                config.episodes,
            )
        space = env.action_space.to(self.device)
        steps = env.episode_length
        shape = {name: getattr(config, name) for name in NETWORK_FIELDS}
        with forked_rng(self.device, derive_seed(seed, "grpo", "torch")):
            policy = self._build(
                env.observation_spec, space, max_steps=steps, shape=shape, seed=seed
            )
            learner = _GRPOLearner(
                config,
                policy.network,
                space,
                generator=make_generator(self.device, derive_seed(seed, "grpo", "minibatch")),
            )
            episode = 0
            for update in range(updates):
                parts: list[Trajectories] = []
                for group in range(config.groups_per_update):
                    group_seed = episode_seed(seed, update * config.groups_per_update + group)
                    for _ in range(config.group_episodes):
                        rollout = _Rollout(policy)
                        run_episode(
                            env,
                            policy,
                            seed=group_seed,
                            tracker=tracker,
                            step_offset=episode * steps,
                            episode=episode,
                            deterministic=False,
                            on_step=rollout.on_step,
                        )
                        parts.append(rollout.trajectories(group))
                        episode += 1
                metrics = learner.update(Trajectories.cat(parts), config.groups_per_update)
                tracker.log_step(episode * steps - 1, metrics)
        policy.reset()
        tracker.flush()
        return self.policy()

    def policy(self) -> Policy:
        """Return the history policy.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._policy is None:
            raise not_fitted(self)
        return self._policy

    def save(self, path: Path) -> None:
        """Write the spaces, ``max_steps``, the network shape and the network state to ``path``.

        Raises:
            RuntimeError: If neither :meth:`fit` nor :meth:`load` ran yet.
        """
        if self._policy is None or self._observation_spec is None:
            raise not_fitted(self)
        network = self._policy.network
        spec = {name: list(shape) for name, shape in self._observation_spec.items()}
        state = {
            "observation_spec": spec,
            "max_steps": network.max_steps,
            "network_shape": dict(self._network_shape),
            "network": network.state_dict(),
        }
        save_state(path, {**space_state(self._policy.space), **state})

    def load(self, path: Path) -> None:
        """Rebuild the network with the saved shape and restore its state from ``path``."""
        state = load_state(path)
        spec = {name: tuple(shape) for name, shape in state["observation_spec"].items()}
        policy = self._build(
            spec,
            space_from_state(state, self.device),
            max_steps=int(state["max_steps"]),
            shape=dict(state["network_shape"]),
            seed=0,
        )
        policy.network.load_state_dict(state["network"])
