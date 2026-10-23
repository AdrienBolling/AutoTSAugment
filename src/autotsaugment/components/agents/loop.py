"""The episode loop that every agent and every experiment uses.

One call runs one full episode of an :class:`~autotsaugment.core.environment.Environment`:
reset, ``E`` steps with a policy, then one evaluation on the comparison set. The loop logs each
step and the episode through the tracker. It keeps every metric on the device: the tracker
moves them to the host once per episode.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from autotsaugment.core.agent import Policy
from autotsaugment.core.environment import Environment, StepResult
from autotsaugment.core.featurizer import Observation
from autotsaugment.core.seeding import derive_seed
from autotsaugment.core.tracker import Tracker

StepCallback = Callable[[Observation, Tensor, StepResult], Mapping[str, Tensor] | None]
"""Called after each step with the observation before the step, the action and the result.

The callback can store transitions or train a learner. The metrics it returns, if any, go into
the log of that step. They must be device tensors: the loop does not move them to the host.
"""


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of one episode for all k lockstep environments.

    Attributes:
        returns: Sum of the rewards over the episode, shape ``(k,)``.
        final_skill: Skill per dataset on the comparison set from ``env.evaluate()``,
            shape ``(k, d)``. Lower is better.
        aggregate_curve: Aggregate skill on the reward subset after each step, shape
            ``(E, k)``. NaN when the environment does not report ``info["aggregate"]``.
        steps: Number of steps of the episode, E.
    """

    returns: Tensor
    final_skill: Tensor
    aggregate_curve: Tensor
    steps: int


def episode_seed(seed: int, episode: int) -> int:
    """Return the seed of episode ``episode`` of a run with root seed ``seed``."""
    return derive_seed(seed, "episode", episode)


def geometric_mean(skill: Tensor) -> Tensor:
    """Return the geometric mean over the last dimension, the aggregate of a skill matrix.

    Args:
        skill: Skill scores, shape ``(..., d)``, positive.

    Returns:
        Shape ``(...)``. Same definition as :func:`autotsaugment.core.metrics.aggregate`.
    """
    return torch.exp(torch.log(skill).mean(dim=-1))


def step_metrics(result: StepResult, batch_size: int) -> dict[str, Tensor]:
    """Return the metrics of one step, as means over the k environments.

    Every ``info`` entry with shape ``(k,)`` is logged under its own name, for example the
    aggregate on the reward subset and the train statistics. Entries with more dimensions (per
    dataset scores, the action) stay out of the step log.

    Args:
        result: The result of one call to ``env.step``.
        batch_size: The number of lockstep environments, k.
    """
    metrics = {"reward": result.reward.detach().float().mean()}
    for key, value in result.info.items():
        if key in metrics or value.ndim != 1 or value.shape[0] != batch_size:
            continue
        metrics[key] = value.detach().float().mean()
    return metrics


def episode_metrics(
    returns: Tensor, final_skill: Tensor, dataset_names: tuple[str, ...]
) -> dict[str, Tensor]:
    """Return the metrics of one episode, as means over the k environments.

    Args:
        returns: Episode returns, shape ``(k,)``.
        final_skill: Comparison skill per dataset, shape ``(k, d)``.
        dataset_names: One name per dataset, in the order of ``final_skill``.
    """
    metrics = {
        "return": returns.detach().mean(),
        "comparison/aggregate": geometric_mean(final_skill.detach()).mean(),
    }
    for index, name in enumerate(dataset_names):
        metrics[f"comparison/{name}"] = final_skill[:, index].detach().mean()
    return metrics


def run_episode(
    env: Environment[Any],
    policy: Policy,
    *,
    seed: int,
    tracker: Tracker,
    step_offset: int = 0,
    episode: int = 0,
    deterministic: bool = True,
    on_step: StepCallback | None = None,
) -> EpisodeResult:
    """Run one full episode of ``env`` with ``policy`` and log it.

    The loop calls ``env.reset(seed=seed)``, then exactly ``env.episode_length`` steps (every
    episode has a fixed length, so the loop never reads ``done``), then ``env.evaluate()``. It
    does not synchronize with the host. The tracker receives one ``log_step`` per step and one
    ``log_episode`` at the end, which flushes the tracker once per episode.

    Args:
        env: The batched environment.
        policy: Maps the observation to the action, shape ``(k, a)``.
        seed: Seed of this episode. Use :func:`episode_seed` to derive it from a run seed.
        tracker: Receives the step and episode metrics.
        step_offset: Global step index of the first step, for the step log.
        episode: Global episode index, for the episode log.
        deterministic: Passed to the policy.
        on_step: Optional callback after each step, see :data:`StepCallback`.

    Returns:
        The returns, the final skill and the aggregate curve of the episode.
    """
    observation = env.reset(seed=seed)
    batch_size = env.batch_size
    returns = torch.zeros(batch_size, device=env.device)
    missing = torch.full((batch_size,), float("nan"), device=env.device)
    curve: list[Tensor] = []
    for step in range(env.episode_length):
        action = policy(observation, deterministic=deterministic)
        result = env.step(action)
        returns = returns + result.reward
        curve.append(result.info.get("aggregate", missing).detach())
        metrics = step_metrics(result, batch_size)
        if on_step is not None:
            extra = on_step(observation, action, result)
            if extra is not None:
                metrics.update(extra)
        tracker.log_step(step_offset + step, metrics)
        observation = result.observation
    final_skill = env.evaluate()
    tracker.log_episode(episode, episode_metrics(returns, final_skill, env.dataset_names))
    return EpisodeResult(
        returns=returns,
        final_skill=final_skill,
        aggregate_curve=torch.stack(curve),
        steps=env.episode_length,
    )
