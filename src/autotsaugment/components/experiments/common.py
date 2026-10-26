"""Shared parts of the experiments: run construction, policy evaluation, summaries.

Evaluation protocol of :func:`evaluate_policy`: one full episode per evaluation seed ``s``,
with the episode seed ``derive_seed(root_seed, "eval", s)`` and a deterministic policy. These
seeds differ from the training seeds of the agents (``derive_seed(seed, "episode", i)``), and
two policies evaluated on the same seeds see the same reward windows at every step. Each
episode trains the K lockstep tsmodels, so one seed gives K samples. All values stay on the
device until one :func:`~autotsaugment.core.tracker.to_host` call at the end.

Summary keys (:attr:`EvalSummary.metrics`). Skill and aggregate values are skill-normalized WQL,
so lower is better:

- ``comparison/aggregate/mean``: the primary metric. The aggregate (geometric mean over the
  datasets) of the comparison skill from ``env.evaluate()``, averaged over the K tsmodels,
  then over the seeds.
- ``comparison/aggregate/std``: the sample standard deviation over the seeds of the same
  K-mean. ``0.0`` for one seed.
- ``comparison/aggregate/seed<s>``: the K-mean of evaluation seed ``s``.
- ``comparison/<dataset>/mean``: the comparison skill of one dataset, averaged over the K
  tsmodels and the seeds.
- ``curve/auc/mean``: the area under the aggregate curve, the secondary metric. The aggregate
  on the reward subset after each step, averaged over the steps, the K tsmodels and the seeds.
- ``return/mean``: the episode return, averaged over the K tsmodels and the seeds.
- ``eval/seeds``: the number of evaluation seeds.
- ``eval/wall_time_s``: the wall time of the evaluation, in seconds.
"""

import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import hydra
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import Field, field_validator
from torch import Tensor

from autotsaugment.components.agents.loop import geometric_mean, run_episode
from autotsaugment.components.tracking.null import NullTracker
from autotsaugment.core.agent import Agent, Policy
from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.environment import Environment
from autotsaugment.core.seeding import derive_seed, seed_everything
from autotsaugment.core.tracker import Tracker, to_host

CONFIG_DIR = Path(__file__).resolve().parents[4] / "configs"
"""The Hydra config directory of the repository, ``configs/``."""

EVAL_PREFIX = "eval/"
"""Tracker prefix of the evaluation episodes, so they stay apart from the training episodes."""

AGENT_CONFIG_FILE = "agent.yaml"
"""File in a checkpoint directory with the agent config of the run that wrote it."""


class ExperimentConfig(ComponentConfig):
    """Config shared by all experiments.

    Attributes:
        name: The experiment name. The MLflow experiment and the output files use it.
    """

    name: str = Field(min_length=1)


class EvaluatedConfig(ExperimentConfig):
    """Config of an experiment that evaluates policies on a list of seeds.

    Attributes:
        eval_seeds: The evaluation seeds, distinct. One episode per seed.
        metrics_dir: Directory of the metrics file, relative to the working directory.
    """

    eval_seeds: tuple[int, ...] = Field(default=(0, 1, 2), min_length=1)
    metrics_dir: str = "metrics"

    @field_validator("eval_seeds")
    @classmethod
    def _distinct(cls, seeds: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(seeds)) != len(seeds):
            msg = f"eval_seeds must be distinct, got {list(seeds)}"
            raise ValueError(msg)
        return seeds


def plain(value: object) -> object:
    """Return ``value`` with OmegaConf containers converted to resolved Python containers."""
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def experiment_config[C: ComponentConfig](config_type: type[C], leaves: Mapping[str, Any]) -> C:
    """Validate the leaves of an experiment YAML file, as Hydra passes them, into a config.

    Hydra instantiates the experiment with ``_recursive_=False``, so lists and mappings arrive
    as OmegaConf containers. They are converted before the validation.
    """
    return config_type.model_validate({key: plain(value) for key, value in leaves.items()})


def resolve_path(path: str | Path) -> Path:
    """Return ``path``, relative to the current working directory when it is not absolute.

    Hydra runs with ``hydra.job.chdir: false``, so the working directory is the directory of
    the command, usually the repository root, not the Hydra output directory.
    """
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else Path.cwd() / resolved


@runtime_checkable
class FailureAware(Protocol):
    """A tracker that can record that its run failed, for example the MLflow tracker."""

    def mark_failed(self) -> None:
        """Make the next ``close`` end the run as failed."""
        ...


def close_failed(tracker: Tracker) -> None:
    """Mark the run of ``tracker`` as failed when the tracker supports it, then close it."""
    if isinstance(tracker, FailureAware):
        tracker.mark_failed()
    tracker.close()


@contextmanager
def closing_tracker(tracker: Tracker) -> Generator[Tracker]:
    """Close ``tracker`` at the end of the block, as failed when the block raises."""
    try:
        yield tracker
    except BaseException:
        close_failed(tracker)
        raise
    tracker.close()


def build_tracker(run_config: DictConfig) -> Tracker:
    """Instantiate the tracker of the run, or a :class:`NullTracker` without a tracker group.

    Raises:
        TypeError: If ``run_config.tracker`` does not build a :class:`Tracker`.
    """
    node = run_config.get("tracker")
    if node is None:
        return NullTracker()
    tracker = hydra.utils.instantiate(node)
    if not isinstance(tracker, Tracker):
        msg = f"cfg.tracker must build a Tracker, got {type(tracker).__name__}"
        raise TypeError(msg)
    return tracker


def build_agent(node: DictConfig) -> Agent[Any]:
    """Instantiate an agent from its config node, for example ``run_config.agent``.

    Raises:
        TypeError: If ``node`` does not build an :class:`Agent`.
    """
    agent = hydra.utils.instantiate(node)
    if not isinstance(agent, Agent):
        msg = f"the agent config must build an Agent, got {type(agent).__name__}"
        raise TypeError(msg)
    return agent


def build_env(run_config: DictConfig) -> Environment[Any]:
    """Instantiate the environment of the run. This loads the data and the weights.

    Raises:
        TypeError: If ``run_config.env`` does not build an :class:`Environment`.
    """
    env = hydra.utils.instantiate(run_config.env)
    if not isinstance(env, Environment):
        msg = f"cfg.env must build an Environment, got {type(env).__name__}"
        raise TypeError(msg)
    return env


def run_params(run_config: DictConfig, *, exclude: Sequence[str] = ()) -> dict[str, object]:
    """Return the resolved run config as a nested dict, without ``hydra`` and ``exclude``.

    Raises:
        TypeError: If the run config is not a mapping.
    """
    container = OmegaConf.to_container(run_config, resolve=True)
    if not isinstance(container, dict):
        msg = f"the run config must be a mapping, got {type(container).__name__}"
        raise TypeError(msg)
    skip = {"hydra", *exclude}
    return {str(key): value for key, value in container.items() if key not in skip}


@dataclass(frozen=True)
class Run:
    """The components of one run.

    Attributes:
        env: The environment.
        agent: The agent of ``run_config.agent``.
        tracker: The tracker. The experiment closes it.
    """

    env: Environment[Any]
    agent: Agent[Any]
    tracker: Tracker


def start_tracker(run_config: DictConfig, *, exclude: Sequence[str] = ()) -> Tracker:
    """Seed the global streams, build the tracker and log the resolved config as parameters.

    Args:
        run_config: The composed run config.
        exclude: Top-level groups to leave out of the parameters.
    """
    seed_everything(int(run_config.seed))
    tracker = build_tracker(run_config)
    try:
        tracker.log_params(run_params(run_config, exclude=exclude))
    except BaseException:
        close_failed(tracker)
        raise
    return tracker


def build_run(run_config: DictConfig) -> Run:
    """Build the tracker, the agent and the environment of a run, in this order.

    The global random streams are seeded from ``run_config.seed`` first. The tracker comes
    first, so a failure in the agent or the environment ends its run as failed. The
    environment comes last because it loads the data and the weights.

    Args:
        run_config: The composed run config: ``seed``, ``env``, ``agent`` and ``tracker``.
    """
    tracker = start_tracker(run_config)
    try:
        agent = build_agent(run_config.agent)
        env = build_env(run_config)
    except BaseException:
        close_failed(tracker)
        raise
    return Run(env=env, agent=agent, tracker=tracker)


def synchronize(device: torch.device) -> None:
    """Wait for the queued work on ``device``, so a wall time measures the work itself."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class EvalSummary:
    """The result of :func:`evaluate_policy`.

    Attributes:
        metrics: The summary, keys in the module docstring.
        episodes: The number of episodes run, one per seed.
        steps: The number of steps run.
    """

    metrics: dict[str, float]
    episodes: int
    steps: int


def evaluate_policy(
    env: Environment[Any],
    policy: Policy,
    *,
    seeds: Sequence[int],
    tracker: Tracker,
    root_seed: int,
    step_offset: int = 0,
    episode_offset: int = 0,
) -> EvalSummary:
    """Run one deterministic episode of ``policy`` per seed and summarize the results.

    Args:
        env: The environment.
        policy: The policy to evaluate.
        seeds: The evaluation seeds, distinct.
        tracker: Receives the step and episode metrics of each episode.
        root_seed: The run seed. Each episode seed derives from it and the evaluation seed.
        step_offset: The tracker step of the first step.
        episode_offset: The tracker episode index of the first episode.

    Returns:
        The summary metrics on the host, see the module docstring.

    Raises:
        ValueError: If ``seeds`` is empty or holds a seed twice.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        msg = f"evaluate_policy needs distinct seeds, got {list(seeds)}"
        raise ValueError(msg)
    start = time.perf_counter()
    aggregates: list[Tensor] = []
    skills: list[Tensor] = []
    curves: list[Tensor] = []
    returns: list[Tensor] = []
    for index, seed in enumerate(seeds):
        result = run_episode(
            env,
            policy,
            seed=derive_seed(root_seed, "eval", seed),
            tracker=tracker,
            step_offset=step_offset + index * env.episode_length,
            episode=episode_offset + index,
            deterministic=True,
        )
        final_skill = result.final_skill.detach().float()
        aggregates.append(geometric_mean(final_skill).mean())
        skills.append(final_skill.mean(dim=0))
        curves.append(result.aggregate_curve.detach().float().mean())
        returns.append(result.returns.detach().float().mean())
    aggregate = torch.stack(aggregates)
    skill = torch.stack(skills)
    spread = aggregate.std() if len(seeds) > 1 else torch.zeros_like(aggregate[0])
    metrics: dict[str, Tensor] = {
        "comparison/aggregate/mean": aggregate.mean(),
        "comparison/aggregate/std": spread,
        "curve/auc/mean": torch.stack(curves).mean(),
        "return/mean": torch.stack(returns).mean(),
    }
    for seed, value in zip(seeds, aggregate.unbind(), strict=True):
        metrics[f"comparison/aggregate/seed{seed}"] = value
    for index, name in enumerate(env.dataset_names):
        metrics[f"comparison/{name}/mean"] = skill[:, index].mean()
    summary = to_host(metrics)
    summary["eval/seeds"] = len(seeds)
    summary["eval/wall_time_s"] = time.perf_counter() - start
    return EvalSummary(metrics=summary, episodes=len(seeds), steps=len(seeds) * env.episode_length)


def prefixed(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    """Return ``metrics`` with ``prefix`` in front of every key."""
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def log_summary(tracker: Tracker, metrics: Mapping[str, float]) -> None:
    """Log the final metrics of an experiment once, at step 0, and flush.

    The summary keys do not collide with the step keys, so each summary key holds one value.
    """
    tracker.log_step(0, metrics)
    tracker.flush()
