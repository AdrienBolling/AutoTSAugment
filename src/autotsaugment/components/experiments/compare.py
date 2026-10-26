"""Compare experiment: evaluate several policies on the same seeds with one environment.

Each entry of ``policies`` names an agent config file in ``configs/agent/`` and, optionally, a
checkpoint directory that ``train_agent`` wrote:

- With a checkpoint, the agent loads it. When the directory holds ``agent.yaml``, the agent is
  built from that file, so the networks match the trained ones also after command-line
  overrides. Its ``_target_`` must equal the one of the named config file.
- Without a checkpoint, the agent is fitted on the environment with the run seed. This is
  instant for the training-free baselines (``fixed_uniform``, ``random``). CMA-ES and the RL
  agents train here for their full ``config.episodes``, which is allowed but slow: prefer
  ``train_agent`` and a checkpoint.

Every agent config and every checkpoint is checked when the experiment is built, before any
data loads or any episode runs. One environment instance serves all policies, and every policy
runs on the same ``eval_seeds``, so the reward windows are the same at every step. The
environment config of the run must be the one the checkpoints were trained with.

Outputs: ``<metrics_dir>/<name>.json`` with every summary key under ``<policy>/`` (keys in
:mod:`autotsaugment.components.experiments.common`) and ``wall_time_s``; one table at INFO; in
the tracker, each policy logs under ``<policy>/`` (its fit under ``<policy>/fit/``).
"""

import copy
import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict
from pydantic import Field, field_validator

from autotsaugment.components.agents.policies import CHECKPOINT_NAME
from autotsaugment.components.experiments.common import (
    AGENT_CONFIG_FILE,
    CONFIG_DIR,
    EvaluatedConfig,
    build_agent,
    build_env,
    closing_tracker,
    evaluate_policy,
    experiment_config,
    log_summary,
    prefixed,
    resolve_path,
    start_tracker,
)
from autotsaugment.components.tracking.prefixed import PrefixedTracker
from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.experiment import Experiment, write_metrics

_log = logging.getLogger(__name__)

AGENT_CONFIG_DIR = CONFIG_DIR / "agent"
"""Directory of the agent config files that the ``agent`` field of an entry names."""

_POLICY_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


class PolicyEntry(ComponentConfig):
    """One policy of the comparison.

    Attributes:
        agent: Name of an agent config file in ``configs/agent/``, without ``.yaml``.
        checkpoint: A directory that ``agent.save`` wrote, relative to the working directory.
            ``None`` fits the agent on the environment instead.
    """

    agent: str = Field(min_length=1)
    checkpoint: str | None = None


class CompareConfig(EvaluatedConfig):
    """Config of :class:`CompareExperiment`.

    Attributes:
        name: The experiment name, also the name of the metrics file.
        eval_seeds: The evaluation seeds, the same for every policy.
        policies: The policies by name. A name is the prefix of its metric keys, so it holds
            only letters, digits, ``_``, ``.`` and ``-``.
        metrics_dir: Directory of the metrics file, relative to the working directory.
    """

    name: str = "compare"
    policies: dict[str, PolicyEntry] = Field(min_length=1)

    @field_validator("policies")
    @classmethod
    def _names(cls, policies: dict[str, PolicyEntry]) -> dict[str, PolicyEntry]:
        for name in policies:
            if not _POLICY_NAME.match(name):
                msg = f"policy name {name!r} must match {_POLICY_NAME.pattern}"
                raise ValueError(msg)
        return policies


def load_mapping(path: Path) -> DictConfig:
    """Load a YAML file that holds one mapping.

    Raises:
        TypeError: If the file does not hold a mapping.
    """
    node = OmegaConf.load(path)
    if not isinstance(node, DictConfig):
        msg = f"{path} must hold a mapping, got {type(node).__name__}"
        raise TypeError(msg)
    return node


def agent_node(run_config: DictConfig, policy: str, entry: PolicyEntry) -> DictConfig:
    """Return the agent config node of one entry, inside a copy of the run config.

    The node sits under the key ``agent`` of a deep copy of ``run_config``, so interpolations
    such as ``${device}`` resolve against the run.

    Args:
        run_config: The composed run config.
        policy: The name of the entry, for the error messages.
        entry: The entry.

    Raises:
        FileNotFoundError: If the agent config file, the checkpoint directory or its agent
            state is missing.
        ValueError: If the ``agent.yaml`` of the checkpoint builds another agent class than
            the named config file.
    """
    path = AGENT_CONFIG_DIR / f"{entry.agent}.yaml"
    if not path.is_file():
        names = sorted(p.stem for p in AGENT_CONFIG_DIR.glob("*.yaml"))
        msg = f"policy {policy!r}: no agent config {path}; choose one of {names}"
        raise FileNotFoundError(msg)
    node = load_mapping(path)
    if entry.checkpoint is not None:
        checkpoint = resolve_path(entry.checkpoint)
        if not (checkpoint / CHECKPOINT_NAME).is_file():
            msg = (
                f"policy {policy!r}: no checkpoint at {checkpoint / CHECKPOINT_NAME}; "
                "run experiment=train_agent first, or fix the path"
            )
            raise FileNotFoundError(msg)
        saved = checkpoint / AGENT_CONFIG_FILE
        if saved.is_file():
            stored = load_mapping(saved)
            if stored.get("_target_") != node.get("_target_"):
                msg = (
                    f"policy {policy!r}: {saved} builds {stored.get('_target_')}, "
                    f"but agent={entry.agent} builds {node.get('_target_')}"
                )
                raise ValueError(msg)
            node = stored
            if "device" in node:
                node.device = "${device}"
    root = copy.deepcopy(run_config)
    with open_dict(root):
        root.agent = node
    return root.agent


def comparison_table(rows: Mapping[str, Mapping[str, float]]) -> str:
    """Return one text line per policy: the aggregate mean, its std and the curve AUC."""
    width = max(len("policy"), *(len(name) for name in rows))
    lines = [f"{'policy':<{width}}  {'aggregate':>9}  {'std':>8}  {'auc':>8}"]
    lines.extend(
        f"{name:<{width}}  {metrics['comparison/aggregate/mean']:>9.4f}  "
        f"{metrics['comparison/aggregate/std']:>8.4f}  {metrics['curve/auc/mean']:>8.4f}"
        for name, metrics in rows.items()
    )
    return "\n".join(lines)


class CompareExperiment(Experiment):
    """Evaluate every policy of ``policies`` on ``eval_seeds`` and write one table.

    The ``agent`` group of the run is not used. Its parameters stay out of the tracker.

    Args:
        run_config: The composed run config, resolved.
        **config: The leaves of ``configs/experiment/compare.yaml``, validated as
            :class:`CompareConfig`.
    """

    def __init__(self, *, run_config: DictConfig, **config: Any) -> None:
        self.config = experiment_config(CompareConfig, config)
        self.run_config = run_config
        self._nodes = {
            name: agent_node(run_config, name, entry)
            for name, entry in self.config.policies.items()
        }

    def run(self) -> dict[str, float]:
        """Run the experiment.

        Returns:
            The summary of every policy under ``<policy>/``, and ``wall_time_s``, the wall
            time of the whole experiment.
        """
        start = time.perf_counter()
        seed = int(self.run_config.seed)
        tracker = start_tracker(self.run_config, exclude=("agent",))
        with closing_tracker(tracker):
            agents = {name: build_agent(node) for name, node in self._nodes.items()}
            for name, entry in self.config.policies.items():
                if entry.checkpoint is not None:
                    agents[name].load(resolve_path(entry.checkpoint))
            env = build_env(self.run_config)
            rows: dict[str, dict[str, float]] = {}
            metrics: dict[str, float] = {}
            for name, entry in self.config.policies.items():
                agent = agents[name]
                if entry.checkpoint is None:
                    fit_tracker = PrefixedTracker(tracker, f"{name}/fit/")
                    policy = agent.fit(env, tracker=fit_tracker, seed=seed)
                else:
                    policy = agent.policy()
                summary = evaluate_policy(
                    env,
                    policy,
                    seeds=self.config.eval_seeds,
                    tracker=PrefixedTracker(tracker, f"{name}/"),
                    root_seed=seed,
                )
                rows[name] = summary.metrics
                metrics.update(prefixed(summary.metrics, f"{name}/"))
            metrics["wall_time_s"] = time.perf_counter() - start
            path = resolve_path(self.config.metrics_dir) / f"{self.config.name}.json"
            write_metrics(path, metrics)
            tracker.log_artifact(path)
            log_summary(tracker, metrics)
        _log.info(
            "compare: %d policies on seeds %s (lower is better)\n%s",
            len(rows),
            list(self.config.eval_seeds),
            comparison_table(rows),
        )
        return metrics
