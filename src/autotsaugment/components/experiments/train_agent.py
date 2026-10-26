"""Train-agent experiment: fit one agent, save it, evaluate its policy on fresh seeds.

Outputs. Relative paths are relative to the working directory, the repository root under
Hydra, not to the Hydra output directory:

- ``<checkpoint_dir>/<label>/``: the agent state (``agent.pt``, from ``agent.save``) and the
  agent config of the run (``agent.yaml``). The compare experiment loads both.
- ``<metrics_dir>/<name>-<label>.json``: the metrics, for ``dvc metrics``.
- The tracker run: the training episodes without a prefix, the evaluation episodes under
  ``eval/``, the checkpoint directory and the metrics file as artifacts, and the summary.

``checkpoints/`` is ignored by git, so DVC cannot track a checkpoint there. The log gives the
commands that copy a keeper checkpoint to ``models/`` and track it with DVC.
"""

import logging
import re
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from autotsaugment.components.experiments.common import (
    AGENT_CONFIG_FILE,
    EVAL_PREFIX,
    EvaluatedConfig,
    build_run,
    closing_tracker,
    evaluate_policy,
    experiment_config,
    log_summary,
    resolve_path,
    synchronize,
)
from autotsaugment.components.tracking.prefixed import PrefixedTracker
from autotsaugment.core.experiment import Experiment, write_metrics

_log = logging.getLogger(__name__)

KEEPER_DIR = "models"
"""Directory, tracked by git, of the DVC files of the keeper checkpoints."""

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class TrainAgentConfig(EvaluatedConfig):
    """Config of :class:`TrainAgentExperiment`.

    Attributes:
        name: The experiment name, the first part of the metrics file name.
        eval_seeds: The evaluation seeds, one episode each.
        checkpoint_dir: Parent directory of the checkpoint, relative to the working directory.
        metrics_dir: Directory of the metrics file, relative to the working directory.
        label: Name of the checkpoint directory and suffix of the metrics file. ``None`` uses
            the agent class name in snake case without ``Agent``: ``sac``, ``ppo``, ``cmaes``,
            ``fixed_action``, ``random``.
    """

    name: str = "train_agent"
    checkpoint_dir: str = "checkpoints/train_agent"
    label: str | None = None


def agent_label(agent: object) -> str:
    """Return the class name of ``agent`` in snake case, without the ``Agent`` suffix.

    Examples: ``SACAgent`` gives ``sac``, ``FixedActionAgent`` gives ``fixed_action``.
    """
    name = type(agent).__name__
    name = name.removesuffix("Agent") or name
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def display_path(path: Path) -> str:
    """Return ``path`` relative to the working directory when it is below it."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def dvc_commands(checkpoint: Path, metrics: Path, keeper: str) -> list[str]:
    """Return the commands that track a keeper checkpoint and its metrics with DVC and git.

    Args:
        checkpoint: The checkpoint directory.
        metrics: The metrics file.
        keeper: Name of the keeper copy in :data:`KEEPER_DIR`.
    """
    target = f"{KEEPER_DIR}/{keeper}"
    return [
        f"dvc add {display_path(checkpoint)} -o {target}",
        f"git add {target}.dvc {KEEPER_DIR}/.gitignore {display_path(metrics)}",
        f'git commit -m "chore(models): add {keeper}"',
        "dvc push",
    ]


class TrainAgentExperiment(Experiment):
    """Fit the agent of the run, save it, and evaluate its policy on ``eval_seeds``.

    The agent trains for its own ``config.episodes`` with the run seed. The evaluation uses
    the protocol of :func:`~autotsaugment.components.experiments.common.evaluate_policy`.

    Args:
        run_config: The composed run config, resolved.
        **config: The leaves of ``configs/experiment/train_agent.yaml``, validated as
            :class:`TrainAgentConfig`.
    """

    def __init__(self, *, run_config: DictConfig, **config: Any) -> None:
        self.config = experiment_config(TrainAgentConfig, config)
        self.run_config = run_config

    def run(self) -> dict[str, float]:
        """Run the experiment.

        Returns:
            The evaluation summary (keys in :mod:`autotsaugment.components.experiments.common`),
            ``train/wall_time_s``, ``train/episodes`` and ``wall_time_s``, the wall time of the
            whole experiment.
        """
        start = time.perf_counter()
        seed = int(self.run_config.seed)
        run = build_run(self.run_config)
        with closing_tracker(run.tracker) as tracker:
            label = self.config.label or agent_label(run.agent)
            fit_start = time.perf_counter()
            policy = run.agent.fit(run.env, tracker=tracker, seed=seed)
            synchronize(run.env.device)
            train_time = time.perf_counter() - fit_start

            checkpoint = resolve_path(self.config.checkpoint_dir) / label
            if checkpoint.exists():
                _log.info("train_agent: overwrite the checkpoint in %s", checkpoint)
            run.agent.save(checkpoint)
            OmegaConf.save(self.run_config.agent, checkpoint / AGENT_CONFIG_FILE)
            tracker.log_artifact(checkpoint)

            summary = evaluate_policy(
                run.env,
                policy,
                seeds=self.config.eval_seeds,
                tracker=PrefixedTracker(tracker, EVAL_PREFIX),
                root_seed=seed,
            )
            metrics = {
                **summary.metrics,
                "train/wall_time_s": train_time,
                "train/episodes": int(run.agent.config.episodes),
                "wall_time_s": time.perf_counter() - start,
            }
            keeper = f"{self.config.name}-{label}"
            path = resolve_path(self.config.metrics_dir) / f"{keeper}.json"
            write_metrics(path, metrics)
            tracker.log_artifact(path)
            log_summary(tracker, metrics)
        _log.info(
            "train_agent: %s, aggregate %.4f +- %.4f over %d seeds, checkpoint in %s",
            label,
            metrics["comparison/aggregate/mean"],
            metrics["comparison/aggregate/std"],
            summary.episodes,
            checkpoint,
        )
        _log.info(
            "to keep this checkpoint, run from the repository root:\n  %s",
            "\n  ".join(dvc_commands(checkpoint, path, keeper)),
        )
        return metrics
