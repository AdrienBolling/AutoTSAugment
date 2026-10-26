"""Smoke experiment: build the run, fit the agent, evaluate its policy for a few episodes.

With the defaults of ``configs/experiment/smoke.yaml`` (fake env, no tracker, CPU, fixed
uniform agent), ``uv run autotsaugment-run`` finishes in seconds. With ``env=default
device=cuda`` it is the first check of the real environment.
"""

import logging
import time
from typing import Any

from omegaconf import DictConfig
from pydantic import Field

from autotsaugment.components.experiments.common import (
    EVAL_PREFIX,
    ExperimentConfig,
    build_run,
    closing_tracker,
    evaluate_policy,
    experiment_config,
    log_summary,
)
from autotsaugment.components.tracking.prefixed import PrefixedTracker
from autotsaugment.core.experiment import Experiment

_log = logging.getLogger(__name__)


class SmokeConfig(ExperimentConfig):
    """Config of :class:`SmokeExperiment`.

    Attributes:
        name: The experiment name.
        episodes: Evaluation episodes, with the evaluation seeds ``0 .. episodes - 1``.
    """

    name: str = "smoke"
    episodes: int = Field(default=1, ge=1)


class SmokeExperiment(Experiment):
    """Fit the agent of the run, then evaluate its policy for ``episodes`` episodes.

    The agent trains for its own ``config.episodes`` (zero for the baselines). The
    evaluation episodes log under ``eval/``.

    Args:
        run_config: The composed run config, resolved.
        **config: The leaves of ``configs/experiment/smoke.yaml``, validated as
            :class:`SmokeConfig`.
    """

    def __init__(self, *, run_config: DictConfig, **config: Any) -> None:
        self.config = experiment_config(SmokeConfig, config)
        self.run_config = run_config

    def run(self) -> dict[str, float]:
        """Run the experiment.

        Returns:
            The evaluation summary (keys in :mod:`autotsaugment.components.experiments.common`)
            and ``wall_time_s``, the wall time of the whole experiment.
        """
        start = time.perf_counter()
        seed = int(self.run_config.seed)
        run = build_run(self.run_config)
        with closing_tracker(run.tracker) as tracker:
            policy = run.agent.fit(run.env, tracker=tracker, seed=seed)
            summary = evaluate_policy(
                run.env,
                policy,
                seeds=range(self.config.episodes),
                tracker=PrefixedTracker(tracker, EVAL_PREFIX),
                root_seed=seed,
            )
            metrics = {**summary.metrics, "wall_time_s": time.perf_counter() - start}
            log_summary(tracker, metrics)
        _log.info(
            "smoke: aggregate %.4f over %d episodes in %.1f s",
            metrics["comparison/aggregate/mean"],
            summary.episodes,
            metrics["wall_time_s"],
        )
        return metrics
