"""A tracker that drops everything. The default for tests."""

from collections.abc import Mapping
from pathlib import Path

from autotsaugment.core.tracker import Metrics, Tracker


class NullTracker(Tracker):
    """Accepts every call and records nothing."""

    def __init__(self) -> None:
        """Create the tracker."""

    def log_params(self, params: Mapping[str, object]) -> None:
        """Drop the parameters."""

    def log_step(self, step: int, metrics: Metrics) -> None:
        """Drop the step metrics."""

    def log_episode(self, episode: int, metrics: Metrics) -> None:
        """Drop the episode metrics."""

    def log_artifact(self, path: Path) -> None:
        """Drop the artifact."""

    def flush(self) -> None:
        """Do nothing."""

    def close(self) -> None:
        """Do nothing."""
