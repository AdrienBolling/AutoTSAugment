"""A view of a tracker that puts a prefix in front of every key.

Experiments use it to keep the phases of one run apart in the same backend: the evaluation
episodes of ``train_agent`` log under ``eval/``, and each policy of ``compare`` logs under
``<policy>/``. Step and episode indices then start at zero in each phase.
"""

from collections.abc import Mapping
from pathlib import Path

from autotsaugment.core.tracker import Metrics, Tracker


class PrefixedTracker(Tracker):
    """Forwards every call to ``tracker``, with ``prefix`` in front of each key.

    The view does not own the tracker: :meth:`close` only flushes. The owner of ``tracker``
    closes it.

    Args:
        tracker: The tracker that receives the calls.
        prefix: The prefix of every metric and parameter key, for example ``"eval/"``.
    """

    def __init__(self, tracker: Tracker, prefix: str) -> None:
        self.tracker = tracker
        self.prefix = prefix

    def _prefixed[V](self, values: Mapping[str, V]) -> dict[str, V]:
        return {f"{self.prefix}{key}": value for key, value in values.items()}

    def log_params(self, params: Mapping[str, object]) -> None:
        """Forward the parameters with prefixed top-level keys."""
        self.tracker.log_params(self._prefixed(params))

    def log_step(self, step: int, metrics: Metrics) -> None:
        """Forward the step metrics with prefixed keys. No host synchronization."""
        self.tracker.log_step(step, self._prefixed(metrics))

    def log_episode(self, episode: int, metrics: Metrics) -> None:
        """Forward the episode metrics with prefixed keys. The tracker flushes."""
        self.tracker.log_episode(episode, self._prefixed(metrics))

    def log_artifact(self, path: Path) -> None:
        """Forward the artifact unchanged."""
        self.tracker.log_artifact(path)

    def flush(self) -> None:
        """Flush the tracker."""
        self.tracker.flush()

    def close(self) -> None:
        """Flush the tracker, but leave it open for its owner."""
        self.tracker.flush()
