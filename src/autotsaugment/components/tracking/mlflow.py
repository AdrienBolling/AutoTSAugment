"""MLflow tracker: a local store under ``mlruns/``, step metrics written once per episode.

The tracker writes to an MLflow store through :class:`mlflow.MlflowClient`. It never uses the
global MLflow run of the fluent API, so two trackers in one process do not interfere.

Store: a ``tracking_uri`` without a scheme, for example ``mlruns``, is a local directory. A
relative directory is relative to the current working directory, which is the repository root
under Hydra (``hydra.job.chdir: false``). The directory holds the SQLite database
``mlflow.db`` and the run artifacts in ``artifacts/``. MLflow 3.17 refuses the plain file store
unless ``MLFLOW_ALLOW_FILE_STORE=true``, so the local store is SQLite. A ``tracking_uri`` with
a scheme (``sqlite://``, ``http://``, ``file://``) goes to MLflow unchanged. Open the local
store with ``uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db``.

Performance: :meth:`MLflowTracker.log_step` only appends the metrics to a buffer. The tensors
stay on their device, untouched. :meth:`MLflowTracker.flush` moves the whole buffer to the host
with one transfer per device, then writes it with ``log_batch``. The episode loop calls
:meth:`MLflowTracker.log_episode` once per episode, which flushes.
"""

import importlib
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.tracker import Metrics, Tracker

PARAMS_PER_BATCH = 100
"""Parameters per ``log_batch`` call, the MLflow limit."""

METRICS_PER_BATCH = 1000
"""Metric values per ``log_batch`` call, the MLflow limit."""

MAX_PARAM_LENGTH = 6000
"""Characters of a parameter value that MLflow accepts. Longer values are truncated."""

STORE_FILE = "mlflow.db"
"""SQLite database inside a local store directory."""

ARTIFACT_DIR = "artifacts"
"""Artifact root inside a local store directory."""

EPISODE_PREFIX = "episode/"
"""Prefix of the episode metrics. Their MLflow step is the episode index."""

GIT_COMMIT_TAG = "mlflow.source.git.commit"
"""The MLflow run tag of the git commit of the working directory."""


class MLflowTrackerConfig(ComponentConfig):
    """Config of :class:`MLflowTracker`.

    Attributes:
        tracking_uri: A local store directory, or an MLflow tracking URI with a scheme.
        experiment_name: The MLflow experiment. The tracker creates it when it is missing.
        run_name: The MLflow run name. ``None`` lets MLflow choose a random name.
        tags: Extra run tags.
    """

    tracking_uri: str = "mlruns"
    experiment_name: str = "default"
    run_name: str | None = None
    tags: dict[str, str] | None = None


def resolve_store(tracking_uri: str) -> tuple[str, str | None]:
    """Return the MLflow store URI and the artifact location of ``tracking_uri``.

    Args:
        tracking_uri: A directory, absolute or relative to the current working directory, or
            a URI with a scheme.

    Returns:
        For a directory: ``sqlite:///<dir>/mlflow.db`` and ``file://<dir>/artifacts``. The
        directory is created. For a URI: the URI unchanged, and ``None``, so the store chooses
        the artifact location.
    """
    if "://" in tracking_uri:
        return tracking_uri, None
    root = Path(tracking_uri).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / STORE_FILE}", (root / ARTIFACT_DIR).as_uri()


def flatten_params(params: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    """Return ``params`` as one level of dotted keys with string values.

    Nested mappings become dotted keys, for example ``env.config.episode.epochs``. Other
    values, lists included, become their ``str``, truncated to :data:`MAX_PARAM_LENGTH`.

    Args:
        params: Nested parameters, for example a resolved Hydra config.
        prefix: Prefix of every key.
    """
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)[:MAX_PARAM_LENGTH]
    return flat


def _now_ms() -> int:
    """Return the host time in milliseconds, the MLflow timestamp unit."""
    return time.time_ns() // 1_000_000


class MLflowTracker(Tracker):
    """Writes the parameters, metrics and artifacts of one run to an MLflow store.

    The constructor creates the experiment when it is missing and starts a new run, tagged
    with the user, the source and the git commit of the working directory. The ``mlflow``
    package is imported here, not at module import, because the config tests import every
    ``_target_`` module and ``mlflow`` takes about 0.6 s to import.

    Args:
        **config: The fields of :class:`MLflowTrackerConfig`, the leaves of
            ``configs/tracker/mlflow.yaml``.
    """

    def __init__(self, **config: Any) -> None:
        self.config = MLflowTrackerConfig.model_validate(config)
        self._entities = importlib.import_module("mlflow.entities")
        tracking = importlib.import_module("mlflow.tracking")
        context = importlib.import_module("mlflow.tracking.context.registry")
        self.store_uri, artifact_location = resolve_store(self.config.tracking_uri)
        self._client: Any = tracking.MlflowClient(tracking_uri=self.store_uri)
        experiment = self._client.get_experiment_by_name(self.config.experiment_name)
        if experiment is None:
            experiment_id = self._client.create_experiment(
                self.config.experiment_name, artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
        tags = context.resolve_tags(dict(self.config.tags or {}))
        # MLflow reads the commit from the entry-point script in .venv/bin and finds none.
        commit = importlib.import_module("mlflow.utils.git_utils").get_git_commit(str(Path.cwd()))
        if commit is not None:
            tags.setdefault(GIT_COMMIT_TAG, commit)
        run = self._client.create_run(experiment_id, run_name=self.config.run_name, tags=tags)
        self.run_id: str = run.info.run_id
        self._buffer: list[tuple[int, int, Metrics]] = []
        self._status = "FINISHED"
        self._closed = False

    def log_params(self, params: Mapping[str, object]) -> None:
        """Write ``params`` as flat string parameters, in batches of 100.

        MLflow parameters are immutable: a second call with the same key and another value
        fails.
        """
        param = self._entities.Param
        items = list(flatten_params(params).items())
        for start in range(0, len(items), PARAMS_PER_BATCH):
            chunk = items[start : start + PARAMS_PER_BATCH]
            self._client.log_batch(self.run_id, params=[param(k, v) for k, v in chunk])

    def log_step(self, step: int, metrics: Metrics) -> None:
        """Append the metrics of one step to the buffer. No host synchronization.

        The tensors stay untouched on their device until :meth:`flush`. The timestamp is the
        host time of this call.
        """
        self._buffer.append((step, _now_ms(), dict(metrics)))

    def log_episode(self, episode: int, metrics: Metrics) -> None:
        """Buffer the episode metrics under ``episode/`` at step ``episode``, then flush."""
        prefixed = {f"{EPISODE_PREFIX}{key}": value for key, value in metrics.items()}
        self._buffer.append((episode, _now_ms(), prefixed))
        self.flush()

    def log_artifact(self, path: Path) -> None:
        """Copy a file, or a directory under its own name, into the run artifacts."""
        if path.is_dir():
            self._client.log_artifacts(self.run_id, str(path), artifact_path=path.name)
        else:
            self._client.log_artifact(self.run_id, str(path))

    def flush(self) -> None:
        """Move the buffer to the host in one transfer per device and write it.

        The values go to ``log_batch`` in batches of 1000. NaN values are written as NaN.
        """
        if not self._buffer:
            return
        keys: list[tuple[str, int, int]] = []
        values: list[float] = []
        pending: dict[torch.device, list[tuple[int, Tensor]]] = {}
        for step, timestamp, metrics in self._buffer:
            for key, value in metrics.items():
                keys.append((key, timestamp, step))
                if isinstance(value, Tensor):
                    slot = (len(values), value.detach().reshape(()))
                    pending.setdefault(value.device, []).append(slot)
                    values.append(math.nan)
                else:
                    values.append(float(value))
        self._buffer = []
        for slots in pending.values():
            host = torch.stack([tensor for _, tensor in slots]).to("cpu", torch.float64)
            for (index, _), value in zip(slots, host.tolist(), strict=True):
                values[index] = value
        metric = self._entities.Metric
        entries = [
            metric(key, value, timestamp, step)
            for (key, timestamp, step), value in zip(keys, values, strict=True)
        ]
        for start in range(0, len(entries), METRICS_PER_BATCH):
            chunk = entries[start : start + METRICS_PER_BATCH]
            self._client.log_batch(self.run_id, metrics=chunk)

    def mark_failed(self) -> None:
        """End the run with the status ``FAILED`` at :meth:`close`, not ``FINISHED``."""
        self._status = "FAILED"

    def close(self) -> None:
        """Flush, then end the run. A second call does nothing."""
        if self._closed:
            return
        self._closed = True
        try:
            self.flush()
        finally:
            self._client.set_terminated(self.run_id, status=self._status)
