"""Featurizers of the curriculum environment.

:class:`StatsFeaturizer` builds the four observation groups of decision 4 of the plan. Each
group is a float32 tensor with a leading k, one row per lockstep environment:

- ``progress`` (2): ``epoch / E`` and ``series so far / (N * E)``.
- ``train_stats`` (5): training loss, EMA of the loss, least-squares slope of the loss over
  the last ``loss_window`` epochs, gradient norm, learning rate.
- ``reward_vector`` (3 * d): skill per dataset, change of the skill since the previous step,
  best (lowest) skill so far in the episode.
- ``action_history`` (a): EMA of the past actions, with weight ``action_ema`` on the newest.

How the initial scores reach the featurizer: the contract keeps ``reset() -> Observation``.
:meth:`StatsFeaturizer.reset` forgets the episode and returns an observation with no scores.
The environment then calls the featurizer with an :class:`EpochRecord` with ``epoch = 0``
that holds the scores of the untrained tsmodels. A record with ``epoch = 0`` sets the scores
only: it does not change the training statistics, the series count or the action history.
"""

import torch
from jaxtyping import Float
from pydantic import Field
from torch import Tensor

from autotsaugment.components.env.history import RecentValues
from autotsaugment.core.featurizer import (
    EpochRecord,
    Featurizer,
    FeaturizerConfig,
    Observation,
    ObservationSpec,
)
from autotsaugment.core.typechecking import checked

FeatureBatch = Float[Tensor, "k f"]
"""One observation group, one row per lockstep environment."""

PROGRESS = "progress"
TRAIN_STATS = "train_stats"
REWARD_VECTOR = "reward_vector"
ACTION_HISTORY = "action_history"
TRAIN_FEATURES = ("loss", "loss_ema", "loss_slope", "grad_norm", "lr")
"""Columns of the ``train_stats`` group, in order."""


class StatsFeaturizerConfig(FeaturizerConfig):
    """Config of :class:`StatsFeaturizer`.

    Attributes:
        loss_window: Epochs used for the least-squares slope of the loss.
        loss_ema: Weight of the newest loss in the loss EMA.
        action_ema: Weight of the newest action in the action history.
        normalize: Apply a running mean and variance per feature. The statistics merge every
            observation over steps, environments and episodes, and never freeze.
        epsilon: Added to the running variance before the square root.
    """

    loss_window: int = Field(default=5, ge=2)
    loss_ema: float = Field(default=0.5, gt=0.0, le=1.0)
    action_ema: float = Field(default=0.5, gt=0.0, le=1.0)
    normalize: bool = True
    epsilon: float = Field(default=1e-8, gt=0.0)


class RunningMoments:
    """Running mean and variance of each feature, on the device.

    The update merges one batch at a time with the parallel form of Welford's algorithm
    (Chan, Golub and LeVeque). The count is a host integer, so nothing synchronizes.

    Args:
        n_features: Number of features, f.
        device: Device of the buffers.
        epsilon: Added to the variance before the square root in :meth:`normalize`.
    """

    def __init__(self, n_features: int, *, device: torch.device, epsilon: float) -> None:
        self._count = 0
        self._mean = torch.zeros(n_features, device=device)
        self._m2 = torch.zeros(n_features, device=device)
        self._epsilon = epsilon

    @property
    def count(self) -> int:
        """Return the number of rows merged so far."""
        return self._count

    @property
    def mean(self) -> Tensor:
        """Return the running mean, shape ``(f,)``."""
        return self._mean

    @property
    def variance(self) -> Tensor:
        """Return the running population variance, shape ``(f,)``, zeros before any update."""
        return self._m2 / max(self._count, 1)

    @checked
    def update(self, batch: FeatureBatch) -> None:
        """Merge the rows of ``batch``, shape ``(k, f)``, into the statistics."""
        rows = int(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_m2 = ((batch - batch_mean) ** 2).sum(dim=0)
        total = self._count + rows
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * (rows / total)
        self._m2 = self._m2 + batch_m2 + delta**2 * (self._count * rows / total)
        self._count = total

    @checked
    def normalize(self, batch: FeatureBatch) -> FeatureBatch:
        """Return ``(batch - mean) / sqrt(variance + epsilon)``, or ``batch`` before any update."""
        if self._count == 0:
            return batch
        return (batch - self._mean) / torch.sqrt(self.variance + self._epsilon)


class StatsFeaturizer(Featurizer[StatsFeaturizerConfig]):
    """Hand-made statistics of the training run, in four groups. See the module docstring.

    Args:
        config: The featurizer config.
        n_datasets: Number of benchmark datasets, d.
        action_dim: Dimension of the action, a.
        batch_envs: Number of lockstep environments, k.
        device: Device of every buffer and observation.

    Raises:
        ValueError: If a size is less than 1.
    """

    def __init__(
        self,
        *,
        config: StatsFeaturizerConfig,
        n_datasets: int,
        action_dim: int,
        batch_envs: int,
        device: torch.device | str,
    ) -> None:
        for name, value in (
            ("n_datasets", n_datasets),
            ("action_dim", action_dim),
            ("batch_envs", batch_envs),
        ):
            if value < 1:
                msg = f"{name} must be at least 1, got {value}"
                raise ValueError(msg)
        self.config = config
        self._n_datasets = n_datasets
        self._action_dim = action_dim
        self._batch_envs = batch_envs
        self._device = torch.device(device)
        self._moments = {
            name: RunningMoments(shape[0], device=self._device, epsilon=config.epsilon)
            for name, shape in self.observation_spec.items()
        }
        self._losses = RecentValues(config.loss_window)
        self._clear_episode()

    def _clear_episode(self) -> None:
        self._losses.clear()
        self._steps = 0
        self._has_scores = False
        self._series_seen = self._zeros()
        self._loss_ema = self._zeros()
        self._scores = self._zeros(self._n_datasets)
        self._best = self._zeros(self._n_datasets)
        self._action_ema = self._zeros(self._action_dim)

    @property
    def observation_spec(self) -> ObservationSpec:
        """Return the shape of each group, without the leading k."""
        return {
            PROGRESS: (2,),
            TRAIN_STATS: (len(TRAIN_FEATURES),),
            REWARD_VECTOR: (3 * self._n_datasets,),
            ACTION_HISTORY: (self._action_dim,),
        }

    @property
    def moments(self) -> dict[str, RunningMoments]:
        """Return the running statistics of each group, used when ``normalize`` is true."""
        return self._moments

    def _zeros(self, *features: int) -> Tensor:
        return torch.zeros((self._batch_envs, *features), device=self._device)

    def reset(self) -> Observation:
        """Forget the episode and return the observation before the first step.

        The running statistics of the normalization persist across episodes. The returned
        observation has no scores yet; the environment passes them next, through a record
        with ``epoch = 0``.
        """
        self._clear_episode()
        observation = {name: self._zeros(*shape) for name, shape in self.observation_spec.items()}
        return self._finish(observation, update=False)

    @checked
    def __call__(self, record: EpochRecord) -> Observation:
        """Return the observation after ``record`` and update the running statistics.

        Raises:
            ValueError: If the scores or the action of ``record`` have the wrong shape.
        """
        self._check(record)
        if record.epoch == 0:
            progress = self._zeros(2)
            train_stats = self._zeros(len(TRAIN_FEATURES))
        else:
            progress = self._progress(record)
            train_stats = self._train_stats(record)
            weight = self.config.action_ema
            action = record.action.detach().float()
            self._action_ema = weight * action + (1.0 - weight) * self._action_ema
            self._steps += 1
        observation = {
            PROGRESS: progress,
            TRAIN_STATS: train_stats,
            REWARD_VECTOR: self._reward_vector(record),
            ACTION_HISTORY: self._action_ema,
        }
        return self._finish(observation, update=True)

    def _check(self, record: EpochRecord) -> None:
        expected = {
            "scores": (self._batch_envs, self._n_datasets),
            "action": (self._batch_envs, self._action_dim),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(record, name).shape)
            if actual != shape:
                msg = f"record.{name} has shape {actual}, expected {shape}"
                raise ValueError(msg)

    def _progress(self, record: EpochRecord) -> Tensor:
        epoch_fraction = torch.full(
            (self._batch_envs,), record.epoch / record.epochs, device=self._device
        )
        series = record.train.get("series")
        if series is None:
            # Without a series count, every step counts as the same number of series.
            series_fraction = epoch_fraction
        else:
            series = series.detach().float()
            self._series_seen = self._series_seen + series
            series_fraction = self._series_seen / (series * record.epochs)
        return torch.stack((epoch_fraction, series_fraction), dim=-1)

    def _train_stats(self, record: EpochRecord) -> Tensor:
        loss = record.train["loss"].detach().float()
        self._losses.append(loss)
        if self._steps == 0:
            self._loss_ema = loss
        else:
            weight = self.config.loss_ema
            self._loss_ema = weight * loss + (1.0 - weight) * self._loss_ema
        columns = (
            loss,
            self._loss_ema,
            self._losses.slope(),
            record.train["grad_norm"].detach().float(),
            record.train["lr"].detach().float(),
        )
        return torch.stack(columns, dim=-1)

    def _reward_vector(self, record: EpochRecord) -> Tensor:
        scores = record.scores.detach().float()
        if self._has_scores:
            change = scores - self._scores
            best = torch.minimum(self._best, scores)
        else:
            change = torch.zeros_like(scores)
            best = scores
        self._scores = scores
        self._best = best
        self._has_scores = True
        return torch.cat((scores, change, best), dim=-1)

    def _finish(self, observation: Observation, *, update: bool) -> Observation:
        if not self.config.normalize:
            return {name: value.clone() for name, value in observation.items()}
        normalized: Observation = {}
        for name, value in observation.items():
            moments = self._moments[name]
            if update:
                moments.update(value)
            normalized[name] = moments.normalize(value)
        return normalized
