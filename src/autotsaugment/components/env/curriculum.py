"""The curriculum environment: K tsmodels trained in lockstep on synthetic series.

One step is one training epoch (section 3.1 of the plan):

1. Clip the action to the action space.
2. Sample N series per environment with the augmentation method.
3. Train tsmodel i for one epoch on the series of environment i, for i in 0..K-1.
4. Draw one reward subset, the same for all K tsmodels, and score every tsmodel on it.
5. Build the :class:`EpochRecord`, then the reward and the observation.

Random streams: each stream has its own generator, derived from the ``reset`` seed with
:func:`derive_seed`. ``("tsmodel", i)`` resets tsmodel i, ``("train", i)`` drives its
training, ``"augmentation"`` drives the series, and ``"reward_subset"`` draws the reward
windows. The reward subset has its own stream, so two episodes with the same seed score the
same windows at every step, whatever the actions. Rewards of two policies are then paired.

The initial scores reach the featurizer through a record with ``epoch = 0``: ``reset``
calls ``featurizer.reset()``, then ``featurizer(record_0)``, and returns that observation.

The environment never logs. Trackers belong to agents and experiments, which read
:attr:`StepResult.info`.
"""

from collections.abc import Callable
from dataclasses import fields
from typing import Any

import torch
from pydantic import Field
from torch import Tensor

from autotsaugment.components.env.scoring import score
from autotsaugment.core.augmentation import AugmentationMethod
from autotsaugment.core.benchmark import Benchmark, WindowSet
from autotsaugment.core.environment import Environment, EnvironmentConfig, StepResult
from autotsaugment.core.featurizer import EpochRecord, Featurizer, Observation, ObservationSpec
from autotsaugment.core.reward import RewardFunction
from autotsaugment.core.seeding import derive_seed, make_generator
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tsmodel import TrainStats, TSModel, TSModelFactory
from autotsaugment.core.typechecking import ActionBatch, PerDatasetK, checked

SERIES_KEY = "series"
"""Key of ``EpochRecord.train`` that holds the series trained on in the step, shape ``(k,)``."""

TRAIN_KEYS = (*(field.name for field in fields(TrainStats)), SERIES_KEY)
"""Keys of ``EpochRecord.train``: the :class:`TrainStats` fields, then the series count."""


class CurriculumEnvConfig(EnvironmentConfig):
    """Config of :class:`CurriculumEnv`.

    The fields are those of :class:`EnvironmentConfig`, with lower bounds.

    Attributes:
        reward_subset_per_dataset: Reward windows scored per step and dataset. The subset is
            redrawn every step and is the same for all K tsmodels. ``None`` scores the full
            reward set.
        score_batch_size: Windows per forward pass when scoring.
    """

    reward_subset_per_dataset: int | None = Field(default=128, ge=1)
    score_batch_size: int = Field(default=1024, ge=1)


def _canonical(device: torch.device | str) -> torch.device:
    """Return ``device`` with an explicit index for CUDA, so equal devices compare equal."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class CurriculumEnv(Environment[CurriculumEnvConfig]):
    """K lockstep training runs of a tsmodel on series from one augmentation method.

    See the module docstring for the step, the random streams and how the initial scores
    reach the featurizer.

    ``StepResult.info`` holds, each with a leading k:

    - ``scores`` ``(k, d)`` and ``aggregate`` ``(k,)`` on the reward subset of the step;
    - ``train/<name>`` ``(k,)`` for every key of :data:`TRAIN_KEYS`;
    - ``action`` ``(k, a)``, the clipped action;
    - ``reward_rows`` ``(k, m)``, int64, the rows of the reward set scored in the step. All
      k rows are equal, because all tsmodels use the same subset.

    Args:
        config: The environment config.
        device: Device of every tensor the environment returns.
        tsmodel: Factory that builds one tsmodel; called K times.
        augmentation: The synthetic series generator, on ``device``.
        benchmark: The real windows, on ``device``.
        featurizer: Factory that builds the featurizer from ``n_datasets``, ``action_dim``,
            ``batch_envs`` and ``device``. Hydra gives it with ``_partial_: true``.
        reward: The reward function.

    Raises:
        TypeError: If ``featurizer`` is a featurizer instance and not a factory.
        ValueError: If a component lives on another device than ``device``.
    """

    def __init__(
        self,
        *,
        config: CurriculumEnvConfig,
        device: torch.device | str,
        tsmodel: TSModelFactory,
        augmentation: AugmentationMethod[Any],
        benchmark: Benchmark[Any],
        featurizer: Callable[..., Featurizer[Any]],
        reward: RewardFunction[Any],
    ) -> None:
        if isinstance(featurizer, Featurizer):
            msg = "featurizer must be a factory; set `_partial_: true` in its config"
            raise TypeError(msg)
        self.config = config
        self._device = torch.device(device)
        self._canonical_device = _canonical(self._device)
        self._augmentation = augmentation
        self._benchmark = benchmark
        self._reward = reward
        self._tsmodels = tuple(tsmodel() for _ in range(self.batch_size))
        self._check_devices()
        self._featurizer = featurizer(
            n_datasets=len(benchmark.dataset_names),
            action_dim=augmentation.action_space.dim,
            batch_envs=self.batch_size,
            device=self._device,
        )
        reward_windows = benchmark.reward_windows
        # Built once here: `torch.nonzero` synchronizes, so it must stay out of `step`.
        self._rows_by_dataset = tuple(
            torch.nonzero(reward_windows.dataset_index == dataset, as_tuple=False).squeeze(-1)
            for dataset in range(reward_windows.n_datasets)
        )
        self._all_rows = torch.arange(reward_windows.n_windows, device=self._device)
        self._epoch: int | None = None
        self._augmentation_generator = make_generator(self._device, 0)
        self._subset_generator = make_generator(self._device, 0)
        self._train_generators = tuple(make_generator(self._device, 0) for _ in self._tsmodels)

    def _check_devices(self) -> None:
        expected = self._canonical_device
        found = {
            "augmentation": self._augmentation.device,
            "benchmark reward windows": self._benchmark.reward_windows.device,
            "benchmark comparison windows": self._benchmark.comparison_windows.device,
            **{f"tsmodel {i}": model.device for i, model in enumerate(self._tsmodels)},
        }
        for name, device in found.items():
            if _canonical(device) != expected:
                msg = f"the {name} live on {device}, but the environment device is {self._device}"
                raise ValueError(msg)

    @property
    def device(self) -> torch.device:
        """Return the device of all tensors."""
        return self._device

    @property
    def action_space(self) -> BoxSpace:
        """Return the action space of the augmentation method."""
        return self._augmentation.action_space

    def uniform_action(self) -> Tensor:
        """Return the uniform action of the augmentation method, shape ``(a,)``."""
        return self._augmentation.uniform_action()

    @property
    def observation_spec(self) -> ObservationSpec:
        """Return the shape of each observation entry, without the leading k."""
        return self._featurizer.observation_spec

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return the benchmark dataset names, the order of the score vectors."""
        return self._benchmark.dataset_names

    @property
    def tsmodels(self) -> tuple[TSModel[Any], ...]:
        """Return the K tsmodels, for experiments that checkpoint one of them."""
        return self._tsmodels

    @property
    def epoch(self) -> int | None:
        """Return the steps done in the current episode, or ``None`` before the first reset."""
        return self._epoch

    def _zeros(self, *shape: int) -> Tensor:
        return torch.zeros((self.batch_size, *shape), device=self._device)

    def _draw_reward_subset(self) -> tuple[Tensor, WindowSet]:
        """Return the rows of the reward subset and its windows.

        Same rule as ``WindowSet.subset``, without its ``torch.nonzero``: at most
        ``reward_subset_per_dataset`` random rows per dataset, all rows of smaller datasets.
        """
        windows = self._benchmark.reward_windows
        per_dataset = self.config.reward_subset_per_dataset
        if per_dataset is None:
            return self._all_rows, windows
        keep: list[Tensor] = []
        for rows in self._rows_by_dataset:
            n_rows = rows.numel()
            if n_rows > per_dataset:
                order = torch.randperm(n_rows, generator=self._subset_generator, device=rows.device)
                keep.append(rows[order[:per_dataset]])
            else:
                keep.append(rows)
        selected = torch.cat(keep)
        return selected, windows.select(selected)

    def _score_all(self, windows: WindowSet) -> tuple[Tensor, Tensor]:
        """Return the scores ``(k, d)`` and the aggregates ``(k,)`` of the K tsmodels."""
        results = [
            score(model, windows, batch_size=self.config.score_batch_size)
            for model in self._tsmodels
        ]
        scores = torch.stack([skill for skill, _ in results])
        aggregates = torch.stack([value for _, value in results])
        return scores, aggregates

    @checked
    def reset(self, *, seed: int) -> Observation:
        """Start a new episode: reset the K tsmodels, score them once, return the observation.

        Args:
            seed: Root seed of the episode. Every random stream derives from it.
        """
        for i, model in enumerate(self._tsmodels):
            model.reset(seed=derive_seed(seed, "tsmodel", i))
        self._augmentation_generator = make_generator(
            self._device, derive_seed(seed, "augmentation")
        )
        self._subset_generator = make_generator(self._device, derive_seed(seed, "reward_subset"))
        self._train_generators = tuple(
            make_generator(self._device, derive_seed(seed, "train", i))
            for i in range(self.batch_size)
        )
        self._epoch = 0
        _, windows = self._draw_reward_subset()
        scores, aggregate = self._score_all(windows)
        self._reward.reset(aggregate)
        self._featurizer.reset()
        record = EpochRecord(
            epoch=0,
            epochs=self.episode_length,
            train={name: self._zeros() for name in TRAIN_KEYS},
            scores=scores,
            aggregate=aggregate,
            action=self._zeros(self.action_space.dim),
        )
        return self._featurizer(record)

    @checked
    def step(self, action: ActionBatch) -> StepResult:
        """Generate, train one epoch, score, and return the result.

        Args:
            action: Shape ``(k, a)``, on :attr:`device`. Values outside the action space are
                clipped.

        Raises:
            RuntimeError: Before the first :meth:`reset`, or after the last step of an
                episode.
            ValueError: If ``action`` has the wrong shape or lives on another device.
        """
        if self._epoch is None:
            msg = "call reset() before step()"
            raise RuntimeError(msg)
        if self._epoch >= self.episode_length:
            msg = f"the episode ended after {self.episode_length} steps; call reset()"
            raise RuntimeError(msg)
        expected = (self.batch_size, self.action_space.dim)
        if tuple(action.shape) != expected:
            msg = f"action has shape {tuple(action.shape)}, expected {expected}"
            raise ValueError(msg)
        if _canonical(action.device) != self._canonical_device:
            msg = f"action lives on {action.device}, expected {self._device}"
            raise ValueError(msg)
        action = self.action_space.clip(action.detach().float())
        n_series = self.config.episode.series_per_epoch
        series = self._augmentation.sample(action, n_series, generator=self._augmentation_generator)
        stats = [
            model.train_epoch(series[i], generator=self._train_generators[i]).as_dict()
            for i, model in enumerate(self._tsmodels)
        ]
        train = {
            name: torch.stack([values[name] for values in stats]).detach().float()
            for name in stats[0]
        }
        train[SERIES_KEY] = torch.full((self.batch_size,), float(n_series), device=self._device)
        rows, windows = self._draw_reward_subset()
        scores, aggregate = self._score_all(windows)
        self._epoch += 1
        record = EpochRecord(
            epoch=self._epoch,
            epochs=self.episode_length,
            train=train,
            scores=scores,
            aggregate=aggregate,
            action=action,
        )
        reward = self._reward(record)
        observation = self._featurizer(record)
        done = torch.full(
            (self.batch_size,),
            self._epoch >= self.episode_length,
            dtype=torch.bool,
            device=self._device,
        )
        info = {
            "scores": scores,
            "aggregate": aggregate,
            **{f"train/{name}": value for name, value in train.items()},
            "action": action,
            "reward_rows": rows.unsqueeze(0).expand(self.batch_size, -1),
        }
        return StepResult(observation=observation, reward=reward, done=done, info=info)

    @checked
    def evaluate(self) -> PerDatasetK:
        """Return the skill per dataset on the full comparison set, shape ``(k, d)``.

        Called once at the end of an episode. Never used for the reward.
        """
        scores, _ = self._score_all(self._benchmark.comparison_windows)
        return scores
