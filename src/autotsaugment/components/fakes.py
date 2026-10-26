"""Fast fake components that honour every contract. For tests, smoke runs and agent work.

The fakes are real, small computations, not stubs: the fake tsmodel trains two parameters
with autograd, and the fake benchmark has a trend the fake augmentation can match, so the
reward responds to the action.
"""

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pydantic import Field
from torch import Tensor

from autotsaugment.core.agent import Policy
from autotsaugment.core.augmentation import AugmentationConfig, AugmentationMethod
from autotsaugment.core.benchmark import Benchmark, BenchmarkConfig, WindowSet
from autotsaugment.core.environment import Environment, EnvironmentConfig, StepResult
from autotsaugment.core.featurizer import Observation, ObservationSpec
from autotsaugment.core.metrics import pinball_loss, pinball_sum, seasonal_naive, target_abs_sum
from autotsaugment.core.seeding import make_generator
from autotsaugment.core.spaces import BoxSpace
from autotsaugment.core.tracker import Metrics, Tracker, to_host
from autotsaugment.core.tsmodel import TrainStats, TSModel, TSModelConfig
from autotsaugment.core.typechecking import (
    Action,
    ActionBatch,
    Context,
    QuantileLevels,
    Quantiles,
    SeriesBatch,
    SeriesBatchK,
    checked,
)

# --- augmentation --------------------------------------------------------------------------


class FakeAugmentationConfig(AugmentationConfig):
    """Config of :class:`FakeAugmentation`."""

    series_length: int = 64
    noise_scale: float = 0.1


class FakeAugmentation(AugmentationMethod[FakeAugmentationConfig]):
    """Random walk with drift. The action sets level, drift, noise and a nuisance sine.

    ``drift per step = DRIFT_SCALE * action[1]``, so one action dimension controls the one
    statistic the fake tsmodel learns. A fake benchmark with drift ``0.2`` is matched by
    ``action[1] = 0.2 / DRIFT_SCALE``.
    """

    NAMES = ("level", "trend", "log_noise", "sine_amp")
    DRIFT_SCALE = 0.3

    def __init__(self, *, config: FakeAugmentationConfig, device: torch.device | str) -> None:
        """Create the method on ``device``."""
        self.config = config
        self._device = torch.device(device)
        dim = len(self.NAMES)
        self._space = BoxSpace(
            low=torch.full((dim,), -1.0, device=self._device),
            high=torch.full((dim,), 1.0, device=self._device),
            names=self.NAMES,
        )
        self._steps = torch.arange(config.series_length, device=self._device, dtype=torch.float32)

    @property
    def action_space(self) -> BoxSpace:
        """Return the 4-D box."""
        return self._space

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return self._device

    def uniform_action(self) -> Action:
        """Return zeros: no level, no drift, unit noise, no sine."""
        return torch.zeros(self._space.dim, device=self._device)

    @checked
    def sample(self, action: ActionBatch, n: int, *, generator: torch.Generator) -> SeriesBatchK:
        """Return ``(k, n, t)`` series."""
        k = action.shape[0]
        level = action[:, 0].view(k, 1, 1)
        drift = (self.DRIFT_SCALE * action[:, 1]).view(k, 1, 1)
        noise = torch.exp(action[:, 2]).view(k, 1, 1) * self.config.noise_scale
        amplitude = (0.5 * torch.sigmoid(4.0 * action[:, 3]) - 0.25).clamp(min=0.0).view(k, 1, 1)
        steps = self._steps.view(1, 1, -1)
        length = steps.shape[-1]
        white = torch.randn((k, n, length), generator=generator, device=self._device)
        walk = torch.cumsum(white, dim=-1)
        sine = amplitude * torch.sin(2 * math.pi * steps / 16.0)
        return level + drift * steps + noise * walk + sine

    def describe(self, action: Action) -> dict[str, float]:
        """Return the action values by name."""
        return dict(zip(self.NAMES, action.detach().cpu().tolist(), strict=True))


# --- tsmodel -------------------------------------------------------------------------------


class FakeTSModelConfig(TSModelConfig):
    """Config of :class:`FakeTSModel`."""

    context_length: int = 32
    horizon_length: int = 8
    batch_size: int = 16
    learning_rate: float = 0.1
    n_levels: int = Field(default=9, ge=3)


class FakeTSModel(TSModel[FakeTSModelConfig]):
    """Forecasts ``last value + drift * step`` with a learned drift and a learned spread.

    Two parameters, trained with autograd on the pinball loss. A model trained on synthetic
    series with the benchmark's trend scores better on the benchmark.
    """

    def __init__(
        self, *, config: FakeTSModelConfig, device: torch.device | str, episode_epochs: int
    ) -> None:
        """Create the model on ``device``."""
        self.config = config
        self._device = torch.device(device)
        self._episode_epochs = episode_epochs
        levels = torch.linspace(0.05, 0.95, config.n_levels, device=self._device)
        self._levels = levels
        self._z = torch.erfinv(2 * levels - 1) * math.sqrt(2.0)
        self._drift = torch.zeros((), device=self._device, requires_grad=True)
        self._log_spread = torch.zeros((), device=self._device, requires_grad=True)
        self._optimizer = torch.optim.SGD([self._drift, self._log_spread], lr=config.learning_rate)

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return self._device

    @property
    def quantile_levels(self) -> QuantileLevels:
        """Return the levels."""
        return self._levels

    def reset(self, *, seed: int) -> None:
        """Zero both parameters and the optimizer state."""
        del seed
        with torch.no_grad():
            self._drift.zero_()
            self._log_spread.zero_()
        self._optimizer = torch.optim.SGD(
            [self._drift, self._log_spread], lr=self.config.learning_rate
        )

    def _predict(self, context: Tensor, horizon: int) -> Tensor:
        last = torch.nan_to_num(context[:, -1:], nan=0.0)
        steps = torch.arange(1, horizon + 1, device=self._device, dtype=context.dtype)
        median = last + self._drift * steps
        spread = torch.exp(self._log_spread)
        return median.unsqueeze(-1) + spread * self._z

    @checked
    def train_epoch(self, series: SeriesBatch, *, generator: torch.Generator) -> TrainStats:
        """One pass of SGD over random windows of ``series``."""
        n, length = series.shape
        c, h = self.config.context_length, self.config.horizon_length
        if length < c + h:
            msg = f"series length {length} is shorter than context + horizon = {c + h}"
            raise ValueError(msg)
        order = torch.randperm(n, generator=generator, device=self._device)
        starts = torch.randint(
            0, length - c - h + 1, (n,), generator=generator, device=self._device
        )
        offsets = torch.arange(c + h, device=self._device)
        windows = series[order][:, None, :].expand(n, 1, length)
        windows = torch.gather(windows[:, 0, :], 1, starts[:, None] + offsets[None, :])
        losses: list[Tensor] = []
        grads: list[Tensor] = []
        for start in range(0, n, self.config.batch_size):
            batch = windows[start : start + self.config.batch_size]
            forecast = self._predict(batch[:, :c], h)
            loss = pinball_loss(forecast, batch[:, c:], self._levels).mean()
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_([self._drift, self._log_spread], float("inf"))
            self._optimizer.step()
            losses.append(loss.detach())
            grads.append(grad.detach())
        stacked = torch.stack(losses)
        return TrainStats(
            loss=stacked.mean(),
            loss_last=stacked[-1],
            grad_norm=torch.stack(grads).mean(),
            # torch.full fills on the device; torch.tensor would copy from the host and sync.
            lr=torch.full((), self.config.learning_rate, device=self._device),
            n_steps=torch.full((), float(len(losses)), device=self._device),
        )

    @checked
    def forecast(self, context: Context, horizon: int) -> Quantiles:
        """Return ``(n, horizon, q)`` quantiles without gradient."""
        with torch.no_grad():
            return self._predict(context, horizon)

    def state_dict(self) -> dict[str, Any]:
        """Return both parameters."""
        return {
            "drift": self._drift.detach().clone(),
            "log_spread": self._log_spread.detach().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore both parameters."""
        with torch.no_grad():
            self._drift.copy_(state["drift"])
            self._log_spread.copy_(state["log_spread"])

    def n_trainable_parameters(self) -> int:
        """Return 2."""
        return 2


# --- benchmark -----------------------------------------------------------------------------


class FakeBenchmarkConfig(BenchmarkConfig):
    """Config of :class:`FakeBenchmark`.

    Attributes:
        n_datasets: Datasets, each a random walk with its own drift ``0.1 * (index + 1)``.
        context_length: Context steps c of every window.
        horizon: Horizon steps h of every window.
        noise_scale: Standard deviation of one random-walk step.
        seed: Seed of the generated data.
    """

    n_datasets: int = 3
    context_length: int = 32
    horizon: int = 8
    noise_scale: float = 0.1
    reward_windows_per_dataset: int = 32
    comparison_windows_per_dataset: int = 32
    seed: int = 1234


class FakeBenchmark(Benchmark[FakeBenchmarkConfig]):
    """Random walks with drift, fixed seed. The naive forecaster repeats the last value.

    A forecaster that learns the mean drift beats the naive forecaster on every dataset, so
    the skill of a trained fake tsmodel is below 1.0.
    """

    DOMAINS = ("Energy", "Nature", "Sales", "Healthcare", "Transport", "Web/CloudOps", "Econ/Fin")
    DRIFT_PER_DATASET = 0.1
    LEVELS = 9

    def __init__(self, *, config: FakeBenchmarkConfig, device: torch.device | str) -> None:
        """Generate both window sets on ``device``."""
        self.config = config
        self._device = torch.device(device)
        generator = make_generator(self._device, config.seed)
        self._reward = self._make(config.reward_windows_per_dataset, generator)
        self._comparison = self._make(config.comparison_windows_per_dataset, generator)

    @property
    def mean_drift(self) -> float:
        """Return the drift averaged over datasets, the optimum of the fake tsmodel."""
        d = self.config.n_datasets
        return self.DRIFT_PER_DATASET * (d + 1) / 2.0

    def _process(self, per_dataset: int, length: int, generator: torch.Generator) -> Tensor:
        """Return ``(d * per_dataset, length)`` random walks with the per-dataset drift."""
        d = self.config.n_datasets
        steps = torch.arange(length, device=self._device, dtype=torch.float32).view(1, 1, -1)
        dataset = torch.arange(d, device=self._device, dtype=torch.float32).view(d, 1, 1)
        drift = self.DRIFT_PER_DATASET * (dataset + 1)
        level = torch.randn((d, per_dataset, 1), generator=generator, device=self._device)
        white = torch.randn((d, per_dataset, length), generator=generator, device=self._device)
        walk = self.config.noise_scale * torch.cumsum(white, dim=-1)
        values = level + drift * steps + walk
        return values.reshape(d * per_dataset, length)

    def sample_series(self, per_dataset: int, length: int, *, generator: torch.Generator) -> Tensor:
        """Return series from the benchmark's own process, for tests that need a matching source."""
        return self._process(per_dataset, length, generator)

    def _make(self, per_dataset: int, generator: torch.Generator) -> WindowSet:
        cfg = self.config
        c, h, d = cfg.context_length, cfg.horizon, cfg.n_datasets
        values = self._process(per_dataset, c + h, generator)
        context, target = values[:, :c], values[:, c:]
        levels = torch.linspace(0.05, 0.95, self.LEVELS, device=self._device)
        naive = seasonal_naive(context, h, 1).unsqueeze(-1).expand(-1, -1, levels.numel())
        names = tuple(f"fake_{i}" for i in range(d))
        domains = tuple(self.DOMAINS[i % len(self.DOMAINS)] for i in range(d))
        return WindowSet(
            context=context,
            target=target,
            dataset_index=torch.arange(d, device=self._device).repeat_interleave(per_dataset),
            naive_loss_sum=pinball_sum(naive, target, levels),
            abs_sum=target_abs_sum(target),
            dataset_names=names,
            domains=domains,
            quantile_levels=self.quantile_levels,
        )

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Return the levels of the naive loss, the levels of the default fake tsmodel."""
        step = 0.9 / (self.LEVELS - 1)
        return tuple(round(0.05 + i * step, 6) for i in range(self.LEVELS))

    @property
    def reward_windows(self) -> WindowSet:
        """Return the reward set."""
        return self._reward

    @property
    def comparison_windows(self) -> WindowSet:
        """Return the comparison set."""
        return self._comparison


# --- environment ---------------------------------------------------------------------------


class FakeEnvironmentConfig(EnvironmentConfig):
    """Config of :class:`FakeEnvironment`.

    Attributes:
        action_dim: Dimension of the action.
        n_datasets: Width of the score vectors.
        shared_target: Draw one hidden target per episode for all K environments. A static
            action can then be optimal in every slot, as in the real environment, where the K
            slots are exchangeable.
    """

    action_dim: int = 3
    n_datasets: int = 2
    shared_target: bool = False


class FakeEnvironment(Environment[FakeEnvironmentConfig]):
    """A batched quadratic bandit with a hidden target per environment, for agent tests.

    The observation holds the progress and the hidden target, so a policy can reach the
    optimum. The reward is ``-||action - target||^2``. The skill on evaluate is the mean
    reward of the episode mapped to ``[1, 2)``; lower is better.

    ``StepResult.info`` holds, each with a leading k, the layout of the curriculum
    environment for the loop and the trackers:

    - ``aggregate`` ``(k,)``: ``2 * sigmoid(-4 * reward)``, the map of :meth:`evaluate`
      applied to the reward of the step;
    - ``scores`` ``(k, d)``: the aggregate repeated over the ``n_datasets`` columns, so
      their geometric mean is the aggregate;
    - ``distance`` ``(k,)``: ``||action - target||``.
    """

    def __init__(self, *, config: FakeEnvironmentConfig, device: torch.device | str) -> None:
        """Create the environment on ``device``."""
        self.config = config
        self._device = torch.device(device)
        a = config.action_dim
        self._space = BoxSpace(
            low=torch.full((a,), -1.0, device=self._device),
            high=torch.full((a,), 1.0, device=self._device),
            names=tuple(f"a{i}" for i in range(a)),
        )
        self._target = torch.zeros((self.batch_size, a), device=self._device)
        self._epoch = 0
        self._return = torch.zeros(self.batch_size, device=self._device)

    @property
    def device(self) -> torch.device:
        """Return the device."""
        return self._device

    @property
    def action_space(self) -> BoxSpace:
        """Return the box."""
        return self._space

    def uniform_action(self) -> Tensor:
        """Return zeros, shape ``(action_dim,)``."""
        return torch.zeros(self.config.action_dim, device=self._device)

    @property
    def observation_spec(self) -> ObservationSpec:
        """Return the two entries."""
        return {"progress": (2,), "target": (self.config.action_dim,)}

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return fake names."""
        return tuple(f"fake_{i}" for i in range(self.config.n_datasets))

    def _observation(self) -> Observation:
        progress = torch.tensor(
            [self._epoch / self.episode_length, 0.0], device=self._device
        ).expand(self.batch_size, 2)
        return {"progress": progress.clone(), "target": self._target.clone()}

    def reset(self, *, seed: int) -> Observation:
        """Draw a new hidden target per environment, or one for all with ``shared_target``."""
        generator = make_generator(self._device, seed)
        draws = 1 if self.config.shared_target else self.batch_size
        target = self._space.sample(draws, generator=generator) * 0.5
        self._target = target.expand(self.batch_size, -1).clone()
        self._epoch = 0
        self._return = torch.zeros(self.batch_size, device=self._device)
        return self._observation()

    @staticmethod
    def _skill(reward: Tensor) -> Tensor:
        """Map a reward (or a mean reward), at most 0, to a skill in ``[1, 2)``."""
        return 2.0 * torch.sigmoid(-reward * 4.0)

    def _per_dataset(self, skill: Tensor) -> Tensor:
        return skill.unsqueeze(-1).expand(-1, self.config.n_datasets).clone()

    def step(self, action: Tensor) -> StepResult:
        """Pay the negative squared distance to the target."""
        action = self._space.clip(action)
        reward = -((action - self._target) ** 2).sum(dim=-1)
        self._epoch += 1
        self._return = self._return + reward
        done = torch.full(
            (self.batch_size,), self._epoch >= self.episode_length, device=self._device
        )
        aggregate = self._skill(reward)
        info = {
            "aggregate": aggregate,
            "scores": self._per_dataset(aggregate),
            "distance": (-reward).sqrt(),
        }
        return StepResult(self._observation(), reward, done, info)

    def evaluate(self) -> Tensor:
        """Return a skill in ``[1, 2)`` per dataset, lower when the return is higher."""
        mean_return = self._return / max(self._epoch, 1)
        return self._per_dataset(self._skill(mean_return))


class ConstantPolicy:
    """Returns the same action for every environment."""

    def __init__(self, action: Tensor, batch_size: int) -> None:
        """Store ``action`` of shape ``(a,)``."""
        self._action = action
        self._batch_size = batch_size

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> Tensor:
        """Return the action repeated k times."""
        del observation, deterministic
        return self._action.unsqueeze(0).expand(self._batch_size, -1).clone()


def as_policy(policy: Policy) -> Policy:
    """Return ``policy`` unchanged; a typing helper for tests."""
    return policy


# --- tracker -------------------------------------------------------------------------------


class MemoryTracker(Tracker):
    """Keeps every call in memory, converted to floats on flush. For tests."""

    def __init__(self) -> None:
        """Create empty buffers."""
        self.params: dict[str, object] = {}
        self.steps: list[tuple[int, dict[str, float]]] = []
        self.episodes: list[tuple[int, dict[str, float]]] = []
        self.artifacts: list[Path] = []
        self._pending_steps: list[tuple[int, Metrics]] = []
        self._pending_episodes: list[tuple[int, Metrics]] = []
        self.flushes = 0

    def log_params(self, params: Mapping[str, object]) -> None:
        """Store the parameters."""
        self.params.update(params)

    def log_step(self, step: int, metrics: Metrics) -> None:
        """Buffer the step without touching the host."""
        self._pending_steps.append((step, dict(metrics)))

    def log_episode(self, episode: int, metrics: Metrics) -> None:
        """Buffer the episode, then flush."""
        self._pending_episodes.append((episode, dict(metrics)))
        self.flush()

    def log_artifact(self, path: Path) -> None:
        """Store the path."""
        self.artifacts.append(path)

    def flush(self) -> None:
        """Convert every buffered metric to floats."""
        self.steps.extend((step, to_host(metrics)) for step, metrics in self._pending_steps)
        self.episodes.extend(
            (episode, to_host(metrics)) for episode, metrics in self._pending_episodes
        )
        self._pending_steps.clear()
        self._pending_episodes.clear()
        self.flushes += 1

    def close(self) -> None:
        """Flush."""
        self.flush()
