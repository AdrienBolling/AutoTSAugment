import pytest
import torch

from autotsaugment.components.env.scoring import Forecaster, score
from autotsaugment.components.fakes import (
    FakeBenchmark,
    FakeBenchmarkConfig,
    FakeTSModel,
    FakeTSModelConfig,
)
from autotsaugment.core.benchmark import WindowSet
from autotsaugment.core.metrics import seasonal_naive


class NaiveForecaster:
    """Repeats the last context value at every level: the forecaster of the naive loss."""

    def __init__(self, device: torch.device, n_levels: int = FakeBenchmark.LEVELS) -> None:
        self.device = device
        self.quantile_levels = torch.linspace(0.05, 0.95, n_levels, device=device)
        self.calls = 0

    def forecast(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        self.calls += 1
        point = seasonal_naive(context, horizon, 1)
        return point.unsqueeze(-1).expand(-1, -1, self.quantile_levels.numel())


class PerfectForecaster(NaiveForecaster):
    """Returns the true target of the windows in scoring order, at every level."""

    def __init__(self, device: torch.device, windows: WindowSet) -> None:
        super().__init__(device)
        self._target = windows.target
        self._cursor = 0

    def forecast(self, context: torch.Tensor, horizon: int) -> torch.Tensor:
        n = context.shape[0]
        target = self._target[self._cursor : self._cursor + n, :horizon]
        self._cursor += n
        return target.unsqueeze(-1).expand(-1, -1, self.quantile_levels.numel())


@pytest.fixture
def windows(device: torch.device) -> WindowSet:
    return FakeBenchmark(config=FakeBenchmarkConfig(), device=device).reward_windows


def test_naive_forecaster_has_skill_exactly_one(device: torch.device, windows: WindowSet):
    model = NaiveForecaster(device)
    skill, agg = score(model, windows, batch_size=7)
    assert model.calls == -(-windows.n_windows // 7)
    assert skill.shape == (windows.n_datasets,)
    assert torch.equal(skill, torch.ones(windows.n_datasets, device=device))
    assert agg.shape == ()
    assert agg.item() == 1.0


def test_perfect_forecaster_has_skill_zero(device: torch.device, windows: WindowSet):
    skill, agg = score(PerfectForecaster(device, windows), windows, batch_size=10)
    assert torch.equal(skill, torch.zeros(windows.n_datasets, device=device))
    assert agg.item() == 0.0


def test_score_does_not_depend_on_the_batch_size(device: torch.device, windows: WindowSet):
    model = FakeTSModel(config=FakeTSModelConfig(), device=device, episode_epochs=1)
    model.reset(seed=0)
    small, small_agg = score(model, windows, batch_size=5)
    large, large_agg = score(model, windows, batch_size=1000)
    torch.testing.assert_close(small, large)
    torch.testing.assert_close(small_agg, large_agg)
    assert torch.isfinite(small).all()


def test_tsmodel_is_a_forecaster(device: torch.device):
    model = FakeTSModel(config=FakeTSModelConfig(), device=device, episode_epochs=1)
    assert isinstance(model, Forecaster)


def test_score_rejects_bad_input(device: torch.device, windows: WindowSet):
    model = NaiveForecaster(device)
    with pytest.raises(ValueError, match="batch_size"):
        score(model, windows, batch_size=0)
    empty = windows.select(torch.zeros(0, dtype=torch.long, device=device))
    with pytest.raises(ValueError, match="empty"):
        score(model, empty, batch_size=8)
