import pytest
import torch

from autotsaugment.components.fakes import (
    FakeAugmentation,
    FakeAugmentationConfig,
    FakeBenchmark,
    FakeBenchmarkConfig,
    FakeEnvironment,
    FakeEnvironmentConfig,
    FakeTSModel,
    FakeTSModelConfig,
    MemoryTracker,
)
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.metrics import aggregate, pinball_sum, skill_scores, wql_per_dataset


def test_fake_augmentation_shapes_and_determinism(device: torch.device):
    method = FakeAugmentation(config=FakeAugmentationConfig(), device=device)
    action = method.uniform_action().unsqueeze(0).repeat(2, 1)
    g1 = torch.Generator(device=device).manual_seed(3)
    g2 = torch.Generator(device=device).manual_seed(3)
    s1 = method.sample(action, 7, generator=g1)
    s2 = method.sample(action, 7, generator=g2)
    assert s1.shape == (2, 7, 64)
    torch.testing.assert_close(s1, s2)
    assert torch.isfinite(s1).all()
    assert set(method.describe(method.uniform_action())) == set(method.NAMES)


def test_fake_tsmodel_learns_the_trend(device: torch.device, generator: torch.Generator):
    model = FakeTSModel(config=FakeTSModelConfig(), device=device, episode_epochs=5)
    model.reset(seed=0)
    t = torch.arange(64.0, device=device)
    series = (0.5 * t).unsqueeze(0).repeat(64, 1)  # drift 0.5 per step
    first = model.train_epoch(series, generator=generator)
    for _ in range(20):
        last = model.train_epoch(series, generator=generator)
    assert last.loss.item() < first.loss.item()
    assert abs(model.state_dict()["drift"].item() - 0.5) < 0.2
    forecast = model.forecast(series[:3, :32], horizon=8)
    assert forecast.shape == (3, 8, 9)
    assert torch.all(forecast[..., 1:] >= forecast[..., :-1])


def test_fake_benchmark_naive_skill_is_one(device: torch.device):
    bench = FakeBenchmark(config=FakeBenchmarkConfig(), device=device)
    ws = bench.reward_windows
    assert ws.n_datasets == 3
    assert ws.n_windows == 96
    naive_wql = wql_per_dataset(ws.naive_loss_sum, ws.abs_sum, ws.dataset_index, 3, 9)
    torch.testing.assert_close(skill_scores(naive_wql, naive_wql), torch.ones(3))
    assert bench.comparison_windows.n_windows == 96
    assert not torch.equal(bench.comparison_windows.context, ws.context)


def aggregate_skill(model: FakeTSModel, ws) -> torch.Tensor:  # noqa: ANN001  # reason: WindowSet import kept local to the fixture-free test
    d, q = ws.n_datasets, model.quantile_levels.numel()
    naive_wql = wql_per_dataset(ws.naive_loss_sum, ws.abs_sum, ws.dataset_index, d, q)
    forecast = model.forecast(ws.context, ws.horizon)
    loss = pinball_sum(forecast, ws.target, model.quantile_levels)
    return aggregate(
        skill_scores(wql_per_dataset(loss, ws.abs_sum, ws.dataset_index, d, q), naive_wql)
    )


def test_fake_tsmodel_training_improves_the_aggregate(
    device: torch.device, generator: torch.Generator
):
    bench = FakeBenchmark(config=FakeBenchmarkConfig(), device=device)
    model = FakeTSModel(config=FakeTSModelConfig(), device=device, episode_epochs=5)
    model.reset(seed=0)
    before = aggregate_skill(model, bench.reward_windows)
    series = bench.sample_series(32, 64, generator=generator)
    for _ in range(30):
        model.train_epoch(series, generator=generator)
    after = aggregate_skill(model, bench.reward_windows)
    assert torch.isfinite(after)
    assert after.item() < before.item()
    assert after.item() < 1.0  # beats the naive forecaster
    assert abs(model.state_dict()["drift"].item() - bench.mean_drift) < 0.05


def test_fake_reward_responds_to_the_action(device: torch.device):
    bench = FakeBenchmark(config=FakeBenchmarkConfig(), device=device)
    method = FakeAugmentation(config=FakeAugmentationConfig(), device=device)
    results = {}
    for name, trend in {"matched": bench.mean_drift / method.DRIFT_SCALE, "opposite": -0.5}.items():
        generator = torch.Generator(device=device).manual_seed(0)
        model = FakeTSModel(config=FakeTSModelConfig(), device=device, episode_epochs=5)
        model.reset(seed=0)
        action = torch.tensor([[0.0, trend, 0.0, 0.0]], device=device)
        for _ in range(20):
            series = method.sample(action, 64, generator=generator)[0]
            model.train_epoch(series, generator=generator)
        results[name] = aggregate_skill(model, bench.reward_windows).item()
    assert results["matched"] < 1.0 < results["opposite"]


def test_fake_environment_episode(device: torch.device):
    config = FakeEnvironmentConfig(
        episode=EpisodeConfig(epochs=4, series_per_epoch=1, batch_envs=3)
    )
    env = FakeEnvironment(config=config, device=device)
    obs = env.reset(seed=0)
    assert obs["target"].shape == (3, 3)
    assert obs["progress"].shape == (3, 2)
    for step in range(4):
        result = env.step(obs["target"])  # the optimal action
        obs = result.observation
        assert result.reward.shape == (3,)
        torch.testing.assert_close(result.reward, torch.zeros(3))
        assert bool(result.done.all()) == (step == 3)
    skill = env.evaluate()
    assert skill.shape == (3, 2)
    torch.testing.assert_close(skill, torch.ones(3, 2))


def test_fake_environment_has_an_empty_state(device: torch.device):
    env = FakeEnvironment(config=FakeEnvironmentConfig(), device=device)
    assert env.state_dict() == {}
    env.load_state_dict({})
    with pytest.raises(ValueError, match="has no state"):
        env.load_state_dict({"featurizer/x": torch.zeros(1)})


def test_fake_environment_uniform_action_is_zero(device: torch.device):
    env = FakeEnvironment(config=FakeEnvironmentConfig(action_dim=5), device=device)
    action = env.uniform_action()
    torch.testing.assert_close(action, torch.zeros(5))
    assert env.action_space.contains(action.unsqueeze(0))


def test_memory_tracker_flushes_once_per_episode(device: torch.device):
    tracker = MemoryTracker()
    for step in range(3):
        tracker.log_step(step, {"reward": torch.tensor(float(step), device=device), "x": 1})
    assert tracker.steps == []
    tracker.log_episode(0, {"skill": torch.tensor(0.5)})
    assert len(tracker.steps) == 3
    assert tracker.steps[2][1] == {"reward": 2.0, "x": 1.0}
    assert tracker.episodes == [(0, {"skill": 0.5})]
    assert tracker.flushes == 1
