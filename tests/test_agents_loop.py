import pytest
import torch

from autotsaugment.components.agents.loop import (
    episode_seed,
    geometric_mean,
    run_episode,
)
from autotsaugment.components.agents.policies import (
    ConstantActionPolicy,
    SlotActionPolicy,
    observation_batch_size,
)
from autotsaugment.components.fakes import (
    FakeEnvironment,
    FakeEnvironmentConfig,
    MemoryTracker,
)
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.environment import StepResult
from autotsaugment.core.featurizer import Observation
from autotsaugment.core.seeding import derive_seed


def make_env(device: torch.device, *, batch_envs: int = 4, epochs: int = 8) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


class TargetPolicy:
    """Outputs the hidden target that the fake observation holds: the optimal policy."""

    def __call__(self, observation: Observation, *, deterministic: bool = True) -> torch.Tensor:
        return observation["target"].clone()


def test_run_episode_shapes_and_logs(device: torch.device):
    env = make_env(device)
    tracker = MemoryTracker()
    action = torch.full((3,), 0.25, device=device)
    result = run_episode(
        env, ConstantActionPolicy(action), seed=7, tracker=tracker, step_offset=16, episode=2
    )
    assert result.returns.shape == (4,)
    assert result.final_skill.shape == (4, 2)
    assert result.aggregate_curve.shape == (8, 4)
    assert result.steps == 8
    assert tracker.flushes == 1
    assert [step for step, _ in tracker.steps] == list(range(16, 24))
    assert len(tracker.episodes) == 1
    episode, metrics = tracker.episodes[0]
    assert episode == 2
    assert set(metrics) == {
        "return",
        "comparison/aggregate",
        "comparison/fake_0",
        "comparison/fake_1",
    }
    torch.testing.assert_close(torch.tensor(metrics["return"]), result.returns.mean())
    step_metrics = tracker.steps[0][1]
    assert set(step_metrics) == {"reward", "distance"}


def test_run_episode_without_aggregate_gives_a_nan_curve(device: torch.device):
    env = make_env(device)
    result = run_episode(env, TargetPolicy(), seed=0, tracker=MemoryTracker())
    assert torch.isnan(result.aggregate_curve).all()
    torch.testing.assert_close(result.returns, torch.zeros(4))
    torch.testing.assert_close(result.final_skill, torch.ones(4, 2))


def test_run_episode_reads_the_aggregate_from_info(device: torch.device):
    class AggregateEnvironment(FakeEnvironment):
        def step(self, action: torch.Tensor) -> StepResult:
            result = super().step(action)
            info = {**result.info, "aggregate": result.info["distance"] + 1.0}
            return StepResult(result.observation, result.reward, result.done, info)

    env = AggregateEnvironment(config=make_env(device).config, device=device)
    tracker = MemoryTracker()
    result = run_episode(env, TargetPolicy(), seed=0, tracker=tracker)
    torch.testing.assert_close(result.aggregate_curve, torch.ones(8, 4))
    assert tracker.steps[0][1]["aggregate"] == pytest.approx(1.0)


def test_run_episode_callback_sees_every_step_and_adds_metrics(device: torch.device):
    env = make_env(device)
    tracker = MemoryTracker()
    seen: list[tuple[int, int]] = []

    def on_step(
        observation: Observation, action: torch.Tensor, result: StepResult
    ) -> dict[str, torch.Tensor]:
        seen.append((observation_batch_size(observation), action.shape[0]))
        return {"extra": result.reward.sum()}

    run_episode(env, TargetPolicy(), seed=0, tracker=tracker, on_step=on_step)
    assert seen == [(4, 4)] * 8
    assert all("extra" in metrics for _, metrics in tracker.steps)


def test_episode_seed_is_derived():
    assert episode_seed(3, 5) == derive_seed(3, "episode", 5)
    assert episode_seed(3, 5) != episode_seed(3, 6)


def test_geometric_mean_matches_core_aggregate():
    skill = torch.tensor([[0.5, 2.0], [1.0, 1.0]])
    torch.testing.assert_close(geometric_mean(skill), torch.tensor([1.0, 1.0]))


def test_slot_policy_checks_the_batch(device: torch.device):
    env = make_env(device)
    observation = env.reset(seed=0)
    policy = SlotActionPolicy(torch.zeros(4, 3, device=device))
    assert policy(observation).shape == (4, 3)
    with pytest.raises(ValueError, match="holds 2 actions"):
        SlotActionPolicy(torch.zeros(2, 3, device=device))(observation)
    with pytest.raises(ValueError, match=r"shape \(k, a\)"):
        SlotActionPolicy(torch.zeros(3, device=device))
    with pytest.raises(ValueError, match=r"shape \(a,\)"):
        ConstantActionPolicy(torch.zeros(4, 3, device=device))
