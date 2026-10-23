from pathlib import Path

import pytest
import torch

from autotsaugment.components.agents.cmaes import (
    CMAESAgent,
    CMAESAgentConfig,
    default_population,
)
from autotsaugment.components.agents.loop import episode_seed
from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig, MemoryTracker
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.featurizer import Observation


class SharedTargetEnvironment(FakeEnvironment):
    """The fake bandit with one hidden target for all K slots.

    ``FakeEnvironment`` draws one target per slot. A static action cannot match K different
    targets, so this subclass copies the target of slot 0 to every slot. The K slots of the
    real environment are exchangeable in the same way.
    """

    def __init__(self, *, config: FakeEnvironmentConfig, device: torch.device) -> None:
        super().__init__(config=config, device=device)
        self.seeds: list[int] = []

    def reset(self, *, seed: int) -> Observation:
        super().reset(seed=seed)
        self.seeds.append(seed)
        self._target = self._target[:1].expand_as(self._target).clone()
        return self._observation()


def make_env(device: torch.device) -> SharedTargetEnvironment:
    episode = EpisodeConfig(epochs=8, series_per_epoch=1, batch_envs=4)
    return SharedTargetEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


def test_default_population_is_a_multiple_of_k():
    assert default_population(3, 4) == 8  # 4 + floor(3 ln 3) = 7, rounded up
    assert default_population(38, 4) == 16  # 4 + floor(3 ln 38) = 14
    assert default_population(3, 1) == 7


def test_cmaes_finds_the_hidden_target(device: torch.device, tmp_path: Path):
    env = make_env(device)
    tracker = MemoryTracker()
    config = CMAESAgentConfig(episodes=12, sigma0=0.3, fixed_seed=True)
    agent = CMAESAgent(config=config, device=device)
    with pytest.raises(RuntimeError, match="call fit"):
        agent.policy()
    policy = agent.fit(env, tracker=tracker, seed=0)
    assert len(tracker.episodes) == 12
    assert tracker.flushes == 12
    assert set(env.seeds) == {episode_seed(0, 0)}
    target = env.reset(seed=episode_seed(0, 0))["target"][0]
    assert agent.action is not None
    assert (agent.action - target).abs().max() < 0.25
    first = policy(env.reset(seed=1))
    second = policy(env.reset(seed=2))
    assert first.shape == (4, 3)
    assert env.action_space.contains(first)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, agent.action.expand(4, 3))
    agent.save(tmp_path / "cmaes")
    other = CMAESAgent(config=config, device=device)
    other.load(tmp_path / "cmaes")
    torch.testing.assert_close(other.policy()(env.reset(seed=1)), first)


def test_cmaes_without_fixed_seed_advances_the_seed(device: torch.device):
    env = make_env(device)
    config = CMAESAgentConfig(episodes=3, fixed_seed=False)
    CMAESAgent(config=config, device=device).fit(env, tracker=MemoryTracker(), seed=5)
    assert env.seeds == [episode_seed(5, i) for i in range(3)]


def test_cmaes_partial_generation(device: torch.device):
    # One episode evaluates half of a population of 8: CMA-ES never updates its mean.
    env = make_env(device)
    mean_agent = CMAESAgent(
        config=CMAESAgentConfig(episodes=1, population=8, final="mean"), device=device
    )
    mean_agent.fit(env, tracker=MemoryTracker(), seed=0)
    assert mean_agent.action is not None
    torch.testing.assert_close(mean_agent.action, torch.zeros(3))
    # The best candidate still comes from the evaluated half.
    best_agent = CMAESAgent(config=CMAESAgentConfig(episodes=1, population=8), device=device)
    best_agent.fit(env, tracker=MemoryTracker(), seed=0)
    assert best_agent.action is not None
    assert not torch.equal(best_agent.action, torch.zeros(3))


def test_cmaes_with_k_not_dividing_the_population(device: torch.device):
    env = make_env(device)
    config = CMAESAgentConfig(episodes=4, population=6, sigma0=0.3)
    agent = CMAESAgent(config=config, device=device)
    policy = agent.fit(env, tracker=MemoryTracker(), seed=0)
    assert env.action_space.contains(policy(env.reset(seed=0)))
