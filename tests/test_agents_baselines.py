from pathlib import Path

import pytest
import torch

from autotsaugment.components.agents.fixed import FixedActionAgent, FixedActionAgentConfig
from autotsaugment.components.agents.policies import uniform_action
from autotsaugment.components.agents.random import RandomAgent, RandomAgentConfig
from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig, MemoryTracker
from autotsaugment.core.config import EpisodeConfig


def make_env(device: torch.device, *, batch_envs: int = 4, epochs: int = 8) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


class UniformFakeEnvironment(FakeEnvironment):
    """A fake environment that exposes the uniform action of its augmentation method."""

    def uniform_action(self) -> torch.Tensor:
        return torch.full((self.config.action_dim,), 0.5, device=self.device)


def test_fixed_agent_without_action_returns_zeros(device: torch.device):
    env = make_env(device)
    agent = FixedActionAgent(config=FixedActionAgentConfig(), device=device)
    with pytest.raises(RuntimeError, match="call fit"):
        agent.policy()
    policy = agent.fit(env, tracker=MemoryTracker(), seed=0)
    observation = env.reset(seed=1)
    action = policy(observation)
    assert action.shape == (4, 3)
    assert env.action_space.contains(action)
    torch.testing.assert_close(action, torch.zeros(4, 3))


def test_fixed_agent_uses_the_uniform_action_when_the_env_has_one(device: torch.device):
    env = UniformFakeEnvironment(config=make_env(device).config, device=device)
    torch.testing.assert_close(uniform_action(env), torch.full((3,), 0.5))
    agent = FixedActionAgent(config=FixedActionAgentConfig(), device=device)
    action = agent.fit(env, tracker=MemoryTracker(), seed=0)(env.reset(seed=0))
    torch.testing.assert_close(action, torch.full((4, 3), 0.5))


def test_fixed_agent_with_action_clips_and_round_trips(device: torch.device, tmp_path: Path):
    env = make_env(device)
    config = FixedActionAgentConfig(action=[0.5, -2.0, 0.25])
    agent = FixedActionAgent(config=config, device=device)
    assert agent.policy()(env.reset(seed=0)).shape == (4, 3)
    policy = agent.fit(env, tracker=MemoryTracker(), seed=0)
    action = policy(env.reset(seed=0))
    torch.testing.assert_close(action, torch.tensor([[0.5, -1.0, 0.25]]).expand(4, 3))
    agent.save(tmp_path / "fixed")
    other = FixedActionAgent(config=FixedActionAgentConfig(), device=device)
    other.load(tmp_path / "fixed")
    torch.testing.assert_close(other.policy()(env.reset(seed=0)), action)


def test_fixed_agent_rejects_a_wrong_dimension(device: torch.device):
    agent = FixedActionAgent(config=FixedActionAgentConfig(action=[0.0, 0.0]), device=device)
    with pytest.raises(ValueError, match="2 values"):
        agent.fit(make_env(device), tracker=MemoryTracker(), seed=0)


def test_random_agent_actions_are_in_the_box_and_seeded(device: torch.device):
    env = make_env(device)
    observation = env.reset(seed=0)
    agent = RandomAgent(config=RandomAgentConfig(), device=device)
    with pytest.raises(RuntimeError, match="call fit"):
        agent.policy()
    first = agent.fit(env, tracker=MemoryTracker(), seed=3)
    actions = torch.stack([first(observation) for _ in range(50)])
    assert actions.shape == (50, 4, 3)
    assert env.action_space.contains(actions.reshape(-1, 3))
    assert actions.std() > 0.3
    second = RandomAgent(config=RandomAgentConfig(), device=device)
    again = second.fit(env, tracker=MemoryTracker(), seed=3)
    torch.testing.assert_close(torch.stack([again(observation) for _ in range(50)]), actions)


def test_random_agent_round_trip_continues_the_stream(device: torch.device, tmp_path: Path):
    env = make_env(device)
    observation = env.reset(seed=0)
    agent = RandomAgent(config=RandomAgentConfig(), device=device)
    policy = agent.fit(env, tracker=MemoryTracker(), seed=0)
    policy(observation)
    agent.save(tmp_path / "random")
    expected = policy(observation)
    other = RandomAgent(config=RandomAgentConfig(), device=device)
    other.load(tmp_path / "random")
    torch.testing.assert_close(other.policy()(observation), expected)


def test_baselines_need_no_episode(device: torch.device):
    tracker = MemoryTracker()
    env = make_env(device)
    FixedActionAgent(config=FixedActionAgentConfig(), device=device).fit(
        env, tracker=tracker, seed=0
    )
    RandomAgent(config=RandomAgentConfig(), device=device).fit(env, tracker=tracker, seed=0)
    assert tracker.steps == []
    assert tracker.episodes == []
