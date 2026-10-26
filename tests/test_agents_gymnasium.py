import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from autotsaugment.components.agents.gymnasium_adapter import GymEnv
from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig
from autotsaugment.core.config import EpisodeConfig


def make_env(device: torch.device, *, batch_envs: int = 1, epochs: int = 8) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


def test_check_env_passes(device: torch.device):
    check_env(GymEnv(make_env(device)), skip_render_check=True)


def test_episode_terminates_at_step_e(device: torch.device):
    env = GymEnv(make_env(device), seed=0)
    observation, info = env.reset(seed=4)
    assert info == {}
    assert set(observation) == {"progress", "target"}
    assert observation["target"].dtype == np.float32
    flags = []
    for _ in range(8):
        observation, reward, terminated, truncated, _ = env.step(observation["target"])
        assert reward == pytest.approx(0.0)
        assert not truncated
        flags.append(terminated)
    assert flags == [False] * 7 + [True]


def test_seeded_reset_is_deterministic(device: torch.device):
    env = GymEnv(make_env(device))
    first, _ = env.reset(seed=1)
    second, _ = env.reset()
    again, _ = env.reset(seed=1)
    np.testing.assert_array_equal(first["target"], again["target"])
    assert not np.array_equal(first["target"], second["target"])


def test_needs_one_environment(device: torch.device):
    with pytest.raises(ValueError, match="batch_size 1"):
        GymEnv(make_env(device, batch_envs=2))
