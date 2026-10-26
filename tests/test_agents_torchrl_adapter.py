import torch
from torchrl.envs.utils import check_env_specs

from autotsaugment.components.agents.loop import episode_seed
from autotsaugment.components.agents.torchrl.adapter import TorchRLEnv
from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig
from autotsaugment.core.config import EpisodeConfig


def make_env(device: torch.device, *, batch_envs: int = 4, epochs: int = 8) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


def test_check_env_specs_passes(device: torch.device):
    env = TorchRLEnv(make_env(device), seed=0)
    check_env_specs(env)
    assert env.batch_size == torch.Size([4])
    assert env.action_spec.shape == torch.Size([4, 3])
    assert set(env.observation_spec.keys()) == {"progress", "target"}


def test_rollout_shapes_and_reset_after_e_steps(device: torch.device):
    env = TorchRLEnv(make_env(device), seed=0)
    env.set_seed(3)
    rollout = env.rollout(16, break_when_any_done=False)
    assert rollout.batch_size == torch.Size([4, 16])
    assert rollout["action"].shape == (4, 16, 3)
    assert rollout["target"].shape == (4, 16, 3)
    assert rollout["next", "reward"].shape == (4, 16, 1)
    done = rollout["next", "done"].squeeze(-1)
    expected = torch.zeros(16, dtype=torch.bool)
    expected[[7, 15]] = True
    assert torch.equal(done, expected.expand(4, 16))
    assert torch.equal(rollout["next", "terminated"], rollout["next", "done"])
    progress = rollout["progress"][..., 0]
    torch.testing.assert_close(progress[:, 0], torch.zeros(4))
    torch.testing.assert_close(progress[:, 8], torch.zeros(4))
    torch.testing.assert_close(progress[:, 7], torch.full((4,), 7 / 8))
    assert env.episode == 2
    # Episode 1 of the rollout used the second derived seed: a new target.
    fake = make_env(device)
    first = fake.reset(seed=episode_seed(3, 0))["target"]
    second = fake.reset(seed=episode_seed(3, 1))["target"]
    torch.testing.assert_close(rollout["target"][:, 0], first)
    torch.testing.assert_close(rollout["target"][:, 8], second)


def test_reward_is_the_negative_squared_distance(device: torch.device):
    env = TorchRLEnv(make_env(device), seed=0)
    data = env.reset()
    data["action"] = data["target"].clone()
    step = env.step(data)
    torch.testing.assert_close(step["next", "reward"], torch.zeros(4, 1))
