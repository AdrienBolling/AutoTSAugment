from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from autotsaugment.components.agents.loop import episode_seed, run_episode
from autotsaugment.components.agents.policies import ConstantActionPolicy
from autotsaugment.components.agents.random import RandomAgent, RandomAgentConfig
from autotsaugment.components.agents.torchrl.networks import (
    flatten_observation,
    observation_dim,
    squash,
)
from autotsaugment.components.agents.torchrl.ppo import PPOAgent, PPOAgentConfig
from autotsaugment.components.agents.torchrl.sac import SACAgent, SACAgentConfig
from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig, MemoryTracker
from autotsaugment.components.tracking.null import NullTracker
from autotsaugment.core.agent import Policy
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.environment import StepResult
from autotsaugment.core.featurizer import Observation

EVAL_SEEDS = [episode_seed(10_000, i) for i in range(4)]


def make_env(device: torch.device, *, batch_envs: int = 4, epochs: int = 8) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


def mean_distance(env: FakeEnvironment, policy: Policy, seeds: list[int]) -> float:
    """Mean distance between the action and the hidden target over fresh episodes."""
    distances: list[torch.Tensor] = []

    def on_step(observation: Observation, action: torch.Tensor, result: StepResult) -> None:
        distances.append(result.info["distance"].mean())

    for seed in seeds:
        run_episode(env, policy, seed=seed, tracker=NullTracker(), on_step=on_step)
    return float(torch.stack(distances).mean())


def random_distance(env: FakeEnvironment) -> float:
    random_policy = RandomAgent(config=RandomAgentConfig(), device=env.device).fit(
        env, tracker=NullTracker(), seed=0
    )
    return mean_distance(env, random_policy, EVAL_SEEDS)


def zero_distance(env: FakeEnvironment) -> float:
    zero = ConstantActionPolicy(torch.zeros(env.action_space.dim, device=env.device))
    return mean_distance(env, zero, EVAL_SEEDS)


def check_policy(env: FakeEnvironment, policy: Policy) -> None:
    observation = env.reset(seed=123)
    action = policy(observation)
    assert action.shape == (4, 3)
    assert env.action_space.contains(action)
    torch.testing.assert_close(policy(observation), action)
    sampled = policy(observation, deterministic=False)
    assert env.action_space.contains(sampled)
    assert not torch.equal(sampled, action)


# --- networks ------------------------------------------------------------------------------


def test_flatten_observation_uses_sorted_keys():
    observation = {"b": torch.ones(2, 3), "a": torch.zeros(2, 1, 2)}
    flat = flatten_observation(observation)
    assert flat.shape == (2, 5)
    torch.testing.assert_close(flat[:, :2], torch.zeros(2, 2))
    assert observation_dim({"b": (3,), "a": (1, 2)}) == 5
    with pytest.raises(ValueError, match="no entry"):
        flatten_observation({})


def test_squash_maps_into_the_box():
    low, high = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 4.0])
    values = squash(torch.tensor([[0.0, 0.0], [50.0, -50.0]]), low, high)
    torch.testing.assert_close(values, torch.tensor([[0.0, 2.0], [1.0, 0.0]]))


# --- SAC -----------------------------------------------------------------------------------


def small_sac(**overrides: object) -> SACAgentConfig:
    values: dict[str, object] = {
        "hidden_sizes": [64, 64],
        "warmup_steps": 32,
        "batch_size": 64,
        "episodes": 40,
    }
    values.update(overrides)
    return SACAgentConfig.model_validate(values)


def test_sac_config_needs_a_buffer_of_one_batch():
    with pytest.raises(ValidationError, match="replay_capacity"):
        SACAgentConfig(replay_capacity=8, batch_size=16)
    config = SACAgentConfig(replay_capacity=16, batch_size=16)
    assert config.replay_capacity == config.batch_size


def test_sac_smoke(device: torch.device, tmp_path: Path):
    env = make_env(device)
    tracker = MemoryTracker()
    agent = SACAgent(config=small_sac(episodes=2, warmup_steps=4, batch_size=16), device=device)
    with pytest.raises(RuntimeError, match="call fit"):
        agent.policy()
    state = torch.get_rng_state()
    policy = agent.fit(env, tracker=tracker, seed=0)
    assert torch.equal(torch.get_rng_state(), state)
    assert len(tracker.episodes) == 2
    assert len(tracker.steps) == 16
    assert "sac/loss_qvalue" not in tracker.steps[2][1]
    assert {"sac/loss_actor", "sac/loss_qvalue", "sac/alpha"} <= set(tracker.steps[-1][1])
    check_policy(env, policy)
    agent.save(tmp_path / "sac")
    other = SACAgent(config=small_sac(), device=device)
    other.load(tmp_path / "sac")
    observation = env.reset(seed=5)
    torch.testing.assert_close(other.policy()(observation), policy(observation))


def test_sac_is_reproducible(device: torch.device):
    actions = []
    for _ in range(2):
        env = make_env(device)
        agent = SACAgent(config=small_sac(episodes=3, warmup_steps=4, batch_size=16), device=device)
        policy = agent.fit(env, tracker=NullTracker(), seed=7)
        actions.append(policy(env.reset(seed=1)))
    torch.testing.assert_close(actions[0], actions[1])


def test_sac_beats_the_random_policy(device: torch.device):
    env = make_env(device)
    policy = SACAgent(config=small_sac(), device=device).fit(env, tracker=NullTracker(), seed=0)
    assert mean_distance(env, policy, EVAL_SEEDS) < random_distance(env)


def test_sac_learns_the_target(device: torch.device):
    # Four updates per step: the policy must beat the best static action (zero) by half.
    env = make_env(device)
    agent = SACAgent(config=small_sac(updates_per_step=4), device=device)
    policy = agent.fit(env, tracker=NullTracker(), seed=0)
    assert mean_distance(env, policy, EVAL_SEEDS) < 0.5 * zero_distance(env)


# --- PPO -----------------------------------------------------------------------------------


def small_ppo(**overrides: object) -> PPOAgentConfig:
    values: dict[str, object] = {
        "hidden_sizes": [64, 64],
        "episodes_per_update": 4,
        "episodes": 60,
    }
    values.update(overrides)
    return PPOAgentConfig.model_validate(values)


def test_ppo_smoke(device: torch.device, tmp_path: Path):
    env = make_env(device)
    tracker = MemoryTracker()
    agent = PPOAgent(config=small_ppo(episodes=3, episodes_per_update=2), device=device)
    with pytest.raises(RuntimeError, match="call fit"):
        agent.policy()
    state = torch.get_rng_state()
    policy = agent.fit(env, tracker=tracker, seed=0)
    assert torch.equal(torch.get_rng_state(), state)
    assert len(tracker.episodes) == 3
    updates = [step for step, metrics in tracker.steps if "ppo/loss_objective" in metrics]
    assert updates == [15, 23]
    check_policy(env, policy)
    agent.save(tmp_path / "ppo")
    other = PPOAgent(config=small_ppo(), device=device)
    other.load(tmp_path / "ppo")
    observation = env.reset(seed=5)
    torch.testing.assert_close(other.policy()(observation), policy(observation))


def test_ppo_is_reproducible(device: torch.device):
    actions = []
    for _ in range(2):
        env = make_env(device)
        agent = PPOAgent(config=small_ppo(episodes=4, episodes_per_update=2), device=device)
        policy = agent.fit(env, tracker=NullTracker(), seed=7)
        actions.append(policy(env.reset(seed=1)))
    torch.testing.assert_close(actions[0], actions[1])


def test_ppo_beats_the_random_policy(device: torch.device):
    env = make_env(device)
    policy = PPOAgent(config=small_ppo(), device=device).fit(env, tracker=NullTracker(), seed=0)
    assert mean_distance(env, policy, EVAL_SEEDS) < random_distance(env)


def test_ppo_learns_the_target(device: torch.device):
    # A larger learning rate: the policy must beat the best static action (zero) by half.
    env = make_env(device)
    agent = PPOAgent(config=small_ppo(learning_rate=1e-3), device=device)
    policy = agent.fit(env, tracker=NullTracker(), seed=0)
    assert mean_distance(env, policy, EVAL_SEEDS) < 0.5 * zero_distance(env)
