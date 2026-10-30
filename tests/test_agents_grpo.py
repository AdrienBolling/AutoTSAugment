import logging
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from pydantic import ValidationError

from autotsaugment.components.agents.grpo import (
    GRPOAgent,
    GRPOAgentConfig,
    Trajectories,
    group_advantages,
    group_moments,
    returns_to_go,
)
from autotsaugment.components.agents.loop import episode_seed, run_episode
from autotsaugment.components.agents.policies import ConstantActionPolicy
from autotsaugment.components.agents.random import RandomAgent, RandomAgentConfig
from autotsaugment.components.agents.transformer import HistoryPolicy
from autotsaugment.components.experiments.common import CONFIG_DIR, evaluate_policy
from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig, MemoryTracker
from autotsaugment.components.tracking.null import NullTracker
from autotsaugment.core.agent import Policy, StatefulPolicy
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.environment import Environment, StepResult
from autotsaugment.core.featurizer import Observation

EVAL_SEEDS = [episode_seed(10_000, i) for i in range(4)]
METRIC_KEYS = {
    "grpo/loss_policy",
    "grpo/kl_approx",
    "grpo/clip_fraction",
    "grpo/entropy",
    "grpo/group_return_std",
    "grpo/scale_mean",
}


def make_env(device: torch.device, *, batch_envs: int = 4, epochs: int = 8) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device=device)


def small_grpo(**overrides: object) -> GRPOAgentConfig:
    values: dict[str, object] = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "group_episodes": 2,
        "episodes": 4,
    }
    values.update(overrides)
    return GRPOAgentConfig.model_validate(values)


def episode_actions(env: Environment[Any], policy: Policy, seed: int) -> torch.Tensor:
    """The deterministic actions of one episode, shape ``(E, k, a)``."""
    actions: list[torch.Tensor] = []

    def on_step(observation: Observation, action: torch.Tensor, result: StepResult) -> None:
        actions.append(action)

    run_episode(env, policy, seed=seed, tracker=NullTracker(), on_step=on_step)
    return torch.stack(actions)


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


# --- config --------------------------------------------------------------------------------


def test_config_checks_the_network_shape():
    with pytest.raises(ValidationError, match="multiple of n_heads"):
        GRPOAgentConfig(d_model=30, n_heads=4)
    with pytest.raises(ValidationError, match="above min_scale"):
        GRPOAgentConfig(init_scale=0.01, min_scale=0.01)
    with pytest.raises(ValidationError, match="advantage"):
        GRPOAgentConfig.model_validate({"advantage": "episode"})
    config = GRPOAgentConfig()
    assert (config.group_episodes, config.groups_per_update, config.epochs) == (2, 1, 4)
    assert (config.d_model, config.n_layers, config.n_heads) == (256, 4, 8)
    assert config.trajectories_per_minibatch is None
    assert config.episodes == 200


# --- advantages ----------------------------------------------------------------------------


def test_returns_to_go():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    torch.testing.assert_close(
        returns_to_go(rewards, 1.0), torch.tensor([[6.0, 5.0, 3.0], [1.0, 1.0, 1.0]])
    )
    torch.testing.assert_close(
        returns_to_go(rewards, 0.5), torch.tensor([[2.75, 3.5, 3.0], [0.25, 0.5, 1.0]])
    )
    torch.testing.assert_close(returns_to_go(rewards, 0.0), rewards)


def batch_of(rewards: torch.Tensor, group: list[int]) -> Trajectories:
    n, steps = rewards.shape
    return Trajectories(
        tokens=torch.zeros(n, steps, 3),
        samples=torch.zeros(n, steps, 2),
        rewards=rewards,
        group=torch.tensor(group),
    )


def test_group_moments_use_the_group_ids():
    values = torch.tensor([1.0, 10.0, 3.0, 30.0, 5.0])
    mean, std = group_moments(values, torch.tensor([0, 1, 0, 1, 0]), 2)
    torch.testing.assert_close(mean, torch.tensor([3.0, 20.0]))
    torch.testing.assert_close(std, torch.stack([values[[0, 2, 4]].std(), values[[1, 3]].std()]))


def test_step_advantages_are_standardized_per_group_and_step():
    generator = torch.Generator().manual_seed(0)
    rewards = torch.randn(8, 5, generator=generator)
    group = [0, 0, 0, 0, 1, 1, 1, 1]
    advantage = group_advantages(
        batch_of(rewards, group), 2, mode="step", gamma=1.0, normalize_std=True
    )
    assert advantage.shape == (8, 5)
    returns = returns_to_go(rewards, 1.0)
    for rows in (slice(0, 4), slice(4, 8)):
        expected = (returns[rows] - returns[rows].mean(dim=0)) / (returns[rows].std(dim=0) + 1e-6)
        torch.testing.assert_close(advantage[rows], expected)
        torch.testing.assert_close(advantage[rows].mean(dim=0), torch.zeros(5))
    centered = group_advantages(
        batch_of(rewards, group), 2, mode="step", gamma=1.0, normalize_std=False
    )
    torch.testing.assert_close(centered[:4], returns[:4] - returns[:4].mean(dim=0))


def test_outcome_advantages_broadcast_the_episode_return():
    rewards = torch.tensor([[1.0, 0.0], [0.0, 3.0], [2.0, 2.0], [0.0, 0.0]])
    advantage = group_advantages(
        batch_of(rewards, [0, 0, 0, 0]), 1, mode="outcome", gamma=0.5, normalize_std=True
    )
    episode_return = torch.tensor([1.0, 3.0, 4.0, 0.0])
    expected = (episode_return - 2.0) / (episode_return.std() + 1e-6)
    torch.testing.assert_close(advantage, expected.unsqueeze(1).expand(4, 2))


# --- agent ---------------------------------------------------------------------------------


def test_grpo_needs_two_trajectories_per_group(device: torch.device):
    env = make_env(device, batch_envs=1)
    agent = GRPOAgent(config=small_grpo(group_episodes=1), device=device)
    with pytest.raises(ValueError, match="at least two trajectories"):
        agent.fit(env, tracker=NullTracker(), seed=0)


def test_grpo_smoke(device: torch.device, tmp_path: Path, caplog: pytest.LogCaptureFixture):
    env = make_env(device)
    tracker = MemoryTracker()
    agent = GRPOAgent(config=small_grpo(episodes=5), device=device)
    with pytest.raises(RuntimeError, match="call fit"):
        agent.policy()
    state = torch.get_rng_state()
    with caplog.at_level(logging.WARNING, logger="autotsaugment.components.agents.grpo"):
        policy = agent.fit(env, tracker=tracker, seed=0)
    assert torch.equal(torch.get_rng_state(), state)
    assert "drops the last 1 of 5 episodes" in caplog.text
    assert [episode for episode, _ in tracker.episodes] == [0, 1, 2, 3]
    updates = [(step, metrics) for step, metrics in tracker.steps if "grpo/loss_policy" in metrics]
    assert [step for step, _ in updates] == [15, 31]
    for _, metrics in updates:
        assert set(metrics) == METRIC_KEYS
        assert metrics["grpo/scale_mean"] == pytest.approx(0.3, abs=0.05)
        assert metrics["grpo/group_return_std"] > 0.0
    assert tracker._pending_steps == []
    assert isinstance(policy, HistoryPolicy)
    assert policy.steps == 0
    observation = env.reset(seed=123)
    action = policy(observation)
    assert action.shape == (4, 3)
    assert env.action_space.contains(action)
    agent.save(tmp_path / "grpo")
    other = GRPOAgent(config=small_grpo(d_model=64), device=device)
    other.load(tmp_path / "grpo")
    torch.testing.assert_close(
        episode_actions(env, other.policy(), seed=5), episode_actions(env, policy, seed=5)
    )


def test_grpo_minibatches_of_whole_trajectories(device: torch.device):
    env = make_env(device)
    tracker = MemoryTracker()
    config = small_grpo(trajectories_per_minibatch=3, advantage="outcome", kl_coef=0.1)
    GRPOAgent(config=config, device=device).fit(env, tracker=tracker, seed=0)
    updates = [metrics for _, metrics in tracker.steps if "grpo/loss_policy" in metrics]
    assert len(updates) == 2
    assert all(set(metrics) == METRIC_KEYS for metrics in updates)


def test_grpo_policy_is_deterministic_by_default(device: torch.device):
    env = make_env(device)
    policy = GRPOAgent(config=small_grpo(), device=device).fit(env, tracker=NullTracker(), seed=0)
    assert isinstance(policy, StatefulPolicy)
    observation = env.reset(seed=1)
    policy.reset()
    action = policy(observation)
    policy.reset()
    torch.testing.assert_close(policy(observation), action)
    policy.reset()
    sampled = policy(observation, deterministic=False)
    assert env.action_space.contains(sampled)
    assert not torch.equal(sampled, action)


def test_grpo_is_reproducible(device: torch.device):
    actions = []
    for _ in range(2):
        env = make_env(device)
        agent = GRPOAgent(config=small_grpo(learning_rate=1e-3), device=device)
        policy = agent.fit(env, tracker=NullTracker(), seed=7)
        actions.append(episode_actions(env, policy, seed=1))
    torch.testing.assert_close(actions[0], actions[1])


def test_grpo_beats_the_random_policy(device: torch.device):
    env = make_env(device)
    agent = GRPOAgent(config=small_grpo(episodes=40), device=device)
    policy = agent.fit(env, tracker=NullTracker(), seed=0)
    assert mean_distance(env, policy, EVAL_SEEDS) < random_distance(env)


def test_grpo_learns_the_target(device: torch.device):
    # The plain fake env: K different targets per episode give the update a spread of targets.
    # (A shared target gives one target per update, and the policy then tracks the last target
    # instead of learning the map.) The fake env is a bandit, each reward depends on the action
    # of its own step only, so gamma = 0 removes the noise of the later steps from the return.
    env = make_env(device)
    config = small_grpo(episodes=200, learning_rate=1e-3, gamma=0.0)
    policy = GRPOAgent(config=config, device=device).fit(env, tracker=NullTracker(), seed=0)
    assert mean_distance(env, policy, EVAL_SEEDS) < 0.5 * zero_distance(env)


# --- GPU -----------------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode:UserWarning")
def test_grpo_on_the_real_env_on_the_gpu(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    root = Path(os.environ.get("AUTOTSAUGMENT_GIFTEVAL_ROOT", CONFIG_DIR.parent / "data/gifteval"))
    if not (root / "hub").is_dir():
        pytest.skip(f"needs the GIFT-Eval shards under {root}; set AUTOTSAUGMENT_GIFTEVAL_ROOT")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            "run",
            overrides=[
                "env=default",
                "agent=grpo",
                "device=cuda",
                f"env.benchmark.config.root={root}",
                "env.config.episode.epochs=2",
                "env.config.episode.series_per_epoch=256",
                "agent.config.group_episodes=2",
                "agent.config.episodes=2",
            ],
        )
    env = instantiate(cfg.env)
    agent = instantiate(cfg.agent)
    assert isinstance(agent, GRPOAgent)
    tracker = MemoryTracker()
    agent.fit(env, tracker=tracker, seed=0)
    updates = [metrics for _, metrics in tracker.steps if "grpo/loss_policy" in metrics]
    assert len(updates) == 1
    assert set(updates[0]) == METRIC_KEYS
    assert all(torch.isfinite(torch.tensor(value)) for value in updates[0].values())
    agent.save(tmp_path / "grpo")
    loaded = GRPOAgent(config=agent.config, device="cuda")
    loaded.load(tmp_path / "grpo")
    policy = loaded.policy()
    # The acting path makes no host synchronization.
    observation = env.reset(seed=0)
    torch.cuda.synchronize()
    try:
        torch.cuda.set_sync_debug_mode("error")
        action = policy(observation, deterministic=False)
        action = policy(observation)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert action.device.type == "cuda"
    assert env.action_space.contains(action)
    summary = evaluate_policy(env, policy, seeds=[0], tracker=NullTracker(), root_seed=0)
    assert torch.isfinite(torch.tensor(summary.metrics["comparison/aggregate/mean"]))
    del env, agent, loaded, policy
    torch.cuda.empty_cache()
