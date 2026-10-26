from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir

from autotsaugment.components.fakes import FakeEnvironment, FakeEnvironmentConfig
from autotsaugment.components.tracking.null import NullTracker
from autotsaugment.core.agent import Agent, AgentConfig
from autotsaugment.core.config import EpisodeConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
AGENTS = sorted(p.stem for p in (CONFIGS / "agent").glob("*.yaml"))


def build(name: str) -> object:
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose("run", overrides=[f"agent={name}", "device=cpu"])
    return hydra.utils.instantiate(cfg.agent)


def test_every_agent_file_is_tested():
    assert AGENTS == ["cmaes", "fixed_uniform", "ppo", "random", "sac"]


@pytest.mark.parametrize("name", AGENTS)
def test_instantiate_builds_the_agent(name: str):
    agent = build(name)
    assert isinstance(agent, Agent)
    assert isinstance(agent.config, AgentConfig)
    assert getattr(agent, "device", None) == torch.device("cpu")


@pytest.mark.parametrize("name", ["fixed_uniform", "random"])
def test_baselines_from_hydra_fit_without_episodes(name: str):
    agent = build(name)
    assert isinstance(agent, Agent)
    assert agent.config.episodes == 0
    episode = EpisodeConfig(epochs=2, series_per_epoch=1, batch_envs=2)
    env = FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device="cpu")
    policy = agent.fit(env, tracker=NullTracker(), seed=0)
    assert policy(env.reset(seed=0)).shape == (2, 3)


def test_hydra_values_reach_the_configs():
    sac = build("sac")
    assert isinstance(sac, Agent)
    assert sac.config.model_dump()["hidden_sizes"] == (256, 256)
    assert sac.config.model_dump()["target_entropy"] == "auto"
    assert sac.config.model_dump()["learning_rate"] == pytest.approx(3e-4)
    cmaes = build("cmaes")
    assert isinstance(cmaes, Agent)
    assert cmaes.config.model_dump()["population"] is None
