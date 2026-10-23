"""Every config group composes, and every `_target_` names an importable object once it exists."""

import importlib
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, ListConfig, OmegaConf

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))
AGENTS = sorted(p.stem for p in (CONFIGS / "agent").glob("*.yaml"))
ENVS = sorted(p.stem for p in (CONFIGS / "env").glob("*.yaml"))


def targets(node: object) -> list[str]:
    """Return every `_target_` string below `node`, without resolving interpolations."""
    if isinstance(node, DictConfig | ListConfig):
        node = OmegaConf.to_container(node, resolve=False)
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "_target_":
                found.append(str(value))
            else:
                found.extend(targets(value))
    elif isinstance(node, list | tuple):
        for value in node:
            found.extend(targets(value))
    return found


def importable(target: str) -> bool:
    module, _, attr = target.rpartition(".")
    try:
        return hasattr(importlib.import_module(module), attr)
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("agent", AGENTS)
@pytest.mark.parametrize("env", ENVS)
def test_compose(experiment: str, agent: str, env: str):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose("run", overrides=[f"experiment={experiment}", f"agent={agent}", f"env={env}"])
    assert cfg.seed == 0
    assert cfg.device in {"cpu", "cuda"}
    OmegaConf.resolve(cfg.env)
    OmegaConf.resolve(cfg.agent)
    assert cfg.env._target_
    assert cfg.agent._target_
    assert cfg.env.tsmodel._partial_ is True
    assert cfg.env.config.episode.epochs >= 1
    for target in targets(cfg):
        module = target.rpartition(".")[0]
        if "." not in module:
            continue
        # Modules that already exist must expose the attribute. Missing modules are work in progress.
        try:
            mod = importlib.import_module(module)
        except ModuleNotFoundError:
            continue
        assert hasattr(mod, target.rpartition(".")[2]), target


def test_fake_env_targets_exist():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose("run", overrides=["experiment=smoke", "env=fake"])
    assert cfg.device == "cpu"
    for target in targets(cfg.env):
        if "fakes" in target:
            assert importable(target), target
