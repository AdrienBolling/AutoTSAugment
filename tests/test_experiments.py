import json
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from mlflow import MlflowClient
from omegaconf import DictConfig

from autotsaugment.components.experiments.common import (
    AGENT_CONFIG_FILE,
    CONFIG_DIR,
    ENV_STATE_FILE,
    EVAL_PREFIX,
    build_tracker,
    closing_tracker,
    evaluate_policy,
    experiment_config,
    load_env_state,
    save_env_state,
)
from autotsaugment.components.experiments.compare import comparison_table
from autotsaugment.components.experiments.smoke import SmokeConfig
from autotsaugment.components.experiments.train_agent import agent_label, dvc_commands
from autotsaugment.components.fakes import (
    ConstantPolicy,
    FakeEnvironment,
    FakeEnvironmentConfig,
    MemoryTracker,
)
from autotsaugment.components.tracking.null import NullTracker
from autotsaugment.components.tracking.prefixed import PrefixedTracker
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.experiment import Experiment

pytest_plugins = ("pytester",)

# SQLAlchemy warns about a loader strategy inside MLflow's SQL store. Not our code.
pytestmark = pytest.mark.filterwarnings("ignore:The ``noload`` loader strategy:DeprecationWarning")

REPO = CONFIG_DIR.parent
FAST = ("env=fake", "device=cpu", "tracker=none")
DATASETS = ("fake_0", "fake_1", "fake_2")  # configs/env/benchmark/fake.yaml: n_datasets 3


def summary_keys(seeds: tuple[int, ...], datasets: tuple[str, ...] = DATASETS) -> set[str]:
    """The keys that the docstring of components/experiments/common.py documents."""
    return {
        "comparison/aggregate/mean",
        "comparison/aggregate/std",
        *(f"comparison/aggregate/seed{seed}" for seed in seeds),
        *(f"comparison/{name}/mean" for name in datasets),
        "curve/auc/mean",
        "return/mean",
        "eval/seeds",
        "eval/wall_time_s",
    }


def run_config(tmp_path: Path, *overrides: str) -> DictConfig:
    # `output_dir` interpolates the Hydra runtime, which only the CLI sets.
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose("run", overrides=[*overrides, f"output_dir={tmp_path}"])


def build(cfg: DictConfig) -> Experiment:
    experiment = instantiate(cfg.experiment, run_config=cfg, _recursive_=False)
    assert isinstance(experiment, Experiment)
    return experiment


def train(tmp_path: Path, *overrides: str) -> dict[str, float]:
    cfg = run_config(
        tmp_path,
        "experiment=train_agent",
        *FAST,
        f"experiment.checkpoint_dir={tmp_path / 'checkpoints'}",
        f"experiment.metrics_dir={tmp_path / 'metrics'}",
        *overrides,
    )
    return build(cfg).run()


def read_json(path: Path) -> dict[str, float]:
    return json.loads(path.read_text())


def fake_env(batch_envs: int = 2, epochs: int = 3) -> FakeEnvironment:
    episode = EpisodeConfig(epochs=epochs, series_per_epoch=1, batch_envs=batch_envs)
    return FakeEnvironment(config=FakeEnvironmentConfig(episode=episode), device="cpu")


# --- helpers ---------------------------------------------------------------------------------


def test_evaluate_policy_summarizes_every_seed():
    env = fake_env()
    tracker = MemoryTracker()
    policy = ConstantPolicy(torch.full((3,), 0.25), env.batch_size)
    summary = evaluate_policy(
        env, policy, seeds=(4, 7), tracker=tracker, root_seed=1, step_offset=10, episode_offset=2
    )
    names = env.dataset_names
    assert set(summary.metrics) == summary_keys((4, 7), names)
    assert (summary.episodes, summary.steps) == (2, 6)
    assert summary.metrics["eval/seeds"] == 2
    per_seed = torch.tensor([summary.metrics[f"comparison/aggregate/seed{s}"] for s in (4, 7)])
    assert summary.metrics["comparison/aggregate/mean"] == pytest.approx(per_seed.mean().item())
    assert summary.metrics["comparison/aggregate/std"] == pytest.approx(per_seed.std().item())
    assert [episode for episode, _ in tracker.episodes] == [2, 3]
    assert [step for step, _ in tracker.steps] == list(range(10, 16))
    again = evaluate_policy(env, policy, seeds=(4, 7), tracker=NullTracker(), root_seed=1)
    for key in summary_keys((4, 7), names) - {"eval/wall_time_s"}:
        assert again.metrics[key] == summary.metrics[key], key


def test_evaluate_policy_with_one_seed_has_zero_std():
    env = fake_env()
    policy = ConstantPolicy(torch.zeros(3), env.batch_size)
    summary = evaluate_policy(env, policy, seeds=[0], tracker=NullTracker(), root_seed=0)
    assert summary.metrics["comparison/aggregate/std"] == 0.0
    assert all(math.isfinite(value) for value in summary.metrics.values())


@pytest.mark.parametrize("seeds", [(), (1, 1)])
def test_evaluate_policy_needs_distinct_seeds(seeds: tuple[int, ...]):
    env = fake_env()
    policy = ConstantPolicy(torch.zeros(3), env.batch_size)
    with pytest.raises(ValueError, match="distinct seeds"):
        evaluate_policy(env, policy, seeds=seeds, tracker=NullTracker(), root_seed=0)


def test_prefixed_tracker_prefixes_every_key():
    inner = MemoryTracker()
    view = PrefixedTracker(inner, "eval/")
    view.log_params({"seed": 0})
    view.log_step(3, {"reward": torch.tensor(1.0)})
    view.log_episode(1, {"return": torch.tensor(2.0)})
    view.log_artifact(Path("model"))
    view.close()
    assert inner.params == {"eval/seed": 0}
    assert inner.steps == [(3, {"eval/reward": 1.0})]
    assert inner.episodes == [(1, {"eval/return": 2.0})]
    assert inner.artifacts == [Path("model")]
    assert inner.flushes == 2  # log_episode and close flush; close leaves the tracker open


class FailingAwareTracker(MemoryTracker):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False
        self.closed = 0

    def mark_failed(self) -> None:
        self.failed = True

    def close(self) -> None:
        super().close()
        self.closed += 1


def test_closing_tracker_marks_a_failed_run():
    tracker = FailingAwareTracker()
    with closing_tracker(tracker):
        pass
    assert (tracker.failed, tracker.closed) == (False, 1)
    tracker = FailingAwareTracker()
    message = "boom"
    with pytest.raises(RuntimeError, match=message), closing_tracker(tracker):
        raise RuntimeError(message)
    assert (tracker.failed, tracker.closed) == (True, 1)


def test_without_a_tracker_group_the_run_uses_the_null_tracker(tmp_path: Path):
    cfg = run_config(tmp_path, "experiment=train_agent", "~tracker")
    assert "tracker" not in cfg
    assert isinstance(build_tracker(cfg), NullTracker)


def test_experiment_leaves_are_validated(tmp_path: Path):
    config = experiment_config(SmokeConfig, {"name": "smoke", "episodes": 2})
    assert config == SmokeConfig(episodes=2)
    cfg = run_config(tmp_path, "experiment=smoke", "+experiment.episode=2")
    with pytest.raises(InstantiationException, match="episode"):
        build(cfg)
    cfg = run_config(tmp_path, "experiment=train_agent", *FAST, "experiment.eval_seeds=[1,1]")
    with pytest.raises(InstantiationException, match="distinct"):
        build(cfg)


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("SACAgent", "sac"),
        ("PPOAgent", "ppo"),
        ("CMAESAgent", "cmaes"),
        ("FixedActionAgent", "fixed_action"),
        ("RandomAgent", "random"),
        ("Agent", "agent"),
    ],
)
def test_agent_label_is_the_snake_case_class_name(name: str, label: str):
    assert agent_label(type(name, (), {})()) == label


def test_dvc_commands_copy_the_keeper_out_of_the_ignored_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "checkpoints" / "train_agent" / "sac"
    metrics = tmp_path / "metrics" / "train_agent-sac.json"
    assert dvc_commands(checkpoint, metrics, "train_agent-sac") == [
        "dvc add checkpoints/train_agent/sac -o models/train_agent-sac",
        "git add models/train_agent-sac.dvc models/.gitignore metrics/train_agent-sac.json",
        'git commit -m "chore(models): add train_agent-sac"',
        "dvc push",
    ]


def test_comparison_table_has_one_line_per_policy():
    row = {"comparison/aggregate/mean": 0.5, "comparison/aggregate/std": 0.1, "curve/auc/mean": 0.7}
    lines = comparison_table({"uniform": row, "a_long_name": row}).splitlines()
    assert lines[0].split() == ["policy", "aggregate", "std", "auc"]
    assert lines[2].split() == ["a_long_name", "0.5000", "0.1000", "0.7000"]


# --- experiments through Hydra ---------------------------------------------------------------


def test_smoke_with_the_defaults_returns_the_documented_keys(tmp_path: Path):
    cfg = run_config(tmp_path, "experiment=smoke")
    assert (cfg.device, cfg.tracker._target_.rpartition(".")[2]) == ("cpu", "NullTracker")
    metrics = build(cfg).run()
    assert set(metrics) == summary_keys((0,)) | {"wall_time_s"}
    assert metrics["eval/seeds"] == 1
    assert metrics["comparison/aggregate/std"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())


def test_smoke_logs_to_mlflow(tmp_path: Path):
    store = tmp_path / "mlruns"
    cfg = run_config(
        tmp_path, "experiment=smoke", "tracker=mlflow", f"tracker.tracking_uri={store}"
    )
    metrics = build(cfg).run()
    client = MlflowClient(tracking_uri=f"sqlite:///{store / 'mlflow.db'}")
    experiment = client.get_experiment_by_name("smoke")
    assert experiment is not None
    [run] = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    assert run.data.params["env.config.episode.epochs"] == "5"
    assert run.data.params["experiment._target_"].endswith("SmokeExperiment")
    assert run.data.metrics["comparison/aggregate/mean"] == pytest.approx(
        metrics["comparison/aggregate/mean"]
    )
    rewards = client.get_metric_history(run.info.run_id, f"{EVAL_PREFIX}reward")
    assert sorted(m.step for m in rewards) == list(range(5))
    [episode] = client.get_metric_history(run.info.run_id, f"episode/{EVAL_PREFIX}return")
    assert episode.step == 0


def test_train_agent_writes_the_metrics_and_the_checkpoint(tmp_path: Path):
    metrics = train(tmp_path, "agent=fixed_uniform")
    assert set(metrics) == summary_keys((0, 1, 2)) | {
        "train/wall_time_s",
        "train/episodes",
        "wall_time_s",
    }
    assert metrics["train/episodes"] == 0
    assert read_json(tmp_path / "metrics" / "train_agent-fixed_action.json") == metrics
    checkpoint = tmp_path / "checkpoints" / "fixed_action"
    assert (checkpoint / "agent.pt").is_file()
    assert (checkpoint / AGENT_CONFIG_FILE).is_file()
    state = load_env_state(checkpoint)  # the fake env has a stats featurizer with moments
    assert state is not None
    assert all(key.startswith("featurizer/") for key in state)


def test_train_agent_with_cmaes_and_a_label(tmp_path: Path):
    metrics = train(
        tmp_path,
        "agent=cmaes",
        "agent.config.episodes=1",
        "experiment.eval_seeds=[3]",
        "experiment.label=cma_short",
    )
    assert metrics["train/episodes"] == 1
    assert set(metrics) >= summary_keys((3,))
    assert (tmp_path / "metrics" / "train_agent-cma_short.json").is_file()
    assert (tmp_path / "checkpoints" / "cma_short" / "agent.pt").is_file()


@pytest.fixture
def trained(tmp_path: Path) -> tuple[Path, dict[str, float]]:
    """A fixed-action checkpoint with another action than the uniform one."""
    metrics = train(tmp_path, "agent=fixed_uniform", "agent.config.action=[0.5,-0.5,0.25,0.0]")
    return tmp_path / "checkpoints" / "fixed_action", metrics


def compare(tmp_path: Path, *overrides: str) -> DictConfig:
    return run_config(
        tmp_path,
        "experiment=compare",
        *FAST,
        f"experiment.metrics_dir={tmp_path / 'metrics'}",
        *overrides,
    )


def test_compare_baselines_and_a_checkpoint(tmp_path: Path, trained: tuple[Path, Any]):
    checkpoint, trained_metrics = trained
    cfg = compare(
        tmp_path,
        "+experiment.policies.trained.agent=fixed_uniform",
        f"+experiment.policies.trained.checkpoint={checkpoint}",
    )
    metrics = build(cfg).run()
    policies = ("uniform", "random", "trained")
    expected = {f"{p}/{key}" for p in policies for key in summary_keys((0, 1, 2))}
    assert set(metrics) == expected | {"wall_time_s"}
    assert read_json(tmp_path / "metrics" / "compare.json") == metrics
    # Same seeds, same root seed, deterministic policy: the same numbers as train_agent.
    for key in summary_keys((0, 1, 2)) - {"eval/wall_time_s"}:
        assert metrics[f"trained/{key}"] == pytest.approx(trained_metrics[key]), key
    assert metrics["trained/comparison/aggregate/mean"] != pytest.approx(
        metrics["uniform/comparison/aggregate/mean"]
    )


def test_compare_checks_the_entries_before_any_episode(tmp_path: Path, trained: tuple[Path, Any]):
    checkpoint, _ = trained
    missing = compare(
        tmp_path,
        "+experiment.policies.sac.agent=sac",
        f"+experiment.policies.sac.checkpoint={tmp_path / 'nothing'}",
    )
    with pytest.raises(InstantiationException, match="no checkpoint"):
        build(missing)
    unknown = compare(tmp_path, "+experiment.policies.x.agent=not_an_agent")
    with pytest.raises(InstantiationException, match="no agent config"):
        build(unknown)
    mismatch = compare(
        tmp_path,
        "+experiment.policies.wrong.agent=random",
        f"+experiment.policies.wrong.checkpoint={checkpoint}",
    )
    with pytest.raises(InstantiationException, match="builds"):
        build(mismatch)
    assert not (tmp_path / "metrics" / "compare.json").exists()


def test_compare_reproduces_a_trained_sac_policy(tmp_path: Path):
    # Small networks: compare must rebuild them from agent.yaml, not from configs/agent/sac.yaml.
    trained = train(
        tmp_path,
        "agent=sac",
        "agent.config.episodes=2",
        "agent.config.hidden_sizes=[8]",
        "experiment.eval_seeds=[0,1]",
    )
    checkpoint = tmp_path / "checkpoints" / "sac"
    assert (checkpoint / ENV_STATE_FILE).is_file()
    cfg = compare(
        tmp_path,
        "experiment.eval_seeds=[0,1]",
        "+experiment.policies.sac.agent=sac",
        f"+experiment.policies.sac.checkpoint={checkpoint}",
    )
    metrics = build(cfg).run()
    # uniform and random run first and move the featurizer moments; env.pt restores them.
    for key in summary_keys((0, 1)) - {"eval/wall_time_s"}:
        assert metrics[f"sac/{key}"] == pytest.approx(trained[key]), key
    (checkpoint / ENV_STATE_FILE).unlink()
    without = build(cfg).run()
    assert without["sac/comparison/aggregate/mean"] != pytest.approx(
        trained["comparison/aggregate/mean"]
    )


def test_compare_rejects_a_foreign_env_state_before_any_episode(
    tmp_path: Path, trained: tuple[Path, Any]
):
    checkpoint, _ = trained
    torch.save({"featurizer/unknown": torch.zeros(2)}, checkpoint / ENV_STATE_FILE)
    cfg = compare(
        tmp_path,
        "+experiment.policies.trained.agent=fixed_uniform",
        f"+experiment.policies.trained.checkpoint={checkpoint}",
    )
    with pytest.raises(ValueError, match="unknown"):
        build(cfg).run()
    assert not (tmp_path / "metrics" / "compare.json").exists()


def test_an_empty_env_state_removes_a_stale_file(tmp_path: Path):
    stale = tmp_path / ENV_STATE_FILE
    stale.write_bytes(b"old")
    assert save_env_state(fake_env(), tmp_path) is None
    assert not stale.exists()
    assert load_env_state(tmp_path) is None


# --- command line ----------------------------------------------------------------------------


def test_cli_runs_the_smoke_experiment(pytester: pytest.Pytester, tmp_path: Path):
    uv = shutil.which("uv")
    assert uv is not None
    out = tmp_path / "run"
    result = pytester.run(
        uv,
        "run",
        "--project",
        str(REPO),
        "autotsaugment-run",
        "experiment=smoke",
        f"hydra.run.dir={out}",
        timeout=120,
    )
    assert result.ret == 0, "\n".join(result.errlines)
    metrics = read_json(out / "metrics.json")
    assert set(metrics) == summary_keys((0,)) | {"wall_time_s"}
