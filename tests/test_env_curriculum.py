import functools
from pathlib import Path
from typing import Any

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir

from autotsaugment.components.env.curriculum import TRAIN_KEYS, CurriculumEnv, CurriculumEnvConfig
from autotsaugment.components.env.featurizers import StatsFeaturizer, StatsFeaturizerConfig
from autotsaugment.components.env.rewards import ImprovementReward, ImprovementRewardConfig
from autotsaugment.components.env.scoring import score
from autotsaugment.components.fakes import (
    FakeAugmentation,
    FakeAugmentationConfig,
    FakeBenchmark,
    FakeBenchmarkConfig,
    FakeTSModel,
    FakeTSModelConfig,
)
from autotsaugment.core.config import EpisodeConfig
from autotsaugment.core.environment import StepResult
from autotsaugment.core.typechecking import ActionBatch, SeriesBatchK

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EPOCHS, SERIES, K = 5, 64, 2


class RecordingAugmentation(FakeAugmentation):
    """Keeps a copy of every batch of series it samples."""

    def __init__(self, *, config: FakeAugmentationConfig, device: torch.device | str) -> None:
        super().__init__(config=config, device=device)
        self.samples: list[torch.Tensor] = []

    def sample(self, action: ActionBatch, n: int, *, generator: torch.Generator) -> SeriesBatchK:
        series = super().sample(action, n, generator=generator)
        self.samples.append(series.clone())
        return series


def make_env(
    device: torch.device | str = "cpu",
    *,
    subset: int | None = 16,
    reward_config: ImprovementRewardConfig | None = None,
) -> CurriculumEnv:
    config = CurriculumEnvConfig(
        episode=EpisodeConfig(epochs=EPOCHS, series_per_epoch=SERIES, batch_envs=K),
        reward_subset_per_dataset=subset,
        score_batch_size=40,
    )
    return CurriculumEnv(
        config=config,
        device=device,
        tsmodel=functools.partial(
            FakeTSModel, config=FakeTSModelConfig(), device=device, episode_epochs=EPOCHS
        ),
        augmentation=RecordingAugmentation(config=FakeAugmentationConfig(), device=device),
        benchmark=FakeBenchmark(config=FakeBenchmarkConfig(), device=device),
        featurizer=functools.partial(StatsFeaturizer, config=StatsFeaturizerConfig()),
        reward=ImprovementReward(config=reward_config or ImprovementRewardConfig()),
    )


def constant_action(trend: float, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([[0.0, trend, 0.0, 0.0]] * K, device=device)


MATCHED = FakeBenchmark(config=FakeBenchmarkConfig(), device="cpu").mean_drift / (
    FakeAugmentation.DRIFT_SCALE
)
OPPOSITE = -0.5


def run_episode(env: CurriculumEnv, action: torch.Tensor, seed: int = 0) -> list[StepResult]:
    env.reset(seed=seed)
    return [env.step(action) for _ in range(env.episode_length)]


def recorded(env: CurriculumEnv) -> list[torch.Tensor]:
    augmentation = env._augmentation
    assert isinstance(augmentation, RecordingAugmentation)
    return augmentation.samples


def test_every_output_has_the_documented_shape():
    env = make_env()
    d, a = len(env.dataset_names), env.action_space.dim
    assert (d, a) == (3, 4)
    assert env.dataset_names == ("fake_0", "fake_1", "fake_2")
    assert len(env.tsmodels) == K
    observation = env.reset(seed=0)
    assert env.epoch == 0
    assert set(observation) == set(env.observation_spec)
    for name, shape in env.observation_spec.items():
        assert observation[name].shape == (K, *shape), name
    result = env.step(constant_action(0.3))
    for name, shape in env.observation_spec.items():
        assert result.observation[name].shape == (K, *shape), name
        assert result.observation[name].dtype == torch.float32, name
        assert torch.isfinite(result.observation[name]).all(), name
    assert result.reward.shape == (K,)
    assert result.done.shape == (K,)
    assert result.done.dtype == torch.bool
    for name, value in result.info.items():
        assert value.shape[0] == K, name
    assert result.info["scores"].shape == (K, d)
    assert result.info["aggregate"].shape == (K,)
    assert result.info["action"].shape == (K, a)
    assert result.info["reward_rows"].shape == (K, 16 * d)
    for name in TRAIN_KEYS:
        assert result.info[f"train/{name}"].shape == (K,), name
    torch.testing.assert_close(result.info["train/series"], torch.full((K,), float(SERIES)))
    assert recorded(env)[0].shape == (K, SERIES, 64)


def test_uniform_action_is_the_augmentation_uniform_action():
    env = make_env()
    action = env.uniform_action()
    assert action.shape == (env.action_space.dim,)
    assert action.device == env.device
    torch.testing.assert_close(action, env._augmentation.uniform_action())


def test_done_only_at_the_last_step():
    env = make_env()
    results = run_episode(env, constant_action(0.3))
    for step, result in enumerate(results, start=1):
        assert result.done.tolist() == [step == EPOCHS] * K, step
    with pytest.raises(RuntimeError, match="reset"):
        env.step(constant_action(0.3))
    env.reset(seed=1)
    assert env.epoch == 0
    assert not env.step(constant_action(0.3)).done.any()


def test_step_needs_a_reset():
    with pytest.raises(RuntimeError, match="reset"):
        make_env().step(constant_action(0.3))


def test_rewards_telescope_to_the_aggregate_change():
    env = make_env(subset=None)
    env.reset(seed=3)
    windows = env._benchmark.reward_windows
    initial = torch.stack([score(model, windows, batch_size=64)[1] for model in env.tsmodels])
    results = [env.step(constant_action(MATCHED)) for _ in range(EPOCHS)]
    total = torch.stack([result.reward for result in results]).sum(dim=0)
    final = results[-1].info["aggregate"]
    torch.testing.assert_close(total, initial - final)
    assert results[-1].info["reward_rows"].shape == (K, windows.n_windows)


def test_same_seed_same_episode_and_other_seed_other_series():
    action = constant_action(0.4)
    first, second, other = make_env(), make_env(), make_env()
    runs = [run_episode(env, action, seed) for env, seed in ((first, 7), (second, 7), (other, 8))]
    for a, b in zip(runs[0], runs[1], strict=True):
        assert torch.equal(a.reward, b.reward)
        for name in a.observation:
            assert torch.equal(a.observation[name], b.observation[name]), name
    for a, b in zip(recorded(first), recorded(second), strict=True):
        assert torch.equal(a, b)
    assert not torch.equal(recorded(first)[0], recorded(other)[0])
    assert not torch.equal(runs[0][-1].reward, runs[2][-1].reward)


def test_reward_subset_is_shared_by_all_models_and_redrawn_every_step():
    env = make_env(subset=16)
    first, second = run_episode(env, constant_action(0.3))[:2]
    rows_1, rows_2 = first.info["reward_rows"], second.info["reward_rows"]
    for rows in (rows_1, rows_2):
        assert torch.equal(rows, rows[:1].expand_as(rows))
        dataset_index = env._benchmark.reward_windows.dataset_index[rows[0]]
        assert torch.bincount(dataset_index).tolist() == [16, 16, 16]
    assert not torch.equal(rows_1[0], rows_2[0])


def test_the_reward_responds_to_the_action():
    returns = {}
    for name, trend in (("matched", MATCHED), ("opposite", OPPOSITE)):
        results = run_episode(make_env(), constant_action(trend), seed=0)
        returns[name] = torch.stack([result.reward for result in results]).sum(dim=0)
    assert (returns["matched"] > returns["opposite"]).all()


def test_evaluate_after_a_matched_episode():
    env = make_env()
    env.reset(seed=0)
    before = env.evaluate()
    assert before.shape == (K, len(env.dataset_names))
    assert torch.isfinite(before).all()
    run_episode(env, constant_action(MATCHED), seed=0)
    after = env.evaluate()
    assert after.shape == (K, 3)
    assert torch.isfinite(after).all()
    assert (after < 1.0).all()


def test_actions_are_clipped_and_checked():
    env = make_env()
    env.reset(seed=0)
    result = env.step(torch.full((K, 4), 3.0))
    torch.testing.assert_close(result.info["action"], torch.ones(K, 4))
    with pytest.raises(ValueError, match="shape"):
        env.step(torch.zeros(K, 5))
    with pytest.raises(ValueError, match="shape"):
        env.step(torch.zeros(K + 1, 4))


def test_state_round_trip_through_torch_save(tmp_path: Path):
    trained = make_env()
    for seed in (0, 1):
        run_episode(trained, constant_action(0.3), seed=seed)
    state = trained.state_dict()
    assert state
    assert all(key.startswith("featurizer/") for key in state)
    path = tmp_path / "env_state.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored, fresh = make_env(), make_env()
    restored.load_state_dict(loaded)
    observations = [env.reset(seed=5) for env in (trained, restored, fresh)]
    from_trained, from_restored, from_fresh = observations
    for name in trained.observation_spec:
        torch.testing.assert_close(from_restored[name], from_trained[name])
    assert not torch.allclose(from_fresh["reward_vector"], from_trained["reward_vector"])
    action = constant_action(MATCHED)
    first, second = trained.step(action), restored.step(action)
    for name in trained.observation_spec:
        torch.testing.assert_close(second.observation[name], first.observation[name])


def test_wrong_state_keys_are_rejected():
    env = make_env()
    state = env.state_dict()
    with pytest.raises(ValueError, match="prefix"):
        env.load_state_dict({**state, "tsmodel/weights": torch.zeros(1)})
    with pytest.raises(ValueError, match="unknown keys"):
        env.load_state_dict({**state, "featurizer/extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="missing keys"):
        env.load_state_dict({})


def test_components_on_another_device_are_rejected():
    benchmark = FakeBenchmark(config=FakeBenchmarkConfig(), device="cpu")
    with pytest.raises(ValueError, match="tsmodel 0"):
        CurriculumEnv(
            config=CurriculumEnvConfig(),
            device="cpu",
            tsmodel=functools.partial(
                FakeTSModel, config=FakeTSModelConfig(), device="meta", episode_epochs=1
            ),
            augmentation=FakeAugmentation(config=FakeAugmentationConfig(), device="cpu"),
            benchmark=benchmark,
            featurizer=functools.partial(StatsFeaturizer, config=StatsFeaturizerConfig()),
            reward=ImprovementReward(config=ImprovementRewardConfig()),
        )


def test_a_featurizer_instance_is_rejected():
    # Typed as Any: the point of the test is a wrong argument that Hydra can produce.
    featurizer: Any = StatsFeaturizer(
        config=StatsFeaturizerConfig(), n_datasets=3, action_dim=4, batch_envs=K, device="cpu"
    )
    with pytest.raises(TypeError, match="_partial_"):
        CurriculumEnv(
            config=CurriculumEnvConfig(),
            device="cpu",
            tsmodel=functools.partial(
                FakeTSModel, config=FakeTSModelConfig(), device="cpu", episode_epochs=1
            ),
            augmentation=FakeAugmentation(config=FakeAugmentationConfig(), device="cpu"),
            benchmark=FakeBenchmark(config=FakeBenchmarkConfig(), device="cpu"),
            featurizer=featurizer,
            reward=ImprovementReward(config=ImprovementRewardConfig()),
        )


def test_hydra_builds_the_fake_environment():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
        cfg = compose("run", overrides=["experiment=smoke", "env=fake", "device=cpu"])
    env = hydra.utils.instantiate(cfg.env)
    assert isinstance(env, CurriculumEnv)
    assert env.device == torch.device("cpu")
    assert env.batch_size == cfg.env.config.episode.batch_envs
    observation = env.reset(seed=0)
    assert set(observation) == set(env.observation_spec)
    action = env.action_space.sample(env.batch_size, generator=torch.Generator().manual_seed(0))
    result = env.step(action)
    assert result.reward.shape == (env.batch_size,)
    assert torch.isfinite(result.reward).all()


@pytest.mark.gpu
@pytest.mark.filterwarnings("ignore:Synchronization debug mode:UserWarning")
def test_a_fake_episode_on_the_gpu_never_synchronizes():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    device = torch.device("cuda")
    env = make_env(device)
    action = constant_action(MATCHED, device)
    torch.cuda.synchronize()
    try:
        torch.cuda.set_sync_debug_mode("error")
        env.reset(seed=0)
        results = [env.step(action) for _ in range(env.episode_length)]
        skill = env.evaluate()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert results[-1].done.all()
    assert results[-1].reward.device.type == "cuda"
    for value in results[-1].observation.values():
        assert value.device.type == "cuda"
    assert torch.isfinite(skill).all()
    del env, results, skill, action
    torch.cuda.empty_cache()
