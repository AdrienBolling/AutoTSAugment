import pytest
import torch

from autotsaugment.components.env.featurizers import (
    TRAIN_FEATURES,
    RunningMoments,
    StatsFeaturizer,
    StatsFeaturizerConfig,
)
from autotsaugment.components.env.history import RecentValues
from autotsaugment.core.featurizer import (
    EpochRecord,
    Featurizer,
    FeaturizerConfig,
    Observation,
    ObservationSpec,
)

K, D, A, E = 2, 3, 4, 10


def make(device: torch.device, **config: object) -> StatsFeaturizer:
    return StatsFeaturizer(
        config=StatsFeaturizerConfig.model_validate(config),
        n_datasets=D,
        action_dim=A,
        batch_envs=K,
        device=device,
    )


def record(
    epoch: int,
    *,
    loss: float | torch.Tensor = 1.0,
    scores: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
    series: float | None = 64.0,
) -> EpochRecord:
    loss = loss if isinstance(loss, torch.Tensor) else torch.full((K,), loss)
    train = {"loss": loss, "grad_norm": torch.full((K,), 0.5), "lr": torch.full((K,), 1e-3)}
    if series is not None:
        train["series"] = torch.full((K,), series)
    scores = torch.ones(K, D) if scores is None else scores
    return EpochRecord(
        epoch=epoch,
        epochs=E,
        train=train,
        scores=scores,
        aggregate=scores.log().mean(dim=-1).exp(),
        action=torch.zeros(K, A) if action is None else action,
    )


def assert_matches_spec(featurizer: StatsFeaturizer, observation: dict[str, torch.Tensor]):
    assert set(observation) == set(featurizer.observation_spec)
    for name, shape in featurizer.observation_spec.items():
        assert observation[name].shape == (K, *shape), name
        assert observation[name].dtype == torch.float32, name


def test_observation_matches_the_spec(device: torch.device):
    featurizer = make(device)
    assert featurizer.observation_spec == {
        "progress": (2,),
        "train_stats": (len(TRAIN_FEATURES),),
        "reward_vector": (3 * D,),
        "action_history": (A,),
    }
    assert_matches_spec(featurizer, featurizer.reset())
    assert_matches_spec(featurizer, featurizer(record(0)))
    assert_matches_spec(featurizer, featurizer(record(1)))


def test_reset_observation_holds_only_the_initial_scores(device: torch.device):
    featurizer = make(device, normalize=False)
    reset = featurizer.reset()
    for value in reset.values():
        assert torch.equal(value, torch.zeros_like(value))
    scores = torch.tensor([[0.9, 1.2, 1.0], [1.1, 0.8, 1.3]])
    initial = featurizer(record(0, scores=scores, loss=7.0))
    torch.testing.assert_close(initial["reward_vector"][:, :D], scores)
    torch.testing.assert_close(initial["reward_vector"][:, D : 2 * D], torch.zeros(K, D))
    torch.testing.assert_close(initial["reward_vector"][:, 2 * D :], scores)
    for name in ("progress", "train_stats", "action_history"):
        assert torch.equal(initial[name], torch.zeros_like(initial[name])), name


def test_normalized_values_are_finite_from_the_first_step(device: torch.device):
    featurizer = make(device, normalize=True)
    observations = [featurizer.reset(), featurizer(record(0)), featurizer(record(1))]
    for observation in observations:
        for name, value in observation.items():
            assert torch.isfinite(value).all(), name
    # Equal rows have zero variance; the epsilon guard maps them to zero, not to NaN.
    torch.testing.assert_close(observations[1]["reward_vector"], torch.zeros(K, 3 * D))


def test_slope_of_a_decreasing_loss_is_negative(device: torch.device):
    featurizer = make(device, normalize=False, loss_window=3)
    featurizer.reset()
    featurizer(record(0))
    observations = [
        featurizer(record(epoch, loss=loss))
        for epoch, loss in enumerate([9.0, 5.0, 4.0, 3.0, 2.0], start=1)
    ]
    column = TRAIN_FEATURES.index("loss_slope")
    # Window [9, 5] at epoch 2: slope -4. Window [4, 3, 2] at epoch 5: slope -1.
    torch.testing.assert_close(observations[1]["train_stats"][:, column], torch.full((K,), -4.0))
    torch.testing.assert_close(observations[-1]["train_stats"][:, column], torch.full((K,), -1.0))


def test_train_stats_columns(device: torch.device):
    featurizer = make(device, normalize=False, loss_ema=0.25)
    featurizer.reset()
    featurizer(record(0))
    first = featurizer(record(1, loss=4.0))["train_stats"]
    second = featurizer(record(2, loss=8.0))["train_stats"]
    torch.testing.assert_close(first[0], torch.tensor([4.0, 4.0, 0.0, 0.5, 1e-3]))
    torch.testing.assert_close(second[0], torch.tensor([8.0, 5.0, 4.0, 0.5, 1e-3]))


def test_action_ema_after_one_step(device: torch.device):
    featurizer = make(device, normalize=False, action_ema=0.3)
    featurizer.reset()
    featurizer(record(0))
    action = torch.tensor([[0.5, -1.0, 0.2, 0.0], [1.0, 1.0, -0.4, 0.3]])
    first = featurizer(record(1, action=action))
    torch.testing.assert_close(first["action_history"], 0.3 * action)
    second = featurizer(record(2, action=-action))
    torch.testing.assert_close(second["action_history"], 0.3 * -action + 0.7 * 0.3 * action)


def test_progress_and_best_scores(device: torch.device):
    featurizer = make(device, normalize=False)
    featurizer.reset()
    featurizer(record(0, scores=torch.full((K, D), 1.0)))
    featurizer(record(1, scores=torch.full((K, D), 0.6)))
    third = featurizer(record(2, scores=torch.full((K, D), 0.8)))
    torch.testing.assert_close(third["progress"], torch.full((K, 2), 2 / E))
    vector = third["reward_vector"]
    torch.testing.assert_close(vector[:, D : 2 * D], torch.full((K, D), 0.2))
    torch.testing.assert_close(vector[:, 2 * D :], torch.full((K, D), 0.6))
    # Without a series count, the second progress entry follows the epoch.
    featurizer.reset()
    observation = featurizer(record(3, series=None))
    torch.testing.assert_close(observation["progress"], torch.full((K, 2), 3 / E))


def test_reset_forgets_the_episode_but_keeps_the_statistics(device: torch.device):
    featurizer = make(device, normalize=True)
    featurizer.reset()
    featurizer(record(0))
    featurizer(record(1, action=torch.ones(K, A)))
    count = featurizer.moments["action_history"].count
    featurizer.reset()
    assert featurizer.moments["action_history"].count == count
    raw = make(device, normalize=False)
    raw.reset()
    raw(record(1, action=torch.ones(K, A)))
    raw.reset()
    torch.testing.assert_close(raw(record(1))["action_history"], torch.zeros(K, A))


def test_running_moments_match_the_batch_statistics(device: torch.device):
    data = torch.randn(30, 5, generator=torch.Generator().manual_seed(0)) * 3.0 + 2.0
    moments = RunningMoments(5, device=device, epsilon=1e-8)
    for chunk in data.split(4):
        moments.update(chunk)
    assert moments.count == 30
    torch.testing.assert_close(moments.mean, data.mean(dim=0))
    torch.testing.assert_close(moments.variance, data.var(dim=0, correction=0))
    normalized = moments.normalize(data)
    torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(5), atol=1e-5, rtol=0)


def feed_an_episode(featurizer: StatsFeaturizer, *, offset: float = 0.0) -> None:
    generator = torch.Generator().manual_seed(int(offset * 10))
    featurizer.reset()
    featurizer(record(0, scores=torch.rand(K, D, generator=generator) + 0.5))
    for epoch in range(1, 4):
        featurizer(
            record(
                epoch,
                loss=torch.rand(K, generator=generator) + offset,
                scores=torch.rand(K, D, generator=generator) + 0.5,
                action=torch.rand(K, A, generator=generator) * 2.0 - 1.0,
            )
        )


def test_state_round_trip_reproduces_the_observation(device: torch.device):
    trained = make(device)
    feed_an_episode(trained, offset=1.0)
    feed_an_episode(trained, offset=2.0)
    state = trained.state_dict()
    assert set(state) == {
        f"{group}/{name}" for group in trained.observation_spec for name in ("mean", "m2", "count")
    }
    assert state["progress/count"].item() == trained.moments["progress"].count
    fresh, restored = make(device), make(device)
    restored.load_state_dict(state)
    probe = record(2, loss=3.0, scores=torch.full((K, D), 0.7), action=torch.ones(K, A))
    observations = []
    for featurizer in (trained, fresh, restored):
        featurizer.reset()
        featurizer(record(0, scores=torch.full((K, D), 0.9)))
        observations.append(featurizer(probe))
    from_trained, from_fresh, from_restored = observations
    for name in trained.observation_spec:
        torch.testing.assert_close(from_restored[name], from_trained[name])
    assert not torch.allclose(from_fresh["train_stats"], from_trained["train_stats"])


def test_state_is_a_copy(device: torch.device):
    featurizer = make(device)
    feed_an_episode(featurizer, offset=1.0)
    state = featurizer.state_dict()
    before = {key: value.clone() for key, value in state.items()}
    feed_an_episode(featurizer, offset=2.0)
    for key, value in state.items():
        torch.testing.assert_close(value, before[key])


def test_wrong_state_is_rejected(device: torch.device):
    featurizer = make(device)
    feed_an_episode(featurizer, offset=1.0)
    state = featurizer.state_dict()
    with pytest.raises(ValueError, match="missing keys \\['progress/mean'\\]"):
        featurizer.load_state_dict({k: v for k, v in state.items() if k != "progress/mean"})
    with pytest.raises(ValueError, match="unknown keys \\['extra'\\]"):
        featurizer.load_state_dict({**state, "extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="action_history/mean has shape"):
        featurizer.load_state_dict({**state, "action_history/mean": torch.zeros(A + 1)})
    with pytest.raises(ValueError, match="count is negative"):
        featurizer.load_state_dict({**state, "progress/count": torch.tensor(-1)})
    other = make(device)
    with pytest.raises(ValueError, match="shape"):
        other.load_state_dict({**state, "reward_vector/m2": torch.zeros(1)})
    assert other.moments["reward_vector"].count == 0


class StatelessFeaturizer(Featurizer[FeaturizerConfig]):
    """The smallest featurizer: the default state methods of the contract."""

    config = FeaturizerConfig()

    @property
    def observation_spec(self) -> ObservationSpec:
        return {"progress": (1,)}

    def reset(self) -> Observation:
        return {"progress": torch.zeros(K, 1)}

    def __call__(self, record: EpochRecord) -> Observation:
        return {"progress": torch.full((K, 1), record.epoch / record.epochs)}


def test_the_default_featurizer_state_is_empty():
    featurizer = StatelessFeaturizer()
    assert featurizer.state_dict() == {}
    featurizer.load_state_dict({})
    with pytest.raises(ValueError, match="has no state"):
        featurizer.load_state_dict({"progress/mean": torch.zeros(1)})


def test_record_with_wrong_shapes_is_rejected(device: torch.device):
    featurizer = make(device)
    featurizer.reset()
    with pytest.raises(ValueError, match="scores"):
        featurizer(record(1, scores=torch.ones(K, D + 1)))


def test_recent_values_slope_and_variance():
    window = RecentValues(3)
    window.append(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(window.slope(), torch.zeros(2))
    torch.testing.assert_close(window.variance(), torch.zeros(2))
    for value in ([3.0, 2.0], [5.0, 2.0], [7.0, 2.0]):
        window.append(torch.tensor(value))
    assert len(window) == 3
    torch.testing.assert_close(window.slope(), torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(window.variance(), torch.tensor([8.0 / 3.0, 0.0]))
    window.clear()
    with pytest.raises(RuntimeError, match="no value"):
        window.slope()
    with pytest.raises(ValueError, match="at least 2"):
        RecentValues(1)
