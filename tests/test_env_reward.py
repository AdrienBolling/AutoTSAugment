import pytest
import torch

from autotsaugment.components.env.rewards import ImprovementReward, ImprovementRewardConfig
from autotsaugment.core.featurizer import EpochRecord


def record(aggregate: list[float], loss: list[float], epoch: int = 1) -> EpochRecord:
    k = len(aggregate)
    return EpochRecord(
        epoch=epoch,
        epochs=10,
        train={"loss": torch.tensor(loss)},
        scores=torch.ones(k, 3),
        aggregate=torch.tensor(aggregate),
        action=torch.zeros(k, 4),
    )


def test_a_decrease_of_the_aggregate_is_a_positive_reward():
    reward = ImprovementReward(config=ImprovementRewardConfig(scale=2.0))
    reward.reset(torch.tensor([1.0, 1.0]))
    first = reward(record([0.8, 1.1], [0.5, 0.5]))
    assert first.shape == (2,)
    torch.testing.assert_close(first, torch.tensor([0.4, -0.2]))
    # The previous aggregate is now the one of the last step.
    second = reward(record([0.7, 1.1], [0.5, 0.5], epoch=2))
    torch.testing.assert_close(second, torch.tensor([0.2, 0.0]))


def test_quality_terms_are_zero_by_default():
    reward = ImprovementReward(config=ImprovementRewardConfig())
    reward.reset(torch.tensor([1.0]))
    values = [reward(record([1.0], [loss], epoch=i + 1)) for i, loss in enumerate([3.0, 1.0, 5.0])]
    for value in values:
        torch.testing.assert_close(value, torch.zeros(1))


def test_the_variance_term_penalizes_a_noisy_loss():
    config = ImprovementRewardConfig(loss_variance_weight=0.5, loss_window=3)
    reward = ImprovementReward(config=config)
    reward.reset(torch.tensor([1.0, 1.0]))
    losses = [[1.0, 2.0], [3.0, 2.0], [5.0, 2.0], [7.0, 2.0]]
    values = [reward(record([1.0, 1.0], loss, epoch=i + 1)) for i, loss in enumerate(losses)]
    torch.testing.assert_close(values[0], torch.zeros(2))  # one value: zero variance
    # Last step: window [3, 5, 7] has population variance 8/3; the constant loss has none.
    torch.testing.assert_close(values[-1], torch.tensor([-0.5 * 8.0 / 3.0, 0.0]))


def test_the_slope_term_rewards_a_decreasing_loss():
    config = ImprovementRewardConfig(loss_slope_weight=0.1, loss_window=4)
    reward = ImprovementReward(config=config)
    reward.reset(torch.tensor([1.0, 1.0]))
    losses = [[4.0, 1.0], [3.0, 2.0], [2.0, 3.0]]
    values = [reward(record([1.0, 1.0], loss, epoch=i + 1)) for i, loss in enumerate(losses)]
    # Slope -1 per epoch gives +0.1; slope +1 per epoch gives -0.1.
    torch.testing.assert_close(values[-1], torch.tensor([0.1, -0.1]))


def test_reset_starts_a_new_episode():
    config = ImprovementRewardConfig(loss_variance_weight=1.0)
    reward = ImprovementReward(config=config)
    reward.reset(torch.tensor([1.0]))
    reward(record([0.5], [1.0]))
    reward(record([0.5], [9.0], epoch=2))
    reward.reset(torch.tensor([2.0]))
    value = reward(record([1.5], [1.0]))
    torch.testing.assert_close(value, torch.tensor([0.5]))


def test_the_reward_needs_a_reset():
    reward = ImprovementReward(config=ImprovementRewardConfig())
    with pytest.raises(RuntimeError, match="reset"):
        reward(record([1.0], [1.0]))
