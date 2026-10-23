import math

import pytest
import torch

from autotsaugment.core.metrics import (
    aggregate,
    pinball_loss,
    pinball_sum,
    seasonal_naive,
    skill_scores,
    target_abs_sum,
    wql_per_dataset,
)


def test_pinball_median_is_half_absolute_error():
    quantiles = torch.zeros(2, 3, 1)
    target = torch.tensor([[1.0, -2.0, 4.0], [0.0, 0.5, -0.5]])
    loss = pinball_loss(quantiles, target, torch.tensor([0.5]))
    torch.testing.assert_close(loss.squeeze(-1), 0.5 * target.abs())


def test_pinball_asymmetry():
    levels = torch.tensor([0.1, 0.9])
    target = torch.tensor([[1.0]])
    under = pinball_loss(torch.zeros(1, 1, 2), target, levels)[0, 0]  # forecast below target
    over = pinball_loss(torch.full((1, 1, 2), 2.0), target, levels)[0, 0]  # forecast above target
    torch.testing.assert_close(under, torch.tensor([0.1, 0.9]))
    torch.testing.assert_close(over, torch.tensor([0.9, 0.1]))


def test_nan_target_gives_zero_loss_and_zero_scale():
    target = torch.tensor([[1.0, float("nan")]])
    loss = pinball_loss(torch.zeros(1, 2, 3), target, torch.tensor([0.1, 0.5, 0.9]))
    assert loss[0, 1].sum() == 0.0
    assert target_abs_sum(target).item() == 1.0


def test_wql_perfect_forecast_is_zero_and_naive_skill_is_one():
    levels = torch.tensor([0.25, 0.5, 0.75])
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    perfect = target.unsqueeze(-1).expand(-1, -1, 3)
    index = torch.tensor([0, 0, 1])
    model = wql_per_dataset(
        pinball_sum(perfect, target, levels), target_abs_sum(target), index, 2, 3
    )
    torch.testing.assert_close(model, torch.zeros(2))
    naive = perfect + 1.0
    naive_wql = wql_per_dataset(
        pinball_sum(naive, target, levels), target_abs_sum(target), index, 2, 3
    )
    torch.testing.assert_close(skill_scores(naive_wql, naive_wql), torch.ones(2))


def test_wql_matches_definition():
    levels = torch.tensor([0.5])
    target = torch.tensor([[2.0, 2.0]])
    forecast = torch.zeros(1, 2, 1)
    wql = wql_per_dataset(
        pinball_sum(forecast, target, levels), target_abs_sum(target), torch.tensor([0]), 1, 1
    )
    # 2 * sum(0.5 * |2|) / (1 * sum|2|) = 2 * 2 / 4 = 1.0
    torch.testing.assert_close(wql, torch.tensor([1.0]))


def test_aggregate_is_geometric_mean():
    skill = torch.tensor([1.0, 4.0])
    assert math.isclose(aggregate(skill).item(), 2.0, rel_tol=1e-6)
    assert aggregate(torch.ones(5)).item() == pytest.approx(1.0)


def test_seasonal_naive_repeats_last_season():
    context = torch.arange(10.0).unsqueeze(0)
    forecast = seasonal_naive(context, horizon=5, season=3)
    torch.testing.assert_close(forecast, torch.tensor([[7.0, 8.0, 9.0, 7.0, 8.0]]))
    last = seasonal_naive(context, horizon=3, season=1)
    torch.testing.assert_close(last, torch.tensor([[9.0, 9.0, 9.0]]))
    clipped = seasonal_naive(context, horizon=2, season=100)
    torch.testing.assert_close(clipped, torch.tensor([[0.0, 1.0]]))
