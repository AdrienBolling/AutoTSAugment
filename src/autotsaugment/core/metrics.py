"""Metric definitions shared by the reward, the scorer and the trainer.

All functions are pure torch, run on any device and never synchronize with the host.
Definitions follow Chronos and GIFT-Eval:

- pinball loss at level ``q``: ``max(q * (y - f), (q - 1) * (y - f))``
- WQL of one dataset: ``2 * sum(pinball over windows, horizon, levels) / (Q * sum |y|)``
- skill: ``WQL_model / WQL_seasonal_naive`` on the same windows
- aggregate: geometric mean of the skill over datasets
"""

import torch

from autotsaugment.core.typechecking import (
    Context,
    DatasetIndex,
    PerDataset,
    PerWindow,
    QuantileLevels,
    Quantiles,
    Scalar,
    Target,
    checked,
)


@checked
def pinball_loss(quantiles: Quantiles, target: Target, levels: QuantileLevels) -> Quantiles:
    """Return the pinball loss per window, horizon step and quantile level.

    Args:
        quantiles: Forecast quantiles, shape ``(n, h, q)``.
        target: Observed values, shape ``(n, h)``.
        levels: Quantile levels in ``(0, 1)``, shape ``(q,)``.

    Returns:
        The loss, shape ``(n, h, q)``. Missing targets (NaN) give a loss of zero.
    """
    error = target.unsqueeze(-1) - quantiles
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    return torch.nan_to_num(loss, nan=0.0)


@checked
def pinball_sum(quantiles: Quantiles, target: Target, levels: QuantileLevels) -> PerWindow:
    """Return the pinball loss summed over horizon and levels, per window, shape ``(n,)``."""
    return pinball_loss(quantiles, target, levels).sum(dim=(-1, -2))


@checked
def target_abs_sum(target: Target) -> PerWindow:
    """Return ``sum |y|`` over the horizon per window, NaN counted as zero, shape ``(n,)``."""
    return torch.nan_to_num(target, nan=0.0).abs().sum(dim=-1)


@checked
def wql_per_dataset(
    loss_sum: PerWindow,
    abs_sum: PerWindow,
    dataset_index: DatasetIndex,
    n_datasets: int,
    n_levels: int,
) -> PerDataset:
    """Return the weighted quantile loss of each dataset, shape ``(d,)``.

    Args:
        loss_sum: Pinball loss summed over horizon and levels, per window.
        abs_sum: ``sum |y|`` over the horizon, per window.
        dataset_index: Dataset id of each window, in ``[0, n_datasets)``.
        n_datasets: Number of datasets, d.
        n_levels: Number of quantile levels, Q.

    Returns:
        ``2 * sum(loss) / (Q * sum |y|)`` per dataset. A dataset without windows gives NaN.
    """
    zeros = torch.zeros(n_datasets, dtype=loss_sum.dtype, device=loss_sum.device)
    loss = zeros.scatter_add(0, dataset_index, loss_sum)
    scale = zeros.scatter_add(0, dataset_index, abs_sum)
    return 2.0 * loss / (n_levels * scale)


@checked
def skill_scores(model_wql: PerDataset, naive_wql: PerDataset) -> PerDataset:
    """Return ``model_wql / naive_wql`` per dataset. Lower is better, 1.0 is the naive forecaster."""
    return model_wql / naive_wql


@checked
def aggregate(skill: PerDataset) -> Scalar:
    """Return the geometric mean of the skill scores over datasets."""
    return torch.exp(torch.log(skill).mean())


@checked
def seasonal_naive(context: Context, horizon: int, season: int) -> Target:
    """Return the seasonal-naive point forecast, shape ``(n, horizon)``.

    The forecast at step ``t`` of the horizon repeats the context value one season back,
    ``context[-season + (t mod season)]``. ``season`` is clipped to the context length.

    Args:
        context: Observed context, shape ``(n, c)``.
        horizon: Number of steps to forecast, h.
        season: Season length in steps, at least 1.
    """
    length = context.shape[-1]
    season = max(1, min(season, length))
    steps = torch.arange(horizon, device=context.device)
    index = length - season + (steps % season)
    return context[:, index]
