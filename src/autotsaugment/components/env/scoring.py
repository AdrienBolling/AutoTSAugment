"""Skill scores of one forecaster on one window set.

The scorer forwards the windows in batches under ``torch.no_grad``, with bf16 autocast when the
model runs on CUDA, and reduces the losses with :mod:`autotsaugment.core.metrics`. It never
synchronizes with the host.
"""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from autotsaugment.core.benchmark import WindowSet
from autotsaugment.core.metrics import aggregate, pinball_sum, skill_scores, wql_per_dataset
from autotsaugment.core.typechecking import (
    Context,
    PerDataset,
    QuantileLevels,
    Quantiles,
    Scalar,
    checked,
)


@runtime_checkable
class Forecaster(Protocol):
    """The part of a tsmodel that :func:`score` uses. Every ``TSModel`` satisfies it."""

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""

    @property
    def quantile_levels(self) -> QuantileLevels:
        """Return the quantile levels of :meth:`forecast`, shape ``(q,)``."""

    def forecast(self, context: Context, horizon: int) -> Quantiles:
        """Return quantile forecasts without gradient, shape ``(n, horizon, q)``."""


@checked
def score(model: Forecaster, windows: WindowSet, *, batch_size: int) -> tuple[PerDataset, Scalar]:
    """Return the skill of ``model`` per dataset of ``windows``, and the aggregate skill.

    The skill of a dataset is the WQL of the model divided by the WQL of the seasonal-naive
    forecaster on the same windows. The naive WQL comes from ``windows.naive_loss_sum`` and
    ``windows.abs_sum`` through the same :func:`wql_per_dataset` call as the model WQL. Both
    calls use the number of levels of ``model.quantile_levels``, so the naive loss of the
    window set must come from the same levels, ``windows.quantile_levels``. Then a naive
    forecaster scores exactly 1.0. The function does not compare the levels, because the
    comparison reads them on the host; ``CurriculumEnv`` compares them once, at construction.

    Args:
        model: The forecaster to score. Its forecast must have the horizon of ``windows``.
        windows: The windows to forecast, on the device of the model.
        batch_size: Windows per forward pass.

    Returns:
        The skill per dataset, shape ``(d,)``, and its geometric mean over datasets, shape
        ``()``. Lower is better; 1.0 is the seasonal-naive forecaster.

    Raises:
        ValueError: If ``batch_size`` is less than 1 or ``windows`` holds no window.
    """
    if batch_size < 1:
        msg = f"batch_size must be at least 1, got {batch_size}"
        raise ValueError(msg)
    if windows.n_windows == 0:
        msg = "cannot score an empty window set"
        raise ValueError(msg)
    levels = model.quantile_levels.float()
    n_levels = int(levels.shape[0])
    loss_sums: list[Tensor] = []
    use_bf16 = model.device.type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=use_bf16),
    ):
        for start in range(0, windows.n_windows, batch_size):
            stop = start + batch_size
            quantiles = model.forecast(windows.context[start:stop], windows.horizon)
            target = windows.target[start:stop]
            loss_sums.append(pinball_sum(quantiles.float(), target, levels))
    model_wql = wql_per_dataset(
        torch.cat(loss_sums), windows.abs_sum, windows.dataset_index, windows.n_datasets, n_levels
    )
    naive_wql = wql_per_dataset(
        windows.naive_loss_sum, windows.abs_sum, windows.dataset_index, windows.n_datasets, n_levels
    )
    skill = skill_scores(model_wql, naive_wql)
    return skill, aggregate(skill)
