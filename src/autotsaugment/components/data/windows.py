"""Splits, window cutting, padding, naive scores and ``WindowSet`` assembly.

Splits, per series of length ``T``:

- ``test_start = floor(0.9 * T)``. ``[test_start, T)`` is the official GIFT-Eval test part.
  No window reads it.
- reward split ``[0, reward_end)`` with ``reward_end = floor(reward_fraction * test_start)``.
- comparison split ``[reward_end, test_start)``.

Both floors use integer arithmetic on the decimal value of the fractions, so they are exact.

Windows: a window has a horizon of ``h`` steps that starts at index ``t`` and a context of at
most ``c`` steps that ends at ``t``. The horizon lies fully inside its split. The context reads
``[max(context_start, t - c), t)``, where ``context_start`` is the first index the split lets a
context read:

- reward split: ``0``, so the whole reward window lies inside the reward split.
- comparison split: ``0`` when ``comparison_context_may_cross`` is set (the default), so the
  context may read the end of the reward split, as a GIFT-Eval test window reads the training
  part; ``reward_end`` otherwise, so the whole window lies inside the comparison split.

No window reads ``[test_start, T)``. The context can be shorter than ``c`` near
``context_start``. Candidate horizons are anchored at the end of the split, ``t = split_end - h -
j * stride`` for ``j = 0, 1, ...``, and need ``t >= split_start`` and ``t - context_start >=
season``. A candidate is valid when its target has at least one observed value and its context
at least ``season`` observed values. A dataset with fewer than ``min_windows`` valid windows in
a split raises :class:`TooFewWindowsError`.

Padding: the context is right-aligned in a NaN row of ``max_context_length`` columns and the
target is left-aligned in a NaN row of ``max_horizon`` columns, so one tensor holds all
frequencies.

Naive scores: the seasonal-naive forecast repeats the last season of the context, after a
forward fill of the context NaN (the stored context keeps its NaN). With at least ``season``
observed values in the context, the forward fill leaves no NaN in the last season, so the
naive forecast is finite.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from autotsaugment.components.data.arrow import SeriesArrays
from autotsaugment.core.benchmark import WindowSet
from autotsaugment.core.metrics import pinball_sum, seasonal_naive, target_abs_sum
from autotsaugment.core.seeding import derive_seed, make_generator
from autotsaugment.core.typechecking import (
    Context,
    PerWindow,
    QuantileLevels,
    Target,
    checked,
)

REWARD: Final = "reward"
COMPARISON: Final = "comparison"
SPLITS: Final = (REWARD, COMPARISON)
USABLE_FRACTION: Final = Fraction(9, 10)  # the last 10% of each series is the official test part

WINDOW_FIELDS: Final = ("context", "target", "naive_loss_sum", "abs_sum", "series", "horizon_start")

FlatValues = Float[Tensor, "total"]
WindowIndex = Int64[Tensor, "n"]


class TooFewWindowsError(ValueError):
    """A split of a dataset has fewer valid windows than the minimum, possibly none."""


@dataclass(frozen=True)
class WindowSpec:
    """Window lengths of one frequency.

    Attributes:
        context_length: Maximum context steps, c.
        horizon: Horizon steps, h.
        season: Season length of the seasonal-naive forecast, and minimum number of observed
            context values.
        stride: Steps between the horizon starts of two candidate windows.
        max_context_length: Width of the padded context.
        max_horizon: Width of the padded target.
    """

    context_length: int
    horizon: int
    season: int
    stride: int
    max_context_length: int
    max_horizon: int

    def __post_init__(self) -> None:
        """Validate the lengths."""
        if min(self.context_length, self.horizon, self.season, self.stride) < 1:
            msg = f"every length of {self} must be at least 1"
            raise ValueError(msg)
        if self.context_length > self.max_context_length or self.horizon > self.max_horizon:
            msg = f"{self}: context or horizon exceeds the padded width"
            raise ValueError(msg)
        if self.season > self.context_length:
            msg = f"{self}: the season is longer than the context"
            raise ValueError(msg)


def _exact_fraction(value: float | Fraction) -> Fraction:
    """Return ``value`` as the fraction of its decimal text, for example 0.85 -> 17/20."""
    if isinstance(value, Fraction):
        return value
    return Fraction(repr(value)).limit_denominator(10**6)


def usable_ends(lengths: Tensor) -> Tensor:
    """Return ``floor(0.9 * T)`` per series, the first index of the official test part."""
    return lengths * USABLE_FRACTION.numerator // USABLE_FRACTION.denominator


@dataclass(frozen=True)
class SplitBounds:
    """Index bounds of one split, per series, each int64 ``(n_series,)``.

    Attributes:
        start: First index of a horizon.
        end: End (exclusive) of the horizons.
        context_start: First index a context may read.
    """

    start: Tensor
    end: Tensor
    context_start: Tensor


def split_bounds(
    lengths: Tensor, reward_fraction: float, *, comparison_context_may_cross: bool
) -> dict[str, SplitBounds]:
    """Return the index bounds of each split, per series.

    Args:
        lengths: Series lengths, int64, shape ``(n_series,)``.
        reward_fraction: Share of ``[0, test_start)`` that forms the reward split.
        comparison_context_may_cross: Let comparison contexts read the reward split.

    Returns:
        ``{"reward": ..., "comparison": ...}``.
    """
    test_start = usable_ends(lengths)
    fraction = _exact_fraction(reward_fraction)
    reward_end = test_start * fraction.numerator // fraction.denominator
    zeros = torch.zeros_like(lengths)
    return {
        REWARD: SplitBounds(start=zeros, end=reward_end, context_start=zeros),
        COMPARISON: SplitBounds(
            start=reward_end,
            end=test_start,
            context_start=zeros if comparison_context_may_cross else reward_end,
        ),
    }


def candidate_windows(bounds: SplitBounds, spec: WindowSpec) -> tuple[Tensor, Tensor]:
    """Return every candidate window of a split, before the validity filter.

    Args:
        bounds: The bounds of the split.
        spec: The window lengths.

    Returns:
        ``(series, horizon_start)``, int64, one entry per candidate, sorted by series, then
        by horizon start.
    """
    lowest = torch.maximum(bounds.start, bounds.context_start + spec.season)
    span = bounds.end - spec.horizon - lowest
    count = torch.where(span >= 0, span // spec.stride + 1, torch.zeros_like(span))
    series = torch.repeat_interleave(torch.arange(count.numel()), count)
    first = torch.cumsum(count, dim=0) - count
    rank = torch.arange(series.numel()) - first[series]
    horizon_start = bounds.end[series] - spec.horizon - (count[series] - 1 - rank) * spec.stride
    return series, horizon_start


def valid_windows(
    data: SeriesArrays, bounds: SplitBounds, spec: WindowSpec
) -> tuple[Tensor, Tensor]:
    """Return the candidate windows of a split that pass the validity filter.

    A window is valid when its target has at least one observed value and its context at
    least ``spec.season`` observed values.

    Args:
        data: The series of the dataset.
        bounds: The bounds of the split.
        spec: The window lengths.

    Returns:
        ``(series, horizon_start)`` of the valid windows, in candidate order.
    """
    series, horizon_start = candidate_windows(bounds, spec)
    total = data.values.numel()
    count_type = torch.int32 if total < torch.iinfo(torch.int32).max else torch.int64
    observed = torch.zeros(total + 1, dtype=count_type)
    observed[1:] = torch.cumsum(~torch.isnan(data.values), dim=0, dtype=count_type)
    first = data.offsets[:-1][series]
    context_start = torch.maximum(bounds.context_start[series], horizon_start - spec.context_length)
    context_observed = observed[first + horizon_start] - observed[first + context_start]
    target_observed = (
        observed[first + horizon_start + spec.horizon] - observed[first + horizon_start]
    )
    keep = (context_observed >= spec.season) & (target_observed > 0)
    return series[keep], horizon_start[keep]


def subsample(n: int, k: int, generator: torch.Generator) -> Tensor:
    """Return ``min(n, k)`` sorted distinct indices in ``[0, n)``, drawn with ``generator``."""
    if n <= k:
        return torch.arange(n)
    return torch.randperm(n, generator=generator)[:k].sort().values


@checked
def cut_windows(
    values: FlatValues,
    first: WindowIndex,
    horizon_start: WindowIndex,
    context_start: WindowIndex,
    spec: WindowSpec,
) -> tuple[Context, Target]:
    """Return the padded context and target of each window.

    Args:
        values: Flat values of all series.
        first: Index of the first value of the window's series in ``values``.
        horizon_start: First horizon step of each window, in series coordinates.
        context_start: First index the context may read, in series coordinates.
        spec: The window lengths.

    Returns:
        The context ``(n, max_context_length)``, right-aligned, NaN before ``context_start``
        and in the padding, and the target ``(n, max_horizon)``, left-aligned, NaN padded.
    """
    n = horizon_start.numel()
    steps = torch.arange(-spec.context_length, 0)
    position = horizon_start[:, None] + steps[None, :]
    inside = position >= context_start[:, None]
    gathered = values[first[:, None] + position.clamp(min=0)]
    context = torch.full((n, spec.max_context_length), float("nan"), dtype=values.dtype)
    context[:, spec.max_context_length - spec.context_length :] = torch.where(
        inside, gathered, torch.full_like(gathered, float("nan"))
    )
    target = torch.full((n, spec.max_horizon), float("nan"), dtype=values.dtype)
    horizon_steps = torch.arange(spec.horizon)
    target[:, : spec.horizon] = values[(first + horizon_start)[:, None] + horizon_steps[None, :]]
    return context, target


@checked
def forward_fill(context: Context) -> Context:
    """Return ``context`` with each NaN replaced by the last observed value before it.

    NaN values before the first observed value of a row stay NaN.
    """
    columns = torch.arange(context.shape[-1], device=context.device).expand_as(context)
    last = torch.where(torch.isnan(context), torch.zeros_like(columns), columns)
    last = torch.cummax(last, dim=-1).values
    return torch.gather(context, -1, last)


@checked
def naive_scores(
    context: Context, target: Target, levels: QuantileLevels, season: int
) -> tuple[PerWindow, PerWindow]:
    """Return the seasonal-naive pinball loss sum and ``sum |y|`` per window.

    Args:
        context: Padded contexts, right-aligned.
        target: Padded targets, left-aligned; NaN gives zero loss.
        levels: The quantile levels of the pinball loss.
        season: The season length.

    Returns:
        ``(naive_loss_sum, abs_sum)``: the pinball loss of the seasonal-naive point forecast,
        used as every quantile, summed over horizon and levels, and ``sum |y|`` over the horizon.
    """
    point = seasonal_naive(forward_fill(context), target.shape[-1], season)
    quantiles = point.unsqueeze(-1).expand(-1, -1, levels.numel())
    return pinball_sum(quantiles, target, levels), target_abs_sum(target)


@dataclass(frozen=True)
class SplitWindows:
    """The windows of one dataset in one split, on the CPU.

    Attributes:
        context: Padded contexts, float32 ``(n, max_context_length)``.
        target: Padded targets, float32 ``(n, max_horizon)``.
        naive_loss_sum: Seasonal-naive pinball loss sum per window, float32 ``(n,)``.
        abs_sum: ``sum |y|`` per window, float32 ``(n,)``.
        series: Index of the window's series in the dataset, int64 ``(n,)``.
        horizon_start: First horizon step in series coordinates, int64 ``(n,)``.
    """

    context: Tensor
    target: Tensor
    naive_loss_sum: Tensor
    abs_sum: Tensor
    series: Tensor
    horizon_start: Tensor

    @property
    def n_windows(self) -> int:
        """Return the number of windows."""
        return int(self.context.shape[0])

    def as_dict(self) -> dict[str, Tensor]:
        """Return the tensors by field name."""
        return {name: getattr(self, name) for name in WINDOW_FIELDS}

    @classmethod
    def from_dict(cls, tensors: dict[str, Tensor]) -> "SplitWindows":
        """Build from the output of :meth:`as_dict`."""
        missing = set(WINDOW_FIELDS) - set(tensors)
        if missing:
            msg = f"missing window fields: {sorted(missing)}"
            raise KeyError(msg)
        return cls(
            context=tensors["context"],
            target=tensors["target"],
            naive_loss_sum=tensors["naive_loss_sum"],
            abs_sum=tensors["abs_sum"],
            series=tensors["series"],
            horizon_start=tensors["horizon_start"],
        )


def check_window_counts(
    dataset_id: str, counts: Mapping[str, int], minimum: int, detail: str = ""
) -> None:
    """Raise when a split has fewer than ``minimum`` windows.

    Args:
        dataset_id: The id ``"<name>/<freq>"``.
        counts: The number of valid windows of each split.
        minimum: The minimum number of windows per split.
        detail: Extra text for the message.

    Raises:
        TooFewWindowsError: A split has fewer than ``minimum`` windows.
    """
    short = [split for split in SPLITS if counts[split] < minimum]
    if short:
        msg = (
            f"{dataset_id}: too few windows in the {' and '.join(short)} split "
            f"(reward {counts[REWARD]}, comparison {counts[COMPARISON]}, "
            f"min_windows_per_split {minimum}){detail}. Remove the dataset, or change "
            "min_windows_per_split or reward_fraction."
        )
        raise TooFewWindowsError(msg)


def _cut_split(
    data: SeriesArrays,
    spec: WindowSpec,
    windows: tuple[Tensor, Tensor],
    *,
    bounds: SplitBounds,
    levels: Tensor,
) -> SplitWindows:
    """Pad and score the selected windows of one split."""
    series, horizon_start = windows
    context, target = cut_windows(
        data.values, data.offsets[:-1][series], horizon_start, bounds.context_start[series], spec
    )
    naive_loss_sum, abs_sum = naive_scores(context, target, levels, spec.season)
    return SplitWindows(
        context=context,
        target=target,
        naive_loss_sum=naive_loss_sum,
        abs_sum=abs_sum,
        series=series,
        horizon_start=horizon_start,
    )


def build_dataset_windows(
    data: SeriesArrays,
    spec: WindowSpec,
    *,
    dataset_id: str,
    reward_fraction: float,
    comparison_context_may_cross: bool,
    windows_per_split: Mapping[str, int],
    min_windows: int,
    levels: Tensor,
    seed: int,
) -> dict[str, SplitWindows]:
    """Build the reward and comparison windows of one dataset.

    The subsample of each split uses a generator seeded with
    ``derive_seed(seed, dataset_id, split)``, so the windows of one dataset do not depend on
    the other datasets of the config.

    Args:
        data: The series of the dataset.
        spec: The window lengths of the dataset's frequency.
        dataset_id: The id ``"<name>/<freq>"``.
        reward_fraction: Share of the usable part that forms the reward split.
        comparison_context_may_cross: Let comparison contexts read the reward split.
        windows_per_split: Maximum number of windows kept per split.
        min_windows: Minimum number of valid windows per split.
        levels: Quantile levels of the naive pinball loss.
        seed: The root seed of the subsample.

    Returns:
        ``{"reward": ..., "comparison": ...}``.

    Raises:
        TooFewWindowsError: A split has fewer than ``min_windows`` valid windows.
    """
    bounds = split_bounds(
        data.lengths, reward_fraction, comparison_context_may_cross=comparison_context_may_cross
    )
    valid = {split: valid_windows(data, bounds[split], spec) for split in SPLITS}
    lengths = data.lengths.tolist() or [0]
    check_window_counts(
        dataset_id,
        {split: int(valid[split][0].numel()) for split in SPLITS},
        min_windows,
        f"; series lengths {min(lengths)}..{max(lengths)}, a window needs {spec.season} "
        f"observed context steps and {spec.horizon} horizon steps",
    )
    result: dict[str, SplitWindows] = {}
    for split in SPLITS:
        series, horizon_start = valid[split]
        generator = make_generator("cpu", derive_seed(seed, dataset_id, split))
        keep = subsample(series.numel(), windows_per_split[split], generator)
        result[split] = _cut_split(
            data, spec, (series[keep], horizon_start[keep]), bounds=bounds[split], levels=levels
        )
    return result


def assemble_window_set(
    parts: Sequence[SplitWindows],
    *,
    dataset_names: tuple[str, ...],
    domains: tuple[str, ...],
    device: torch.device,
) -> WindowSet:
    """Concatenate the windows of each dataset into one ``WindowSet`` on ``device``.

    Args:
        parts: The windows of each dataset, in dataset index order.
        dataset_names: The dataset ids, in the same order.
        domains: The domain of each dataset, in the same order.
        device: The device of the result.

    Returns:
        The window set; ``dataset_index`` follows the order of ``parts``.
    """
    counts = torch.tensor([part.n_windows for part in parts], dtype=torch.int64)
    window_set = WindowSet(
        context=torch.cat([part.context for part in parts]),
        target=torch.cat([part.target for part in parts]),
        dataset_index=torch.repeat_interleave(torch.arange(len(parts)), counts),
        naive_loss_sum=torch.cat([part.naive_loss_sum for part in parts]),
        abs_sum=torch.cat([part.abs_sum for part in parts]),
        dataset_names=dataset_names,
        domains=domains,
    )
    return window_set.to(device)
