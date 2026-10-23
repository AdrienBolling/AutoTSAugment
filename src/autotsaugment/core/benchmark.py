"""Benchmark contract: fixed windows of real data, scored on the device."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from autotsaugment.core.config import ComponentConfig
from autotsaugment.core.typechecking import Context, DatasetIndex, PerWindow, Target


@dataclass(frozen=True)
class WindowSet:
    """Fixed forecast windows from one split, with the naive scores needed for skill.

    Attributes:
        context: Observed context, shape ``(n, c)``, NaN for missing values and for the
            left padding of datasets with a shorter context.
        target: Values to forecast, shape ``(n, h)``, NaN for missing values and for the
            right padding of datasets with a shorter horizon. NaN gives zero loss.
        dataset_index: Dataset id of each window, shape ``(n,)``, in ``[0, d)``.
        naive_loss_sum: Pinball loss of the seasonal-naive forecast summed over horizon and
            levels, per window, shape ``(n,)``. Lets any subset compute its naive WQL.
        abs_sum: ``sum |y|`` over the horizon per window, shape ``(n,)``.
        dataset_names: Name of each dataset, length d.
        domains: Domain tag of each dataset, length d, for example ``"Energy"``.
    """

    context: Tensor
    target: Tensor
    dataset_index: Tensor
    naive_loss_sum: Tensor
    abs_sum: Tensor
    dataset_names: tuple[str, ...]
    domains: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate shapes and ids."""
        n = self.context.shape[0]
        for name in ("target", "dataset_index", "naive_loss_sum", "abs_sum"):
            if getattr(self, name).shape[0] != n:
                msg = f"{name} has {getattr(self, name).shape[0]} rows, expected {n}"
                raise ValueError(msg)
        if len(self.dataset_names) != len(self.domains):
            msg = "dataset_names and domains must have the same length"
            raise ValueError(msg)

    @property
    def n_windows(self) -> int:
        """Return the number of windows."""
        return int(self.context.shape[0])

    @property
    def n_datasets(self) -> int:
        """Return the number of datasets, d."""
        return len(self.dataset_names)

    @property
    def context_length(self) -> int:
        """Return c."""
        return int(self.context.shape[1])

    @property
    def horizon(self) -> int:
        """Return h."""
        return int(self.target.shape[1])

    @property
    def device(self) -> torch.device:
        """Return the device of the tensors."""
        return self.context.device

    def to(self, device: torch.device | str) -> "WindowSet":
        """Return a copy with every tensor on ``device``."""
        return WindowSet(
            context=self.context.to(device),
            target=self.target.to(device),
            dataset_index=self.dataset_index.to(device),
            naive_loss_sum=self.naive_loss_sum.to(device),
            abs_sum=self.abs_sum.to(device),
            dataset_names=self.dataset_names,
            domains=self.domains,
        )

    def select(self, rows: Tensor) -> "WindowSet":
        """Return the windows at ``rows`` (a 1-D index tensor), same datasets."""
        return WindowSet(
            context=self.context[rows],
            target=self.target[rows],
            dataset_index=self.dataset_index[rows],
            naive_loss_sum=self.naive_loss_sum[rows],
            abs_sum=self.abs_sum[rows],
            dataset_names=self.dataset_names,
            domains=self.domains,
        )

    def subset(self, per_dataset: int, *, generator: torch.Generator) -> "WindowSet":
        """Return at most ``per_dataset`` random windows of each dataset.

        Datasets with fewer windows keep all of them. The result keeps the dataset order.
        """
        keep: list[Tensor] = []
        for dataset in range(self.n_datasets):
            rows = torch.nonzero(self.dataset_index == dataset, as_tuple=False).squeeze(-1)
            if rows.numel() > per_dataset:
                perm = torch.randperm(rows.numel(), generator=generator, device=rows.device)
                rows = rows[perm[:per_dataset]]
            keep.append(rows)
        return self.select(torch.cat(keep))


class BenchmarkConfig(ComponentConfig):
    """Config shared by all benchmarks.

    Window lengths depend on the sampling frequency, as in GIFT-Eval. Every window set pads the
    context on the left and the target on the right with NaN up to the maximum lengths, so one
    tensor holds all datasets and NaN targets contribute zero loss.

    Attributes:
        context_by_freq: Context steps per pandas frequency alias.
        horizon_by_freq: Horizon steps per frequency alias (GIFT-Eval short term).
        season_by_freq: Season length per frequency alias, for the seasonal-naive baseline.
        reward_windows_per_dataset: Windows kept in the reward set, per dataset.
        comparison_windows_per_dataset: Windows kept in the comparison set, per dataset.
        reward_fraction: Share of the non-test part of each series that forms the reward
            split; the rest is the comparison split. The official test part is never used.
    """

    context_by_freq: dict[str, int] = {
        "10S": 512,
        "5T": 512,
        "10T": 512,
        "15T": 512,
        "H": 512,
        "D": 256,
        "W": 104,
        "M": 60,
        "Q": 24,
        "A": 12,
    }
    horizon_by_freq: dict[str, int] = {
        "10S": 48,
        "5T": 48,
        "10T": 48,
        "15T": 48,
        "H": 48,
        "D": 30,
        "W": 8,
        "M": 12,
        "Q": 8,
        "A": 6,
    }
    season_by_freq: dict[str, int] = {
        "10S": 360,
        "5T": 288,
        "10T": 144,
        "15T": 96,
        "H": 24,
        "D": 7,
        "W": 52,
        "M": 12,
        "Q": 4,
        "A": 1,
    }
    reward_windows_per_dataset: int = 512
    comparison_windows_per_dataset: int = 512
    reward_fraction: float = 0.85

    @property
    def max_context_length(self) -> int:
        """Return the context length every window set is padded to."""
        return max(self.context_by_freq.values())

    @property
    def max_horizon(self) -> int:
        """Return the horizon every window set is padded to."""
        return max(self.horizon_by_freq.values())


class Benchmark[ConfigT: BenchmarkConfig](ABC):
    """Real benchmark data as two fixed window sets on the device.

    Implementations take ``__init__(self, *, config, device)`` and build the window sets once,
    from a local cache, with the naive scores precomputed.
    """

    config: ConfigT

    @property
    @abstractmethod
    def reward_windows(self) -> WindowSet:
        """Return the windows scored every step, from the reward split."""

    @property
    @abstractmethod
    def comparison_windows(self) -> WindowSet:
        """Return the windows scored once per episode, from the comparison split."""

    @property
    def dataset_names(self) -> tuple[str, ...]:
        """Return the dataset names, in index order."""
        return self.reward_windows.dataset_names

    @property
    def domains(self) -> tuple[str, ...]:
        """Return the domain tag of each dataset, in index order."""
        return self.reward_windows.domains


def typed_context(window_set: WindowSet) -> tuple[Context, Target, DatasetIndex, PerWindow]:
    """Return the tensors of ``window_set`` with their shape-checked aliases, for scorers."""
    return (
        window_set.context,
        window_set.target,
        window_set.dataset_index,
        window_set.naive_loss_sum,
    )
