"""GIFT-Eval benchmark: fixed reward and comparison windows of real series on the device.

Build order, once per process:

1. The config validates every dataset id before any data loads.
2. Each dataset id with a cache file under ``root/cache`` loads from it: no network and no
   Arrow read.
3. The other dataset ids read their shards from ``root/hub/<repo>/<revision>``. Shards that
   are not there come from the Hub, after a size plan that must fit ``max_total_gb``.
4. The windows of each dataset are cut, scored and written to the cache.
5. A dataset with fewer than ``min_windows_per_split`` windows in a split raises
   :class:`~autotsaugment.components.data.windows.TooFewWindowsError`, also from the cache.
6. The per-dataset windows are concatenated in config order and moved to the device.
"""

import itertools
import logging
from pathlib import Path
from typing import Final, Self

import torch
from pydantic import Field, field_validator, model_validator

from autotsaugment.components.data.arrow import read_series
from autotsaugment.components.data.cache import cache_key, cache_path, load_windows, save_windows
from autotsaugment.components.data.download import (
    DatasetId,
    download,
    find_local_sources,
    hub_dir,
    normalize_freq,
    parse_dataset_id,
    plan_download,
)
from autotsaugment.components.data.windows import (
    COMPARISON,
    REWARD,
    SPLITS,
    USABLE_FRACTION,
    SplitWindows,
    WindowSpec,
    assemble_window_set,
    build_dataset_windows,
    check_window_counts,
)
from autotsaugment.core.benchmark import Benchmark, BenchmarkConfig, WindowSet

_log = logging.getLogger(__name__)

# GIFT-Eval domain of each dataset, keyed by Hub name.
# Source: the "domain" field of notebooks/dataset_properties.json in
# https://github.com/SalesforceAIResearch/gift-eval (commit cfaefe09c6bb, read 2026-10-23).
# That file uses lower-case names and maps saugeenday -> saugeen and drops the
# "_with_missing" suffix (the "pretty_names" table of notebooks/naive.ipynb).
DOMAINS: Final[dict[str, str]] = {
    "LOOP_SEATTLE": "Transport",
    "M_DENSE": "Transport",
    "SZ_TAXI": "Transport",
    "bitbrains_fast_storage": "Web/CloudOps",
    "bitbrains_rnd": "Web/CloudOps",
    "bizitobs_application": "Web/CloudOps",
    "bizitobs_l2c": "Web/CloudOps",
    "bizitobs_service": "Web/CloudOps",
    "car_parts_with_missing": "Sales",
    "covid_deaths": "Healthcare",
    "electricity": "Energy",
    "ett1": "Energy",
    "ett2": "Energy",
    "hierarchical_sales": "Sales",
    "hospital": "Healthcare",
    "jena_weather": "Nature",
    "kdd_cup_2018_with_missing": "Nature",
    "m4_daily": "Econ/Fin",
    "m4_hourly": "Econ/Fin",
    "m4_monthly": "Econ/Fin",
    "m4_quarterly": "Econ/Fin",
    "m4_weekly": "Econ/Fin",
    "m4_yearly": "Econ/Fin",
    "restaurant": "Sales",
    "saugeenday": "Nature",
    "solar": "Energy",
    "temperature_rain_with_missing": "Nature",
    "us_births": "Healthcare",
}

# The 21 quantile levels of Timestral: 0.01, 0.05, 0.10, ..., 0.95, 0.99.
TIMESTRAL_QUANTILE_LEVELS: Final = (0.01, *(round(0.05 * i, 2) for i in range(1, 20)), 0.99)


class GiftEvalBenchmarkConfig(BenchmarkConfig):
    """Config of :class:`GiftEvalBenchmark`.

    Attributes:
        root: Data root. Shards go to ``root/hub/<repo>/<revision>`` (``<repo>`` is
            ``repo_id`` with ``/`` replaced by ``--``), window caches to ``root/cache``.
        repo_id: Hugging Face dataset repository.
        revision: Commit sha (or branch or tag) of the repository; None is the main branch.
            The YAML files pin a commit sha, so the data cannot change under a run.
        max_total_gb: Budget for everything under ``root``, in GB (10^9 bytes).
        datasets: Dataset ids ``"<name>/<freq>"``; the order is the dataset index order. The
            frequency is a pandas alias or a key of ``context_by_freq``.
        stride_in_horizons: Steps between two candidate windows, in horizons of the frequency.
        comparison_context_may_cross: Let the context of a comparison window read the end of
            the reward split. Only its horizon must lie in the comparison split. False keeps
            the whole comparison window inside the comparison split.
        min_windows_per_split: Minimum number of valid windows of each dataset in each split.
            Fewer raises an error.
        seed: Seed of the window subsample. Keep it fixed across runs so every run scores the
            same windows.
        quantile_levels: Levels of the naive pinball loss, the tsmodel's levels.
    """

    root: Path = Path("data/gifteval")
    repo_id: str = "Salesforce/GiftEval"
    revision: str | None = None
    max_total_gb: float = Field(default=10.0, gt=0.0)
    datasets: tuple[str, ...] = Field(min_length=1)
    stride_in_horizons: int = Field(default=1, ge=1)
    comparison_context_may_cross: bool = True
    min_windows_per_split: int = Field(default=32, ge=1)
    seed: int = 0
    quantile_levels: tuple[float, ...] = TIMESTRAL_QUANTILE_LEVELS

    @field_validator("quantile_levels")
    @classmethod
    def _check_levels(cls, levels: tuple[float, ...]) -> tuple[float, ...]:
        if not levels or any(not 0.0 < level < 1.0 for level in levels):
            msg = f"quantile levels must lie in (0, 1), got {levels}"
            raise ValueError(msg)
        if any(a >= b for a, b in itertools.pairwise(levels)):
            msg = f"quantile levels must be strictly increasing, got {levels}"
            raise ValueError(msg)
        return levels

    @model_validator(mode="after")
    def _check_datasets(self) -> Self:
        if not 0.0 < self.reward_fraction < 1.0:
            msg = f"reward_fraction must lie in (0, 1), got {self.reward_fraction}"
            raise ValueError(msg)
        kept = min(self.reward_windows_per_dataset, self.comparison_windows_per_dataset)
        if kept < self.min_windows_per_split:
            msg = (
                f"min_windows_per_split = {self.min_windows_per_split} exceeds the windows kept "
                f"per dataset ({kept})"
            )
            raise ValueError(msg)
        if self.revision is not None and not self.revision.strip():
            msg = "revision must be None or a non-empty commit sha, branch or tag"
            raise ValueError(msg)
        ids = self.dataset_ids
        for dataset in ids:
            if dataset.name not in DOMAINS:
                msg = f"unknown GIFT-Eval dataset {dataset.name!r}; known: {sorted(DOMAINS)}"
                raise ValueError(msg)
            for table in (self.horizon_by_freq, self.season_by_freq):
                normalize_freq(dataset.freq, table.keys())
            self.window_spec(dataset.freq)  # raises on inconsistent lengths
        duplicates = sorted({str(d) for d in ids if ids.count(d) > 1})
        if duplicates:
            msg = f"duplicate dataset ids: {duplicates}"
            raise ValueError(msg)
        return self

    @property
    def dataset_ids(self) -> tuple[DatasetId, ...]:
        """Return the parsed dataset ids, in config order."""
        return tuple(parse_dataset_id(text, self.context_by_freq.keys()) for text in self.datasets)

    def window_spec(self, freq: str) -> WindowSpec:
        """Return the window lengths of the frequency key ``freq``."""
        horizon = self.horizon_by_freq[freq]
        return WindowSpec(
            context_length=self.context_by_freq[freq],
            horizon=horizon,
            season=self.season_by_freq[freq],
            stride=self.stride_in_horizons * horizon,
            max_context_length=self.max_context_length,
            max_horizon=self.max_horizon,
        )

    def cache_fields(self, dataset: DatasetId) -> dict[str, object]:
        """Return every value that changes the windows of ``dataset``, for the cache key."""
        spec = self.window_spec(dataset.freq)
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "dataset": str(dataset),
            "context_length": spec.context_length,
            "horizon": spec.horizon,
            "season": spec.season,
            "stride": spec.stride,
            "max_context_length": spec.max_context_length,
            "max_horizon": spec.max_horizon,
            "usable_fraction": str(USABLE_FRACTION),
            "reward_fraction": self.reward_fraction,
            "comparison_context_may_cross": self.comparison_context_may_cross,
            "reward_windows_per_dataset": self.reward_windows_per_dataset,
            "comparison_windows_per_dataset": self.comparison_windows_per_dataset,
            "seed": self.seed,
            "quantile_levels": list(self.quantile_levels),
        }


class FrequencyMismatchError(ValueError):
    """The ``freq`` column of the shards does not match the frequency of the dataset id."""


class GiftEvalBenchmark(Benchmark[GiftEvalBenchmarkConfig]):
    """GIFT-Eval datasets as two fixed window sets on the device.

    Each univariate series (each variate of a multivariate item) is split along time: the
    last 10% is never read, the rest splits into the reward part and the comparison part.
    See :mod:`autotsaugment.components.data.windows` for the split and window rules.
    """

    def __init__(self, *, config: GiftEvalBenchmarkConfig, device: torch.device | str) -> None:
        """Build both window sets on ``device``, from the cache when present."""
        self.config = config
        self._device = torch.device(device)
        datasets = config.dataset_ids
        parts = _load_or_build(config, datasets)
        names = tuple(str(dataset) for dataset in datasets)
        domains = tuple(DOMAINS[dataset.name] for dataset in datasets)
        self._reward = assemble_window_set(
            [part[REWARD] for part in parts],
            dataset_names=names,
            domains=domains,
            device=self._device,
        )
        self._comparison = assemble_window_set(
            [part[COMPARISON] for part in parts],
            dataset_names=names,
            domains=domains,
            device=self._device,
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the window sets."""
        return self._device

    @property
    def reward_windows(self) -> WindowSet:
        """Return the reward set."""
        return self._reward

    @property
    def comparison_windows(self) -> WindowSet:
        """Return the comparison set."""
        return self._comparison


def _load_or_build(
    config: GiftEvalBenchmarkConfig, datasets: tuple[DatasetId, ...]
) -> list[dict[str, SplitWindows]]:
    """Return the windows of each dataset id, from the cache or built from the shards."""
    paths = {
        dataset: cache_path(config.root, dataset.stem, cache_key(config.cache_fields(dataset)))
        for dataset in datasets
    }
    missing = [dataset for dataset in datasets if not paths[dataset].is_file()]
    shards = _shard_paths(config, missing) if missing else {}
    levels = torch.tensor(config.quantile_levels, dtype=torch.float32)
    known = config.context_by_freq.keys()
    parts: list[dict[str, SplitWindows]] = []
    for dataset in datasets:
        path = paths[dataset]
        if dataset not in shards:
            _log.info("%s: windows from %s", dataset, path)
            windows = load_windows(path)
            check_window_counts(
                str(dataset),
                {split: windows[split].n_windows for split in SPLITS},
                config.min_windows_per_split,
            )
            parts.append(windows)
            continue
        data = read_series(shards[dataset])
        if normalize_freq(data.freq, known) != dataset.freq:
            msg = f"{dataset}: the shards hold frequency {data.freq!r}, not {dataset.freq!r}"
            raise FrequencyMismatchError(msg)
        windows = build_dataset_windows(
            data,
            config.window_spec(dataset.freq),
            dataset_id=str(dataset),
            reward_fraction=config.reward_fraction,
            comparison_context_may_cross=config.comparison_context_may_cross,
            windows_per_split={
                REWARD: config.reward_windows_per_dataset,
                COMPARISON: config.comparison_windows_per_dataset,
            },
            min_windows=config.min_windows_per_split,
            levels=levels,
            seed=config.seed,
        )
        save_windows(path, str(dataset), windows)
        _log.info(
            "%s: %d series, %d reward and %d comparison windows, cached in %s",
            dataset,
            data.n_series,
            windows[REWARD].n_windows,
            windows[COMPARISON].n_windows,
            path,
        )
        parts.append(windows)
    return parts


def _shard_paths(
    config: GiftEvalBenchmarkConfig, datasets: list[DatasetId]
) -> dict[DatasetId, tuple[Path, ...]]:
    """Return the local shard files of each dataset id, downloaded first when missing."""
    known = config.context_by_freq.keys()
    sources = find_local_sources(
        datasets, config.root, known, repo_id=config.repo_id, revision=config.revision
    )
    absent = [dataset for dataset in datasets if dataset not in sources]
    if absent:
        plan = plan_download(
            repo_id=config.repo_id,
            revision=config.revision,
            root=config.root,
            datasets=absent,
            max_total_gb=config.max_total_gb,
            known=known,
        )
        for line in plan.describe():
            _log.info("download plan: %s", line)
        download(plan)
        sources.update({source.dataset: source for source in plan.sources})
    local = hub_dir(config.root, config.repo_id, config.revision)
    return {
        dataset: tuple(local / shard.path for shard in sources[dataset].shards)
        for dataset in datasets
    }
