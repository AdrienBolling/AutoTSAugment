"""GiftEvalBenchmark on a tiny Hub-shaped tree of Arrow shards, without network.

Every value encodes its position: ``value = 10000 * series + time``, so a test can read back
which series and which time steps a window holds.
"""

import math
import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import hydra.utils
import pyarrow as pa
import pytest
import torch
from omegaconf import OmegaConf
from pyarrow import ipc
from pydantic import ValidationError

from autotsaugment.components.data import download as download_module
from autotsaugment.components.data import gifteval as gifteval_module
from autotsaugment.components.data.arrow import read_series
from autotsaugment.components.data.download import (
    HUB_DIR,
    BudgetExceededError,
    UnknownFrequencyError,
    list_local_files,
)
from autotsaugment.components.data.gifteval import (
    DOMAINS,
    TIMESTRAL_QUANTILE_LEVELS,
    FrequencyMismatchError,
    GiftEvalBenchmark,
    GiftEvalBenchmarkConfig,
)
from autotsaugment.components.data.windows import usable_ends
from autotsaugment.core.benchmark import WindowSet
from autotsaugment.core.metrics import pinball_sum, skill_scores, wql_per_dataset

NAN = float("nan")
CODE = 10000
CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "env" / "benchmark"
ETT1_LENGTHS = (2000, 1500)  # two items of two variates each
COVID_LENGTHS = (400, 300, 250, 330)  # series 3 is all NaN
COVID_NAN = range(96, 160)  # two whole reward targets of series 0
COVID_NULL = 50  # an Arrow null in series 0


def encoded(series: int, length: int) -> list[float]:
    return [float(CODE * series + t) for t in range(length)]


def write_shard(path: Path, columns: dict[str, pa.Array], *, file_format: bool = False) -> None:
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(path), "wb") as sink:
        writer = (
            ipc.new_file(sink, table.schema) if file_format else ipc.new_stream(sink, table.schema)
        )
        with writer:
            writer.write_table(table)


def common(n: int, freq: str) -> dict[str, pa.Array]:
    return {
        "item_id": pa.array([f"item_{i}" for i in range(n)]),
        "start": pa.array([0] * n, type=pa.timestamp("s")),
        "freq": pa.array([freq] * n),
    }


def write_hub(hub: Path) -> None:
    """Write ett1/H (multivariate), ett1/D, covid_deaths (top level, IPC file) and hospital."""
    variates = pa.list_(pa.list_(pa.float32()), 2)
    ett1_targets = [
        [encoded(2 * item + v, length) for v in range(2)]
        for item, length in enumerate(ETT1_LENGTHS)
    ]
    write_shard(
        hub / "ett1" / "H" / "data-00000-of-00001.arrow",
        {
            **common(2, "h"),
            "target": pa.array(ett1_targets, type=variates),
            "past_feat_dynamic_real": pa.array(ett1_targets, type=variates),
        },
    )
    write_shard(
        hub / "ett1" / "D" / "data-00000-of-00001.arrow",
        {**common(1, "D"), "target": pa.array([encoded(0, 800)], type=pa.list_(pa.float32()))},
    )
    covid: list[list[float | None]] = [list(encoded(s, n)) for s, n in enumerate(COVID_LENGTHS)]
    for t in COVID_NAN:
        covid[0][t] = NAN
    covid[0][COVID_NULL] = None
    covid[3] = [NAN] * COVID_LENGTHS[3]
    write_shard(
        hub / "covid_deaths" / "data-00000-of-00001.arrow",
        {**common(4, "D"), "target": pa.array(covid, type=pa.list_(pa.float32()))},
        file_format=True,
    )
    write_shard(  # hospital is monthly on the Hub; this shard says daily
        hub / "hospital" / "data-00000-of-00001.arrow",
        {**common(1, "D"), "target": pa.array([encoded(0, 300)], type=pa.list_(pa.float32()))},
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        msg = "network access in a CPU test"
        raise AssertionError(msg)

    monkeypatch.setattr(download_module, "list_hub_files", refuse)
    monkeypatch.setattr(download_module, "snapshot_download", refuse)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    write_hub(tmp_path / "root" / HUB_DIR)
    return tmp_path / "root"


def make_config(root: Path, *datasets: str, **fields: object) -> GiftEvalBenchmarkConfig:
    values: dict[str, object] = {
        "root": root,
        "datasets": datasets,
        "reward_windows_per_dataset": 16,
        "comparison_windows_per_dataset": 4,
    }
    return GiftEvalBenchmarkConfig.model_validate(values | fields)


def rows_of(window_set: WindowSet, dataset: int) -> torch.Tensor:
    return torch.nonzero(window_set.dataset_index == dataset).squeeze(-1)


def splits(benchmark: GiftEvalBenchmark) -> Iterator[tuple[str, WindowSet]]:
    yield "reward", benchmark.reward_windows
    yield "comparison", benchmark.comparison_windows


# --- parsing -------------------------------------------------------------------------------


def test_read_series_splits_variates_and_keeps_nan(root: Path):
    ett1 = read_series([root / HUB_DIR / "ett1" / "H" / "data-00000-of-00001.arrow"])
    assert ett1.freq == "h"
    assert ett1.n_series == 4
    assert ett1.lengths.tolist() == [2000, 2000, 1500, 1500]
    for series in range(4):
        assert ett1.series(series)[:3].tolist() == [CODE * series + t for t in range(3)]
    assert ett1.values.dtype == torch.float32

    covid = read_series([root / HUB_DIR / "covid_deaths" / "data-00000-of-00001.arrow"])
    assert covid.freq == "D"
    assert covid.lengths.tolist() == list(COVID_LENGTHS)
    first = covid.series(0)
    assert torch.isnan(first[COVID_NULL])  # an Arrow null becomes NaN
    assert torch.isnan(first[list(COVID_NAN)]).all()
    assert int(torch.isnan(first).sum()) == len(COVID_NAN) + 1
    assert torch.isnan(covid.series(3)).all()


def test_read_series_rejects_mixed_frequencies(root: Path):
    shards = [
        root / HUB_DIR / "ett1" / "H" / "data-00000-of-00001.arrow",
        root / HUB_DIR / "ett1" / "D" / "data-00000-of-00001.arrow",
    ]
    with pytest.raises(ValueError, match="several frequencies"):
        read_series(shards)


# --- windows -------------------------------------------------------------------------------


def test_shapes_padding_and_device(root: Path):
    benchmark = GiftEvalBenchmark(
        config=make_config(root, "ett1/H", "covid_deaths/D"), device="cpu"
    )
    config = benchmark.config
    for _, window_set in splits(benchmark):
        n = window_set.n_windows
        assert window_set.context.shape == (n, config.max_context_length)
        assert window_set.target.shape == (n, config.max_horizon)
        assert window_set.context.dtype == torch.float32
        assert window_set.dataset_index.dtype == torch.int64
        assert window_set.device == torch.device("cpu")
        covid = rows_of(window_set, 1)
        context_pad = config.max_context_length - config.context_by_freq["D"]
        assert torch.isnan(window_set.context[covid, :context_pad]).all()
        assert torch.isnan(window_set.target[covid, config.horizon_by_freq["D"] :]).all()
        ett1 = rows_of(window_set, 0)
        assert torch.isfinite(window_set.target[ett1]).all()
        assert torch.isfinite(window_set.naive_loss_sum).all()
        assert torch.isfinite(window_set.abs_sum).all()
        assert not torch.isinf(window_set.context).any()


def test_no_window_reads_the_test_part_or_crosses_the_split_boundary(root: Path):
    config = make_config(
        root,
        "ett1/H",
        "covid_deaths/D",
        reward_windows_per_dataset=1000,
        comparison_windows_per_dataset=1000,
    )
    benchmark = GiftEvalBenchmark(config=config, device="cpu")
    lengths = {0: torch.tensor(ETT1_LENGTHS).repeat_interleave(2), 1: torch.tensor(COVID_LENGTHS)}
    width = config.max_context_length
    for split, window_set in splits(benchmark):
        for row in range(window_set.n_windows):
            dataset = int(window_set.dataset_index[row])
            values = torch.cat([window_set.context[row], window_set.target[row]])
            columns = torch.nonzero(torch.isfinite(values)).squeeze(-1)
            codes = values[columns].long()
            series = codes // CODE
            times = codes % CODE
            assert torch.all(series == series[0])
            # column j of the joined row is time horizon_start - width + j
            assert torch.all(times - columns == times[0] - columns[0])
            test_start = int(usable_ends(lengths[dataset][series[0]]))
            reward_end = test_start * 17 // 20
            assert int(times.max()) < test_start
            if split == "reward":
                assert int(times.max()) < reward_end
            else:
                assert int(times.min()) >= reward_end
            horizon_start = int(times[0] - columns[0]) + width
            assert torch.all(times[columns < width] < horizon_start)


def reference_count(values: list[float], start: int, end: int, freq: str) -> int:
    """Count valid windows with plain loops, independent of the vectorized code."""
    config = GiftEvalBenchmarkConfig(datasets=("ett1/H",))
    c, h, season = (
        config.context_by_freq[freq],
        config.horizon_by_freq[freq],
        config.season_by_freq[freq],
    )
    count = 0
    t = end - h
    while t - start >= season:
        context = values[max(start, t - c) : t]
        target = values[t : t + h]
        observed = sum(not math.isnan(v) for v in context)
        if observed >= season and any(not math.isnan(v) for v in target):
            count += 1
        t -= h
    return count


def test_window_counts_match_a_loop_reference(root: Path):
    benchmark = GiftEvalBenchmark(
        config=make_config(
            root,
            "ett1/H",
            "covid_deaths/D",
            reward_windows_per_dataset=1000,
            comparison_windows_per_dataset=1000,
        ),
        device="cpu",
    )
    shards = {
        0: (root / HUB_DIR / "ett1" / "H" / "data-00000-of-00001.arrow", "H"),
        1: (root / HUB_DIR / "covid_deaths" / "data-00000-of-00001.arrow", "D"),
    }
    expected = {"reward": {0: 0, 1: 0}, "comparison": {0: 0, 1: 0}}
    for dataset, (path, freq) in shards.items():
        data = read_series([path])
        for series in range(data.n_series):
            values = data.series(series).tolist()
            test_start = len(values) * 9 // 10
            reward_end = test_start * 17 // 20
            expected["reward"][dataset] += reference_count(values, 0, reward_end, freq)
            expected["comparison"][dataset] += reference_count(values, reward_end, test_start, freq)
    assert expected["comparison"][1] == 2  # by hand: one window in series 0 and in series 1
    for split, window_set in splits(benchmark):
        for dataset in (0, 1):
            assert rows_of(window_set, dataset).numel() == expected[split][dataset]


def test_naive_skill_is_one_by_construction(root: Path):
    benchmark = GiftEvalBenchmark(
        config=make_config(root, "ett1/H", "covid_deaths/D"), device="cpu"
    )
    config = benchmark.config
    levels = torch.tensor(config.quantile_levels)
    seasons = (config.season_by_freq["H"], config.season_by_freq["D"])
    for _, window_set in splits(benchmark):
        direct = torch.zeros(window_set.n_windows)
        for row in range(window_set.n_windows):
            season = seasons[int(window_set.dataset_index[row])]
            filled = window_set.context[row].tolist()
            for i in range(1, len(filled)):
                if math.isnan(filled[i]):
                    filled[i] = filled[i - 1]
            last_season = filled[-season:]
            forecast = torch.tensor([last_season[j % season] for j in range(config.max_horizon)])
            assert torch.isfinite(forecast).all()
            quantiles = forecast[None, :, None].expand(1, -1, levels.numel())
            direct[row] = pinball_sum(quantiles, window_set.target[row : row + 1], levels)[0]
        d, q = window_set.n_datasets, levels.numel()
        stored = wql_per_dataset(
            window_set.naive_loss_sum, window_set.abs_sum, window_set.dataset_index, d, q
        )
        recomputed = wql_per_dataset(direct, window_set.abs_sum, window_set.dataset_index, d, q)
        torch.testing.assert_close(stored, recomputed)
        torch.testing.assert_close(skill_scores(recomputed, stored), torch.ones(d))


def test_subsample_of_a_dataset_does_not_depend_on_the_others(root: Path):
    alone = GiftEvalBenchmark(config=make_config(root, "ett1/H"), device="cpu")
    shutil.rmtree(root / "cache")
    together = GiftEvalBenchmark(config=make_config(root, "covid_deaths/D", "ett1/H"), device="cpu")
    for (_, single), (_, both) in zip(splits(alone), splits(together), strict=True):
        rows = rows_of(both, 1)
        torch.testing.assert_close(both.context[rows], single.context, equal_nan=True)
        torch.testing.assert_close(both.target[rows], single.target, equal_nan=True)
        torch.testing.assert_close(both.naive_loss_sum[rows], single.naive_loss_sum)
    assert alone.reward_windows.n_windows == 16  # 108 candidates, 16 kept
    assert alone.comparison_windows.n_windows == 4  # 16 candidates, 4 kept
    other_seed = GiftEvalBenchmark(config=make_config(root, "ett1/H", seed=1), device="cpu")
    assert not torch.equal(other_seed.reward_windows.target, alone.reward_windows.target)


def test_dataset_order_names_and_domains(root: Path):
    benchmark = GiftEvalBenchmark(
        config=make_config(root, "covid_deaths/D", "ett1/h"), device="cpu"
    )
    assert benchmark.dataset_names == ("covid_deaths/D", "ett1/H")
    assert benchmark.domains == ("Healthcare", "Energy")
    for _, window_set in splits(benchmark):
        assert window_set.dataset_names == benchmark.dataset_names
        index = window_set.dataset_index
        assert torch.all(index[1:] >= index[:-1])
        assert set(index.tolist()) == {0, 1}
    assert len(set(DOMAINS.values())) == 7
    assert len(DOMAINS) == 28


# --- cache ---------------------------------------------------------------------------------


def test_cache_round_trip_reads_no_arrow(root: Path, monkeypatch: pytest.MonkeyPatch):
    config = make_config(root, "ett1/H", "covid_deaths/D")
    first = GiftEvalBenchmark(config=config, device="cpu")
    names = sorted(path.name for path in (root / "cache").iterdir())
    assert len(names) == 2
    assert re.fullmatch(r"covid_deaths__D__[0-9a-f]{40}\.pt", names[0])
    assert re.fullmatch(r"ett1__H__[0-9a-f]{40}\.pt", names[1])

    def fail(*args: object, **kwargs: object) -> None:
        msg = "no Arrow read and no shard lookup with a complete cache"
        raise AssertionError(msg)

    monkeypatch.setattr(gifteval_module, "read_series", fail)
    monkeypatch.setattr(gifteval_module, "find_local_sources", fail)
    second = GiftEvalBenchmark(config=config, device="cpu")
    for (_, a), (_, b) in zip(splits(first), splits(second), strict=True):
        torch.testing.assert_close(a.context, b.context, equal_nan=True, rtol=0, atol=0)
        torch.testing.assert_close(a.target, b.target, equal_nan=True, rtol=0, atol=0)
        torch.testing.assert_close(a.naive_loss_sum, b.naive_loss_sum, rtol=0, atol=0)
        torch.testing.assert_close(a.abs_sum, b.abs_sum, rtol=0, atol=0)
        assert torch.equal(a.dataset_index, b.dataset_index)
    with pytest.raises(AssertionError, match="no Arrow read"):  # a new window field, a new key
        GiftEvalBenchmark(config=make_config(root, "ett1/H", stride_in_horizons=2), device="cpu")


# --- errors and downloads ------------------------------------------------------------------


def test_missing_frequency_lists_the_available_ones(root: Path, monkeypatch: pytest.MonkeyPatch):
    listing = list_local_files(root / HUB_DIR)
    monkeypatch.setattr(download_module, "list_hub_files", lambda repo_id: listing)
    with pytest.raises(UnknownFrequencyError, match=r"ett1 has the frequencies \['D', 'H'\]"):
        GiftEvalBenchmark(config=make_config(root, "ett1/15T"), device="cpu")
    with pytest.raises(UnknownFrequencyError, match=r"covid_deaths has the frequencies \['D'\]"):
        GiftEvalBenchmark(config=make_config(root, "covid_deaths/H"), device="cpu")


def test_frequency_column_must_match_the_id(root: Path):
    with pytest.raises(FrequencyMismatchError, match="hospital/M: the shards hold frequency 'D'"):
        GiftEvalBenchmark(config=make_config(root, "hospital/M"), device="cpu")


def test_missing_shards_are_planned_and_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    remote = tmp_path / "remote"
    write_hub(remote)
    listing = list_local_files(remote)
    calls: list[list[str]] = []

    def fake_snapshot(
        repo_id: str, *, repo_type: str, allow_patterns: list[str], local_dir: Path
    ) -> None:
        assert repo_type == "dataset"
        calls.append(allow_patterns)
        for pattern in allow_patterns:
            target = Path(local_dir) / pattern
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote / pattern, target)

    monkeypatch.setattr(download_module, "list_hub_files", lambda repo_id: listing)
    monkeypatch.setattr(download_module, "snapshot_download", fake_snapshot)
    root = tmp_path / "root"
    with pytest.raises(BudgetExceededError, match="exceeds max_total_gb"):
        GiftEvalBenchmark(config=make_config(root, "ett1/H", max_total_gb=1e-6), device="cpu")
    assert calls == []
    benchmark = GiftEvalBenchmark(config=make_config(root, "ett1/H"), device="cpu")
    assert calls == [["ett1/H/data-00000-of-00001.arrow"]]
    assert benchmark.reward_windows.n_windows == 16


# --- config --------------------------------------------------------------------------------


def test_config_defaults_and_validation(tmp_path: Path):
    config = GiftEvalBenchmarkConfig(datasets=("ett1/H",))
    assert config.root == Path("data/gifteval")
    assert len(config.quantile_levels) == 21
    assert config.quantile_levels == TIMESTRAL_QUANTILE_LEVELS
    assert config.quantile_levels[:3] == (0.01, 0.05, 0.1)
    assert config.quantile_levels[-2:] == (0.95, 0.99)
    bad: list[dict[str, object]] = [
        {"datasets": ()},
        {"datasets": ("not_a_dataset/H",)},
        {"datasets": ("ett1",)},
        {"datasets": ("ett1/2H",)},
        {"datasets": ("ett1/H", "ett1/h")},
        {"datasets": ("ett1/H",), "reward_fraction": 1.0},
        {"datasets": ("ett1/H",), "quantile_levels": (0.5, 0.1)},
        {"datasets": ("ett1/H",), "stride_in_horizons": 0},
        {"datasets": ("ett1/H",), "max_total_gb": 0.0},
        {"datasets": ("ett1/H",), "unknown_field": 1},
    ]
    for fields in bad:
        with pytest.raises(ValidationError):
            GiftEvalBenchmarkConfig.model_validate(fields)


@pytest.mark.parametrize(("name", "n_datasets"), [("gifteval_small", 8), ("gifteval_full", 47)])
def test_yaml_configs_validate(name: str, n_datasets: int):
    node = OmegaConf.load(CONFIGS / f"{name}.yaml")
    assert node._target_ == "autotsaugment.components.data.gifteval.GiftEvalBenchmark"
    config = hydra.utils.instantiate(node.config)
    assert isinstance(config, GiftEvalBenchmarkConfig)
    assert len(config.dataset_ids) == n_datasets
    assert {DOMAINS[dataset.name] for dataset in config.dataset_ids} == set(DOMAINS.values())


@pytest.mark.gpu
def test_window_sets_live_on_the_gpu(root: Path):
    benchmark = GiftEvalBenchmark(config=make_config(root, "ett1/H"), device="cuda")
    for _, window_set in splits(benchmark):
        assert window_set.device.type == "cuda"
        assert window_set.dataset_index.device.type == "cuda"
