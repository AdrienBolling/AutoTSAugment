import importlib.util
from pathlib import Path
from types import ModuleType

import hydra.utils
import pytest
import torch
from omegaconf import OmegaConf

from autotsaugment.components.data import download as download_module
from autotsaugment.components.data.download import (
    HUB_DIR,
    BudgetExceededError,
    DatasetId,
    DatasetSource,
    HubFile,
    UnknownDatasetError,
    UnknownFrequencyError,
    download,
    find_local_sources,
    list_hub_files,
    normalize_freq,
    parse_dataset_id,
    plan_download,
    resolve_sources,
)
from autotsaugment.components.data.gifteval import GiftEvalBenchmark, GiftEvalBenchmarkConfig
from autotsaugment.components.data.windows import EmptySplitError
from autotsaugment.core.benchmark import BenchmarkConfig
from autotsaugment.core.metrics import skill_scores, wql_per_dataset

REPO = Path(__file__).resolve().parents[1]
KNOWN = tuple(BenchmarkConfig().context_by_freq)
LISTING = (
    HubFile("ett1/H/data-00000-of-00001.arrow", 100),
    HubFile("ett1/15T/data-00000-of-00001.arrow", 400),
    HubFile("covid_deaths/data-00000-of-00001.arrow", 50),
    HubFile("jena_weather/data-00000-of-00001.arrow", 1000),  # copy of 10T, ignored
    HubFile("jena_weather/10T/data-00000-of-00002.arrow", 500),
    HubFile("jena_weather/10T/data-00001-of-00002.arrow", 500),
    HubFile("electricity/15T/cache-1318815c7ce7bc45.arrow", 9999),  # not a shard
    HubFile("covid_deaths/dataset_info.json", 10),  # not a shard
)
ETT1_H = DatasetId("ett1", "H")
COVID_D = DatasetId("covid_deaths", "D")
JENA_10T = DatasetId("jena_weather", "10T")


@pytest.mark.parametrize(
    ("alias", "key"),
    [
        ("H", "H"),
        ("h", "H"),
        ("1H", "H"),
        ("15T", "15T"),
        ("15min", "15T"),
        ("10min", "10T"),
        ("5T", "5T"),
        ("10S", "10S"),
        ("10s", "10S"),
        ("D", "D"),
        ("W", "W"),
        ("W-SUN", "W"),
        ("W-THU", "W"),
        ("M", "M"),
        ("ME", "M"),
        ("Q", "Q"),
        ("Q-DEC", "Q"),
        ("QE-DEC", "Q"),
        ("A", "A"),
        ("A-DEC", "A"),
        ("Y", "A"),
        ("YE", "A"),
    ],
)
def test_normalize_freq(alias: str, key: str):
    assert normalize_freq(alias, KNOWN) == key


@pytest.mark.parametrize("alias", ["", "B", "min", "30min", "2W", "7D", "H-DEC", "ms", "W-"])
def test_normalize_freq_fails_loudly(alias: str):
    with pytest.raises(UnknownFrequencyError):
        normalize_freq(alias, KNOWN)


def test_parse_dataset_id():
    dataset = parse_dataset_id("ett1/h", KNOWN)
    assert dataset == ETT1_H
    assert str(dataset) == "ett1/H"
    assert dataset.stem == "ett1__H"
    for text in ("ett1", "/H", "ett1/", "a/b/H"):
        with pytest.raises(ValueError, match="does not have the form"):
            parse_dataset_id(text, KNOWN)


def test_resolve_sources_learns_the_layout():
    ett1, covid, jena = resolve_sources([ETT1_H, COVID_D, JENA_10T], LISTING, KNOWN)
    assert (ett1.directory, ett1.size, ett1.complete) == ("ett1/H", 100, True)
    assert (covid.directory, covid.size) == ("covid_deaths", 50)
    assert (jena.directory, jena.size, len(jena.shards), jena.complete) == (
        "jena_weather/10T",
        1000,
        2,
        True,
    )


def test_resolve_sources_lists_the_available_frequencies():
    with pytest.raises(UnknownFrequencyError, match=r"ett1 has the frequencies \['15T', 'H'\]"):
        resolve_sources([DatasetId("ett1", "D")], LISTING, KNOWN)
    with pytest.raises(UnknownFrequencyError, match=r"covid_deaths has the frequencies \['D'\]"):
        resolve_sources([DatasetId("covid_deaths", "H")], LISTING, KNOWN)
    with pytest.raises(UnknownFrequencyError, match=r"jena_weather has the frequencies \['10T'\]"):
        resolve_sources([DatasetId("jena_weather", "H")], LISTING, KNOWN)
    with pytest.raises(UnknownDatasetError, match="'solar' is not in the listing"):
        resolve_sources([DatasetId("solar", "H")], LISTING, KNOWN)


def test_incomplete_source():
    shard = HubFile("jena_weather/10T/data-00000-of-00002.arrow", 500)
    assert not DatasetSource(
        dataset=JENA_10T, directory="jena_weather/10T", shards=(shard,)
    ).complete


def write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)


def test_plan_counts_present_shards_and_existing_bytes(tmp_path: Path):
    write_bytes(tmp_path / HUB_DIR / "ett1/H/data-00000-of-00001.arrow", 100)
    write_bytes(tmp_path / "cache" / "other.pt", 30)
    plan = plan_download(
        repo_id="r",
        root=tmp_path,
        datasets=[ETT1_H, COVID_D],
        max_total_gb=1.0,
        known=KNOWN,
        files=LISTING,
    )
    assert plan.missing == (HubFile("covid_deaths/data-00000-of-00001.arrow", 50),)
    assert (plan.download_bytes, plan.existing_bytes, plan.total_bytes) == (50, 130, 180)
    assert plan.within_budget
    lines = plan.describe()
    assert len(lines) == 6
    assert lines[0].startswith("ett1/H")
    assert lines[0].endswith("present")
    assert lines[1].startswith("covid_deaths/D")
    assert lines[1].endswith("to download")


def test_planner_refuses_a_plan_over_the_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def refuse(*args: object, **kwargs: object) -> None:
        msg = "no download over the budget"
        raise AssertionError(msg)

    monkeypatch.setattr(download_module, "snapshot_download", refuse)
    with pytest.raises(BudgetExceededError, match="exceeds max_total_gb"):
        plan_download(
            repo_id="r",
            root=tmp_path,
            datasets=[ETT1_H, COVID_D],
            max_total_gb=149e-9,  # 149 bytes for a 150-byte plan
            known=KNOWN,
            files=LISTING,
        )
    write_bytes(tmp_path / "other.bin", 60)  # what is under root counts too
    plan = plan_download(
        repo_id="r",
        root=tmp_path,
        datasets=[COVID_D],
        max_total_gb=100e-9,
        known=KNOWN,
        files=LISTING,
        enforce_budget=False,
    )
    assert not plan.within_budget
    assert "EXCEEDED" in plan.describe()[-1]
    with pytest.raises(BudgetExceededError):
        download(plan)


def test_download_requests_exactly_the_missing_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        download_module,
        "snapshot_download",
        lambda repo_id, **kwargs: calls.append((repo_id, kwargs)),
    )
    plan = plan_download(
        repo_id="r",
        root=tmp_path,
        datasets=[JENA_10T],
        max_total_gb=1.0,
        known=KNOWN,
        files=LISTING,
    )
    download(plan)
    assert calls == [
        (
            "r",
            {
                "repo_type": "dataset",
                "allow_patterns": [
                    "jena_weather/10T/data-00000-of-00002.arrow",
                    "jena_weather/10T/data-00001-of-00002.arrow",
                ],
                "local_dir": tmp_path / HUB_DIR,
            },
        )
    ]


def test_find_local_sources_needs_complete_shards(tmp_path: Path):
    write_bytes(tmp_path / HUB_DIR / "ett1/H/data-00000-of-00001.arrow", 10)
    write_bytes(tmp_path / HUB_DIR / "jena_weather/10T/data-00000-of-00002.arrow", 10)
    found = find_local_sources([ETT1_H, JENA_10T, COVID_D], tmp_path, KNOWN)
    assert set(found) == {ETT1_H}


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "download_benchmarks", REPO / "scripts" / "download_benchmarks.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_prints_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(download_module, "list_hub_files", lambda repo_id: LISTING)
    script = load_script()
    code = script.main(["--plan", "--root", str(tmp_path), "ett1/H", "covid_deaths/D"])
    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    assert lines[0].split() == ["ett1/H", "0.00", "MB", "to", "download"]
    assert lines[1].startswith("covid_deaths/D")
    assert lines[3].split()[:3] == ["to", "download", "0.00"]
    assert not (tmp_path / HUB_DIR).exists()
    over = ["--root", str(tmp_path), "--max-total-gb", "1e-9", "ett1/H"]
    assert script.main(["--plan", *over]) == 1
    assert script.main(over) == 1
    assert "refused" in capsys.readouterr().err


# --- network -------------------------------------------------------------------------------


@pytest.mark.network
def test_network_listing_layout():
    files = list_hub_files("Salesforce/GiftEval")
    assert all(file.path.endswith(".arrow") and "/cache-" not in file.path for file in files)
    with pytest.raises(UnknownFrequencyError, match=r"\['15T', 'D', 'H', 'W'\]"):
        resolve_sources([DatasetId("ett1", "10T")], files, KNOWN)
    (covid,) = resolve_sources([COVID_D], files, KNOWN)
    assert covid.directory == "covid_deaths"


@pytest.mark.network
def test_network_covid_deaths_end_to_end(tmp_path: Path):
    plan = plan_download(
        repo_id="Salesforce/GiftEval",
        root=tmp_path,
        datasets=[COVID_D],
        max_total_gb=0.01,
        known=KNOWN,
    )
    assert plan.download_bytes == plan.sources[0].size > 0
    download(plan)
    assert (tmp_path / HUB_DIR / plan.missing[0].path).stat().st_size == plan.missing[0].size

    # The series have 212 steps: at reward_fraction 0.85 the comparison part has 29 steps,
    # less than one daily horizon of 30, so the default split has no comparison window.
    with pytest.raises(EmptySplitError, match="covid_deaths/D: the comparison split"):
        GiftEvalBenchmark(
            config=GiftEvalBenchmarkConfig(root=tmp_path, datasets=("covid_deaths/D",)),
            device="cpu",
        )
    config = GiftEvalBenchmarkConfig(
        root=tmp_path, datasets=("covid_deaths/D",), reward_fraction=0.7, max_total_gb=0.01
    )
    benchmark = GiftEvalBenchmark(config=config, device="cpu")
    assert benchmark.domains == ("Healthcare",)
    expected = {"reward": 512, "comparison": 266}  # 266 series, one comparison window each
    for split, window_set in (
        ("reward", benchmark.reward_windows),
        ("comparison", benchmark.comparison_windows),
    ):
        n = window_set.n_windows
        assert n == expected[split]
        assert window_set.context.shape == (n, 512)
        assert window_set.target.shape == (n, 48)
        assert torch.isnan(window_set.context[:, : 512 - 256]).all()
        assert torch.isfinite(window_set.target[:, :30]).all()
        assert torch.isnan(window_set.target[:, 30:]).all()
        assert not torch.isinf(window_set.context).any()
        assert torch.isfinite(window_set.naive_loss_sum).all()
        assert torch.isfinite(window_set.abs_sum).all()
        naive = wql_per_dataset(
            window_set.naive_loss_sum, window_set.abs_sum, window_set.dataset_index, 1, 21
        )
        assert torch.isfinite(naive).all()
        torch.testing.assert_close(skill_scores(naive, naive), torch.ones(1))


@pytest.mark.network
@pytest.mark.slow
def test_network_default_config_builds(tmp_path: Path):
    node = OmegaConf.load(REPO / "configs" / "env" / "benchmark" / "gifteval_small.yaml")
    node.device = "cpu"
    node.config.root = str(tmp_path)
    benchmark = hydra.utils.instantiate(node)
    assert isinstance(benchmark, GiftEvalBenchmark)
    assert len(benchmark.dataset_names) == 8
    assert len(set(benchmark.domains)) == 7
    for window_set in (benchmark.reward_windows, benchmark.comparison_windows):
        counts = torch.bincount(window_set.dataset_index, minlength=8)
        assert torch.all(counts > 0)
        assert torch.all(counts <= 512)
        assert torch.isfinite(window_set.naive_loss_sum).all()
