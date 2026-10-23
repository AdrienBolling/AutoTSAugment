"""Dataset ids, the GIFT-Eval Hub layout, the download plan and the download.

Hub layout of ``Salesforce/GiftEval``, read from ``HfApi.list_repo_tree`` on 2026-10-23:

- A dataset with several frequencies has one directory per frequency:
  ``<name>/<freq>/data-XXXXX-of-YYYYY.arrow``, for example ``ett1/H/data-00000-of-00001.arrow``.
- A dataset with one frequency keeps its shards at the top: ``<name>/data-XXXXX-of-YYYYY.arrow``.
  The listing does not give its frequency, so :data:`SINGLE_FREQUENCY` holds it.
- The other files (``dataset_info.json``, ``state.json``, ``cache-*.arrow``, temporary files)
  are not read. ``jena_weather`` also has a top-level shard, a copy of ``jena_weather/10T``;
  it is ignored because the dataset has frequency directories.

The planner sums the shard sizes of the selected datasets from the Hub metadata and refuses to
start when that sum, plus the bytes already under ``root``, exceeds the budget. The download
writes the shards to ``root/hub`` with the same relative paths as on the Hub.
"""

import logging
import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.hf_api import RepoFile

_log = logging.getLogger(__name__)

BYTES_PER_GB: Final = 10**9
HUB_DIR: Final = "hub"
SHARD_PATTERN: Final = re.compile(
    r"^(?P<name>[^/]+)/(?:(?P<sub>[^/]+)/)?data-(?P<index>\d{5})-of-(?P<count>\d{5})\.arrow$"
)

# Frequency of the datasets whose shards have no frequency directory on the Hub.
# Source: the "frequency" field of notebooks/dataset_properties.json in
# https://github.com/SalesforceAIResearch/gift-eval (commit cfaefe09c6bb, read 2026-10-23).
# The GIFT-Eval notebooks read this field for exactly these datasets. The keys are Hub names;
# the JSON file uses lower-case names without the "_with_missing" suffix.
SINGLE_FREQUENCY: Final[dict[str, str]] = {
    "bizitobs_application": "10S",
    "bizitobs_service": "10S",
    "car_parts_with_missing": "M",
    "covid_deaths": "D",
    "hospital": "M",
    "m4_daily": "D",
    "m4_hourly": "H",
    "m4_monthly": "M",
    "m4_quarterly": "Q",
    "m4_weekly": "W",
    "m4_yearly": "A",
    "restaurant": "D",
    "temperature_rain_with_missing": "D",
}

# Pandas offset aliases (old and new spellings) to the unit letter of the frequency keys.
_UNIT_KEYS: Final[dict[str, str]] = {
    "S": "S",
    "s": "S",
    "T": "T",
    "min": "T",
    "H": "H",
    "h": "H",
    "D": "D",
    "W": "W",
    "M": "M",
    "ME": "M",
    "MS": "M",
    "Q": "Q",
    "QE": "Q",
    "QS": "Q",
    "A": "A",
    "AS": "A",
    "Y": "A",
    "YE": "A",
    "YS": "A",
}
_ANCHORED_UNITS: Final = frozenset({"W", "Q", "A"})
_FREQ_PATTERN: Final = re.compile(
    r"^(?P<mult>\d*)(?P<unit>[A-Za-z]+)(?:-(?P<anchor>[A-Za-z]{3}))?$"
)


class UnknownFrequencyError(ValueError):
    """A frequency alias has no key in ``context_by_freq``, or a dataset lacks it."""


class UnknownDatasetError(ValueError):
    """A dataset name is not in the repository."""


class BudgetExceededError(RuntimeError):
    """The planned download does not fit in ``max_total_gb``."""


def normalize_freq(alias: str, known: Collection[str]) -> str:
    """Return the frequency key of a pandas frequency alias.

    Examples: ``h`` and ``H`` give ``H``; ``15min`` and ``15T`` give ``15T``; ``ME`` and ``M``
    give ``M``; ``W-SUN`` gives ``W``; ``YE``, ``A`` and ``A-DEC`` give ``A``.

    Args:
        alias: A pandas frequency alias, as in the ``freq`` column or a Hub directory name.
        known: The accepted keys, normally the keys of ``BenchmarkConfig.context_by_freq``.

    Returns:
        The key, an element of ``known``.

    Raises:
        UnknownFrequencyError: The alias is not recognized or its key is not in ``known``.
    """
    match = _FREQ_PATTERN.match(alias.strip())
    unit = _UNIT_KEYS.get(match["unit"]) if match is not None else None
    if match is None or unit is None or (match["anchor"] and unit not in _ANCHORED_UNITS):
        msg = f"unknown frequency alias {alias!r}; known keys: {sorted(known)}"
        raise UnknownFrequencyError(msg)
    multiple = int(match["mult"]) if match["mult"] else 1
    key = unit if multiple == 1 else f"{multiple}{unit}"
    if key not in known:
        msg = f"frequency alias {alias!r} maps to {key!r}, which is not in {sorted(known)}"
        raise UnknownFrequencyError(msg)
    return key


@dataclass(frozen=True)
class DatasetId:
    """One GIFT-Eval dataset at one frequency.

    Attributes:
        name: Hub directory name, for example ``"ett1"``.
        freq: Frequency key, a key of ``context_by_freq``, for example ``"H"``.
    """

    name: str
    freq: str

    def __str__(self) -> str:
        """Return the id ``"<name>/<freq>"``."""
        return f"{self.name}/{self.freq}"

    @property
    def stem(self) -> str:
        """Return ``"<name>__<freq>"``, for file names."""
        return f"{self.name}__{self.freq}"


def parse_dataset_id(text: str, known: Collection[str]) -> DatasetId:
    """Parse ``"<name>/<freq>"`` and normalize the frequency.

    Args:
        text: The id, for example ``"ett1/H"`` or ``"m4_weekly/W-SUN"``.
        known: The accepted frequency keys.

    Returns:
        The parsed id with the frequency key.

    Raises:
        ValueError: The text does not have the form ``"<name>/<freq>"``.
        UnknownFrequencyError: The frequency is not recognized.
    """
    name, _, freq = text.partition("/")
    if not name or not freq or "/" in freq:
        msg = f"dataset id {text!r} does not have the form '<name>/<freq>'"
        raise ValueError(msg)
    return DatasetId(name=name, freq=normalize_freq(freq, known))


@dataclass(frozen=True)
class HubFile:
    """One shard file of the repository.

    Attributes:
        path: Path relative to the repository root, with ``/`` separators.
        size: Size in bytes.
    """

    path: str
    size: int


@dataclass(frozen=True)
class DatasetSource:
    """The shards of one dataset id.

    Attributes:
        dataset: The dataset id.
        directory: The directory of the shards, relative to the repository root.
        shards: The shard files, sorted by path.
    """

    dataset: DatasetId
    directory: str
    shards: tuple[HubFile, ...]

    @property
    def size(self) -> int:
        """Return the total size of the shards in bytes."""
        return sum(shard.size for shard in self.shards)

    @property
    def complete(self) -> bool:
        """Return True when every shard ``data-i-of-n`` for ``i < n`` is present."""
        counts: set[int] = set()
        for shard in self.shards:
            match = SHARD_PATTERN.match(shard.path)
            if match is not None:
                counts.add(int(match["count"]))
        return len(counts) == 1 and counts.pop() == len(self.shards)


def list_hub_files(repo_id: str) -> tuple[HubFile, ...]:
    """Return the shard files of the repository with their sizes, from the Hub metadata."""
    entries = HfApi().list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    return tuple(
        HubFile(path=entry.path, size=entry.size)
        for entry in entries
        if isinstance(entry, RepoFile) and SHARD_PATTERN.match(entry.path)
    )


def list_local_files(hub_root: Path) -> tuple[HubFile, ...]:
    """Return the shard files under ``hub_root``, the local copy of the repository."""
    if not hub_root.is_dir():
        return ()
    files = (
        HubFile(path=path.relative_to(hub_root).as_posix(), size=path.stat().st_size)
        for path in hub_root.rglob("data-*.arrow")
        if path.is_file()
    )
    return tuple(file for file in files if SHARD_PATTERN.match(file.path))


def _layout(files: Iterable[HubFile]) -> dict[str, dict[str | None, list[HubFile]]]:
    """Group shard files by dataset name and frequency directory (None at the top)."""
    layout: dict[str, dict[str | None, list[HubFile]]] = {}
    for file in files:
        match = SHARD_PATTERN.match(file.path)
        if match is not None:
            layout.setdefault(match["name"], {}).setdefault(match["sub"], []).append(file)
    return layout


def available_frequencies(
    name: str, directories: Collection[str | None], known: Collection[str]
) -> dict[str, str | None]:
    """Return the frequency keys of one dataset, each with its directory (None at the top).

    Args:
        name: The dataset name.
        directories: The frequency directories with shards, None for top-level shards.
        known: The accepted frequency keys.

    Returns:
        A map from frequency key to frequency directory. Directories whose name is not a known
        frequency are left out.

    Raises:
        UnknownDatasetError: The dataset has only top-level shards and no entry in
            :data:`SINGLE_FREQUENCY`.
    """
    subs = [sub for sub in directories if sub is not None]
    if not subs:
        if name not in SINGLE_FREQUENCY:
            msg = f"{name!r} has no frequency directory and no entry in SINGLE_FREQUENCY"
            raise UnknownDatasetError(msg)
        return {normalize_freq(SINGLE_FREQUENCY[name], known): None}
    frequencies: dict[str, str | None] = {}
    for sub in subs:
        try:
            frequencies[normalize_freq(sub, known)] = sub
        except UnknownFrequencyError:
            _log.warning("%s/%s: frequency directory without a known key, ignored", name, sub)
    return frequencies


def resolve_sources(
    datasets: Sequence[DatasetId], files: Iterable[HubFile], known: Collection[str]
) -> tuple[DatasetSource, ...]:
    """Find the shards of each dataset id in a file listing.

    Args:
        datasets: The dataset ids.
        files: A listing of shard files, from the Hub or from the local copy.
        known: The accepted frequency keys.

    Returns:
        One source per dataset id, in the same order.

    Raises:
        UnknownDatasetError: A dataset name is not in the listing.
        UnknownFrequencyError: A dataset does not have the requested frequency; the message
            lists the frequencies it has.
    """
    layout = _layout(files)
    sources: list[DatasetSource] = []
    for dataset in datasets:
        directories = layout.get(dataset.name)
        if directories is None:
            msg = f"dataset {dataset.name!r} is not in the listing; names: {sorted(layout)}"
            raise UnknownDatasetError(msg)
        frequencies = available_frequencies(dataset.name, directories, known)
        if dataset.freq not in frequencies:
            msg = (
                f"{dataset} does not exist: {dataset.name} has the frequencies "
                f"{sorted(frequencies)}"
            )
            raise UnknownFrequencyError(msg)
        sub = frequencies[dataset.freq]
        directory = dataset.name if sub is None else f"{dataset.name}/{sub}"
        shards = tuple(sorted(directories[sub], key=lambda file: file.path))
        sources.append(DatasetSource(dataset=dataset, directory=directory, shards=shards))
    return tuple(sources)


def find_local_sources(
    datasets: Sequence[DatasetId], root: Path, known: Collection[str]
) -> dict[DatasetId, DatasetSource]:
    """Return the dataset ids whose complete shards are already under ``root/hub``.

    Args:
        datasets: The dataset ids to look for.
        root: The data root.
        known: The accepted frequency keys.

    Returns:
        A map from dataset id to its local source, for the ids found complete.
    """
    files = list_local_files(root / HUB_DIR)
    found: dict[DatasetId, DatasetSource] = {}
    for dataset in datasets:
        try:
            (source,) = resolve_sources([dataset], files, known)
        except (UnknownDatasetError, UnknownFrequencyError):
            continue
        if source.complete:
            found[dataset] = source
    return found


def directory_size(root: Path) -> int:
    """Return the total size in bytes of the files under ``root``, 0 if it does not exist."""
    if not root.is_dir():
        return 0
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


@dataclass(frozen=True)
class DownloadPlan:
    """What a download adds under ``root`` and whether it fits the budget.

    Attributes:
        repo_id: The Hub dataset repository.
        root: The data root; shards go to ``root/hub``.
        sources: The shards of each selected dataset id.
        missing: The shards not yet under ``root/hub`` with the Hub size.
        existing_bytes: Bytes under ``root`` before the download.
        budget_bytes: ``max_total_gb`` in bytes.
    """

    repo_id: str
    root: Path
    sources: tuple[DatasetSource, ...]
    missing: tuple[HubFile, ...]
    existing_bytes: int
    budget_bytes: int

    @property
    def download_bytes(self) -> int:
        """Return the bytes the download adds."""
        return sum(file.size for file in self.missing)

    @property
    def total_bytes(self) -> int:
        """Return the bytes under ``root`` after the download."""
        return self.existing_bytes + self.download_bytes

    @property
    def within_budget(self) -> bool:
        """Return True when the total stays inside the budget."""
        return self.total_bytes <= self.budget_bytes

    def check_budget(self) -> None:
        """Raise when the total exceeds the budget.

        Raises:
            BudgetExceededError: The download plus the bytes under ``root`` exceed the budget.
        """
        if not self.within_budget:
            msg = (
                f"the download needs {self.download_bytes / BYTES_PER_GB:.3f} GB and "
                f"{self.root} holds {self.existing_bytes / BYTES_PER_GB:.3f} GB: the total "
                f"{self.total_bytes / BYTES_PER_GB:.3f} GB exceeds max_total_gb = "
                f"{self.budget_bytes / BYTES_PER_GB:.3f} GB"
            )
            raise BudgetExceededError(msg)

    def describe(self) -> list[str]:
        """Return one line per dataset id with its size, then the totals."""
        missing = set(self.missing)
        lines = []
        for source in self.sources:
            state = "to download" if any(shard in missing for shard in source.shards) else "present"
            lines.append(f"{source.dataset!s:<36} {source.size / 1e6:>10.2f} MB  {state}")
        selected = sum(source.size for source in self.sources)
        lines.extend(
            [
                f"{'selected datasets':<36} {selected / 1e6:>10.2f} MB",
                f"{'to download':<36} {self.download_bytes / 1e6:>10.2f} MB",
                f"{'already under ' + str(self.root):<36} {self.existing_bytes / 1e6:>10.2f} MB",
                (
                    f"{'total after download':<36} {self.total_bytes / 1e6:>10.2f} MB"
                    f"  (budget {self.budget_bytes / BYTES_PER_GB:.2f} GB"
                    f"{'' if self.within_budget else ', EXCEEDED'})"
                ),
            ]
        )
        return lines


def plan_download(
    *,
    repo_id: str,
    root: Path,
    datasets: Sequence[DatasetId],
    max_total_gb: float,
    known: Collection[str],
    files: Iterable[HubFile] | None = None,
    enforce_budget: bool = True,
) -> DownloadPlan:
    """Plan the download of the shards of ``datasets`` to ``root/hub``.

    Args:
        repo_id: The Hub dataset repository.
        root: The data root.
        datasets: The dataset ids to download.
        max_total_gb: The budget for everything under ``root``, in GB (10^9 bytes).
        known: The accepted frequency keys.
        files: The Hub listing; None lists the repository through the Hub API.
        enforce_budget: Raise when the plan exceeds the budget.

    Returns:
        The plan.

    Raises:
        BudgetExceededError: ``enforce_budget`` is set and the plan exceeds the budget.
    """
    listing = list_hub_files(repo_id) if files is None else tuple(files)
    sources = resolve_sources(datasets, listing, known)
    hub_root = root / HUB_DIR
    missing = dict.fromkeys(
        shard
        for source in sources
        for shard in source.shards
        if not _is_present(hub_root / shard.path, shard.size)
    )
    plan = DownloadPlan(
        repo_id=repo_id,
        root=root,
        sources=sources,
        missing=tuple(missing),
        existing_bytes=directory_size(root),
        budget_bytes=int(max_total_gb * BYTES_PER_GB),
    )
    if enforce_budget:
        plan.check_budget()
    return plan


def _is_present(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def download(plan: DownloadPlan) -> None:
    """Download the missing shards of ``plan`` to ``root/hub``.

    Raises:
        BudgetExceededError: The plan exceeds the budget.
    """
    plan.check_budget()
    if not plan.missing:
        return
    _log.info(
        "downloading %d shards, %.1f MB, from %s",
        len(plan.missing),
        plan.download_bytes / 1e6,
        plan.repo_id,
    )
    snapshot_download(
        plan.repo_id,
        repo_type="dataset",
        allow_patterns=[file.path for file in plan.missing],
        local_dir=plan.root / HUB_DIR,
    )
