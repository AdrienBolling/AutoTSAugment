"""Experiment contract: a run that produces one metrics file."""

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path


class Experiment(ABC):
    """One reproducible run. Hydra instantiates it from ``configs/experiment/*.yaml``."""

    @abstractmethod
    def run(self) -> dict[str, float]:
        """Execute the experiment and return its summary metrics."""


def write_metrics(path: Path, metrics: Mapping[str, float]) -> None:
    """Write ``metrics`` as sorted JSON, the format ``dvc metrics`` reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(metrics.items())), indent=2) + "\n")
