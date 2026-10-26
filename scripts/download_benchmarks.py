"""Plan or download GIFT-Eval datasets for the benchmark.

Examples:
    uv run python scripts/download_benchmarks.py --plan
    uv run python scripts/download_benchmarks.py ett1/H covid_deaths/D
    uv run python scripts/download_benchmarks.py --config configs/env/benchmark/gifteval_full.yaml

Without dataset ids, the script uses the ``datasets`` list of the config file. ``--plan`` prints
one line per dataset id with its size, then the totals, and downloads nothing. A download that
does not fit ``max_total_gb`` stops before it starts.
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from autotsaugment.components.data.download import BudgetExceededError, download, plan_download
from autotsaugment.components.data.gifteval import GiftEvalBenchmarkConfig

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "env" / "benchmark" / "gifteval_small.yaml"
)


def load_config(path: Path, overrides: dict[str, object]) -> GiftEvalBenchmarkConfig:
    """Return the benchmark config of a Hydra benchmark file, with ``overrides`` applied.

    Args:
        path: A file of ``configs/env/benchmark`` with a ``config:`` block.
        overrides: Config fields that replace the values of the file.
    """
    node = OmegaConf.load(path)
    if not isinstance(node, DictConfig) or "config" not in node:
        msg = f"{path} has no 'config' block"
        raise ValueError(msg)
    fields = OmegaConf.to_container(node.config, resolve=True)
    if not isinstance(fields, dict):
        msg = f"the 'config' block of {path} is not a mapping"
        raise TypeError(msg)
    values = {str(key): value for key, value in fields.items() if key != "_target_"}
    return GiftEvalBenchmarkConfig.model_validate(values | overrides)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit code."""
    parser = argparse.ArgumentParser(description="Plan or download GIFT-Eval datasets.")
    parser.add_argument("datasets", nargs="*", help="dataset ids '<name>/<freq>'")
    parser.add_argument("--plan", action="store_true", help="print the sizes, download nothing")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="benchmark config")
    parser.add_argument("--root", type=Path, help="data root, default: the config value")
    parser.add_argument("--max-total-gb", type=float, help="budget, default: the config value")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("autotsaugment").setLevel(logging.INFO)

    overrides: dict[str, object] = {}
    if args.datasets:
        overrides["datasets"] = tuple(args.datasets)
    if args.root is not None:
        overrides["root"] = args.root
    if args.max_total_gb is not None:
        overrides["max_total_gb"] = args.max_total_gb
    config = load_config(args.config, overrides)

    plan = plan_download(
        repo_id=config.repo_id,
        revision=config.revision,
        root=config.root,
        datasets=config.dataset_ids,
        max_total_gb=config.max_total_gb,
        known=config.context_by_freq.keys(),
        enforce_budget=False,
    )
    for line in plan.describe():
        print(line)
    if args.plan:
        return 0 if plan.within_budget else 1
    try:
        download(plan)
    except BudgetExceededError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    print(f"done: {len(plan.missing)} shards in {plan.hub_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
