"""Command-line entry point: ``autotsaugment-run experiment=... agent=... env=...``."""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from autotsaugment.core.config import RunConfig
from autotsaugment.core.experiment import Experiment, write_metrics

_log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parents[2] / "configs")


def validate(cfg: DictConfig) -> RunConfig:
    """Validate the run-level leaves of the composed config."""
    return RunConfig(seed=cfg.seed, device=cfg.device, output_dir=cfg.get("output_dir", "."))


@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="run")
def main(cfg: DictConfig) -> None:
    """Compose the run config, build the experiment, run it, write its metrics."""
    run = validate(cfg)
    _log.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    experiment = hydra.utils.instantiate(cfg.experiment, run_config=cfg, _recursive_=False)
    if not isinstance(experiment, Experiment):
        msg = f"cfg.experiment must build an Experiment, got {type(experiment).__name__}"
        raise TypeError(msg)
    metrics = experiment.run()
    write_metrics(Path(run.output_dir) / "metrics.json", metrics)
    _log.info("metrics: %s", metrics)


if __name__ == "__main__":
    main()
