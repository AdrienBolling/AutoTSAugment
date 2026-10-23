"""Config bases shared by all components.

Every component owns a frozen Pydantic config. Hydra builds the config object through its
``_target_`` key, so a typo in a YAML file fails at startup, before any GPU work.
"""

from pydantic import BaseModel, ConfigDict, Field


class ComponentConfig(BaseModel):
    """Base class of all component configs: immutable, no unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class EpisodeConfig(ComponentConfig):
    """Budget of one episode.

    Attributes:
        epochs: Steps per episode, E. One step trains the tsmodel for one epoch.
        series_per_epoch: Synthetic series generated and trained on per step, N.
        batch_envs: Independent tsmodels trained in lockstep on one device, K.
    """

    epochs: int = Field(default=50, ge=1)
    series_per_epoch: int = Field(default=4096, ge=1)
    batch_envs: int = Field(default=4, ge=1)


class RunConfig(ComponentConfig):
    """Leaves of ``configs/run.yaml`` that do not belong to one component.

    Attributes:
        seed: Root seed. Every component derives its own seed from it.
        device: Torch device string of the whole run.
        output_dir: Directory of the run, set by Hydra.
    """

    seed: int = 0
    device: str = "cuda"
    output_dir: str = "."
