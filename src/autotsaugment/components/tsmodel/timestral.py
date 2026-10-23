"""Timestral forecasters: the bare architecture from random init, and LoRA on pretrained weights.

Both variants share :class:`TimestralBase`: training windows cut on the device, AdamW, bf16
autocast, gradient clipping, a warmup and cosine schedule over the episode, and an opt-in
``torch.compile`` of the loss. One training step and one forecast never synchronize the host
with the device (see :mod:`autotsaugment.components.tsmodel.network`).

Loss space
----------
Decision: the pinball loss on the quantile knots is computed in the normalized space of the
model, not in raw space.

``TimestralModel.scale`` normalizes each row with causal per-patch statistics: the mean and
standard deviation of every observed value up to the end of the patch, then ``asinh``. The
head forecasts in this normalized space. ``_predict_context_only`` and ``_predict_window``
denormalize the whole block with the frame of the last context patch (the anchor), and
``TimestralForecaster._refine_block`` then moves horizon patch ``j > 0`` to the "advancing
refine frame it was trained in": the statistics of the context plus the medians of patches
``0..j-1``. With the true values in place of the medians, this is the causal statistics of
the true series up to the end of patch ``L + j - 1``. All of these statistics come from the
series itself, so the targets map into the same space: during training the horizon is known,
and :func:`autotsaugment.components.tsmodel.loss.horizon_frames` computes the frame of every
horizon patch from the true window. :meth:`TimestralBase.forecast` inverts the frames with the
forecast medians, as the forecaster does.

Reasons:

1. The loss does not change when a series is scaled or shifted, so each series has the same
   weight in the gradient. KernelSynth draws have unit scale, but real data and the later
   real-data scorer do not.
2. The pretrained weights of the LoRA variant were trained in this frame. Fine-tuning starts
   from their own objective, and the zero-shot forecast of the checkpoint stays valid.
3. ``asinh`` grows like a logarithm, so outliers have a bounded effect on the loss.

The bare variant uses the same frame, so both variants have one loss and one forecast path.
"""

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Self

import torch
from pydantic import Field, field_validator, model_validator
from timestral.config import TimestralConfig
from timestral.loading import load_model
from timestral.model import TimestralModel
from torch import Tensor, nn

from autotsaugment.components.tsmodel.lora import inject_lora, reset_adapters
from autotsaugment.components.tsmodel.loss import normalized_pinball_loss
from autotsaugment.components.tsmodel.network import UnivariateDecoder
from autotsaugment.components.tsmodel.schedule import build_scheduler
from autotsaugment.components.tsmodel.windows import pad_context, random_windows
from autotsaugment.core.tsmodel import TrainStats, TSModel, TSModelConfig
from autotsaugment.core.typechecking import Context, QuantileLevels, Quantiles, SeriesBatch, checked

# --- configs -------------------------------------------------------------------------------


def _plain(value: object) -> object:
    """Return ``value`` with every mapping as a dict and every sequence as a list.

    Hydra passes OmegaConf containers; the architecture must hold plain Python values.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain(item) for item in value]
    return value


class TimestralConfigBase(TSModelConfig):
    """Training config shared by the Timestral variants.

    Attributes:
        warmup_fraction: Share of the episode steps used for the linear warmup.
        min_lr_fraction: Learning rate at the end of the cosine decay, as a share of the peak.
    """

    warmup_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    min_lr_fraction: float = Field(default=0.1, ge=0.0, le=1.0)

    def check_windows(self, *, patch_size: int, block_size: int) -> None:
        """Check the training windows against the architecture.

        Args:
            patch_size: Time steps per patch.
            block_size: Maximum patches decoded in one block.

        Raises:
            ValueError: If a window length is not a positive multiple of ``patch_size``, or
                the horizon is longer than one block.
        """
        for name in ("context_length", "horizon_length"):
            value = getattr(self, name)
            if value < patch_size or value % patch_size:
                msg = f"{name}={value} must be a positive multiple of patch_size={patch_size}"
                raise ValueError(msg)
        if self.horizon_length > block_size * patch_size:
            msg = (
                f"horizon_length={self.horizon_length} is longer than one decode block, "
                f"block_size * patch_size = {block_size * patch_size}"
            )
            raise ValueError(msg)


class TimestralBareConfig(TimestralConfigBase):
    """Config of :class:`TimestralBare`.

    Attributes:
        architecture: Fields of :class:`timestral.TimestralConfig`, as in the ``config.json``
            of a checkpoint.
    """

    architecture: dict[str, Any]

    @field_validator("architecture", mode="before")
    @classmethod
    def _plain_architecture(cls, value: object) -> object:
        return _plain(value)

    @model_validator(mode="after")
    def _check(self) -> Self:
        architecture = self.timestral_config
        self.check_windows(patch_size=architecture.patch_size, block_size=architecture.block_size)
        return self

    @property
    def timestral_config(self) -> TimestralConfig:
        """Return the architecture as a :class:`timestral.TimestralConfig`."""
        return TimestralConfig.from_dict(self.architecture)


class TimestralLoraConfig(TimestralConfigBase):
    """Config of :class:`TimestralLora`.

    Attributes:
        checkpoint: Hugging Face repo id or local directory with ``config.json`` and
            ``model.safetensors``.
        lora_rank: LoRA rank r.
        lora_alpha: LoRA scale numerator; the update is scaled by ``lora_alpha / lora_rank``.
        lora_dropout: Dropout on the adapter input. Must be 0: dropout would draw from the
            global random stream, and every random draw must come from the passed generator.
        lora_targets: Names of the linear modules that get an adapter.
        fully_trainable: Names of the modules trained in full (peft ``modules_to_save``).
        patch_size: Patch size of the checkpoint, to check the windows before the download.
        block_size: Block size of the checkpoint, to check the windows before the download.
    """

    checkpoint: str = "mistralai/Timestral-5M"
    lora_rank: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = Field(default=("wQKV", "wO", "w12", "w3"), min_length=1)
    fully_trainable: tuple[str, ...] = ("head", "representation")
    patch_size: int = Field(default=32, ge=1)
    block_size: int = Field(default=32, ge=1)

    @field_validator("lora_dropout")
    @classmethod
    def _no_dropout(cls, value: float) -> float:
        if value != 0.0:
            msg = "lora_dropout must be 0: dropout draws from the global random stream"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _check(self) -> Self:
        self.check_windows(patch_size=self.patch_size, block_size=self.block_size)
        return self


# --- models --------------------------------------------------------------------------------


class TimestralBase[ConfigT: TimestralConfigBase](TSModel[ConfigT]):
    """Training and forecasting shared by the Timestral variants.

    Subclasses build the network, call ``super().__init__`` and implement :meth:`reset`. The
    parameters that require gradients are the trainable state.

    Args:
        config: The training config.
        model: The network, on ``device``.
        device: Device of the parameters and of every returned tensor.
        episode_epochs: Epochs per episode; the learning-rate schedule spans them.
    """

    def __init__(
        self,
        *,
        config: ConfigT,
        model: TimestralModel,
        device: torch.device,
        episode_epochs: int,
    ) -> None:
        self.config = config
        self._device = device
        self._episode_epochs = episode_epochs
        self._model = model
        self._decoder = UnivariateDecoder(model, device)
        self._levels = torch.tensor(model.config.quantile_knots, dtype=torch.float32, device=device)
        self._loss_fn: Callable[[Tensor], Tensor] = (
            torch.compile(self._batch_loss) if config.compile else self._batch_loss
        )
        self._optimizer = self._new_optimizer()
        self._scheduler: torch.optim.lr_scheduler.LambdaLR | None = None

    @property
    def device(self) -> torch.device:
        """Return the device of the parameters."""
        return self._device

    @property
    def network(self) -> TimestralModel:
        """Return the Timestral network."""
        return self._model

    @property
    def quantile_levels(self) -> QuantileLevels:
        """Return the quantile knots of the network, shape ``(q,)``."""
        return self._levels

    def _trainable(self) -> Iterator[tuple[str, nn.Parameter]]:
        return ((name, p) for name, p in self._model.named_parameters() if p.requires_grad)

    def _new_optimizer(self) -> torch.optim.AdamW:
        """Return AdamW with weight decay on the matrices only, not on norms and biases."""
        decay = [p for _, p in self._trainable() if p.ndim > 1]
        no_decay = [p for _, p in self._trainable() if p.ndim <= 1]
        groups = [
            {"params": params, "weight_decay": weight_decay}
            for params, weight_decay in ((decay, self.config.weight_decay), (no_decay, 0.0))
            if params
        ]
        return torch.optim.AdamW(
            groups, lr=self.config.learning_rate, fused=self._device.type == "cuda"
        )

    def _restart_optimizer(self) -> None:
        """Drop the optimizer state and the schedule. Call after the weights are reset."""
        self._optimizer = self._new_optimizer()
        self._scheduler = None

    def _autocast(self) -> torch.autocast:
        enabled = self.config.mixed_precision and self._device.type == "cuda"
        return torch.autocast(device_type=self._device.type, dtype=torch.bfloat16, enabled=enabled)

    def _batch_loss(self, window: Tensor) -> Tensor:
        """Return the mean normalized pinball loss of a batch of full windows."""
        ps = self._model.config.patch_size
        context_patches = self.config.context_length // ps
        observed = torch.isfinite(window)
        values = torch.where(observed, window, torch.zeros_like(window))
        split = context_patches * ps
        quantiles, _, _ = self._decoder.normalized(
            values[:, :split], observed[:, :split], self.config.horizon_length // ps
        )
        return normalized_pinball_loss(
            quantiles,
            values,
            observed,
            self._levels,
            context_patches=context_patches,
            patch_size=ps,
            minimum_scale=self._model.config.minimum_scale,
            use_asinh=self._model.config.use_asinh,
        )

    @checked
    def train_epoch(self, series: SeriesBatch, *, generator: torch.Generator) -> TrainStats:
        """Train for one epoch on one random window per series.

        Args:
            series: Synthetic series, shape ``(n, t)``, with
                ``t >= context_length + horizon_length``, on :attr:`device`.
            generator: Device generator for window positions and shuffling.

        Returns:
            The epoch statistics as 0-d tensors on :attr:`device`.
        """
        config = self.config
        length = config.context_length + config.horizon_length
        windows = random_windows(series.to(torch.float32), length, generator=generator)
        n = windows.shape[0]
        if self._scheduler is None:
            steps_per_epoch = math.ceil(n / config.batch_size)
            self._scheduler = build_scheduler(
                self._optimizer,
                total_steps=self._episode_epochs * steps_per_epoch,
                warmup_fraction=config.warmup_fraction,
                min_ratio=config.min_lr_fraction,
            )
        params = [p for _, p in self._trainable()]
        max_norm = config.grad_clip if config.grad_clip is not None else math.inf
        self._model.train()
        losses: list[Tensor] = []
        norms: list[Tensor] = []
        for start in range(0, n, config.batch_size):
            with self._autocast():
                loss = self._loss_fn(windows[start : start + config.batch_size])
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norms.append(torch.nn.utils.clip_grad_norm_(params, max_norm))
            self._optimizer.step()
            self._scheduler.step()
            losses.append(loss.detach())
        stacked = torch.stack(losses)
        lr = self._optimizer.param_groups[0]["lr"]
        return TrainStats(
            loss=stacked.mean(),
            loss_last=stacked[-1],
            grad_norm=torch.stack(norms).mean(),
            lr=torch.full((), lr, dtype=torch.float32, device=self._device),
            n_steps=torch.full((), len(losses), dtype=torch.float32, device=self._device),
        )

    @checked
    def forecast(self, context: Context, horizon: int) -> Quantiles:
        """Return quantile forecasts without gradient, shape ``(n, horizon, q)``.

        Args:
            context: Observed context, shape ``(n, c)``. NaN marks a missing value, for
                example the left padding of a short context.
            horizon: Number of steps to forecast, at least 1.

        Raises:
            ValueError: If ``horizon`` is smaller than 1.
        """
        if horizon < 1:
            msg = f"horizon must be at least 1, got {horizon}"
            raise ValueError(msg)
        inputs, observed = pad_context(context.to(torch.float32), self._model.config.patch_size)
        self._model.eval()
        with torch.no_grad(), self._autocast():
            quantiles = self._decoder.forecast(inputs, observed, horizon)
        return quantiles.to(torch.float32)

    def state_dict(self) -> dict[str, Any]:
        """Return a copy of the parameters that require gradients, by name."""
        return {name: p.detach().clone() for name, p in self._trainable()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Copy the values of :meth:`state_dict` into the trainable parameters.

        Raises:
            KeyError: If the names differ from the trainable parameters.
        """
        params = dict(self._trainable())
        if set(params) != set(state):
            missing = sorted(set(params) - set(state))
            unexpected = sorted(set(state) - set(params))
            msg = f"state mismatch: missing {missing}, unexpected {unexpected}"
            raise KeyError(msg)
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(state[name])

    def n_trainable_parameters(self) -> int:
        """Return the number of parameters the optimizer updates."""
        return sum(p.numel() for _, p in self._trainable())


class TimestralBare(TimestralBase[TimestralBareConfig]):
    """The Timestral architecture trained from a random init every episode.

    Args:
        config: The architecture and the training config.
        device: Device of the parameters and of every returned tensor.
        episode_epochs: Epochs per episode; the learning-rate schedule spans them.
    """

    def __init__(
        self, *, config: TimestralBareConfig, device: torch.device | str, episode_epochs: int
    ) -> None:
        device = torch.device(device)
        self._architecture = config.timestral_config
        model = TimestralModel(self._architecture).to(device)
        model.transformer.rope.rebuild_inv_freq()
        super().__init__(config=config, model=model, device=device, episode_epochs=episode_epochs)
        self.reset(seed=0)

    def reset(self, *, seed: int) -> None:
        """Re-initialize every parameter from ``seed`` and restart the optimizer and schedule.

        The new values come from the init of :class:`timestral.TimestralModel`, run on the
        CPU with the CPU generator seeded with ``seed``, so they do not depend on the device.
        The global random state is restored afterwards.
        """
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            fresh = TimestralModel(self._architecture)
        self._model.load_state_dict(fresh.state_dict(), strict=True)
        self._restart_optimizer()


class TimestralLora(TimestralBase[TimestralLoraConfig]):
    """Pretrained Timestral weights with LoRA adapters trained every episode.

    The checkpoint loads once. A CPU copy of the full state (base weights and the copies of
    the fully trainable modules) restores it at every :meth:`reset`.

    Args:
        config: The checkpoint, the adapters and the training config.
        device: Device of the parameters and of every returned tensor.
        episode_epochs: Epochs per episode; the learning-rate schedule spans them.

    Raises:
        ValueError: If the checkpoint has another patch size or block size than the config.
    """

    def __init__(
        self, *, config: TimestralLoraConfig, device: torch.device | str, episode_epochs: int
    ) -> None:
        device = torch.device(device)
        model = load_model(config.checkpoint, device=device)
        architecture = model.config
        if (architecture.patch_size, architecture.block_size) != (
            config.patch_size,
            config.block_size,
        ):
            msg = (
                f"checkpoint {config.checkpoint!r} has patch_size={architecture.patch_size} and "
                f"block_size={architecture.block_size}; the config expects "
                f"{config.patch_size} and {config.block_size}"
            )
            raise ValueError(msg)
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            inject_lora(
                model,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                targets=config.lora_targets,
                fully_trainable=config.fully_trainable,
            )
        self._initial_state = {
            name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()
        }
        super().__init__(config=config, model=model, device=device, episode_epochs=episode_epochs)
        self.reset(seed=0)

    def reset(self, *, seed: int) -> None:
        """Restore the checkpoint, re-initialize the adapters from ``seed``, restart the optimizer.

        After a reset, ``lora_B`` is zero, so the network forecasts as the checkpoint does.
        """
        self._model.load_state_dict(self._initial_state, strict=True)
        reset_adapters(self._model, seed)
        self._restart_optimizer()
