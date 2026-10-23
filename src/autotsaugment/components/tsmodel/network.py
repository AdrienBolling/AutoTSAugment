"""Forward pass and decoding of a Timestral network for univariate rows, on the device.

:meth:`timestral.TimestralModel.predict_block` and :class:`timestral.TimestralForecaster`
synchronize the host with the device on every call: ``torch.unique`` on the group ids,
the tensor that :func:`timestral.model.monotone_quantiles` builds from a Python list, and the
NaN check of the forecaster. :class:`UnivariateDecoder` runs the same computation for rows
that are each their own series (``group_ids = arange(R)``), with the knot widths cached on
the device, so a training step or a forecast never waits for the device.

The decoder calls the submodules of the network (``representation``, ``transformer``,
``head.projection``), so adapters that peft injects into these submodules take part.
"""

import math

import torch
from timestral.layers import TwoStreamMask
from timestral.masks import (
    build_forecast_positions,
    build_forecast_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
)
from timestral.model import TimestralModel
from timestral.scaler import advancing_block_frames, causal_patch_loc_scale, denormalize_quantiles
from torch import Tensor

QUANTILE_CAP_K = 1e4
"""Forecasts stay in ``[min - k * s, max + k * s]`` of the context, ``s`` the anchor scale.

The default of :class:`timestral.TimestralForecaster`.
"""

EXTREME_VALUE = 1e30
"""A fed-back median at or above this magnitude becomes 0, as in the forecaster."""


def knot_widths(levels: tuple[float, ...]) -> list[float]:
    """Return the probability gap of each knot to the next knot toward the median.

    Args:
        levels: Increasing quantile levels that hold 0.5.

    Returns:
        One width per level, 0 for the median.
    """
    median = levels.index(0.5)
    widths = [0.0] * len(levels)
    for i in range(median + 1, len(levels)):
        widths[i] = levels[i] - levels[i - 1]
    for i in range(median):
        widths[i] = levels[i + 1] - levels[i]
    return widths


class UnivariateDecoder:
    """Normalized forward pass, training loss input and forecasts of a Timestral network.

    Args:
        model: The network. The decoder keeps a reference, so in-place weight updates apply.
        device: Device of the parameters.
    """

    def __init__(self, model: TimestralModel, device: torch.device) -> None:
        self.model = model
        config = model.config
        self.patch_size = config.patch_size
        self.num_quantiles = config.num_quantiles
        self.median_index = config.median_index
        self.width_factor = config.quantile_width_factor
        self.widths = torch.tensor(
            knot_widths(config.quantile_knots), dtype=torch.float32, device=device
        )

    def _monotone(self, raw: Tensor) -> Tensor:
        """Return non-crossing quantiles, as :func:`timestral.model.monotone_quantiles`."""
        m = self.median_index
        increments = (
            self.width_factor * self.widths.to(raw.dtype) * torch.nn.functional.softplus(raw)
        )
        median = raw[..., m : m + 1]
        up = torch.cumsum(increments[..., m + 1 :], dim=-1)
        down = torch.flip(torch.cumsum(torch.flip(increments[..., :m], dims=[-1]), dim=-1), [-1])
        big = torch.finfo(raw.dtype).max
        return torch.cat([median - down, median, median + up], dim=-1).clamp(min=-big, max=big)

    def normalized(
        self, inputs: Tensor, observed: Tensor, horizon_patches: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the network on a context and return normalized horizon quantiles.

        Args:
            inputs: Context values, shape ``(R, T)``, ``T`` a multiple of the patch size,
                0 where unobserved.
            observed: Observation mask, shape ``(R, T)``.
            horizon_patches: Patches to forecast, H.

        Returns:
            ``(quantiles, loc, scale)``: quantiles in the normalized space of each horizon
            patch, shape ``(R, H * patch_size, Q)``, at least float32; the causal ``loc`` and
            ``scale`` of the context, each shape ``(R, T)``.
        """
        model = self.model
        config = model.config
        ps = self.patch_size
        rows, steps = inputs.shape
        patches = steps // ps
        device = inputs.device

        scaled, loc, scale = model.scale(inputs, observed)
        enc = model.representation(scaled, observed)
        dec = model.representation.forecast_tokens(rows, horizon_patches)

        obs_per_patch = observed.unflatten(-1, (patches, ps)).any(dim=-1)
        key_valid = compute_enc_key_validity(obs_per_patch, config.attend_missing_patches)
        time_mask = build_forecast_time_mask(patches, horizon_patches, key_valid, device)
        group_ids = torch.arange(rows, device=device)
        variate_mask = build_two_stream_variate_mask(group_ids, key_valid, horizon_patches)
        first_observed = obs_per_patch.to(torch.float32).argmax(dim=-1)
        positions = build_forecast_positions(patches, horizon_patches, first_observed)

        mask: Tensor | TwoStreamMask = time_mask
        if config.split_context_horizon_attn:
            mask = TwoStreamMask(mask=time_mask, enc_len=patches)
        hidden = model.transformer(torch.cat([enc, dec], dim=1), mask, variate_mask, positions)
        raw = model.head.projection(hidden[:, patches:, :])
        raw = raw.reshape(rows, horizon_patches, ps, self.num_quantiles)
        raw = raw.to(torch.promote_types(raw.dtype, torch.float32))
        quantiles = self._monotone(raw)
        return quantiles.reshape(rows, horizon_patches * ps, self.num_quantiles), loc, scale

    def decode_block(self, inputs: Tensor, observed: Tensor, horizon_patches: int) -> Tensor:
        """Forecast one block and return raw-space quantiles in the advancing refine frame.

        Patch 0 uses the frame of the last context patch. Patch ``j`` uses the statistics
        of the context plus the forecast medians of patches ``0..j-1``, as
        :meth:`timestral.TimestralForecaster._refine_block`.

        Args:
            inputs: Context values, shape ``(R, T)``, 0 where unobserved.
            observed: Observation mask, shape ``(R, T)``.
            horizon_patches: Patches to forecast, H.

        Returns:
            Quantiles, shape ``(R, H * patch_size, Q)``, non-decreasing along Q.
        """
        config = self.model.config
        ps = self.patch_size
        quantiles, loc, scale = self.normalized(inputs, observed, horizon_patches)
        anchor_loc, anchor_scale = loc[:, -1:], scale[:, -1:]
        if horizon_patches > 1:
            median = quantiles[..., self.median_index].unflatten(-1, (horizon_patches, ps))
            median = median.to(torch.float64)
            unsquashed = median.sinh() if config.use_asinh else median
            frame_loc, frame_scale = advancing_block_frames(
                inputs, observed, unsquashed, config.minimum_scale
            )
            first = torch.arange(horizon_patches, device=inputs.device) == 0
            frame_loc = torch.where(first, anchor_loc, frame_loc.to(quantiles.dtype))
            frame_scale = torch.where(first, anchor_scale, frame_scale.to(quantiles.dtype))
            anchor_loc = frame_loc.repeat_interleave(ps, dim=-1)
            anchor_scale = frame_scale.repeat_interleave(ps, dim=-1)
        return denormalize_quantiles(
            quantiles, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), config.use_asinh
        )

    def forecast(self, inputs: Tensor, observed: Tensor, horizon: int) -> Tensor:
        """Forecast ``horizon`` steps, block by block, and cap extreme values.

        Each block has at most ``block_size`` patches. After a block, its median joins the
        context as observed values. The result matches the raw forecast of
        :class:`timestral.TimestralForecaster` without the symmetric averaging and the
        non-negative clipping.

        Args:
            inputs: Context values, shape ``(R, T)``, ``T`` a multiple of the patch size,
                0 where unobserved.
            observed: Observation mask, shape ``(R, T)``.
            horizon: Steps to forecast, at least 1.

        Returns:
            Quantiles, shape ``(R, horizon, Q)``.
        """
        config = self.model.config
        ps = self.patch_size
        keep = max(ps, (config.context_length // ps) * ps)
        inputs, observed = inputs[:, -keep:], observed[:, -keep:]
        context, mask = inputs, observed
        remaining = math.ceil(horizon / ps)
        blocks: list[Tensor] = []
        while True:
            block = min(config.block_size, remaining)
            quantiles = self.decode_block(context, mask, block)
            blocks.append(quantiles)
            remaining -= block
            if remaining <= 0:
                break
            median = quantiles[..., self.median_index]
            feedback = torch.where(median.abs() < EXTREME_VALUE, median, torch.zeros_like(median))
            context = torch.cat([context, feedback], dim=-1)[:, -keep:]
            mask = torch.cat([mask, torch.ones_like(feedback, dtype=torch.bool)], dim=-1)[:, -keep:]
        forecast = torch.cat(blocks, dim=1)[:, :horizon]
        return self._cap(forecast, inputs, observed)

    def _cap(self, quantiles: Tensor, inputs: Tensor, observed: Tensor) -> Tensor:
        """Clamp quantiles to the context range widened by ``QUANTILE_CAP_K`` anchor scales."""
        config = self.model.config
        high = inputs.masked_fill(~observed, float("-inf")).amax(-1, keepdim=True)
        low = inputs.masked_fill(~observed, float("inf")).amin(-1, keepdim=True)
        high, low = torch.nan_to_num(high, neginf=0.0), torch.nan_to_num(low, posinf=0.0)
        _, scale = causal_patch_loc_scale(
            inputs, observed, self.patch_size, config.minimum_scale, stabilize_with_global=False
        )
        margin = scale[:, -1:] * QUANTILE_CAP_K
        return quantiles.clamp((low - margin).unsqueeze(-1), (high + margin).unsqueeze(-1))
